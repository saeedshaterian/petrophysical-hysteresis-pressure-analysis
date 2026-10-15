
from pathlib import Path

import numpy as np
import pandas as pd


# ------------------------------------------------------------
# 1. Paths and settings
# ------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "experimental_data.csv"
OUT_DIR = ROOT / "analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PRESSURE_COL = "pressure_psi"
PHI_COL = "porosity_pct"
K_COL = "permeability_mD"
STAGE_COL = "stage"
TIME_COL = "rest_time_hr"

RECOVERY_ANCHORS = {
    "rest_after_5000psi": "5000psi_loading",
    "rest_after_7000psi": "7000psi_loading",
}

# ------------------------------------------------------------
# 2. Load and check the data
# ------------------------------------------------------------

df = pd.read_csv(DATA_PATH)

required = {
    PRESSURE_COL,
    PHI_COL,
    K_COL,
    STAGE_COL,
    TIME_COL,
}

missing = required.difference(df.columns)
if missing:
    raise ValueError(f"Missing required columns: {sorted(missing)}")

df = df.copy()
df[PRESSURE_COL] = pd.to_numeric(df[PRESSURE_COL], errors="raise")
df[PHI_COL] = pd.to_numeric(df[PHI_COL], errors="raise")
df[K_COL] = pd.to_numeric(df[K_COL], errors="raise")
df[TIME_COL] = pd.to_numeric(df[TIME_COL], errors="raise")

if (df[K_COL] <= 0).any():
    raise ValueError("Permeability must be positive for log-ratio modeling.")

p_min = df[PRESSURE_COL].min()
p_max = df[PRESSURE_COL].max()

if p_max <= p_min:
    raise ValueError("Pressure must contain more than one distinct value.")

df["pressure_norm"] = (
    (df[PRESSURE_COL] - p_min) / (p_max - p_min)
)

# ------------------------------------------------------------
# 3. Match every recovery observation to its loading anchor
# ------------------------------------------------------------

recovery_frames = []

for recovery_stage, anchor_stage in RECOVERY_ANCHORS.items():

    recovery = df.loc[
        df[STAGE_COL] == recovery_stage
    ].copy()

    anchor = df.loc[
        df[STAGE_COL] == anchor_stage,
        [PRESSURE_COL, PHI_COL, K_COL],
    ].copy()

    anchor = anchor.rename(
        columns={
            PHI_COL: "phi_anchor_pct",
            K_COL: "k_anchor_mD",
        }
    )

    if recovery.empty or anchor.empty:
        raise ValueError(
            f"Missing recovery or anchor rows: "
            f"{recovery_stage}, {anchor_stage}"
        )

    if anchor[PRESSURE_COL].duplicated().any():
        raise ValueError(
            f"Duplicate pressure levels in anchor stage: {anchor_stage}"
        )

    recovery = recovery.merge(
        anchor,
        on=PRESSURE_COL,
        how="left",
        validate="many_to_one",
    )

    if recovery["phi_anchor_pct"].isna().any():
        raise ValueError(
            f"Some recovery pressures have no matching anchor "
            f"for {recovery_stage}."
        )

    recovery["anchor_stage"] = anchor_stage
    recovery["recovery_stage"] = recovery_stage

    # Changes relative to the matching loading-stage measurement
    recovery["delta_phi_observed_pp"] = (
        recovery[PHI_COL] - recovery["phi_anchor_pct"]
    )

    recovery["log_k_ratio_observed"] = np.log(
        recovery[K_COL] / recovery["k_anchor_mD"]
    )

    # log1p(time) is exactly zero at t = 0
    recovery["log_time"] = np.log1p(recovery[TIME_COL])

    recovery_frames.append(recovery)

rec = pd.concat(recovery_frames, ignore_index=True)

# ------------------------------------------------------------
# 4. Anchored model
#
# delta_phi(t) = (a + b * pressure_norm) * log(1 + t)
#
# log(k(t) / k_anchor) =
#                  (c + d * pressure_norm) * log(1 + t)
#
# Both modeled changes are exactly zero at t = 0.
# ------------------------------------------------------------

def design_matrix(frame):
    z = frame["pressure_norm"].to_numpy(dtype=float)
    tau = frame["log_time"].to_numpy(dtype=float)

    return np.column_stack([
        tau,
        z * tau,
    ])


def fit_linear(X, y):
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    return beta


def predict_stage(train, test):
    X_train = design_matrix(train)
    X_test = design_matrix(test)

    phi_beta = fit_linear(
        X_train,
        train["delta_phi_observed_pp"].to_numpy(dtype=float),
    )

    logk_beta = fit_linear(
        X_train,
        train["log_k_ratio_observed"].to_numpy(dtype=float),
    )

    delta_phi_pred = X_test @ phi_beta
    log_k_ratio_pred = X_test @ logk_beta

    phi_pred = test["phi_anchor_pct"].to_numpy() + delta_phi_pred
    k_pred = (
        test["k_anchor_mD"].to_numpy()
        * np.exp(log_k_ratio_pred)
    )

    return phi_pred, k_pred, phi_beta, logk_beta


def regression_metrics(observed, predicted):
    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    errors = predicted - observed
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    mae = float(np.mean(np.abs(errors)))

    denominator = float(
        np.sum((observed - np.mean(observed)) ** 2)
    )

    r2 = (
        float(1 - np.sum(errors ** 2) / denominator)
        if denominator > 0
        else np.nan
    )

    return {
        "observations": len(observed),
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2,
    }


# ------------------------------------------------------------
# 5. Leave-one-recovery-time-level-out validation
# ------------------------------------------------------------

prediction_frames = []
metric_rows = []
coefficient_rows = []

for stage in RECOVERY_ANCHORS:

    stage_data = rec.loc[
        rec["recovery_stage"] == stage
    ].copy()

    times = sorted(stage_data[TIME_COL].unique())

    if len(times) < 2:
        raise ValueError(
            f"At least two recovery times are required for {stage}."
        )

    for held_time in times:

        train = stage_data.loc[
            stage_data[TIME_COL] != held_time
        ].copy()

        test = stage_data.loc[
            stage_data[TIME_COL] == held_time
        ].copy()

        phi_pred, k_pred, phi_beta, logk_beta = predict_stage(
            train, test
        )

        fold = test.copy()
        fold["validation_type"] = "leave_one_time_level_out"
        fold["held_out_time_hr"] = held_time

        fold["phi_predicted_pct"] = phi_pred
        fold["k_predicted_mD"] = k_pred

        # Reference model: no change from the loading anchor
        fold["phi_anchor_baseline_pred_pct"] = fold["phi_anchor_pct"]
        fold["k_anchor_baseline_pred_mD"] = fold["k_anchor_mD"]

        prediction_frames.append(fold)

        for variable, observed, predicted, baseline in [
            (
                "porosity",
                fold[PHI_COL].to_numpy(),
                phi_pred,
                fold["phi_anchor_pct"].to_numpy(),
            ),
            (
                "permeability",
                fold[K_COL].to_numpy(),
                k_pred,
                fold["k_anchor_mD"].to_numpy(),
            ),
        ]:
            for model_name, model_pred in [
                ("anchored_time_model", predicted),
                ("no_recovery_change_baseline", baseline),
            ]:
                result = regression_metrics(observed, model_pred)

                metric_rows.append({
                    "stage": stage,
                    "held_out_time_hr": held_time,
                    "variable": variable,
                    "model": model_name,
                    **result,
                })

# ------------------------------------------------------------
# 6. Fit the final anchored model to all recovery observations
# ------------------------------------------------------------

for stage in RECOVERY_ANCHORS:

    stage_data = rec.loc[
        rec["recovery_stage"] == stage
    ].copy()

    phi_pred, k_pred, phi_beta, logk_beta = predict_stage(
        stage_data, stage_data
    )

    coefficient_rows.extend([
        {
            "stage": stage,
            "target": "delta_porosity_pp",
            "coefficient": "log_time",
            "value": phi_beta[0],
        },
        {
            "stage": stage,
            "target": "delta_porosity_pp",
            "coefficient": "pressure_norm_x_log_time",
            "value": phi_beta[1],
        },
        {
            "stage": stage,
            "target": "log_permeability_ratio",
            "coefficient": "log_time",
            "value": logk_beta[0],
        },
        {
            "stage": stage,
            "target": "log_permeability_ratio",
            "coefficient": "pressure_norm_x_log_time",
            "value": logk_beta[1],
        },
    ])

# ------------------------------------------------------------
# 7. Save results
# ------------------------------------------------------------

predictions = pd.concat(prediction_frames, ignore_index=True)
metrics = pd.DataFrame(metric_rows)
coefficients = pd.DataFrame(coefficient_rows)

predictions_path = OUT_DIR / "anchored_recovery_holdout_predictions.csv"
metrics_path = OUT_DIR / "anchored_recovery_holdout_metrics.csv"
coefficients_path = OUT_DIR / "anchored_recovery_coefficients.csv"

predictions.to_csv(predictions_path, index=False)
metrics.to_csv(metrics_path, index=False)
coefficients.to_csv(coefficients_path, index=False)

# ------------------------------------------------------------
# 8. Report results
# ------------------------------------------------------------

print("=" * 76)
print("ANCHORED RECOVERY MODEL")
print("=" * 76)

print(f"Input rows: {len(df)}")
print(f"Recovery observations: {len(rec)}")
print(f"Recovery stages: {len(RECOVERY_ANCHORS)}")
print("Validation: leave one observed recovery-time level out")
print("Anchor: matching pressure measurement at the loading stage")

print("\nFOLD-LEVEL HOLDOUT METRICS")
print(
    metrics.sort_values(["stage", "variable", "model"])
    .to_string(index=False, float_format=lambda x: f"{x:.6f}")
)
print(
    "Note: Metrics from separate folds are reported individually; "
    "RMSE, MAE, and R2 are not summed across folds."
)

# Calculate pooled R2 correctly from all held-out predictions
print("\nPOOLED HOLDOUT METRICS")

for stage in RECOVERY_ANCHORS:
    part = predictions.loc[predictions["recovery_stage"] == stage]

    for variable, observed_col, pred_col, base_col in [
        (
            "porosity",
            PHI_COL,
            "phi_predicted_pct",
            "phi_anchor_baseline_pred_pct",
        ),
        (
            "permeability",
            K_COL,
            "k_predicted_mD",
            "k_anchor_baseline_pred_mD",
        ),
    ]:
        anchored = regression_metrics(
            part[observed_col], part[pred_col]
        )
        baseline = regression_metrics(
            part[observed_col], part[base_col]
        )

        print(
            f"{stage} | {variable} | "
            f"anchored RMSE={anchored['RMSE']:.6f}, "
            f"MAE={anchored['MAE']:.6f}, "
            f"R2={anchored['R2']:.6f} | "
            f"anchor baseline RMSE={baseline['RMSE']:.6f}, "
            f"MAE={baseline['MAE']:.6f}, "
            f"R2={baseline['R2']:.6f}"
        )

print("\nFULL-DATA COEFFICIENTS")
print(coefficients.to_string(index=False))

print("\nSAVED FILES")
print(predictions_path)
print(metrics_path)
print(coefficients_path)

print("\nLIMITATIONS")
print(
    "- Time zero is anchored to the loading-stage data; "
    "it is not a separate measured recovery observation."
)
print(
    "- The model assumes a log-time response with pressure-dependent "
    "coefficients and is not a mechanistic constitutive law."
)
print(
    "- Validation tests interpolation among the available time levels, "
    "not extrapolation beyond them or transfer to other rock samples."
)
