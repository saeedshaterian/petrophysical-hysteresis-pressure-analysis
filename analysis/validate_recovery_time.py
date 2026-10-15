
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "experimental_data.csv"

OUTPUT_FILE = (
    ROOT / "analysis" / "recovery_time_holdout_predictions.csv"
)
METRICS_FILE = (
    ROOT / "analysis" / "recovery_time_holdout_metrics.csv"
)

df = pd.read_csv(DATA_FILE)

required = [
    "sample",
    "pressure_psi",
    "porosity_pct",
    "permeability_mD",
    "stage",
    "rest_time_hr",
]

missing = set(required) - set(df.columns)
if missing:
    raise ValueError(f"Missing columns: {sorted(missing)}")

df = df[df["sample"].eq("B9")].copy()

if df.empty:
    raise ValueError("No B9 data found.")

if df[required].isna().any().any():
    raise ValueError("Missing values found.")

if (df["permeability_mD"] <= 0).any():
    raise ValueError("Permeability must be positive.")

if (df["rest_time_hr"] < 0).any():
    raise ValueError("Recovery time cannot be negative.")

p_min = df["pressure_psi"].min()
p_max = df["pressure_psi"].max()

if p_max <= p_min:
    raise ValueError("Invalid pressure range.")

df["p_norm"] = (
    (df["pressure_psi"] - p_min) / (p_max - p_min)
)

df["history_5000"] = df["stage"].isin([
    "5000psi_loading",
    "rest_after_5000psi",
]).astype(float)

df["history_7000"] = df["stage"].isin([
    "7000psi_loading",
    "rest_after_7000psi",
]).astype(float)

df["recovery_5000"] = (
    df["stage"].eq("rest_after_5000psi").astype(float)
)

df["recovery_7000"] = (
    df["stage"].eq("rest_after_7000psi").astype(float)
)

df["log_time"] = np.log1p(df["rest_time_hr"])

df["recovery_time_5000"] = (
    df["recovery_5000"] * df["log_time"]
)

df["recovery_time_7000"] = (
    df["recovery_7000"] * df["log_time"]
)


def design_matrix(frame):
    p = frame["p_norm"].to_numpy()
    h5 = frame["history_5000"].to_numpy()
    h7 = frame["history_7000"].to_numpy()
    r5 = frame["recovery_5000"].to_numpy()
    r7 = frame["recovery_7000"].to_numpy()

    return np.column_stack([
        np.ones(len(frame)),
        p,
        h5,
        h7,
        p * h5,
        p * h7,
        r5,
        r7,
        frame["recovery_time_5000"].to_numpy(),
        frame["recovery_time_7000"].to_numpy(),
    ])


def fit_and_predict(train, test):
    X_train = design_matrix(train)
    X_test = design_matrix(test)

    if np.linalg.matrix_rank(X_train) < X_train.shape[1]:
        raise ValueError(
            "Training design matrix is rank-deficient. "
            "Cannot reliably fit this fold."
        )

    beta_phi, *_ = np.linalg.lstsq(
        X_train,
        train["porosity_pct"].to_numpy(),
        rcond=None,
    )

    beta_logk, *_ = np.linalg.lstsq(
        X_train,
        np.log(train["permeability_mD"].to_numpy()),
        rcond=None,
    )

    predicted_phi = X_test @ beta_phi
    predicted_k = np.exp(X_test @ beta_logk)

    return predicted_phi, predicted_k


# Hold out each observed recovery time separately.
recovery_stages = [
    "rest_after_5000psi",
    "rest_after_7000psi",
]

prediction_parts = []

for stage in recovery_stages:
    stage_times = sorted(
        df.loc[df["stage"].eq(stage), "rest_time_hr"].unique()
    )

    for held_time in stage_times:
        test_mask = (
            df["stage"].eq(stage)
            & np.isclose(df["rest_time_hr"], held_time)
        )

        train = df.loc[~test_mask].copy()
        test = df.loc[test_mask].copy()

        if test.empty:
            continue

        pred_phi, pred_k = fit_and_predict(train, test)

        test["predicted_porosity_pct"] = pred_phi
        test["predicted_permeability_mD"] = pred_k
        test["porosity_error_pp"] = (
            test["predicted_porosity_pct"] - test["porosity_pct"]
        )
        test["permeability_error_mD"] = (
            test["predicted_permeability_mD"]
            - test["permeability_mD"]
        )
        test["held_out_stage"] = stage
        test["held_out_time_hr"] = held_time

        prediction_parts.append(test)


if not prediction_parts:
    raise ValueError("No temporal validation predictions were generated.")

predictions = pd.concat(prediction_parts, ignore_index=True)

predictions.to_csv(OUTPUT_FILE, index=False)


def metrics(group, observed_col, predicted_col):
    observed = group[observed_col].to_numpy()
    predicted = group[predicted_col].to_numpy()
    residual = predicted - observed

    rmse = float(np.sqrt(np.mean(residual**2)))
    mae = float(np.mean(np.abs(residual)))

    denominator = float(np.sum((observed - observed.mean()) ** 2))
    r2 = (
        float(1.0 - np.sum(residual**2) / denominator)
        if denominator > 0
        else np.nan
    )

    return rmse, mae, r2


metric_rows = []

for stage, group in predictions.groupby("held_out_stage"):
    phi_rmse, phi_mae, phi_r2 = metrics(
        group,
        "porosity_pct",
        "predicted_porosity_pct",
    )

    k_rmse, k_mae, k_r2 = metrics(
        group,
        "permeability_mD",
        "predicted_permeability_mD",
    )

    metric_rows.append({
        "stage": stage,
        "observations": len(group),
        "porosity_RMSE_pp": phi_rmse,
        "porosity_MAE_pp": phi_mae,
        "porosity_R2": phi_r2,
        "permeability_RMSE_mD": k_rmse,
        "permeability_MAE_mD": k_mae,
        "permeability_R2": k_r2,
    })


# Overall out-of-fold metrics.
phi_rmse, phi_mae, phi_r2 = metrics(
    predictions,
    "porosity_pct",
    "predicted_porosity_pct",
)

k_rmse, k_mae, k_r2 = metrics(
    predictions,
    "permeability_mD",
    "predicted_permeability_mD",
)

metric_rows.append({
    "stage": "OVERALL",
    "observations": len(predictions),
    "porosity_RMSE_pp": phi_rmse,
    "porosity_MAE_pp": phi_mae,
    "porosity_R2": phi_r2,
    "permeability_RMSE_mD": k_rmse,
    "permeability_MAE_mD": k_mae,
    "permeability_R2": k_r2,
})

metric_df = pd.DataFrame(metric_rows)
metric_df.to_csv(METRICS_FILE, index=False)

print("=" * 72)
print("RECOVERY MODEL: LEAVE-ONE-TIME-LEVEL-OUT VALIDATION")
print("=" * 72)

print(f"Predicted recovery observations: {len(predictions)}")
print(
    "Validation times by stage:",
    predictions.groupby("held_out_stage")[
        "held_out_time_hr"
    ].nunique().to_dict(),
)

print("\nVALIDATION METRICS")
print(
    metric_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}",
    )
)

print(f"\nPredictions saved to: {OUTPUT_FILE}")
print(f"Metrics saved to: {METRICS_FILE}")

print(
    "\nNOTE: This tests interpolation among observed recovery times. "
    "It does not establish reliable extrapolation beyond the observed "
    "time range or validate the model on other rock samples."
)
