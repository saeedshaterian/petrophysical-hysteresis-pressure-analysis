
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "experimental_data.csv"
OUT_DIR = ROOT / "analysis"

df = pd.read_csv(DATA_PATH)

P = "pressure_psi"
PHI = "porosity_pct"
K = "permeability_mD"
STAGE = "stage"
TIME = "rest_time_hr"

ANCHORS = {
    "rest_after_5000psi": "5000psi_loading",
    "rest_after_7000psi": "7000psi_loading",
}

df[P] = pd.to_numeric(df[P], errors="raise")
df[PHI] = pd.to_numeric(df[PHI], errors="raise")
df[K] = pd.to_numeric(df[K], errors="raise")
df[TIME] = pd.to_numeric(df[TIME], errors="raise")

if (df[K] <= 0).any():
    raise ValueError("All permeability values must be positive.")

pmin, pmax = df[P].min(), df[P].max()
df["pressure_norm"] = (df[P] - pmin) / (pmax - pmin)

df["h5"] = df[STAGE].isin(
    ["5000psi_loading", "rest_after_5000psi"]
).astype(float)

df["h7"] = df[STAGE].isin(
    ["7000psi_loading", "rest_after_7000psi"]
).astype(float)

df["r5"] = (df[STAGE] == "rest_after_5000psi").astype(float)
df["r7"] = (df[STAGE] == "rest_after_7000psi").astype(float)
df["log_time"] = np.log1p(df[TIME])

df["recovery_time_5000"] = df["r5"] * df["log_time"]
df["recovery_time_7000"] = df["r7"] * df["log_time"]

df["log_k"] = np.log(df[K])

stage_features = [
    "pressure_norm", "h5", "h7",
]

def make_X(frame, model):
    z = frame["pressure_norm"].to_numpy(dtype=float)
    h5 = frame["h5"].to_numpy(dtype=float)
    h7 = frame["h7"].to_numpy(dtype=float)

    columns = [
        np.ones(len(frame)),
        z,
        h5,
        h7,
        z * h5,
        z * h7,
        frame["r5"].to_numpy(dtype=float),
        frame["r7"].to_numpy(dtype=float),
    ]

    if model == "stage_plus_time":
        columns.extend([
            frame["recovery_time_5000"].to_numpy(dtype=float),
            frame["recovery_time_7000"].to_numpy(dtype=float),
        ])

    return np.column_stack(columns)


def fit_predict(X_train, y_train, X_test):
    beta, _, _, _ = np.linalg.lstsq(X_train, y_train, rcond=None)
    return X_test @ beta


def metrics(y, pred):
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    err = pred - y
    denom = np.sum((y - y.mean()) ** 2)

    return {
        "n": len(y),
        "RMSE": float(np.sqrt(np.mean(err ** 2))),
        "MAE": float(np.mean(np.abs(err))),
        "R2": float(1 - np.sum(err ** 2) / denom)
        if denom > 0 else np.nan,
    }


# Match every recovery row to the same-pressure loading anchor.
anchor_frames = {}

for recovery_stage, anchor_stage in ANCHORS.items():
    anchor = df.loc[
        df[STAGE] == anchor_stage,
        [P, PHI, K],
    ].copy()

    if anchor[P].duplicated().any():
        raise ValueError(f"Duplicate pressures in {anchor_stage}")

    anchor = anchor.rename(columns={
        PHI: "phi_anchor",
        K: "k_anchor",
    })

    recovery = df.loc[df[STAGE] == recovery_stage].copy()
    recovery = recovery.merge(
        anchor, on=P, how="left", validate="many_to_one"
    )

    if recovery["phi_anchor"].isna().any():
        raise ValueError(f"Missing porosity anchors for {recovery_stage}")

    if recovery["k_anchor"].isna().any():
        raise ValueError(f"Missing permeability anchors for {recovery_stage}")

    anchor_frames[recovery_stage] = recovery


prediction_rows = []

for recovery_stage in ANCHORS:
    stage_data = anchor_frames[recovery_stage]
    held_times = sorted(stage_data[TIME].unique())

    for held_time in held_times:
        test_mask = (
            (df[STAGE] == recovery_stage)
            & (df[TIME] == held_time)
        )

        train = df.loc[~test_mask].copy()
        test = df.loc[test_mask].copy()

        if test.empty:
            continue

        # Models fitted to the complete available training data.
        for model in ["stage_only", "stage_plus_time"]:
            X_train = make_X(train, model)
            X_test = make_X(test, model)

            phi_pred = fit_predict(
                X_train, train[PHI].to_numpy(), X_test
            )

            logk_pred = fit_predict(
                X_train, train["log_k"].to_numpy(), X_test
            )

            for i, (_, row) in enumerate(test.iterrows()):
                prediction_rows.append({
                    "stage": recovery_stage,
                    "held_out_time_hr": held_time,
                    "pressure_psi": row[P],
                    "model": model,
                    "phi_observed": row[PHI],
                    "phi_predicted": phi_pred[i],
                    "k_observed": row[K],
                    "k_predicted": np.exp(logk_pred[i]),
                })

        # Anchored model fitted separately within the recovery stage.
        recovery_train = stage_data.loc[
            stage_data[TIME] != held_time
        ].copy()

        recovery_test = stage_data.loc[
            stage_data[TIME] == held_time
        ].copy()

        def anchored_X(frame):
            z = frame["pressure_norm"].to_numpy(dtype=float)
            tau = frame["log_time"].to_numpy(dtype=float)
            return np.column_stack([tau, z * tau])

        Xtr = anchored_X(recovery_train)
        Xte = anchored_X(recovery_test)

        delta_phi = (
            recovery_train[PHI] - recovery_train["phi_anchor"]
        ).to_numpy()

        log_ratio_k = np.log(
            recovery_train[K].to_numpy()
            / recovery_train["k_anchor"].to_numpy()
        )

        phi_change_pred = fit_predict(Xtr, delta_phi, Xte)
        log_ratio_pred = fit_predict(Xtr, log_ratio_k, Xte)

        phi_anchor_pred = (
            recovery_test["phi_anchor"].to_numpy()
            + phi_change_pred
        )

        k_anchor_pred = (
            recovery_test["k_anchor"].to_numpy()
            * np.exp(log_ratio_pred)
        )

        for i, (_, row) in enumerate(recovery_test.iterrows()):
            prediction_rows.append({
                "stage": recovery_stage,
                "held_out_time_hr": held_time,
                "pressure_psi": row[P],
                "model": "anchored_time",
                "phi_observed": row[PHI],
                "phi_predicted": phi_anchor_pred[i],
                "k_observed": row[K],
                "k_predicted": k_anchor_pred[i],
            })

        # No-recovery-change reference model.
        for _, row in recovery_test.iterrows():
            prediction_rows.append({
                "stage": recovery_stage,
                "held_out_time_hr": held_time,
                "pressure_psi": row[P],
                "model": "anchor_baseline",
                "phi_observed": row[PHI],
                "phi_predicted": row["phi_anchor"],
                "k_observed": row[K],
                "k_predicted": row["k_anchor"],
            })


pred = pd.DataFrame(prediction_rows)

# Each row is an out-of-time prediction for one pressure level.
metric_rows = []

for (stage, model), group in pred.groupby(["stage", "model"]):
    for variable, observed_col, predicted_col in [
        ("porosity", "phi_observed", "phi_predicted"),
        ("permeability", "k_observed", "k_predicted"),
    ]:
        result = metrics(group[observed_col], group[predicted_col])

        metric_rows.append({
            "stage": stage,
            "model": model,
            "variable": variable,
            **result,
        })

metrics_df = pd.DataFrame(metric_rows)

pred_path = OUT_DIR / "fair_recovery_holdout_predictions.csv"
metrics_path = OUT_DIR / "fair_recovery_holdout_metrics.csv"

pred.to_csv(pred_path, index=False)
metrics_df.to_csv(metrics_path, index=False)

print("=" * 78)
print("FAIR RECOVERY MODEL COMPARISON")
print("=" * 78)
print("Validation: identical leave-one-time-level-out folds")
print("Predictions: out-of-time predictions only")
print("\nPOOLED HOLDOUT METRICS")
print(
    metrics_df.sort_values(["stage", "variable", "RMSE"]).to_string(
        index=False, float_format=lambda x: f"{x:.6f}"
    )
)

print("\nSAVED FILES")
print(pred_path)
print(metrics_path)

print("\nInterpretation: compare RMSE and MAE within each stage and variable.")
print("R2 can be negative when predictions are worse than the mean baseline.")
print("This is interpolation validation on B9, not independent sample validation.")
