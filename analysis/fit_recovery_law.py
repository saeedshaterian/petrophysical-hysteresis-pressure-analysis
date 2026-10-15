
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "experimental_data.csv"
OUTPUT_FILE = ROOT / "analysis" / "recovery_law_predictions.csv"
COEF_FILE = ROOT / "analysis" / "recovery_law_coefficients.csv"

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

valid_stages = [
    "initial_loading",
    "5000psi_loading",
    "7000psi_loading",
    "rest_after_5000psi",
    "rest_after_7000psi",
]

df = df[df["stage"].isin(valid_stages)].copy()

if df.empty:
    raise ValueError("No usable B9 observations found.")

if df[required].isna().any().any():
    raise ValueError("Missing values found in required columns.")

if (df["permeability_mD"] <= 0).any():
    raise ValueError("Permeability must be positive.")

if (df["rest_time_hr"] < 0).any():
    raise ValueError("Recovery time cannot be negative.")

p_min = df["pressure_psi"].min()
p_max = df["pressure_psi"].max()

if p_max == p_min:
    raise ValueError("Only one measurement pressure is available.")

df["p_norm"] = (
    (df["pressure_psi"] - p_min)
    / (p_max - p_min)
)

# Conditioning history indicators.
df["history_5000"] = df["stage"].isin([
    "5000psi_loading",
    "rest_after_5000psi",
]).astype(float)

df["history_7000"] = df["stage"].isin([
    "7000psi_loading",
    "rest_after_7000psi",
]).astype(float)

# Recovery indicators distinguish recovery measurements
# from the immediate post-conditioning loading-stage data.
df["recovery_5000"] = (
    df["stage"].eq("rest_after_5000psi").astype(float)
)

df["recovery_7000"] = (
    df["stage"].eq("rest_after_7000psi").astype(float)
)

# Logarithmic time transform accommodates a broad time range.
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


term_names = [
    "intercept",
    "normalized_pressure",
    "history_5000",
    "history_7000",
    "pressure_x_history_5000",
    "pressure_x_history_7000",
    "recovery_5000_offset",
    "recovery_7000_offset",
    "recovery_5000_x_log_time",
    "recovery_7000_x_log_time",
]

X = design_matrix(df)

if np.linalg.matrix_rank(X) < X.shape[1]:
    raise ValueError("Design matrix is rank-deficient.")

# Fit full-data models for coefficient inspection.
beta_phi, *_ = np.linalg.lstsq(
    X,
    df["porosity_pct"].to_numpy(),
    rcond=None,
)

beta_logk, *_ = np.linalg.lstsq(
    X,
    np.log(df["permeability_mD"].to_numpy()),
    rcond=None,
)

df["porosity_fitted_pct"] = X @ beta_phi
df["permeability_fitted_mD"] = np.exp(X @ beta_logk)


# ---------------------------------------------------------
# Leave-one-pressure-level-out validation.
# All stages and recovery times at the held-out pressure
# are excluded from training.
# ---------------------------------------------------------

prediction_parts = []
pressures = sorted(df["pressure_psi"].unique())

for held_pressure in pressures:
    train = df[
        ~np.isclose(df["pressure_psi"], held_pressure)
    ].copy()

    test = df[
        np.isclose(df["pressure_psi"], held_pressure)
    ].copy()

    X_train = design_matrix(train)
    X_test = design_matrix(test)

    if np.linalg.matrix_rank(X_train) < X_train.shape[1]:
        raise ValueError(
            f"Rank-deficient training matrix at held-out "
            f"pressure {held_pressure} psi."
        )

    phi_coef, *_ = np.linalg.lstsq(
        X_train,
        train["porosity_pct"].to_numpy(),
        rcond=None,
    )

    logk_coef, *_ = np.linalg.lstsq(
        X_train,
        np.log(train["permeability_mD"].to_numpy()),
        rcond=None,
    )

    test["porosity_predicted_pct"] = X_test @ phi_coef
    test["permeability_predicted_mD"] = np.exp(
        X_test @ logk_coef
    )

    test["porosity_residual_pp"] = (
        test["porosity_pct"]
        - test["porosity_predicted_pct"]
    )

    test["permeability_residual_mD"] = (
        test["permeability_mD"]
        - test["permeability_predicted_mD"]
    )

    test["held_out_pressure_psi"] = held_pressure
    prediction_parts.append(test)


cv = pd.concat(prediction_parts, ignore_index=True)


def report_metrics(observed, predicted):
    residual = observed - predicted
    rmse = float(np.sqrt(np.mean(residual**2)))
    mae = float(np.mean(np.abs(residual)))

    denominator = float(
        np.sum((observed - observed.mean()) ** 2)
    )

    r2 = (
        float(1 - np.sum(residual**2) / denominator)
        if denominator > 0
        else np.nan
    )

    return rmse, mae, r2


phi_metrics = report_metrics(
    cv["porosity_pct"].to_numpy(),
    cv["porosity_predicted_pct"].to_numpy(),
)

k_metrics = report_metrics(
    cv["permeability_mD"].to_numpy(),
    cv["permeability_predicted_mD"].to_numpy(),
)

# Fit quality by stage.
stage_rows = []

for stage, group in cv.groupby("stage"):
    phi_rmse = float(np.sqrt(np.mean(
        group["porosity_residual_pp"].to_numpy() ** 2
    )))

    k_rmse = float(np.sqrt(np.mean(
        group["permeability_residual_mD"].to_numpy() ** 2
    )))

    stage_rows.append({
        "stage": stage,
        "rows": len(group),
        "porosity_RMSE_pp": phi_rmse,
        "permeability_RMSE_mD": k_rmse,
    })

stage_metrics = pd.DataFrame(stage_rows)

# Save out-of-fold predictions and coefficients.
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
cv.to_csv(OUTPUT_FILE, index=False)

coefficients = pd.DataFrame({
    "term": term_names,
    "porosity_coefficient": beta_phi,
    "log_permeability_coefficient": beta_logk,
})

coefficients.to_csv(COEF_FILE, index=False)

print("=" * 68)
print("B9 PRESSURE-HISTORY-RECOVERY MODEL")
print("=" * 68)
print(f"Observations: {len(df)}")
print(f"Pressure levels: {len(pressures)}")
print(f"Stages: {df['stage'].nunique()}")
print(f"Recovery time range: {df['rest_time_hr'].min():.0f} to "
      f"{df['rest_time_hr'].max():.0f} hours")

print("\nLEAVE-ONE-PRESSURE-LEVEL-OUT VALIDATION")
print("Porosity:")
print(f"  RMSE: {phi_metrics[0]:.6f} percentage points")
print(f"  MAE:  {phi_metrics[1]:.6f} percentage points")
print(f"  R2:   {phi_metrics[2]:.6f}")

print("Permeability:")
print(f"  RMSE: {k_metrics[0]:.6f} mD")
print(f"  MAE:  {k_metrics[1]:.6f} mD")
print(f"  R2:   {k_metrics[2]:.6f}")

print("\nVALIDATION BY STAGE")
print(stage_metrics.to_string(
    index=False,
    float_format=lambda x: f"{x:.6f}",
))

print("\nFULL-DATA COEFFICIENTS")
print(coefficients.to_string(
    index=False,
    float_format=lambda x: f"{x:.8f}",
))

print(f"\nPredictions saved to: {OUTPUT_FILE}")
print(f"Coefficients saved to: {COEF_FILE}")

print(
    "\nNOTE: This is an empirical pressure-history-recovery model. "
    "It does not directly predict mechanical strain or displacement."
)
