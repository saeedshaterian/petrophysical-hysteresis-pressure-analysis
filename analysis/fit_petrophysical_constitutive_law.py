
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "experimental_data.csv"
OUTPUT_FILE = ROOT / "analysis" / "petrophysical_law_predictions.csv"
COEFFICIENT_FILE = ROOT / "analysis" / "petrophysical_law_coefficients.csv"

df = pd.read_csv(DATA_FILE)

stages = [
    "initial_loading",
    "5000psi_loading",
    "7000psi_loading",
]

df = df[
    df["sample"].eq("B9")
    & df["stage"].isin(stages)
].copy()

if df.empty:
    raise ValueError("No B9 conditioning-stage data found.")

if df[["pressure_psi", "porosity_pct", "permeability_mD"]].isna().any().any():
    raise ValueError("Missing values found in the required columns.")

if (df["permeability_mD"] <= 0).any():
    raise ValueError("Permeability must be positive for log fitting.")

# Normalize pressure to improve numerical conditioning.
p_min = df["pressure_psi"].min()
p_max = df["pressure_psi"].max()

if p_max == p_min:
    raise ValueError("Pressure data contain only one distinct value.")

df["p_norm"] = (
    (df["pressure_psi"] - p_min)
    / (p_max - p_min)
)

# Stage indicators: initial_loading is the reference stage.
df["is_5000"] = df["stage"].eq("5000psi_loading").astype(float)
df["is_7000"] = df["stage"].eq("7000psi_loading").astype(float)

# Model terms:
# 1, normalized pressure, stage indicators,
# and pressure-stage interactions.
X = np.column_stack([
    np.ones(len(df)),
    df["p_norm"].to_numpy(),
    df["is_5000"].to_numpy(),
    df["is_7000"].to_numpy(),
    (df["p_norm"] * df["is_5000"]).to_numpy(),
    (df["p_norm"] * df["is_7000"]).to_numpy(),
])

if np.linalg.matrix_rank(X) < X.shape[1]:
    raise ValueError("Design matrix is rank-deficient.")

# Fit porosity directly.
y_phi = df["porosity_pct"].to_numpy()
coef_phi, *_ = np.linalg.lstsq(X, y_phi, rcond=None)
pred_phi = X @ coef_phi

# Fit logarithmic permeability to keep predictions positive.
y_logk = np.log(df["permeability_mD"].to_numpy())
coef_logk, *_ = np.linalg.lstsq(X, y_logk, rcond=None)
pred_logk = X @ coef_logk
pred_k = np.exp(pred_logk)


def metrics(observed, predicted):
    residual = observed - predicted
    rmse = float(np.sqrt(np.mean(residual**2)))
    ss_total = float(np.sum((observed - observed.mean())**2))

    r2 = (
        float(1.0 - np.sum(residual**2) / ss_total)
        if ss_total > 0
        else np.nan
    )

    return rmse, r2


phi_rmse, phi_r2 = metrics(y_phi, pred_phi)
k_rmse, k_r2 = metrics(
    df["permeability_mD"].to_numpy(),
    pred_k,
)

df["porosity_predicted_pct"] = pred_phi
df["porosity_residual_pp"] = y_phi - pred_phi

df["permeability_predicted_mD"] = pred_k
df["permeability_residual_mD"] = (
    df["permeability_mD"].to_numpy() - pred_k
)

df.to_csv(OUTPUT_FILE, index=False)

term_names = [
    "intercept",
    "normalized_pressure",
    "stage_5000_indicator",
    "stage_7000_indicator",
    "pressure_x_stage_5000",
    "pressure_x_stage_7000",
]

coefficients = pd.DataFrame({
    "term": term_names,
    "porosity_coefficient": coef_phi,
    "log_permeability_coefficient": coef_logk,
})

coefficients.to_csv(COEFFICIENT_FILE, index=False)

print("=" * 65)
print("B9 PETROPHYSICAL CONSTITUTIVE-LAW CALIBRATION")
print("=" * 65)
print(f"Data rows: {len(df)}")
print(f"Pressure range: {p_min:.0f} to {p_max:.0f} psi")
print("Model: pressure + conditioning stage + interactions")

print("\nPOROSITY MODEL")
print(f"RMSE: {phi_rmse:.6f} percentage points")
print(f"R2:   {phi_r2:.6f}")

print("\nPERMEABILITY MODEL")
print("Target: natural logarithm of permeability")
print(f"RMSE: {k_rmse:.6f} mD")
print(f"R2:   {k_r2:.6f}")

print("\nCOEFFICIENTS")
print(coefficients.to_string(index=False))

print("\nFiles saved:")
print(OUTPUT_FILE)
print(COEFFICIENT_FILE)

print(
    "\nNOTE: These are empirical petrophysical response laws, "
    "not a calibrated stress-strain constitutive law."
)
