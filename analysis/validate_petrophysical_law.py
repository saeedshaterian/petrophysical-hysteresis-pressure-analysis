
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "experimental_data.csv"
OUTPUT_FILE = ROOT / "analysis" / "petrophysical_law_cv_predictions.csv"

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

if df.duplicated(["sample", "stage", "pressure_psi"]).any():
    raise ValueError("Duplicate sample-stage-pressure rows found.")

if df[["pressure_psi", "porosity_pct", "permeability_mD"]].isna().any().any():
    raise ValueError("Missing values found in model variables.")

if (df["permeability_mD"] <= 0).any():
    raise ValueError("Permeability must be positive.")

p_min = df["pressure_psi"].min()
p_max = df["pressure_psi"].max()

if p_max == p_min:
    raise ValueError("Only one pressure level is available.")

df["p_norm"] = (
    (df["pressure_psi"] - p_min)
    / (p_max - p_min)
)

df["is_5000"] = df["stage"].eq("5000psi_loading").astype(float)
df["is_7000"] = df["stage"].eq("7000psi_loading").astype(float)


def design_matrix(frame):
    p = frame["p_norm"].to_numpy()
    s5 = frame["is_5000"].to_numpy()
    s7 = frame["is_7000"].to_numpy()

    return np.column_stack([
        np.ones(len(frame)),
        p,
        s5,
        s7,
        p * s5,
        p * s7,
    ])


def calculate_metrics(observed, predicted):
    residual = observed - predicted

    rmse = float(np.sqrt(np.mean(residual**2)))
    mae = float(np.mean(np.abs(residual)))

    return rmse, mae


pressures = sorted(df["pressure_psi"].unique())
prediction_frames = []

for held_out_pressure in pressures:
    train = df[
        ~np.isclose(df["pressure_psi"], held_out_pressure)
    ].copy()

    test = df[
        np.isclose(df["pressure_psi"], held_out_pressure)
    ].copy()

    X_train = design_matrix(train)
    X_test = design_matrix(test)

    if np.linalg.matrix_rank(X_train) < X_train.shape[1]:
        raise ValueError(
            f"Rank-deficient training matrix when holding out "
            f"{held_out_pressure} psi."
        )

    # Porosity model
    beta_phi, *_ = np.linalg.lstsq(
        X_train,
        train["porosity_pct"].to_numpy(),
        rcond=None,
    )

    phi_pred = X_test @ beta_phi

    # Log-permeability model
    beta_logk, *_ = np.linalg.lstsq(
        X_train,
        np.log(train["permeability_mD"].to_numpy()),
        rcond=None,
    )

    # Exponentiation returns predictions in mD.
    k_pred = np.exp(X_test @ beta_logk)

    fold = test[
        ["sample", "stage", "pressure_psi",
         "porosity_pct", "permeability_mD"]
    ].copy()

    fold["porosity_predicted_pct"] = phi_pred
    fold["porosity_residual_pp"] = (
        fold["porosity_pct"] - phi_pred
    )

    fold["permeability_predicted_mD"] = k_pred
    fold["permeability_residual_mD"] = (
        fold["permeability_mD"] - k_pred
    )

    fold["held_out_pressure_psi"] = held_out_pressure

    prediction_frames.append(fold)


predictions = pd.concat(
    prediction_frames,
    ignore_index=True,
)

phi_rmse, phi_mae = calculate_metrics(
    predictions["porosity_pct"].to_numpy(),
    predictions["porosity_predicted_pct"].to_numpy(),
)

k_rmse, k_mae = calculate_metrics(
    predictions["permeability_mD"].to_numpy(),
    predictions["permeability_predicted_mD"].to_numpy(),
)

# Out-of-fold R²: compares held-out predictions with the
# overall observed mean, without fitting to held-out rows.
phi_observed = predictions["porosity_pct"].to_numpy()
phi_predicted = predictions["porosity_predicted_pct"].to_numpy()

k_observed = predictions["permeability_mD"].to_numpy()
k_predicted = predictions["permeability_predicted_mD"].to_numpy()


def out_of_fold_r2(observed, predicted):
    denominator = np.sum((observed - observed.mean()) ** 2)

    if denominator == 0:
        return np.nan

    return float(
        1.0 - np.sum((observed - predicted) ** 2) / denominator
    )


phi_r2_cv = out_of_fold_r2(phi_observed, phi_predicted)
k_r2_cv = out_of_fold_r2(k_observed, k_predicted)

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
predictions.to_csv(OUTPUT_FILE, index=False)

print("=" * 65)
print("LEAVE-ONE-PRESSURE-LEVEL-OUT VALIDATION")
print("=" * 65)

print(f"Observations: {len(predictions)}")
print(f"Pressure folds: {len(pressures)}")
print("Each fold holds out all stages at one pressure level.")

print("\nPOROSITY: OUT-OF-FOLD METRICS")
print(f"RMSE: {phi_rmse:.6f} percentage points")
print(f"MAE:  {phi_mae:.6f} percentage points")
print(f"R2:   {phi_r2_cv:.6f}")

print("\nPERMEABILITY: OUT-OF-FOLD METRICS")
print(f"RMSE: {k_rmse:.6f} mD")
print(f"MAE:  {k_mae:.6f} mD")
print(f"R2:   {k_r2_cv:.6f}")

print("\nMETRICS BY CONDITIONING STAGE")
for stage_name, group in predictions.groupby("stage"):
    phi_stage_rmse, _ = calculate_metrics(
        group["porosity_pct"].to_numpy(),
        group["porosity_predicted_pct"].to_numpy(),
    )

    k_stage_rmse, _ = calculate_metrics(
        group["permeability_mD"].to_numpy(),
        group["permeability_predicted_mD"].to_numpy(),
    )

    print(
        f"{stage_name}: "
        f"porosity RMSE={phi_stage_rmse:.6f} pp; "
        f"permeability RMSE={k_stage_rmse:.6f} mD"
    )

print(f"\nPredictions saved to: {OUTPUT_FILE}")
print(
    "\nNOTE: Validation is within B9 and within the observed "
    "pressure range. It does not establish generalization to "
    "other core samples or a mechanical constitutive law."
)
