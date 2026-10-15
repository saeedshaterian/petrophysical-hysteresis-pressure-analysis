from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares


# ---------------------------------------------------------
# 1. Paths and experimental data
# ---------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "experimental_data.csv"
OUTPUT_DIR = ROOT / "analysis"

data = pd.read_csv(DATA_FILE)
data = data[data["sample"].eq("B9")].copy()

required = {
    "pressure_psi",
    "porosity_pct",
    "permeability_mD",
    "stage",
    "rest_time_hr",
}
missing = required.difference(data.columns)

if missing:
    raise ValueError(f"Missing columns: {sorted(missing)}")

print("=" * 65)
print("HISTORY-DEPENDENT PETROPHYSICAL MODEL: B9")
print("=" * 65)
print(f"Data rows: {len(data)}")


# ---------------------------------------------------------
# 2. Match initial and post-7000 psi measurements
# ---------------------------------------------------------

initial = (
    data[data["stage"].eq("initial_loading")]
    .set_index("pressure_psi")
)

loaded = (
    data[data["stage"].eq("7000psi_loading")]
    .set_index("pressure_psi")
)

common_pressures = initial.index.intersection(loaded.index)

if len(common_pressures) < 3:
    raise ValueError("Insufficient matched pressure levels.")

p = common_pressures.to_numpy(dtype=float)

phi0 = initial.loc[common_pressures, "porosity_pct"].to_numpy()
phi_loaded = loaded.loc[
    common_pressures, "porosity_pct"
].to_numpy()

k0 = initial.loc[common_pressures, "permeability_mD"].to_numpy()
k_loaded = loaded.loc[
    common_pressures, "permeability_mD"
].to_numpy()

# Porosity loss is expressed in percentage points.
phi_loss = phi0 - phi_loaded

# Permeability loss is expressed as a percentage.
k_loss_pct = 100.0 * (1.0 - k_loaded / k0)


# ---------------------------------------------------------
# 3. Calibrate a simple pressure-dependent history law
# ---------------------------------------------------------
# x = pressure / maximum measured pressure.
# The intercept represents the average persistent effect.
# The slope captures its variation across current pressure.
#
# This is an empirical constitutive calibration, not yet
# a fully validated plasticity law.

p_scale = p.max()
x = p / p_scale

def linear_law(params, x_values):
    intercept, slope = params
    return intercept + slope * x_values


def fit_linear_law(observed, name):
    result = least_squares(
        lambda params: linear_law(params, x) - observed,
        x0=np.array([np.mean(observed), 0.0]),
    )

    predicted = linear_law(result.x, x)
    rmse = np.sqrt(np.mean((observed - predicted) ** 2))

    print(f"\n{name}")
    print("-" * 45)
    print(f"Intercept: {result.x[0]:.6f}")
    print(f"Slope:     {result.x[1]:.6f}")
    print(f"RMSE:      {rmse:.6f}")

    return result.x, predicted, rmse


phi_params, phi_pred, phi_rmse = fit_linear_law(
    phi_loss, "POROSITY-LOSS LAW"
)

k_params, k_pred, k_rmse = fit_linear_law(
    k_loss_pct, "PERMEABILITY-LOSS LAW"
)


# ---------------------------------------------------------
# 4. Estimate an effective volumetric compaction proxy
# ---------------------------------------------------------
# Assumption: solid-grain volume remains constant.
# Porosity is converted from percent to fraction.
# This is inferred from porosity, not measured strain.

phi0_fraction = phi0 / 100.0
phi_loaded_fraction = phi_loaded / 100.0

effective_volume_change_pct = 100.0 * (
    (1.0 - phi0_fraction)
    / (1.0 - phi_loaded_fraction)
    - 1.0
)

print("\nEFFECTIVE VOLUMETRIC COMPACTION PROXY")
print("-" * 45)
print(
    "Mean inferred volume change: "
    f"{effective_volume_change_pct.mean():.4f}%"
)
print(
    "Range: "
    f"{effective_volume_change_pct.min():.4f}% to "
    f"{effective_volume_change_pct.max():.4f}%"
)
print("Negative values indicate inferred volume contraction.")


# ---------------------------------------------------------
# 5. Check recovery data before fitting time dependence
# ---------------------------------------------------------
# Do not assume that each recovery series starts exactly
# at the matching loading-stage value. Check this first.

print("\nRECOVERY-DATA CONSISTENCY CHECK")
print("-" * 45)

for recovery_stage, loading_stage in [
    ("rest_after_5000psi", "5000psi_loading"),
    ("rest_after_7000psi", "7000psi_loading"),
]:
    recovery = data[data["stage"].eq(recovery_stage)]
    loading = (
        data[data["stage"].eq(loading_stage)]
        .set_index("pressure_psi")
    )

    if recovery.empty:
        print(f"{recovery_stage}: no data")
        continue

    merged = recovery.join(
        loading[["porosity_pct", "permeability_mD"]],
        on="pressure_psi",
        rsuffix="_loading",
    )

    phi_diff = (
        merged["porosity_pct"] - merged["porosity_pct_loading"]
    )
    k_diff = (
        merged["permeability_mD"]
        - merged["permeability_mD_loading"]
    )

    print(f"\n{recovery_stage}")
    print(f"Rows: {len(merged)}")
    print(
        "Porosity difference vs loading stage (percentage points): "
        f"{phi_diff.min():.4f} to {phi_diff.max():.4f}"
    )
    print(
        "Permeability difference vs loading stage (mD): "
        f"{k_diff.min():.4f} to {k_diff.max():.4f}"
    )


# ---------------------------------------------------------
# 6. Save calibrated pressure-response data
# ---------------------------------------------------------

results = pd.DataFrame({
    "pressure_psi": p,
    "normalized_pressure": x,
    "porosity_initial_pct": phi0,
    "porosity_after_7000_pct": phi_loaded,
    "porosity_loss_pct_points": phi_loss,
    "porosity_loss_predicted": phi_pred,
    "permeability_initial_mD": k0,
    "permeability_after_7000_mD": k_loaded,
    "permeability_loss_pct": k_loss_pct,
    "permeability_loss_predicted_pct": k_pred,
    "effective_volume_change_proxy_pct":
        effective_volume_change_pct,
})

output_file = OUTPUT_DIR / "b9_history_law_calibration.csv"
results.to_csv(output_file, index=False)

print("\n" + "=" * 65)
print(f"Calibration data saved to: {output_file}")
print("Calibration complete.")
print("=" * 65)
