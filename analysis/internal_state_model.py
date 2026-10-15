
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------
# Paths and configuration
# ---------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "experimental_data.csv"

OUTPUT_FILE = ROOT / "analysis" / "internal_state_results.csv"
SUMMARY_FILE = ROOT / "analysis" / "internal_state_summary.csv"

SAMPLE_ID = "B9"

STAGES = [
    "initial_loading",
    "5000psi_loading",
    "7000psi_loading",
    "rest_after_5000psi",
    "rest_after_7000psi",
]


# ---------------------------------------------------------
# Load and validate data
# ---------------------------------------------------------

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

df = df[
    df["sample"].eq(SAMPLE_ID)
    & df["stage"].isin(STAGES)
].copy()

if df.empty:
    raise ValueError(f"No usable observations for {SAMPLE_ID}.")

if df[required].isna().any().any():
    raise ValueError("Missing values found in required columns.")

if (df["permeability_mD"] <= 0).any():
    raise ValueError("Permeability must be positive.")

if not df["porosity_pct"].between(0, 100).all():
    raise ValueError("Porosity must be between 0 and 100 percent.")

if (df["rest_time_hr"] < 0).any():
    raise ValueError("Recovery time cannot be negative.")


# ---------------------------------------------------------
# Build pressure-matched reference table
# ---------------------------------------------------------

def stage_table(stage, prefix):
    result = (
        df[df["stage"].eq(stage)]
        [["pressure_psi", "porosity_pct", "permeability_mD"]]
        .copy()
    )

    if result["pressure_psi"].duplicated().any():
        raise ValueError(
            f"Duplicate pressure observations in {stage}."
        )

    return result.rename(columns={
        "porosity_pct": f"phi_{prefix}",
        "permeability_mD": f"k_{prefix}",
    })


initial = stage_table("initial_loading", "initial")
loaded_5000 = stage_table("5000psi_loading", "5000")
loaded_7000 = stage_table("7000psi_loading", "7000")

references = (
    initial
    .merge(loaded_5000, on="pressure_psi", how="inner")
    .merge(loaded_7000, on="pressure_psi", how="inner")
)

if references.empty:
    raise ValueError("No common pressure levels across loading stages.")

# Attach pressure-matched reference values to every observation.
data = df.merge(references, on="pressure_psi", how="inner")

# Convert porosity percentages to fractions.
for name in ["initial", "5000", "7000"]:
    data[f"phi_fraction_{name}"] = data[f"phi_{name}"] / 100.0

data["phi_fraction_current"] = data["porosity_pct"] / 100.0


# ---------------------------------------------------------
# Porosity-derived logarithmic volumetric proxy
# ---------------------------------------------------------

def log_solid_fraction_ratio(phi_reference, phi_current):
    """
    Logarithmic volume-change proxy inferred from porosity,
    assuming constant solid-grain volume.

    This is an inferred proxy, not measured mechanical strain.
    """
    solid_reference = 1.0 - phi_reference / 100.0
    solid_current = 1.0 - phi_current / 100.0

    return np.log(solid_reference / solid_current)


data["eps_v_log_proxy"] = log_solid_fraction_ratio(
    data["phi_initial"],
    data["porosity_pct"],
)

data["eps_v_log_5000"] = log_solid_fraction_ratio(
    data["phi_initial"],
    data["phi_5000"],
)

data["eps_v_log_7000"] = log_solid_fraction_ratio(
    data["phi_initial"],
    data["phi_7000"],
)


# ---------------------------------------------------------
# Internal state variables
# ---------------------------------------------------------

# State after 5000-psi conditioning and its recovery period.
mask_5000 = data["stage"].isin([
    "5000psi_loading",
    "rest_after_5000psi",
])

data["z_phi_5000"] = np.nan

valid_5000 = (
    mask_5000
    & data["eps_v_log_5000"].abs().gt(1e-12)
)

data.loc[valid_5000, "z_phi_5000"] = (
    data.loc[valid_5000, "eps_v_log_proxy"]
    / data.loc[valid_5000, "eps_v_log_5000"]
)

data["z_k_5000"] = np.nan

valid_k5000 = (
    mask_5000
    & np.log(data["k_initial"] / data["k_5000"]).abs().gt(1e-12)
)

data.loc[valid_k5000, "z_k_5000"] = (
    np.log(
        data.loc[valid_k5000, "k_initial"]
        / data.loc[valid_k5000, "permeability_mD"]
    )
    / np.log(
        data.loc[valid_k5000, "k_initial"]
        / data.loc[valid_k5000, "k_5000"]
    )
)

# State after 7000-psi conditioning and its recovery period.
mask_7000 = data["stage"].isin([
    "7000psi_loading",
    "rest_after_7000psi",
])

data["z_phi_7000"] = np.nan

valid_7000 = (
    mask_7000
    & data["eps_v_log_7000"].abs().gt(1e-12)
)

data.loc[valid_7000, "z_phi_7000"] = (
    data.loc[valid_7000, "eps_v_log_proxy"]
    / data.loc[valid_7000, "eps_v_log_7000"]
)

data["z_k_7000"] = np.nan

valid_k7000 = (
    mask_7000
    & np.log(data["k_initial"] / data["k_7000"]).abs().gt(1e-12)
)

data.loc[valid_k7000, "z_k_7000"] = (
    np.log(
        data.loc[valid_k7000, "k_initial"]
        / data.loc[valid_k7000, "permeability_mD"]
    )
    / np.log(
        data.loc[valid_k7000, "k_initial"]
        / data.loc[valid_k7000, "k_7000"]
    )
)


# ---------------------------------------------------------
# Export detailed results
# ---------------------------------------------------------

output_columns = [
    "sample",
    "pressure_psi",
    "stage",
    "rest_time_hr",
    "porosity_pct",
    "permeability_mD",
    "phi_initial",
    "phi_5000",
    "phi_7000",
    "k_initial",
    "k_5000",
    "k_7000",
    "eps_v_log_proxy",
    "eps_v_log_5000",
    "eps_v_log_7000",
    "z_phi_5000",
    "z_k_5000",
    "z_phi_7000",
    "z_k_7000",
]

results = data[output_columns].copy()

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
results.to_csv(OUTPUT_FILE, index=False)


# ---------------------------------------------------------
# Summary by stage
# ---------------------------------------------------------

summary = (
    results.groupby("stage", dropna=False)
    .agg(
        observations=("pressure_psi", "size"),
        mean_porosity_pct=("porosity_pct", "mean"),
        mean_permeability_mD=("permeability_mD", "mean"),
        mean_eps_v_log_proxy=("eps_v_log_proxy", "mean"),
        mean_z_phi_5000=("z_phi_5000", "mean"),
        mean_z_k_5000=("z_k_5000", "mean"),
        mean_z_phi_7000=("z_phi_7000", "mean"),
        mean_z_k_7000=("z_k_7000", "mean"),
    )
    .reset_index()
)

summary.to_csv(SUMMARY_FILE, index=False)


# ---------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------

print("=" * 72)
print("INTERNAL STATE MODEL: PRESSURE HISTORY AND RECOVERY")
print("=" * 72)

print(f"Sample: {SAMPLE_ID}")
print(f"Observations processed: {len(results)}")
print(f"Matched pressure levels: {references['pressure_psi'].nunique()}")

print("\nSTAGE SUMMARY")
print(summary.to_string(
    index=False,
    float_format=lambda x: f"{x:.6f}",
))

print("\nSTATE VARIABLE RANGES")

for column in [
    "z_phi_5000",
    "z_k_5000",
    "z_phi_7000",
    "z_k_7000",
]:
    values = results[column].dropna()

    if len(values):
        print(
            f"{column}: "
            f"min={values.min():.4f}, "
            f"max={values.max():.4f}"
        )

print("\nInterpretation:")
print(
    "- State variables are normalized relative to their "
    "pressure-matched conditioning-stage observations."
)
print(
    "- Values near 1 indicate a response similar to the "
    "corresponding conditioned reference."
)
print(
    "- Values below 1 indicate a smaller inferred response; "
    "values above 1 indicate a larger response."
)
print(
    "- These ratios are not constrained to [0, 1] and must "
    "not be clipped automatically."
)
print(
    "- The porosity-derived volumetric proxy assumes constant "
    "grain volume and is not a direct strain measurement."
)

print(f"\nDetailed results: {OUTPUT_FILE}")
print(f"Stage summary: {SUMMARY_FILE}")
