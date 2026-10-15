
from pathlib import Path

import numpy as np
import pandas as pd


# --------------------------------------------------
# Paths
# --------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "experimental_data.csv"
OUTPUT_FILE = ROOT / "analysis" / "history_state_model.csv"


# --------------------------------------------------
# Load experimental data
# --------------------------------------------------
df = pd.read_csv(DATA_FILE)

required_columns = {
    "sample",
    "pressure_psi",
    "porosity_pct",
    "permeability_mD",
    "stage",
}

missing_columns = required_columns - set(df.columns)

if missing_columns:
    raise ValueError(
        f"Missing required columns: {sorted(missing_columns)}"
    )

stages = [
    "initial_loading",
    "5000psi_loading",
    "7000psi_loading",
]

loading = df[df["stage"].isin(stages)].copy()

# This model currently uses the three conditioning stages.
# Recovery measurements are retained in the original CSV
# and will be modeled separately.


def prepare_stage(stage_name, suffix):
    """Return one row per sample and measurement pressure."""

    subset = loading[loading["stage"] == stage_name].copy()

    if subset.duplicated(["sample", "pressure_psi"]).any():
        raise ValueError(
            f"Duplicate sample-pressure rows in {stage_name}"
        )

    subset = subset[
        [
            "sample",
            "pressure_psi",
            "porosity_pct",
            "permeability_mD",
        ]
    ]

    return subset.rename(
        columns={
            "porosity_pct": f"porosity_{suffix}",
            "permeability_mD": f"permeability_{suffix}",
        }
    )


initial = prepare_stage("initial_loading", "initial")
stage_5000 = prepare_stage("5000psi_loading", "5000")
stage_7000 = prepare_stage("7000psi_loading", "7000")

# Match measurements at the same sample and measurement pressure.
model = initial.merge(
    stage_5000,
    on=["sample", "pressure_psi"],
    how="inner",
    validate="one_to_one",
)

model = model.merge(
    stage_7000,
    on=["sample", "pressure_psi"],
    how="inner",
    validate="one_to_one",
)

if model.empty:
    raise ValueError(
        "No matching sample-pressure measurements were found."
    )

if len(model) != len(initial):
    print(
        "WARNING: Some initial-stage rows did not match "
        "the other conditioning stages."
    )


# --------------------------------------------------
# 1. Porosity changes
# --------------------------------------------------
model["porosity_drop_5000_pp"] = (
    model["porosity_initial"] - model["porosity_5000"]
)

model["porosity_drop_7000_total_pp"] = (
    model["porosity_initial"] - model["porosity_7000"]
)

model["porosity_drop_increment_7000_pp"] = (
    model["porosity_5000"] - model["porosity_7000"]
)


# --------------------------------------------------
# 2. Permeability changes
# --------------------------------------------------
model["permeability_loss_5000_pct"] = (
    100
    * (
        1
        - model["permeability_5000"]
        / model["permeability_initial"]
    )
)

model["permeability_loss_7000_total_pct"] = (
    100
    * (
        1
        - model["permeability_7000"]
        / model["permeability_initial"]
    )
)

model["permeability_loss_increment_7000_pct"] = (
    100
    * (
        1
        - model["permeability_7000"]
        / model["permeability_5000"]
    )
)


# --------------------------------------------------
# 3. Porosity-derived volumetric-change proxies
#
# Assumption: grain volume remains constant.
# These are inferred proxies, not measured strains.
# --------------------------------------------------
solid_initial = 1 - model["porosity_initial"] / 100
solid_5000 = 1 - model["porosity_5000"] / 100
solid_7000 = 1 - model["porosity_7000"] / 100

model["history_state_5000"] = (
    solid_initial / solid_5000 - 1
)

model["history_increment_7000"] = (
    solid_5000 / solid_7000 - 1
)

model["history_state_7000_total"] = (
    solid_initial / solid_7000 - 1
)

# Check whether the two increments reproduce the total state.
model["history_closure_error"] = (
    (1 + model["history_state_5000"])
    * (1 + model["history_increment_7000"])
    - 1
    - model["history_state_7000_total"]
)


# --------------------------------------------------
# 4. Export results
# --------------------------------------------------
model = model.sort_values(["sample", "pressure_psi"])

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
model.to_csv(OUTPUT_FILE, index=False)

display_columns = [
    "sample",
    "pressure_psi",
    "porosity_drop_5000_pp",
    "porosity_drop_7000_total_pp",
    "history_state_5000",
    "history_increment_7000",
    "history_state_7000_total",
    "permeability_loss_5000_pct",
    "permeability_loss_7000_total_pct",
]

print("\nHISTORY-STATE MODEL")
print(f"Matched measurements: {len(model)}")
print(f"Samples: {model['sample'].nunique()}")
print(f"Output: {OUTPUT_FILE}")

print("\nResults:")
print(
    model[display_columns].to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}",
    )
)

print("\nValidation:")
print(
    "Maximum absolute history closure error:",
    f"{model['history_closure_error'].abs().max():.3e}",
)

print(
    "\nMean inferred volumetric proxy after 5000-psi conditioning:",
    f"{100 * model['history_state_5000'].mean():.4f}%",
)

print(
    "Mean incremental proxy from 5000- to 7000-psi conditioning:",
    f"{100 * model['history_increment_7000'].mean():.4f}%",
)

print(
    "Mean total inferred proxy after 7000-psi conditioning:",
    f"{100 * model['history_state_7000_total'].mean():.4f}%",
)
