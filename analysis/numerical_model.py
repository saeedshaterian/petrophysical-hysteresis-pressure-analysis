
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# Paths and data
# ---------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "experimental_data.csv"
OUTPUT_DIR = ROOT / "figures"
OUTPUT_DIR.mkdir(exist_ok=True)

df = pd.read_csv(DATA_FILE)
df = df[df["sample"] == "B9"].copy()

PROPERTY_COLUMNS = {
    "porosity_pct": "Porosity (%)",
    "permeability_mD": "Permeability (mD)",
}

INITIAL_STAGE = "initial_loading"
LOADED_STAGE = "7000psi_loading"
RECOVERY_STAGE = "rest_after_7000psi"

# ---------------------------------------------------------
# Validate and align pressure measurements
# ---------------------------------------------------------

required_stages = {
    INITIAL_STAGE,
    LOADED_STAGE,
    RECOVERY_STAGE,
}

available_stages = set(df["stage"].unique())
missing_stages = required_stages - available_stages

if missing_stages:
    raise ValueError(f"Missing experimental stages: {missing_stages}")

initial = (
    df[df["stage"] == INITIAL_STAGE]
    .groupby("pressure_psi")
)

loaded = (
    df[df["stage"] == LOADED_STAGE]
    .groupby("pressure_psi")
)

recovery = df[df["stage"] == RECOVERY_STAGE].copy()

# Use only pressures measured in both reference stages.
common_pressures = sorted(
    set(initial.groups) & set(loaded.groups)
)

if len(common_pressures) < 3:
    raise ValueError(
        "Insufficient common pressure levels for calibration."
    )

# ---------------------------------------------------------
# Recovery model
#
# y(P,t) = y_loaded(P) + (y_initial(P) - y_loaded(P))
#          * R_inf * (1 - exp(-t/tau))
#
# R_inf: asymptotic recoverable fraction
# tau: characteristic recovery time (hours)
#
# This is a phenomenological model, not yet an FEM model.
# ---------------------------------------------------------

def grid_search_fit(observed_fraction, times):
    """Fit asymptotic recovery fraction and time constant."""

    r_values = np.linspace(0.0, 1.0, 201)
    tau_values = np.geomspace(1.0, 10000.0, 301)

    best = None

    for tau in tau_values:
        time_response = 1.0 - np.exp(-times / tau)

        for r_inf in r_values:
            predicted = r_inf * time_response
            mse = np.mean((observed_fraction - predicted) ** 2)

            if best is None or mse < best["mse"]:
                best = {
                    "r_inf": float(r_inf),
                    "tau_hr": float(tau),
                    "mse": float(mse),
                }

    return best


results = []
prediction_rows = []

for column, label in PROPERTY_COLUMNS.items():
    records = []

    for pressure in common_pressures:
        initial_value = initial.get_group(pressure)[column].mean()
        loaded_value = loaded.get_group(pressure)[column].mean()

        delta = initial_value - loaded_value

        # A near-zero reference difference cannot support
        # a stable normalized recovery estimate.
        if abs(delta) < 1e-12:
            continue

        pressure_data = recovery[
            recovery["pressure_psi"] == pressure
        ]

        for _, row in pressure_data.iterrows():
            if row["rest_time_hr"] <= 0:
                continue

            fraction = (
                row[column] - loaded_value
            ) / delta

            records.append({
                "pressure_psi": pressure,
                "time_hr": float(row["rest_time_hr"]),
                "observed": float(row[column]),
                "initial": float(initial_value),
                "loaded": float(loaded_value),
                "fraction": float(fraction),
            })

    fit_data = pd.DataFrame(records)

    if fit_data.empty:
        print(f"No usable recovery data for {label}")
        continue

    fit = grid_search_fit(
        fit_data["fraction"].to_numpy(),
        fit_data["time_hr"].to_numpy(),
    )

    fit_data["predicted_fraction"] = (
        fit["r_inf"]
        * (
            1.0
            - np.exp(
                -fit_data["time_hr"].to_numpy() / fit["tau_hr"]
            )
        )
    )

    fit_data["predicted"] = (
        fit_data["loaded"]
        + (
            fit_data["initial"] - fit_data["loaded"]
        ) * fit_data["predicted_fraction"]
    )

    rmse = np.sqrt(
        np.mean(
            (fit_data["observed"] - fit_data["predicted"]) ** 2
        )
    )

    results.append({
        "property": label,
        "R_inf": fit["r_inf"],
        "tau_hr": fit["tau_hr"],
        "RMSE": rmse,
        "n_observations": len(fit_data),
    })

    fit_data["property"] = label
    prediction_rows.append(fit_data)

    # Plot measured versus predicted recovery.
    plt.figure(figsize=(8, 5))

    for pressure in common_pressures:
        subset = fit_data[
            fit_data["pressure_psi"] == pressure
        ].sort_values("time_hr")

        if subset.empty:
            continue

        plt.scatter(
            subset["time_hr"],
            subset["observed"],
            s=28,
            label=f"{pressure} psi measured",
        )

        plt.plot(
            subset["time_hr"],
            subset["predicted"],
            linestyle="--",
            linewidth=1.4,
        )

    plt.xlabel("Recovery time (hours)")
    plt.ylabel(label)
    plt.title(f"B9: Measured and modeled recovery - {label}")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=7, ncol=2)
    plt.tight_layout()

    filename = (
        "B9_numerical_recovery_"
        + column.replace("_", "")
        + ".png"
    )

    plt.savefig(OUTPUT_DIR / filename, dpi=300)
    plt.close()

# ---------------------------------------------------------
# Save model results
# ---------------------------------------------------------

if not results:
    raise RuntimeError("No model results were generated.")

results_df = pd.DataFrame(results)
results_df.to_csv(
    OUTPUT_DIR / "B9_numerical_model_parameters.csv",
    index=False,
)

predictions_df = pd.concat(
    prediction_rows,
    ignore_index=True,
)

predictions_df.to_csv(
    OUTPUT_DIR / "B9_numerical_model_predictions.csv",
    index=False,
)

print("\nNUMERICAL RECOVERY MODEL")
print("========================")
print(results_df.to_string(index=False))

print("\nFiles saved in:")
print(OUTPUT_DIR)
import numpy as np
import scipy
import matplotlib
import pandas as pd

print("NumPy:", np.__version__)
print("SciPy:", scipy.__version__)
print("Matplotlib:", matplotlib.__version__)
print("Pandas:", pd.__version__)
