
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "experimental_data.csv"
OUTPUT_FILE = ROOT / "analysis" / "recovery_model_comparison.csv"

df = pd.read_csv(DATA_FILE)
df = df[df["sample"].eq("B9")].copy()

stages = [
    "initial_loading",
    "5000psi_loading",
    "7000psi_loading",
    "rest_after_5000psi",
    "rest_after_7000psi",
]

df = df[df["stage"].isin(stages)].copy()

required = [
    "pressure_psi",
    "porosity_pct",
    "permeability_mD",
    "stage",
    "rest_time_hr",
]

if df[required].isna().any().any():
    raise ValueError("Missing values in required columns.")

if (df["permeability_mD"] <= 0).any():
    raise ValueError("Permeability must be positive.")

if df.duplicated(
    ["sample", "stage", "pressure_psi", "rest_time_hr"]
).any():
    raise ValueError("Duplicate observations detected.")

p_min = df["pressure_psi"].min()
p_max = df["pressure_psi"].max()

df["p"] = (df["pressure_psi"] - p_min) / (p_max - p_min)

# Reference category: initial_loading
stage_order = [
    "5000psi_loading",
    "7000psi_loading",
    "rest_after_5000psi",
    "rest_after_7000psi",
]

for stage in stage_order:
    df[f"stage_{stage}"] = df["stage"].eq(stage).astype(float)
    df[f"pressure_x_{stage}"] = df["p"] * df[f"stage_{stage}"]

# Time model terms: time affects only recovery observations.
df["recovery_5000"] = (
    df["stage"].eq("rest_after_5000psi").astype(float)
)
df["recovery_7000"] = (
    df["stage"].eq("rest_after_7000psi").astype(float)
)
df["log_time"] = np.log1p(df["rest_time_hr"])

df["time_5000"] = df["recovery_5000"] * df["log_time"]
df["time_7000"] = df["recovery_7000"] * df["log_time"]


def matrix(frame, include_time):
    columns = [
        np.ones(len(frame)),
        frame["p"].to_numpy(),
    ]

    # Categorical stage effects and pressure interactions.
    for stage in stage_order:
        columns.append(frame[f"stage_{stage}"].to_numpy())
        columns.append(frame[f"pressure_x_{stage}"].to_numpy())

    if include_time:
        # Time terms are added only to the time-dependent model.
        columns.extend([
            frame["time_5000"].to_numpy(),
            frame["time_7000"].to_numpy(),
        ])

    return np.column_stack(columns)


def fit_predict(train, test, include_time, target):
    X_train = matrix(train, include_time)
    X_test = matrix(test, include_time)

    if np.linalg.matrix_rank(X_train) < X_train.shape[1]:
        raise ValueError("Rank-deficient training design matrix.")

    if target == "porosity":
        y_train = train["porosity_pct"].to_numpy()
        beta, *_ = np.linalg.lstsq(X_train, y_train, rcond=None)
        return X_test @ beta

    y_train = np.log(train["permeability_mD"].to_numpy())
    beta, *_ = np.linalg.lstsq(X_train, y_train, rcond=None)
    return np.exp(X_test @ beta)


predictions = []

for pressure in sorted(df["pressure_psi"].unique()):
    train = df[~np.isclose(df["pressure_psi"], pressure)].copy()
    test = df[np.isclose(df["pressure_psi"], pressure)].copy()

    for model_name, include_time in [
        ("stage_only", False),
        ("stage_plus_recovery_time", True),
    ]:
        test = test.copy()

        test["porosity_prediction"] = fit_predict(
            train, test, include_time, "porosity"
        )
        test["permeability_prediction"] = fit_predict(
            train, test, include_time, "permeability"
        )
        test["model"] = model_name
        test["held_out_pressure_psi"] = pressure

        predictions.append(test)


pred = pd.concat(predictions, ignore_index=True)


def metrics(group, observed_col, predicted_col):
    observed = group[observed_col].to_numpy()
    estimated = group[predicted_col].to_numpy()
    residual = observed - estimated

    rmse = float(np.sqrt(np.mean(residual**2)))
    mae = float(np.mean(np.abs(residual)))

    denominator = float(np.sum((observed - observed.mean())**2))
    r2 = (
        float(1 - np.sum(residual**2) / denominator)
        if denominator > 0
        else np.nan
    )

    return rmse, mae, r2


rows = []

for model_name, group in pred.groupby("model"):
    phi = metrics(
        group, "porosity_pct", "porosity_prediction"
    )
    perm = metrics(
        group, "permeability_mD", "permeability_prediction"
    )

    rows.append({
        "model": model_name,
        "porosity_RMSE_pp": phi[0],
        "porosity_MAE_pp": phi[1],
        "porosity_R2": phi[2],
        "permeability_RMSE_mD": perm[0],
        "permeability_MAE_mD": perm[1],
        "permeability_R2": perm[2],
    })

summary = pd.DataFrame(rows)

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
summary.to_csv(OUTPUT_FILE, index=False)

print("=" * 72)
print("FAIR COMPARISON: RECOVERY TIME vs NO RECOVERY TIME")
print("=" * 72)
print(f"Observations per model: {len(df)}")
print(f"Pressure folds: {df['pressure_psi'].nunique()}")
print(summary.to_string(
    index=False,
    float_format=lambda x: f"{x:.6f}",
))

print(f"\nSaved to: {OUTPUT_FILE}")
print(
    "\nInterpretation: Lower RMSE and MAE indicate better predictions. "
    "The comparison tests predictive usefulness of time terms; "
    "it does not prove a physical recovery mechanism."
)
