
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "experimental_data.csv"
OUTPUT_FILE = ROOT / "analysis" / "temporal_model_comparison.csv"

df = pd.read_csv(DATA_FILE)
df = df[df["sample"].eq("B9")].copy()

if df.empty:
    raise ValueError("No B9 data found.")

if df[
    ["pressure_psi", "porosity_pct", "permeability_mD",
     "stage", "rest_time_hr"]
].isna().any().any():
    raise ValueError("Missing required data.")

if (df["permeability_mD"] <= 0).any():
    raise ValueError("Permeability must be positive.")

p_min = df["pressure_psi"].min()
p_max = df["pressure_psi"].max()

df["p_norm"] = (
    (df["pressure_psi"] - p_min) / (p_max - p_min)
)

df["history_5000"] = df["stage"].isin([
    "5000psi_loading", "rest_after_5000psi"
]).astype(float)

df["history_7000"] = df["stage"].isin([
    "7000psi_loading", "rest_after_7000psi"
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


def design_matrix(frame, include_time):
    p = frame["p_norm"].to_numpy()
    h5 = frame["history_5000"].to_numpy()
    h7 = frame["history_7000"].to_numpy()
    r5 = frame["recovery_5000"].to_numpy()
    r7 = frame["recovery_7000"].to_numpy()

    columns = [
        np.ones(len(frame)),
        p,
        h5,
        h7,
        p * h5,
        p * h7,
        r5,
        r7,
    ]

    if include_time:
        columns.extend([
            frame["recovery_time_5000"].to_numpy(),
            frame["recovery_time_7000"].to_numpy(),
        ])

    return np.column_stack(columns)


def fit_predict(train, test, include_time):
    X_train = design_matrix(train, include_time)
    X_test = design_matrix(test, include_time)

    if np.linalg.matrix_rank(X_train) < X_train.shape[1]:
        raise ValueError("Rank-deficient training matrix.")

    beta_phi, *_ = np.linalg.lstsq(
        X_train, train["porosity_pct"].to_numpy(), rcond=None
    )
    beta_logk, *_ = np.linalg.lstsq(
        X_train, np.log(train["permeability_mD"].to_numpy()),
        rcond=None
    )

    return X_test @ beta_phi, np.exp(X_test @ beta_logk)


predictions = []

for stage in ["rest_after_5000psi", "rest_after_7000psi"]:
    times = sorted(
        df.loc[df["stage"].eq(stage), "rest_time_hr"].unique()
    )

    for time in times:
        mask = (
            df["stage"].eq(stage)
            & np.isclose(df["rest_time_hr"], time)
        )

        train = df.loc[~mask].copy()
        test = df.loc[mask].copy()

        for model_name, use_time in [
            ("stage_only", False),
            ("stage_plus_time", True),
        ]:
            phi_pred, k_pred = fit_predict(train, test, use_time)

            out = test[
                ["stage", "pressure_psi", "rest_time_hr",
                 "porosity_pct", "permeability_mD"]
            ].copy()

            out["model"] = model_name
            out["porosity_predicted_pct"] = phi_pred
            out["permeability_predicted_mD"] = k_pred
            out["porosity_error_pp"] = (
                phi_pred - out["porosity_pct"]
            )
            out["permeability_error_mD"] = (
                k_pred - out["permeability_mD"]
            )
            predictions.append(out)


pred = pd.concat(predictions, ignore_index=True)
pred.to_csv(OUTPUT_FILE, index=False)


def summarize(group):
    phi_err = group["porosity_error_pp"].to_numpy()
    k_err = group["permeability_error_mD"].to_numpy()

    return pd.Series({
        "observations": len(group),
        "porosity_RMSE_pp": np.sqrt(np.mean(phi_err**2)),
        "porosity_MAE_pp": np.mean(np.abs(phi_err)),
        "permeability_RMSE_mD": np.sqrt(np.mean(k_err**2)),
        "permeability_MAE_mD": np.mean(np.abs(k_err)),
    })


summary = (
    pred.groupby(["stage", "model"])
    .apply(summarize, include_groups=False)
    .reset_index()
)

overall = (
    pred.groupby("model")
    .apply(summarize, include_groups=False)
    .reset_index()
)
overall.insert(0, "stage", "OVERALL")

summary = pd.concat([summary, overall], ignore_index=True)
summary.to_csv(
    ROOT / "analysis" / "temporal_model_comparison_metrics.csv",
    index=False,
)

print("=" * 72)
print("FAIR COMPARISON: WITH TIME VS WITHOUT TIME")
print("Same held-out recovery times and observations for both models")
print("=" * 72)
print(summary.to_string(
    index=False,
    float_format=lambda x: f"{x:.6f}",
))
print(f"\nPredictions saved to: {OUTPUT_FILE}")
print("Metrics saved to: analysis/temporal_model_comparison_metrics.csv")
