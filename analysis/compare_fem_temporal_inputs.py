
from pathlib import Path

import numpy as np
import pandas as pd

from skfem import (
    MeshTri,
    Basis,
    FacetBasis,
    ElementTriP1,
    ElementVector,
    BilinearForm,
    LinearForm,
    asm,
    condense,
    solve,
)


# =========================================================
# 1. Configuration
# =========================================================

ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "experimental_data.csv"
OUTPUT_DIR = ROOT / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = (
    OUTPUT_DIR / "fem_recovery_inputs_comparison.csv"
)
SUMMARY_FILE = (
    OUTPUT_DIR / "fem_recovery_inputs_summary.csv"
)

SAMPLE_ID = "B9"

# Illustrative core geometry (m).
R = 0.01885
L = 0.06818

# Illustrative elastic properties, not calibrated to B9.
E = 11.244e9
nu = 0.191

NR = 20
NZ = 40
PSI_TO_PA = 6894.757293

STAGE_ONLY = "stage_only"
STAGE_PLUS_TIME = "stage_plus_time"
ANCHORED_TIME = "anchored_time"
HYBRID_MODEL = "hybrid_recovery_model"

RECOVERY_STAGES = [
    "rest_after_5000psi",
    "rest_after_7000psi",
]

if E <= 0 or not (-1.0 < nu < 0.5):
    raise ValueError("Invalid elastic properties.")

if R <= 0 or L <= 0 or NR < 1 or NZ < 1:
    raise ValueError("Invalid geometry or mesh resolution.")

mu = E / (2.0 * (1.0 + nu))
lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
K_bulk = E / (3.0 * (1.0 - 2.0 * nu))


# =========================================================
# 2. Load and validate experimental data
# =========================================================

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

df = df[df["sample"].eq(SAMPLE_ID)].copy()

if df.empty:
    raise ValueError(f"No data found for {SAMPLE_ID}.")

if df[required].isna().any().any():
    raise ValueError("Missing values in required data.")

if (df["pressure_psi"] <= 0).any():
    raise ValueError("Pressure must be positive.")

if (df["rest_time_hr"] < 0).any():
    raise ValueError("Recovery time cannot be negative.")

if (df["permeability_mD"] <= 0).any():
    raise ValueError("Permeability must be positive.")

if not df["porosity_pct"].between(0, 100).all():
    raise ValueError("Porosity must be between 0 and 100%.")

initial = df.loc[
    df["stage"].eq("initial_loading"),
    ["pressure_psi", "porosity_pct"],
].copy()

if initial.empty or initial["pressure_psi"].duplicated().any():
    raise ValueError(
        "Initial data must contain unique pressure levels."
    )

initial = initial.rename(
    columns={"porosity_pct": "porosity_initial_pct"}
)

pressure_levels = np.sort(initial["pressure_psi"].unique())

# Use the same pressure normalization for both models.
p_min = df["pressure_psi"].min()
p_max = df["pressure_psi"].max()

if p_max <= p_min:
    raise ValueError("At least two distinct pressure levels are required.")


# =========================================================
# 3. Shared feature definitions
#    These match compare_temporal_models.py
# =========================================================

def add_features(frame):
    out = frame.copy()

    out["p_norm"] = (
        (out["pressure_psi"] - p_min) / (p_max - p_min)
    )

    out["history_5000"] = out["stage"].isin([
        "5000psi_loading",
        "rest_after_5000psi",
    ]).astype(float)

    out["history_7000"] = out["stage"].isin([
        "7000psi_loading",
        "rest_after_7000psi",
    ]).astype(float)

    out["recovery_5000"] = (
        out["stage"].eq("rest_after_5000psi").astype(float)
    )

    out["recovery_7000"] = (
        out["stage"].eq("rest_after_7000psi").astype(float)
    )

    out["log_time"] = np.log1p(out["rest_time_hr"])

    out["recovery_time_5000"] = (
        out["recovery_5000"] * out["log_time"]
    )

    out["recovery_time_7000"] = (
        out["recovery_7000"] * out["log_time"]
    )

    return out


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


# =========================================================
# 4. Fit both models on all B9 observations
# =========================================================

df = add_features(df)

models = {}

for model_name, include_time in [
    (STAGE_ONLY, False),
    (STAGE_PLUS_TIME, True),
]:
    X = design_matrix(df, include_time)

    if np.linalg.matrix_rank(X) < X.shape[1]:
        raise ValueError(
            f"Rank-deficient design matrix for {model_name}."
        )

    beta_phi, *_ = np.linalg.lstsq(
        X, df["porosity_pct"].to_numpy(), rcond=None
    )

    beta_logk, *_ = np.linalg.lstsq(
        X, np.log(df["permeability_mD"].to_numpy()), rcond=None
    )

    models[model_name] = {
        "include_time": include_time,
        "beta_phi": beta_phi,
        "beta_logk": beta_logk,
    }

# =========================================================
# 4B. Fit anchored recovery models for porosity
# =========================================================

anchored_models = {}

anchor_stages = {
    "rest_after_5000psi": "5000psi_loading",
    "rest_after_7000psi": "7000psi_loading",
}

for recovery_stage, anchor_stage in anchor_stages.items():
    recovery = df.loc[
        df["stage"].eq(recovery_stage)
    ].copy()

    anchor = df.loc[
        df["stage"].eq(anchor_stage),
        ["pressure_psi", "porosity_pct"],
    ].copy()

    if anchor["pressure_psi"].duplicated().any():
        raise ValueError(
            f"Duplicate pressure anchors for {anchor_stage}"
        )

    anchor = anchor.rename(
        columns={"porosity_pct": "anchor_porosity_pct"}
    )

    recovery = recovery.merge(
        anchor,
        on="pressure_psi",
        how="left",
        validate="many_to_one",
    )

    if recovery["anchor_porosity_pct"].isna().any():
        raise ValueError(
            f"Missing pressure-matched anchors for {recovery_stage}"
        )

    recovery["log_time"] = np.log1p(
        recovery["rest_time_hr"]
    )
    recovery["p_log_time"] = (
        recovery["p_norm"] * recovery["log_time"]
    )

    X = np.column_stack([
        recovery["log_time"].to_numpy(),
        recovery["p_log_time"].to_numpy(),
    ])

    y = (
        recovery["porosity_pct"].to_numpy()
        - recovery["anchor_porosity_pct"].to_numpy()
    )

    beta, *_ = np.linalg.lstsq(X, y, rcond=None)

    anchored_models[recovery_stage] = {
        "anchor_stage": anchor_stage,
        "beta_phi": beta,
    }


# =========================================================
# 5. Build recovery scenarios
# =========================================================

# Time zero is a model scenario, not an observed recovery measurement.
# Other times are restricted to the observed range for each stage.

scenarios = {}

for stage in RECOVERY_STAGES:
    observed_times = np.sort(
        df.loc[
            df["stage"].eq(stage), "rest_time_hr"
        ].unique()
    )

    if len(observed_times) == 0:
        raise ValueError(f"No recovery observations for {stage}.")

    if observed_times.min() < 0 or observed_times.max() > 730:
        raise ValueError(
            f"Unexpected recovery time range for {stage}."
        )

    scenarios[stage] = np.unique(
        np.concatenate(([0.0], observed_times.astype(float)))
    )

scenario_rows = []

for stage, times in scenarios.items():
    for time_hr in times:
        for pressure in pressure_levels:
            scenario_rows.append({
                "sample": SAMPLE_ID,
                "stage": stage,
                "pressure_psi": float(pressure),
                "rest_time_hr": float(time_hr),
                "scenario_type": (
                    "zero_time_scenario"
                    if np.isclose(time_hr, 0.0)
                    else "observed_time_level_scenario"
                ),
            })

forecast = pd.DataFrame(scenario_rows)

# Attach initial porosity at the matching pressure.
forecast = forecast.merge(
    initial,
    on="pressure_psi",
    how="left",
    validate="many_to_one",
)

if forecast["porosity_initial_pct"].isna().any():
    raise ValueError("Missing pressure-matched initial porosity.")

forecast = add_features(forecast)

forecast["phi_initial"] = (
    forecast["porosity_initial_pct"] / 100.0
)


# =========================================================
# 6. Axisymmetric FEM model
# =========================================================

mesh = MeshTri.init_tensor(
    np.linspace(0.0, R, NR + 1),
    np.linspace(0.0, L, NZ + 1),
)

element = ElementVector(ElementTriP1())
basis = Basis(mesh, element)


@BilinearForm
def elasticity(u, v, w):
    err_u = u.grad[0, 0]
    ezz_u = u.grad[1, 1]
    ett_u = u[0] / w.x[0]
    erz_u = 0.5 * (
        u.grad[0, 1] + u.grad[1, 0]
    )

    err_v = v.grad[0, 0]
    ezz_v = v.grad[1, 1]
    ett_v = v[0] / w.x[0]
    erz_v = 0.5 * (
        v.grad[0, 1] + v.grad[1, 0]
    )

    div_u = err_u + ezz_u + ett_u
    div_v = err_v + ezz_v + ett_v

    strain_product = (
        err_u * err_v
        + ezz_u * ezz_v
        + ett_u * ett_v
        + 2.0 * erz_u * erz_v
    )

    return (
        lam * div_u * div_v
        + 2.0 * mu * strain_product
    ) * w.x[0]


@LinearForm
def pressure_load(v, w):
    return (
        -w.p
        * (v[0] * w.n[0] + v[1] * w.n[1])
        * w.x[0]
    )


@LinearForm
def history_load(v, w):
    div_v = (
        v.grad[0, 0]
        + v.grad[1, 1]
        + v[0] / w.x[0]
    )

    return 3.0 * K_bulk * w.eps_star * div_v * w.x[0]


loaded_facets = mesh.facets_satisfying(
    lambda x: (
        np.isclose(x[0], R)
        | np.isclose(x[1], 0.0)
        | np.isclose(x[1], L)
    )
)

axis_facets = mesh.facets_satisfying(
    lambda x: np.isclose(x[0], 0.0)
)

fbasis = FacetBasis(
    mesh, element, facets=loaded_facets
)

K = asm(elasticity, basis)

f_pressure_per_psi = asm(
    pressure_load, fbasis, p=PSI_TO_PA
)

f_history_per_strain = asm(
    history_load, basis, eps_star=1.0
)

radial_axis_dofs = basis.get_dofs(
    facets=axis_facets
).all(["u^1"])

bottom_anchor_dofs = basis.get_dofs(
    lambda x: (
        np.isclose(x[0], 0.0)
        & np.isclose(x[1], 0.0)
    )
).all(["u^2"])

D = np.unique(
    np.concatenate([
        radial_axis_dofs,
        bottom_anchor_dofs,
    ])
)


def solve_system(force):
    u = solve(*condense(K, force, D=D))

    if not np.all(np.isfinite(u)):
        raise RuntimeError("Non-finite FEM solution.")

    return u


u_pressure_per_psi = solve_system(f_pressure_per_psi)
u_history_per_strain = solve_system(f_history_per_strain)

top_facets = mesh.facets_satisfying(
    lambda x: np.isclose(x[1], L)
)
bottom_facets = mesh.facets_satisfying(
    lambda x: np.isclose(x[1], 0.0)
)
side_facets = mesh.facets_satisfying(
    lambda x: np.isclose(x[0], R)
)

top_dofs = basis.get_dofs(
    facets=top_facets
).all(["u^2"])

bottom_dofs = basis.get_dofs(
    facets=bottom_facets
).all(["u^2"])

side_dofs = basis.get_dofs(
    facets=side_facets
).all(["u^1"])


def extract_deformation(u):
    top_mean = float(np.mean(u[top_dofs]))
    bottom_mean = float(np.mean(u[bottom_dofs]))
    side_mean = float(np.mean(u[side_dofs]))

    return {
        "axial_shortening_mm": -(top_mean - bottom_mean) * 1000.0,
        "radial_contraction_mm": -side_mean * 1000.0,
    }


# =========================================================
# 7. Predict porosity and propagate into FEM
# =========================================================

prediction_sets = {}

# Standard models
for model_name, model in models.items():
    X = design_matrix(
        forecast, model["include_time"]
    )

    prediction_sets[model_name] = {
        "porosity": X @ model["beta_phi"],
        "permeability": np.exp(
            X @ model["beta_logk"]
        ),
    }

# Anchored porosity predictions, separately by recovery stage
anchored_phi = np.full(len(forecast), np.nan)

for recovery_stage, model in anchored_models.items():
    mask = forecast["stage"].eq(recovery_stage).to_numpy()
    subset = forecast.loc[mask]

    anchor_stage = model["anchor_stage"]

    anchor = df.loc[
        df["stage"].eq(anchor_stage),
        ["pressure_psi", "porosity_pct"],
    ].rename(
        columns={"porosity_pct": "anchor_porosity_pct"}
    )

    subset = subset.merge(
        anchor,
        on="pressure_psi",
        how="left",
        validate="many_to_one",
    )

    log_time = np.log1p(
        subset["rest_time_hr"].to_numpy()
    )
    p_norm = subset["p_norm"].to_numpy()

    X_anchor = np.column_stack([
        log_time,
        p_norm * log_time,
    ])

    delta_phi = X_anchor @ model["beta_phi"]

    predicted_phi = (
        subset["anchor_porosity_pct"].to_numpy()
        + delta_phi
    )

    anchored_phi[mask] = predicted_phi

if not np.all(np.isfinite(anchored_phi)):
    raise ValueError("Invalid anchored porosity predictions.")

# Hybrid model:
# 5000 psi recovery -> anchored_time
# 7000 psi recovery -> stage_plus_time
hybrid_phi = np.full(len(forecast), np.nan)

mask_5000 = forecast["stage"].eq(
    "rest_after_5000psi"
).to_numpy()

mask_7000 = forecast["stage"].eq(
    "rest_after_7000psi"
).to_numpy()

hybrid_phi[mask_5000] = anchored_phi[mask_5000]
hybrid_phi[mask_7000] = prediction_sets[
    STAGE_PLUS_TIME
]["porosity"][mask_7000]

# Permeability is retained from stage_plus_time for the hybrid
# output, but it does not drive the current FEM calculation.
prediction_sets[ANCHORED_TIME] = {
    "porosity": anchored_phi,
    "permeability": prediction_sets[
        STAGE_PLUS_TIME
    ]["permeability"].copy(),
}

prediction_sets[HYBRID_MODEL] = {
    "porosity": hybrid_phi,
    "permeability": prediction_sets[
        STAGE_PLUS_TIME
    ]["permeability"].copy(),
}

rows = []

for model_name, predictions in prediction_sets.items():
    phi_pred_pct = predictions["porosity"]
    k_pred_mD = predictions["permeability"]

    if not np.all(np.isfinite(phi_pred_pct)):
        raise ValueError(
            f"Invalid porosity predictions: {model_name}"
        )

    if (
        not np.all(np.isfinite(k_pred_mD))
        or (k_pred_mD <= 0).any()
    ):
        raise ValueError(
            f"Invalid permeability predictions: {model_name}"
        )

    if (
        (phi_pred_pct <= 0)
        | (phi_pred_pct >= 100)
    ).any():
        raise ValueError(
            f"{model_name} predicts porosity outside (0, 100%)."
        )

    for i, row in enumerate(
        forecast.itertuples(index=False)
    ):
        phi_initial = float(row.phi_initial)
        phi_pred = float(phi_pred_pct[i]) / 100.0

        eps_v_proxy = np.log(
            (1.0 - phi_initial) / (1.0 - phi_pred)
        )
        eps_star_proxy = eps_v_proxy / 3.0

        pressure_psi = float(row.pressure_psi)

        u_pressure = (
            pressure_psi * u_pressure_per_psi
        )

        u_history = (
            eps_star_proxy * u_history_per_strain
        )

        u_combined = u_pressure + u_history

        pressure_result = extract_deformation(u_pressure)
        history_result = extract_deformation(u_history)
        combined_result = extract_deformation(u_combined)

        rows.append({
            "sample": SAMPLE_ID,
            "model": model_name,
            "stage": row.stage,
            "rest_time_hr": float(row.rest_time_hr),
            "scenario_type": row.scenario_type,
            "pressure_psi": pressure_psi,
            "porosity_initial_pct": float(
                row.porosity_initial_pct
            ),
            "porosity_predicted_pct": float(
                phi_pred_pct[i]
            ),
            "permeability_predicted_mD": float(
                k_pred_mD[i]
            ),
            "eps_v_log_proxy": eps_v_proxy,
            "eps_star_proxy": eps_star_proxy,
            "pressure_only_axial_mm":
                pressure_result["axial_shortening_mm"],
            "pressure_only_radial_mm":
                pressure_result["radial_contraction_mm"],
            "history_only_axial_mm":
                history_result["axial_shortening_mm"],
            "history_only_radial_mm":
                history_result["radial_contraction_mm"],
            "combined_axial_mm":
                combined_result["axial_shortening_mm"],
            "combined_radial_mm":
                combined_result["radial_contraction_mm"],
            "young_modulus_GPa_assumed": E / 1e9,
            "poisson_ratio_assumed": nu,
            "mesh_elements": mesh.nelements,
        })

results = pd.DataFrame(rows)
results.to_csv(OUTPUT_FILE, index=False)

# =========================================================
# 8. Summarize and compare the FEM scenarios
# =========================================================

group_cols = [
    "model",
    "stage",
    "rest_time_hr",
    "scenario_type",
]

summary = (
    results.groupby(group_cols, dropna=False)
    .agg(
        pressure_levels=("pressure_psi", "nunique"),
        mean_porosity_predicted_pct=("porosity_predicted_pct", "mean"),
        mean_permeability_predicted_mD=("permeability_predicted_mD", "mean"),
        mean_eps_v_log_proxy=("eps_v_log_proxy", "mean"),
        mean_pressure_only_axial_mm=("pressure_only_axial_mm", "mean"),
        mean_history_only_axial_mm=("history_only_axial_mm", "mean"),
        mean_combined_axial_mm=("combined_axial_mm", "mean"),
        mean_pressure_only_radial_mm=("pressure_only_radial_mm", "mean"),
        mean_history_only_radial_mm=("history_only_radial_mm", "mean"),
        mean_combined_radial_mm=("combined_radial_mm", "mean"),
    )
    .reset_index()
)

summary.to_csv(SUMMARY_FILE, index=False)

# Direct sensitivity comparison between models for each same scenario.
comparison = summary.pivot(
    index=["stage", "rest_time_hr", "scenario_type"],
    columns="model",
    values=[
        "mean_porosity_predicted_pct",
        "mean_eps_v_log_proxy",
        "mean_combined_axial_mm",
        "mean_combined_radial_mm",
    ],
)

comparison.columns = [
    f"{metric}_{model}" for metric, model in comparison.columns
]
comparison = comparison.reset_index()

for metric in [
    "mean_porosity_predicted_pct",
    "mean_eps_v_log_proxy",
    "mean_combined_axial_mm",
    "mean_combined_radial_mm",
]:
    for reference_model in [
        STAGE_ONLY,
        STAGE_PLUS_TIME,
    ]:
        comparison[
            f"delta_{metric}_hybrid_minus_{reference_model}"
        ] = (
            comparison[f"{metric}_{HYBRID_MODEL}"]
            - comparison[f"{metric}_{reference_model}"]
        )

comparison.to_csv(
    OUTPUT_DIR / "fem_recovery_inputs_comparison.csv",
    index=False,
)


# =========================================================
# 9. Report
# =========================================================

print("=" * 78)
print("FEM SENSITIVITY TO TEMPORAL MODEL CHOICE")
print("=" * 78)
print(f"Sample: {SAMPLE_ID}")
print(f"Experimental observations used for fitting: {len(df)}")
print(f"Forecast scenarios: {len(forecast)}")
print(f"FEM output rows: {len(results)}")
print(f"Mesh elements: {mesh.nelements}")
print(f"Assumed E: {E / 1e9:.3f} GPa")
print(f"Assumed Poisson ratio: {nu:.3f}")

print("\nSCENARIO SUMMARY")
print(
    summary[
        [
            "model",
            "stage",
            "rest_time_hr",
            "scenario_type",
            "mean_porosity_predicted_pct",
            "mean_eps_v_log_proxy",
            "mean_combined_axial_mm",
            "mean_combined_radial_mm",
        ]
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}",
    )
)

print(f"\nDetailed results: {OUTPUT_FILE}")
print(f"Summary: {SUMMARY_FILE}")
print(
    "Model-to-model comparison: "
    "figures/fem_temporal_model_sensitivity_comparison.csv"
)

print(
    "\nLIMITATIONS: FEM elastic parameters are illustrative, not calibrated. "
    "The history strain is inferred from predicted porosity under a "
    "constant-grain-volume assumption. Model forecasts are in-sample "
    "scenarios, not independently validated displacement predictions. "
    "Time zero is a scenario, not a measured recovery point."
)
