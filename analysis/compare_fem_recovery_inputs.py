
from pathlib import Path

import numpy as np
import pandas as pd

from skfem import (
    MeshTri, Basis, FacetBasis,
    ElementTriP1, ElementVector,
    BilinearForm, LinearForm, asm, condense, solve,
)

# =========================================================
# 1. Configuration
# =========================================================

ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "experimental_data.csv"
OUTPUT_DIR = ROOT / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "fem_recovery_inputs_comparison.csv"
SUMMARY_FILE = OUTPUT_DIR / "fem_recovery_inputs_summary.csv"

SAMPLE_ID = "B9"

R = 0.01885
L = 0.06818
E = 11.244e9       # Illustrative, not calibrated
nu = 0.191         # Illustrative, not calibrated
NR = 20
NZ = 40
PSI_TO_PA = 6894.757293

STAGE_ONLY = "stage_only"
STAGE_PLUS_TIME = "stage_plus_time"
ANCHORED_TIME = "anchored_time"
HYBRID = "hybrid_recovery_model"

RECOVERY_STAGES = [
    "rest_after_5000psi",
    "rest_after_7000psi",
]

if E <= 0 or not (-1 < nu < 0.5):
    raise ValueError("Invalid elastic properties.")

mu = E / (2 * (1 + nu))
lam = E * nu / ((1 + nu) * (1 - 2 * nu))
K_bulk = E / (3 * (1 - 2 * nu))

# =========================================================
# 2. Load and validate data
# =========================================================

df = pd.read_csv(DATA_FILE)

required = [
    "sample", "pressure_psi", "porosity_pct",
    "permeability_mD", "stage", "rest_time_hr",
]

missing = set(required) - set(df.columns)
if missing:
    raise ValueError(f"Missing columns: {sorted(missing)}")

df = df.loc[df["sample"].eq(SAMPLE_ID)].copy()

if df.empty:
    raise ValueError(f"No data for sample {SAMPLE_ID}.")

if df[required].isna().any().any():
    raise ValueError("Missing values in required columns.")

if (df["pressure_psi"] <= 0).any():
    raise ValueError("Pressure must be positive.")

if (df["rest_time_hr"] < 0).any():
    raise ValueError("Recovery time cannot be negative.")

if (df["permeability_mD"] <= 0).any():
    raise ValueError("Permeability must be positive.")

if not df["porosity_pct"].between(0, 100).all():
    raise ValueError("Invalid porosity.")

initial = df.loc[
    df["stage"].eq("initial_loading"),
    ["pressure_psi", "porosity_pct"],
].copy()

if initial.empty or initial["pressure_psi"].duplicated().any():
    raise ValueError("Initial pressure levels must be unique.")

initial = initial.rename(
    columns={"porosity_pct": "porosity_initial_pct"}
)

pressure_levels = np.sort(initial["pressure_psi"].unique())
p_min = df["pressure_psi"].min()
p_max = df["pressure_psi"].max()

if p_max <= p_min:
    raise ValueError("At least two pressure levels are required.")

# =========================================================
# 3. Shared features and regression
# =========================================================

def add_features(frame):
    out = frame.copy()
    out["p_norm"] = (
        (out["pressure_psi"] - p_min) / (p_max - p_min)
    )
    out["history_5000"] = out["stage"].isin([
        "5000psi_loading", "rest_after_5000psi"
    ]).astype(float)
    out["history_7000"] = out["stage"].isin([
        "7000psi_loading", "rest_after_7000psi"
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
        np.ones(len(frame)), p, h5, h7,
        p * h5, p * h7, r5, r7,
    ]

    if include_time:
        columns.extend([
            frame["recovery_time_5000"].to_numpy(),
            frame["recovery_time_7000"].to_numpy(),
        ])

    return np.column_stack(columns)


df = add_features(df)
models = {}

for name, use_time in [
    (STAGE_ONLY, False),
    (STAGE_PLUS_TIME, True),
]:
    X = design_matrix(df, use_time)

    if np.linalg.matrix_rank(X) < X.shape[1]:
        raise ValueError(f"Rank-deficient matrix: {name}")

    beta_phi, *_ = np.linalg.lstsq(
        X, df["porosity_pct"].to_numpy(), rcond=None
    )
    beta_logk, *_ = np.linalg.lstsq(
        X, np.log(df["permeability_mD"].to_numpy()), rcond=None
    )

    models[name] = {
        "beta_phi": beta_phi,
        "beta_logk": beta_logk,
        "include_time": use_time,
    }

# =========================================================
# 4. Build scenarios at observed times plus model time zero
# =========================================================

scenario_rows = []

for stage in RECOVERY_STAGES:
    times = np.sort(
        df.loc[df["stage"].eq(stage), "rest_time_hr"].unique()
    )
    if len(times) == 0:
        raise ValueError(f"No data for {stage}.")

    for time_hr in np.unique(np.concatenate(([0.0], times))):
        for pressure in pressure_levels:
            scenario_rows.append({
                "sample": SAMPLE_ID,
                "stage": stage,
                "pressure_psi": float(pressure),
                "rest_time_hr": float(time_hr),
                "scenario_type": (
                    "zero_time_scenario"
                    if np.isclose(time_hr, 0)
                    else "observed_time_level_scenario"
                ),
            })

forecast = pd.DataFrame(scenario_rows)
forecast = forecast.merge(
    initial, on="pressure_psi", how="left",
    validate="many_to_one",
)
forecast = add_features(forecast)
forecast["phi_initial"] = forecast["porosity_initial_pct"] / 100

# Predictions from the two standard models
predictions = {}

for name, model in models.items():
    X = design_matrix(forecast, model["include_time"])
    predictions[name] = {
        "phi": X @ model["beta_phi"],
        "k": np.exp(X @ model["beta_logk"]),
    }

# =========================================================
# 5. Fit anchored recovery model for porosity
# =========================================================

anchor_stages = {
    "rest_after_5000psi": "5000psi_loading",
    "rest_after_7000psi": "7000psi_loading",
}

anchored_phi = np.full(len(forecast), np.nan)

for recovery_stage, anchor_stage in anchor_stages.items():
    recovery = df.loc[
        df["stage"].eq(recovery_stage)
    ].copy()

    anchor = df.loc[
        df["stage"].eq(anchor_stage),
        ["pressure_psi", "porosity_pct"],
    ].rename(
        columns={"porosity_pct": "anchor_phi"}
    )

    recovery = recovery.merge(
        anchor, on="pressure_psi", how="left",
        validate="many_to_one",
    )

    if recovery["anchor_phi"].isna().any():
        raise ValueError(f"Missing anchors for {recovery_stage}")

    log_time = np.log1p(recovery["rest_time_hr"].to_numpy())
    p_norm = recovery["p_norm"].to_numpy()
    X = np.column_stack([log_time, p_norm * log_time])

    y = (
        recovery["porosity_pct"].to_numpy()
        - recovery["anchor_phi"].to_numpy()
    )

    beta, *_ = np.linalg.lstsq(X, y, rcond=None)

    mask = forecast["stage"].eq(recovery_stage).to_numpy()
    subset = forecast.loc[mask]

    anchor_forecast = subset.merge(
        anchor, on="pressure_psi", how="left",
        validate="many_to_one",
    )

    t = np.log1p(anchor_forecast["rest_time_hr"].to_numpy())
    p = anchor_forecast["p_norm"].to_numpy()
    X_forecast = np.column_stack([t, p * t])

    anchored_phi[mask] = (
        anchor_forecast["anchor_phi"].to_numpy()
        + X_forecast @ beta
    )

if not np.isfinite(anchored_phi).all():
    raise ValueError("Invalid anchored predictions.")

predictions[ANCHORED_TIME] = {
    "phi": anchored_phi,
    "k": predictions[STAGE_PLUS_TIME]["k"].copy(),
}

hybrid_phi = np.full(len(forecast), np.nan)
mask_5000 = forecast["stage"].eq(
    "rest_after_5000psi"
).to_numpy()
mask_7000 = forecast["stage"].eq(
    "rest_after_7000psi"
).to_numpy()

hybrid_phi[mask_5000] = anchored_phi[mask_5000]
hybrid_phi[mask_7000] = predictions[
    STAGE_PLUS_TIME
]["phi"][mask_7000]

predictions[HYBRID] = {
    "phi": hybrid_phi,
    "k": predictions[STAGE_PLUS_TIME]["k"].copy(),
}

# =========================================================
# 6. Axisymmetric FEM
# =========================================================

mesh = MeshTri.init_tensor(
    np.linspace(0, R, NR + 1),
    np.linspace(0, L, NZ + 1),
)
element = ElementVector(ElementTriP1())
basis = Basis(mesh, element)


@BilinearForm
def elasticity(u, v, w):
    err_u = u.grad[0, 0]
    ezz_u = u.grad[1, 1]
    ett_u = u[0] / w.x[0]
    erz_u = 0.5 * (u.grad[0, 1] + u.grad[1, 0])

    err_v = v.grad[0, 0]
    ezz_v = v.grad[1, 1]
    ett_v = v[0] / w.x[0]
    erz_v = 0.5 * (v.grad[0, 1] + v.grad[1, 0])

    div_u = err_u + ezz_u + ett_u
    div_v = err_v + ezz_v + ett_v

    strain_product = (
        err_u * err_v + ezz_u * ezz_v
        + ett_u * ett_v + 2 * erz_u * erz_v
    )
    return (
        lam * div_u * div_v + 2 * mu * strain_product
    ) * w.x[0]


@LinearForm
def pressure_load(v, w):
    return (
        -w.p * (v[0] * w.n[0] + v[1] * w.n[1]) * w.x[0]
    )


@LinearForm
def history_load(v, w):
    div_v = v.grad[0, 0] + v.grad[1, 1] + v[0] / w.x[0]
    return 3 * K_bulk * w.eps_star * div_v * w.x[0]


loaded_facets = mesh.facets_satisfying(
    lambda x: (
        np.isclose(x[0], R)
        | np.isclose(x[1], 0)
        | np.isclose(x[1], L)
    )
)
axis_facets = mesh.facets_satisfying(
    lambda x: np.isclose(x[0], 0)
)

fbasis = FacetBasis(mesh, element, facets=loaded_facets)
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
    lambda x: np.isclose(x[0], 0) & np.isclose(x[1], 0)
).all(["u^2"])

D = np.unique(np.concatenate([
    radial_axis_dofs, bottom_anchor_dofs
]))


def solve_system(force):
    u = solve(*condense(K, force, D=D))
    if not np.isfinite(u).all():
        raise RuntimeError("Non-finite FEM solution.")
    return u


u_pressure_per_psi = solve_system(f_pressure_per_psi)
u_history_per_strain = solve_system(f_history_per_strain)

top_dofs = basis.get_dofs(
    facets=mesh.facets_satisfying(lambda x: np.isclose(x[1], L))
).all(["u^2"])
bottom_dofs = basis.get_dofs(
    facets=mesh.facets_satisfying(lambda x: np.isclose(x[1], 0))
).all(["u^2"])
side_dofs = basis.get_dofs(
    facets=mesh.facets_satisfying(lambda x: np.isclose(x[0], R))
).all(["u^1"])


def deformation(u):
    top = float(np.mean(u[top_dofs]))
    bottom = float(np.mean(u[bottom_dofs]))
    side = float(np.mean(u[side_dofs]))

    return (
        -(top - bottom) * 1000,
        -side * 1000,
    )


# =========================================================
# 7. Propagate model predictions through FEM
# =========================================================

rows = []

for model_name, pred in predictions.items():
    phi_pct = pred["phi"]
    k_mD = pred["k"]

    if not np.isfinite(phi_pct).all():
        raise ValueError(f"Invalid porosity: {model_name}")
    if not np.isfinite(k_mD).all() or (k_mD <= 0).any():
        raise ValueError(f"Invalid permeability: {model_name}")
    if ((phi_pct <= 0) | (phi_pct >= 100)).any():
        raise ValueError(f"Porosity out of bounds: {model_name}")

    for i, row in enumerate(forecast.itertuples(index=False)):
        phi_initial = float(row.phi_initial)
        phi_current = float(phi_pct[i]) / 100

        # Inferred proxy under constant grain volume.
        eps_v = np.log(
            (1 - phi_initial) / (1 - phi_current)
        )
        eps_star = eps_v / 3

        u_pressure = float(row.pressure_psi) * u_pressure_per_psi
        u_history = eps_star * u_history_per_strain
        u_combined = u_pressure + u_history

        p_axial, p_radial = deformation(u_pressure)
        h_axial, h_radial = deformation(u_history)
        c_axial, c_radial = deformation(u_combined)

        rows.append({
            "sample": SAMPLE_ID,
            "model": model_name,
            "stage": row.stage,
            "rest_time_hr": row.rest_time_hr,
            "scenario_type": row.scenario_type,
            "pressure_psi": row.pressure_psi,
            "porosity_initial_pct": row.porosity_initial_pct,
            "porosity_predicted_pct": phi_pct[i],
            "permeability_predicted_mD": k_mD[i],
            "eps_v_log_proxy": eps_v,
            "eps_star_proxy": eps_star,
            "pressure_only_axial_mm": p_axial,
            "pressure_only_radial_mm": p_radial,
            "history_only_axial_mm": h_axial,
            "history_only_radial_mm": h_radial,
            "combined_axial_mm": c_axial,
            "combined_radial_mm": c_radial,
            "young_modulus_GPa_assumed": E / 1e9,
            "poisson_ratio_assumed": nu,
            "mesh_elements": mesh.nelements,
        })

results = pd.DataFrame(rows)
results.to_csv(OUTPUT_FILE, index=False)

# =========================================================
# 8. Summary and comparisons
# =========================================================

group_cols = [
    "model", "stage", "rest_time_hr", "scenario_type"
]

summary = (
    results.groupby(group_cols)
    .agg(
        pressure_levels=("pressure_psi", "nunique"),
        mean_porosity_predicted_pct=("porosity_predicted_pct", "mean"),
        mean_permeability_predicted_mD=("permeability_predicted_mD", "mean"),
        mean_eps_v_log_proxy=("eps_v_log_proxy", "mean"),
        mean_combined_axial_mm=("combined_axial_mm", "mean"),
        mean_combined_radial_mm=("combined_radial_mm", "mean"),
    )
    .reset_index()
)
summary.to_csv(SUMMARY_FILE, index=False)

print("=" * 76)
print("FEM RECOVERY INPUT COMPARISON")
print("=" * 76)
print(f"Sample: {SAMPLE_ID}")
print(f"Data rows: {len(df)}")
print(f"Forecast scenarios: {len(forecast)}")
print(f"FEM result rows: {len(results)}")
print(f"Mesh elements: {mesh.nelements}")
print(f"Assumed E: {E / 1e9:.3f} GPa")
print(f"Assumed Poisson ratio: {nu:.3f}")
print("\nSUMMARY")
print(
    summary[
        [
            "model", "stage", "rest_time_hr",
            "mean_porosity_predicted_pct",
            "mean_eps_v_log_proxy",
            "mean_combined_axial_mm",
            "mean_combined_radial_mm",
        ]
    ].to_string(index=False, float_format=lambda x: f"{x:.6f}")
)
print(f"\nDetailed CSV: {OUTPUT_FILE}")
print(f"Summary CSV: {SUMMARY_FILE}")
print(
    "\nLIMITATIONS: E and nu are illustrative. The history strain is "
    "inferred from porosity using a constant-grain-volume assumption. "
    "Displacements are model estimates, not measured values. "
    "Time zero is an anchoring scenario, not an independent measurement."
)
