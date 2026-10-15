
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

OUTPUT_FILE = OUTPUT_DIR / "fem_history_time_sweep.csv"
SUMMARY_FILE = OUTPUT_DIR / "fem_history_time_summary.csv"

SAMPLE_ID = "B9"

# Illustrative geometry, metres.
R = 0.01885
L = 0.06818

# Illustrative elastic parameters, not calibrated to B9.
E = 11.244e9
nu = 0.191

# Mesh resolution.
NR = 20
NZ = 40

PSI_TO_PA = 6894.757293

if E <= 0 or not (-1.0 < nu < 0.5):
    raise ValueError("Invalid elastic parameters.")

if R <= 0 or L <= 0 or NR < 1 or NZ < 1:
    raise ValueError("Invalid geometry or mesh resolution.")

mu = E / (2.0 * (1.0 + nu))
lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
K_bulk = E / (3.0 * (1.0 - 2.0 * nu))


# =========================================================
# 2. Load B9 data and match initial porosity by pressure
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
    raise ValueError(f"No data found for sample {SAMPLE_ID}.")

if df[required].isna().any().any():
    raise ValueError("Missing values in required data.")

if (df["permeability_mD"] <= 0).any():
    raise ValueError("Permeability must be positive.")

if (df["pressure_psi"] <= 0).any():
    raise ValueError("Pressure must be positive.")

if (df["rest_time_hr"] < 0).any():
    raise ValueError("Recovery time cannot be negative.")

if not df["porosity_pct"].between(0, 100).all():
    raise ValueError("Porosity must be between 0 and 100%.")

initial = df[df["stage"].eq("initial_loading")][
    ["pressure_psi", "porosity_pct"]
].copy()

if initial["pressure_psi"].duplicated().any():
    raise ValueError("Initial pressure levels are not unique.")

initial = initial.rename(
    columns={"porosity_pct": "porosity_initial_pct"}
)

# Every observation must have an initial reference at the same pressure.
df = df.merge(
    initial,
    on="pressure_psi",
    how="left",
    validate="many_to_one",
)

if df["porosity_initial_pct"].isna().any():
    raise ValueError(
        "Some observations have no pressure-matched initial reference."
    )

df["phi_initial"] = df["porosity_initial_pct"] / 100.0
df["phi_current"] = df["porosity_pct"] / 100.0


# =========================================================
# 3. Estimate history-dependent volumetric proxy
# =========================================================

# Constant solid-grain volume assumption.
# This is an inferred volume-change proxy, not measured strain.

df["eps_v_log_proxy"] = np.log(
    (1.0 - df["phi_initial"])
    / (1.0 - df["phi_current"])
)

# Isotropic eigenstrain proxy for the illustrative FEM model.
df["eps_star_proxy"] = df["eps_v_log_proxy"] / 3.0

# Convert the measurement pressure to Pa.
df["pressure_Pa"] = df["pressure_psi"] * PSI_TO_PA


# =========================================================
# 4. Axisymmetric finite-element mesh
# =========================================================

mesh = MeshTri.init_tensor(
    np.linspace(0.0, R, NR + 1),
    np.linspace(0.0, L, NZ + 1),
)

element = ElementVector(ElementTriP1())
basis = Basis(mesh, element)


# =========================================================
# 5. Elastic stiffness matrix
# =========================================================

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


# =========================================================
# 6. Pressure and eigenstrain load vectors
# =========================================================

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

    return (
        3.0 * K_bulk * w.eps_star * div_v * w.x[0]
    )


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
    mesh,
    element,
    facets=loaded_facets,
)

K = asm(elasticity, basis)

# Unit pressure response: one psi converted to Pa.
f_pressure_per_psi = asm(
    pressure_load,
    fbasis,
    p=PSI_TO_PA,
)

# Unit isotropic eigenstrain response.
f_history_per_strain = asm(
    history_load,
    basis,
    eps_star=1.0,
)


# =========================================================
# 7. Boundary conditions
# =========================================================

radial_axis_dofs = basis.get_dofs(
    facets=axis_facets
).all(["u^1"])

bottom_axis_dofs = basis.get_dofs(
    lambda x: (
        np.isclose(x[0], 0.0)
        & np.isclose(x[1], 0.0)
    )
).all(["u^2"])

D = np.unique(
    np.concatenate([
        radial_axis_dofs,
        bottom_axis_dofs,
    ])
)


# =========================================================
# 8. Solve reference responses once
# =========================================================

def solve_system(force):
    u = solve(*condense(K, force, D=D))

    if not np.all(np.isfinite(u)):
        raise RuntimeError("Non-finite FEM solution.")

    return u


u_pressure_per_psi = solve_system(f_pressure_per_psi)
u_history_per_strain = solve_system(f_history_per_strain)


# =========================================================
# 9. Extract boundary displacements
# =========================================================

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

side_radial_dofs = basis.get_dofs(
    facets=side_facets
).all(["u^1"])


def extract_deformation(u):
    top_mean = float(np.mean(u[top_dofs]))
    bottom_mean = float(np.mean(u[bottom_dofs]))
    side_mean = float(np.mean(u[side_radial_dofs]))

    return {
        "axial_shortening_mm": -(top_mean - bottom_mean) * 1000.0,
        "radial_contraction_mm": -side_mean * 1000.0,
    }


# =========================================================
# 10. Evaluate every experimental observation
# =========================================================

rows = []

for row in df.itertuples(index=False):
    pressure_psi = float(row.pressure_psi)
    pressure_pa = float(row.pressure_Pa)
    eps_star = float(row.eps_star_proxy)

    # Linear-elastic pressure response.
    u_pressure = pressure_psi * u_pressure_per_psi

    # History contribution inferred from porosity.
    u_history = eps_star * u_history_per_strain

    # Combined response under linear superposition.
    u_combined = u_pressure + u_history

    pressure_result = extract_deformation(u_pressure)
    history_result = extract_deformation(u_history)
    combined_result = extract_deformation(u_combined)

    rows.append({
        "sample": row.sample,
        "stage": row.stage,
        "pressure_psi": pressure_psi,
        "rest_time_hr": float(row.rest_time_hr),
        "porosity_initial_pct": float(row.porosity_initial_pct),
        "porosity_observed_pct": float(row.porosity_pct),
        "permeability_observed_mD": float(row.permeability_mD),
        "eps_v_log_proxy": float(row.eps_v_log_proxy),
        "eps_star_proxy": eps_star,

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
# 11. Summarize results by stage and recovery time
# =========================================================

summary = (
    results.groupby(
        ["stage", "rest_time_hr"],
        dropna=False,
    )
    .agg(
        observations=("pressure_psi", "size"),
        mean_porosity_pct=("porosity_observed_pct", "mean"),
        mean_permeability_mD=("permeability_observed_mD", "mean"),
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


# =========================================================
# 12. Report
# =========================================================

print("=" * 76)
print("FEM PRESSURE-HISTORY-RECOVERY SWEEP")
print("=" * 76)

print(f"Sample: {SAMPLE_ID}")
print(f"Observations processed: {len(results)}")
print(f"Pressure levels: {results['pressure_psi'].nunique()}")
print(f"Stages: {results['stage'].nunique()}")
print(f"Mesh elements: {mesh.nelements}")
print(f"Assumed E: {E / 1e9:.3f} GPa")
print(f"Assumed Poisson ratio: {nu:.3f}")

print("\nMEAN RESULTS BY STAGE AND RECOVERY TIME")

print(
    summary[
        [
            "stage",
            "rest_time_hr",
            "observations",
            "mean_eps_v_log_proxy",
            "mean_pressure_only_axial_mm",
            "mean_history_only_axial_mm",
            "mean_combined_axial_mm",
            "mean_combined_radial_mm",
        ]
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}",
    )
)

print(f"\nDetailed results saved to: {OUTPUT_FILE}")
print(f"Stage summary saved to: {SUMMARY_FILE}")

print(
    "\nLIMITATIONS: Elastic properties are illustrative. "
    "History eigenstrain is inferred from porosity under a "
    "constant-grain-volume assumption. These outputs are model "
    "estimates, not measured displacements or validated predictions."
)
