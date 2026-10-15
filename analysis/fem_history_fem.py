
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


"""
Exploratory axisymmetric FEM model for pressure-history effects
in carbonate reservoir rock sample B9.

Scientific scope
----------------
1. Experimental porosity and permeability values are read from
   data/experimental_data.csv.

2. The elastic modulus and Poisson's ratio are illustrative
   assumptions, not independently calibrated B9 properties.

3. History-related isotropic eigenstrains are inferred from
   porosity changes under a constant-grain-volume assumption.
   They are not directly measured mechanical strains.

4. Linear superposition is used to separate the applied-pressure
   response from the inferred history-related contribution.

5. The resulting displacements are model predictions, not
   experimentally validated deformation measurements.

6. This script does not independently validate the constitutive
   law or establish a unique physical interpretation of the
   porosity-derived eigenstrains.
"""

# ---------------------------------------------------------
# 1. Paths, geometry and assumed elastic properties
# ---------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "experimental_data.csv"
OUTPUT_DIR = ROOT / "figures"
OUTPUT_DIR.mkdir(exist_ok=True)

R = 0.01885
L = 0.06818

# Illustrative values; not calibrated to B9.
E = 11.244e9
nu = 0.191

CURRENT_PRESSURE_PSI = 5700.0

NR = 20
NZ = 40

if E <= 0 or not (-1.0 < nu < 0.5):
    raise ValueError("Invalid elastic properties.")

mu = E / (2.0 * (1.0 + nu))
lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
K_bulk = E / (3.0 * (1.0 - 2.0 * nu))

P = CURRENT_PRESSURE_PSI * 6894.757293


# ---------------------------------------------------------
# 2. Extract matched B9 porosity values
# ---------------------------------------------------------

data = pd.read_csv(DATA_FILE)

b9 = data[data["sample"].eq("B9")].copy()

def get_stage(stage_name):
    stage = (
        b9[b9["stage"].eq(stage_name)]
        .set_index("pressure_psi")
    )

    if stage.index.duplicated().any():
        raise ValueError(
            f"Duplicate pressure values in {stage_name}"
        )

    return stage


initial = get_stage("initial_loading")
stage_5000 = get_stage("5000psi_loading")
stage_7000 = get_stage("7000psi_loading")

pressure_sets = [
    set(initial.index),
    set(stage_5000.index),
    set(stage_7000.index),
]

common = set.intersection(*pressure_sets)

if not any(
    np.isclose(p, CURRENT_PRESSURE_PSI)
    for p in common
):
    raise ValueError(
        "Current pressure is not available in all three stages."
    )


def get_porosity(stage, pressure):
    matches = [
        idx for idx in stage.index
        if np.isclose(idx, pressure)
    ]

    if len(matches) != 1:
        raise ValueError(
            f"Expected one porosity value at {pressure} psi."
        )

    value = float(stage.loc[matches[0], "porosity_pct"])

    if not 0.0 < value < 100.0:
        raise ValueError(f"Invalid porosity value: {value}")

    return value / 100.0


phi_initial = get_porosity(initial, CURRENT_PRESSURE_PSI)
phi_5000 = get_porosity(stage_5000, CURRENT_PRESSURE_PSI)
phi_7000 = get_porosity(stage_7000, CURRENT_PRESSURE_PSI)


# ---------------------------------------------------------
# 3. Estimate stage-specific history proxies
#
# Assumption: grain volume remains constant.
# Logarithmic volume-change proxies are used because
# successive log increments are additive.
#
# These are not measured mechanical strains.
# ---------------------------------------------------------

solid_initial = 1.0 - phi_initial
solid_5000 = 1.0 - phi_5000
solid_7000 = 1.0 - phi_7000

eps_v_log_5000 = np.log(solid_initial / solid_5000)

eps_v_log_increment_7000 = np.log(
    solid_5000 / solid_7000
)

eps_v_log_total_7000 = np.log(
    solid_initial / solid_7000
)

# Convert the volumetric proxies into isotropic eigenstrain
# components for this illustrative linear-elastic calculation.
eps_star_5000 = eps_v_log_5000 / 3.0
eps_star_increment_7000 = eps_v_log_increment_7000 / 3.0
eps_star_total_7000 = (
    eps_star_5000 + eps_star_increment_7000
)

if not np.isclose(
    eps_star_total_7000,
    eps_v_log_total_7000 / 3.0,
    rtol=1e-10,
    atol=1e-12,
):
    raise RuntimeError("History increments do not close.")


# ---------------------------------------------------------
# 4. Axisymmetric mesh
# ---------------------------------------------------------

mesh = MeshTri.init_tensor(
    np.linspace(0.0, R, NR + 1),
    np.linspace(0.0, L, NZ + 1),
)

element = ElementVector(ElementTriP1())
basis = Basis(mesh, element)


# ---------------------------------------------------------
# 5. Axisymmetric elastic stiffness
# ---------------------------------------------------------

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


# ---------------------------------------------------------
# 6. Pressure and history load vectors
# ---------------------------------------------------------

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

f_pressure = asm(
    pressure_load,
    fbasis,
    p=P,
)

f_history_5000 = asm(
    history_load,
    basis,
    eps_star=eps_star_5000,
)

f_history_increment_7000 = asm(
    history_load,
    basis,
    eps_star=eps_star_increment_7000,
)


# ---------------------------------------------------------
# 7. Boundary conditions
# ---------------------------------------------------------

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


# ---------------------------------------------------------
# 8. Solve each contribution and combined scenario
# ---------------------------------------------------------

def solve_system(force):
    solution = solve(*condense(K, force, D=D))

    if not np.all(np.isfinite(solution)):
        raise RuntimeError(
            "FEM solution contains invalid values."
        )

    return solution


u_pressure = solve_system(f_pressure)

u_history_5000 = solve_system(f_history_5000)

u_history_increment_7000 = solve_system(
    f_history_increment_7000
)

# Combined solutions use linear superposition.
u_total_5000 = u_pressure + u_history_5000

u_total_7000 = (
    u_pressure
    + u_history_5000
    + u_history_increment_7000
)


# ---------------------------------------------------------
# 9. Extract signed shortening and radial contraction
# ---------------------------------------------------------

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


def extract_deformation(solution):
    top_mean = float(np.mean(solution[top_dofs]))
    bottom_mean = float(np.mean(solution[bottom_dofs]))

    side_mean = float(
        np.mean(solution[side_radial_dofs])
    )

    # Positive values represent contraction.
    axial_shortening_mm = (
        -(top_mean - bottom_mean) * 1000.0
    )

    radial_contraction_mm = -side_mean * 1000.0

    return axial_shortening_mm, radial_contraction_mm


cases = [
    ("pressure_only", u_pressure),
    ("history_5000_contribution", u_history_5000),
    (
        "history_7000_increment_contribution",
        u_history_increment_7000,
    ),
    ("combined_5000_scenario", u_total_5000),
    ("combined_7000_scenario", u_total_7000),
]

rows = []

for case_name, solution in cases:
    axial_mm, radial_mm = extract_deformation(solution)

    rows.append({
        "case": case_name,
        "current_pressure_psi": CURRENT_PRESSURE_PSI,
        "initial_porosity_pct": phi_initial * 100.0,
        "porosity_after_5000_stage_pct": phi_5000 * 100.0,
        "porosity_after_7000_stage_pct": phi_7000 * 100.0,
        "log_volume_proxy_5000": eps_v_log_5000,
        "log_volume_proxy_increment_7000": (
            eps_v_log_increment_7000
        ),
        "log_volume_proxy_total_7000": (
            eps_v_log_total_7000
        ),
        "isotropic_eigenstrain_used": {
            "pressure_only": 0.0,
            "history_5000_contribution": eps_star_5000,
            "history_7000_increment_contribution": (
                eps_star_increment_7000
            ),
            "combined_5000_scenario": eps_star_5000,
            "combined_7000_scenario": eps_star_total_7000,
        }[case_name],
        "axial_shortening_mm": axial_mm,
        "radial_contraction_mm": radial_mm,
        "young_modulus_GPa_assumed": E / 1e9,
        "poisson_ratio_assumed": nu,
        "mesh_elements": mesh.nelements,
    })


results = pd.DataFrame(rows)

output_file = OUTPUT_DIR / "fem_history_stage_comparison.csv"
results.to_csv(output_file, index=False)


# ---------------------------------------------------------
# 10. Report
# ---------------------------------------------------------

print("=" * 68)
print("STAGE-SEPARATED HISTORY-DEPENDENT FEM")

print("\nMODEL SCOPE AND LIMITATIONS")
print("-" * 68)
print("Elastic modulus and Poisson ratio are illustrative assumptions.")
print("History eigenstrains are inferred from porosity changes.")
print("Constant grain volume is assumed in the porosity conversion.")
print("Displacements are numerical predictions, not measured strains.")
print("Linear superposition is valid only under the assumed linear model.")
print("Independent mechanical validation has not yet been established.")

print("=" * 68)

print(f"Sample: B9")
print(f"Current measurement pressure: {CURRENT_PRESSURE_PSI:.0f} psi")
print(f"Initial porosity: {phi_initial * 100:.4f}%")
print(f"After 5000-psi conditioning: {phi_5000 * 100:.4f}%")
print(f"After 7000-psi conditioning: {phi_7000 * 100:.4f}%")

print("\nInferred logarithmic volume-change proxies:")
print(f"5000-stage contribution: {eps_v_log_5000:.8f}")
print(f"7000-stage increment:    {eps_v_log_increment_7000:.8f}")
print(f"7000-stage total:        {eps_v_log_total_7000:.8f}")

print("\nFEM RESULTS")
print("-" * 68)

print(
    results[
        [
            "case",
            "axial_shortening_mm",
            "radial_contraction_mm",
        ]
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}",
    )
)

# Verify linear superposition numerically.
u_combined_check = (
    u_pressure
    + u_history_5000
    + u_history_increment_7000
)

superposition_error = float(
    np.max(np.abs(u_total_7000 - u_combined_check))
)

print(f"\nSuperposition check: {superposition_error:.3e}")
print(f"Mesh elements: {mesh.nelements}")
print(f"Results saved to: {output_file}")

print(
    "\nIMPORTANT: History eigenstrains are inferred from porosity "
    "under a constant-grain-volume assumption. They are not "
    "independently measured mechanical strains."
)
