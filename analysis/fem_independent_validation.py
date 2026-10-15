
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
OUTPUT_DIR = ROOT / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "fem_independent_validation.csv"

R = 0.01885
L = 0.06818
E = 11.244e9
nu = 0.191
PSI_TO_PA = 6894.757293

NR = 20
NZ = 40
PRESSURES_PSI = [800.0, 3600.0, 5700.0]

mu = E / (2.0 * (1.0 + nu))
lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))


# =========================================================
# 2. Axisymmetric elasticity
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


@LinearForm
def pressure_load(v, w):
    return (
        -w.p
        * (v[0] * w.n[0] + v[1] * w.n[1])
        * w.x[0]
    )


# =========================================================
# 3. Mesh and boundary conditions
# =========================================================

mesh = MeshTri.init_tensor(
    np.linspace(0.0, R, NR + 1),
    np.linspace(0.0, L, NZ + 1),
)

element = ElementVector(ElementTriP1())
basis = Basis(mesh, element)

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


# Nodal coordinates must be defined before they are used.
coordinates = mesh.p

# Constrain radial displacement along the axis.
radial_axis_dofs = basis.get_dofs(
    facets=axis_facets
).all(["u^1"])

# Find the bottom node directly from the mesh coordinates.
bottom_node = np.where(
    np.isclose(coordinates[0], 0.0)
    & np.isclose(coordinates[1], 0.0)
)[0]


if len(bottom_node) != 1:
    raise RuntimeError(
        f"Expected one bottom-axis node, found {len(bottom_node)}"
    )

# Constrain the axial displacement at that node.
bottom_anchor_dofs = basis.nodal_dofs[1, bottom_node]

D = np.unique(
    np.concatenate([
        radial_axis_dofs,
        bottom_anchor_dofs
    ])
)

print("Radial axis DOFs:", len(radial_axis_dofs))
print("Bottom anchor DOFs:", bottom_anchor_dofs)
print("Total constrained DOFs:", len(D))


free_dofs = np.setdiff1d(
    np.arange(basis.N),
    D,
)

# Nodal coordinates and component DOF indices.
coordinates = mesh.p
radial_dofs = basis.nodal_dofs[0]
axial_dofs = basis.nodal_dofs[1]


# =========================================================
# 4. Independent analytical displacement
# =========================================================

def analytical_displacement(pressure_pa):
    """
    Uniform hydrostatic compression solution:
        ur = -p * (1 - 2*nu) * r / E
        uz = -p * (1 - 2*nu) * z / E
    """
    r = coordinates[0]
    z = coordinates[1]

    ur = -pressure_pa * (1.0 - 2.0 * nu) * r / E
    uz = -pressure_pa * (1.0 - 2.0 * nu) * z / E

    return ur, uz


# =========================================================
# 5. Solve and validate
# =========================================================

rows = []

for pressure_psi in PRESSURES_PSI:
    pressure_pa = pressure_psi * PSI_TO_PA

    force = asm(
        pressure_load,
        fbasis,
        p=pressure_pa,
    )

    displacement = solve(
        *condense(K, force, D=D)
    )

    # Residual: K*u - F.
    residual = K @ displacement - force

    free_residual_norm = np.linalg.norm(
        residual[free_dofs]
    )
    force_norm = np.linalg.norm(force)

    relative_free_residual = (
        free_residual_norm / force_norm
        if force_norm > 0
        else np.nan
    )

    # Reactions at constrained DOFs.
    reactions = residual[D]

    max_abs_reaction = (
        np.max(np.abs(reactions))
        if len(reactions)
        else 0.0
    )

    # Compare FEM nodal displacement with analytical values.

    ur_exact, uz_exact = analytical_displacement(
        pressure_pa
    )

    ur_fem = displacement[radial_dofs]
    uz_fem = displacement[axial_dofs]
        # Inspect axial displacement along the cylinder centerline.
    center_nodes = np.where(
        np.isclose(coordinates[0], 0.0)
    )[0]

    center_nodes = center_nodes[
        np.argsort(coordinates[1, center_nodes])
    ]

    print("Centerline axial displacement (mm):")

    for idx in [
        center_nodes[0],
        center_nodes[len(center_nodes) // 2],
        center_nodes[-1],
    ]:
        print(
            f"z = {coordinates[1, idx]:.6f} m, "
            f"uz_FEM = {displacement[axial_dofs[idx]] * 1000:.8f} mm"
        )

    print(f"\nPressure = {pressure_psi:.0f} psi")
    print(
        "Axial FEM range (mm):",
        f"{np.min(uz_fem) * 1000:.8f}",
        "to",
        f"{np.max(uz_fem) * 1000:.8f}",
    )
    print(
        "Axial analytical range (mm):",
        f"{np.min(uz_exact) * 1000:.8f}",
        "to",
        f"{np.max(uz_exact) * 1000:.8f}",
    )
    print(
        "Radial FEM range (mm):",
        f"{np.min(ur_fem) * 1000:.8f}",
        "to",
        f"{np.max(ur_fem) * 1000:.8f}",
    )
    print(
        "Radial analytical range (mm):",
        f"{np.min(ur_exact) * 1000:.8f}",
        "to",
        f"{np.max(ur_exact) * 1000:.8f}",
    )

    radial_error = np.max(np.abs(ur_fem - ur_exact))

    axial_error = np.max(np.abs(uz_fem - uz_exact))

    exact_scale = max(
        np.max(np.abs(ur_exact)),
        np.max(np.abs(uz_exact)),
    )

    max_displacement_error = max(
        radial_error,
        axial_error,
    )

    relative_displacement_error = (
        max_displacement_error / exact_scale
        if exact_scale > 0
        else np.nan
    )

    rows.append({
        "pressure_psi": pressure_psi,
        "mesh_elements": mesh.nelements,
        "mesh_nodes": mesh.nvertices,
        "total_dofs": basis.N,
        "constrained_dofs": len(D),
        "free_residual_norm": free_residual_norm,
        "force_vector_norm": force_norm,
        "relative_free_residual": relative_free_residual,
        "max_abs_constraint_reaction_N": max_abs_reaction,
        "max_radial_displacement_error_m": radial_error,
        "max_axial_displacement_error_m": axial_error,
        "relative_max_displacement_error": (
            relative_displacement_error
        ),
    })


# =========================================================
# 6. Save and print
# =========================================================

results = pd.DataFrame(rows)
results.to_csv(OUTPUT_FILE, index=False)

print("=" * 72)
print("INDEPENDENT FEM VALIDATION")
print("=" * 72)
print(f"Mesh: NR={NR}, NZ={NZ}")
print(f"Elements: {mesh.nelements}")
print(f"Nodes: {mesh.nvertices}")
print(f"Young's modulus: {E / 1e9:.3f} GPa")
print(f"Poisson ratio: {nu:.3f}")

print("\nVALIDATION RESULTS")
print(
    results.to_string(
        index=False,
        float_format=lambda x: f"{x:.6e}",
    )
)

print(f"\nSaved: {OUTPUT_FILE}")
print(
    "\nInterpretation: displacement and residual checks assess "
    "this linear-elastic benchmark only. Constraint reactions "
    "must be interpreted alongside the imposed boundary conditions."
)
