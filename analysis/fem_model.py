
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

# ---------------------------------------------------------
# 1. Model parameters
# ---------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "figures"
OUTPUT_DIR.mkdir(exist_ok=True)

# Illustrative core geometry.
# Replace these values with verified dimensions of the
# specific core when available.
R = 0.01885       # radius (m)
L = 0.06818       # length (m)

# Illustrative elastic properties, not calibrated values.
E = 11.244e9      # Young's modulus (Pa)
nu = 0.191        # Poisson's ratio

PRESSURE_PSI = 5700.0
P = PRESSURE_PSI * 6894.757293  # Pa

# Mesh resolution
NR = 20
NZ = 40

if E <= 0 or not (-1.0 < nu < 0.5):
    raise ValueError("Invalid elastic properties.")

# Lamé parameters
mu = E / (2.0 * (1.0 + nu))
lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
bulk_modulus = E / (3.0 * (1.0 - 2.0 * nu))

# ---------------------------------------------------------
# 2. Axisymmetric mesh
# Coordinates: x = radius, y = axial position
# ---------------------------------------------------------

mesh = MeshTri.init_tensor(
    np.linspace(0.0, R, NR + 1),
    np.linspace(0.0, L, NZ + 1),
)

element = ElementVector(ElementTriP1())
basis = Basis(mesh, element)

# ---------------------------------------------------------
# 3. Axisymmetric linear-elastic weak form
# ---------------------------------------------------------

@BilinearForm
def elasticity(u, v, w):
    # Strains for axisymmetric displacement:
    # u[0] = radial displacement
    # u[1] = axial displacement

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
        err_u * err_v
        + ezz_u * ezz_v
        + ett_u * ett_v
        + 2.0 * erz_u * erz_v
    )

    # The common factor 2*pi cancels from the equations.
    return (
        lam * div_u * div_v
        + 2.0 * mu * strain_product
    ) * w.x[0]


@LinearForm
def pressure_load(v, w):
    # External compressive pressure: traction = -P * normal.
    # Axisymmetric integration includes the radial weight.
    return (
        -w.p
        * (v[0] * w.n[0] + v[1] * w.n[1])
        * w.x[0]
    )


# ---------------------------------------------------------
# 4. Boundary conditions
# ---------------------------------------------------------

# Apply pressure on the cylindrical wall and both end faces.
# The r = 0 axis is a symmetry boundary, not a pressure face.
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
f = asm(pressure_load, fbasis, p=P)

# Radial displacement is zero along the symmetry axis.
radial_axis_dofs = basis.get_dofs(
    facets=axis_facets
).all(["u^1"])

# Fix one axial displacement DOF to remove rigid translation.
bottom_axis = basis.get_dofs(
    lambda x: np.isclose(x[0], 0.0)
    & np.isclose(x[1], 0.0)
).all(["u^2"])

D = np.unique(
    np.concatenate([radial_axis_dofs, bottom_axis])
)

# ---------------------------------------------------------
# 5. Solve the finite-element system
# ---------------------------------------------------------

u = solve(*condense(K, f, D=D))
print("Fixed axial DOFs:", bottom_axis)
print("Fixed axial DOF displacement:", u[bottom_axis])
# Diagnostic checks
top_facets = mesh.facets_satisfying(
    lambda x: np.isclose(x[1], L)
)
bottom_facets = mesh.facets_satisfying(
    lambda x: np.isclose(x[1], 0.0)
)
side_facets = mesh.facets_satisfying(
    lambda x: np.isclose(x[0], R)
)

top_dofs = basis.get_dofs(facets=top_facets).all(["u^2"])
bottom_dofs = basis.get_dofs(facets=bottom_facets).all(["u^2"])
side_radial_dofs = basis.get_dofs(facets=side_facets).all(["u^1"])

print("\nFEM DIAGNOSTICS")
print("---------------")
print("Loaded facets:", len(loaded_facets))
print("Top facets:", len(top_facets))
print("Bottom facets:", len(bottom_facets))
print("Side facets:", len(side_facets))
print("Top axial displacement (min, max):",
      np.min(u[top_dofs]), np.max(u[top_dofs]))
print("Bottom axial displacement (min, max):",
      np.min(u[bottom_dofs]), np.max(u[bottom_dofs]))
print("Side radial displacement (min, max):",
      np.min(u[side_radial_dofs]), np.max(u[side_radial_dofs]))

if not np.all(np.isfinite(u)):
    raise RuntimeError("The FEM solution contains invalid values.")


# ---------------------------------------------------------
# 6. Analytical benchmark
# ---------------------------------------------------------

expected_axial_shortening = P * L / (3.0 * bulk_modulus)
expected_radial_contraction = P * R / (3.0 * bulk_modulus)

top_axial = np.mean(u[top_dofs])
bottom_axial = np.mean(u[bottom_dofs])

axial_shortening = abs(top_axial - bottom_axial)
radial_contraction = abs(np.mean(u[side_radial_dofs]))

shortening_error_pct = (
    abs(axial_shortening - expected_axial_shortening)
    / expected_axial_shortening * 100.0
)

radial_error_pct = (
    abs(radial_contraction - expected_radial_contraction)
    / expected_radial_contraction * 100.0
)

print("\nAXISYMMETRIC ELASTIC FEM VALIDATION")
print("===================================")
print(f"Analytical axial shortening: {expected_axial_shortening * 1e3:.6f} mm")
print(f"FEM axial shortening:        {axial_shortening * 1e3:.6f} mm")
print(f"Axial error:                 {shortening_error_pct:.4f}%")
print()
print(f"Analytical radial contraction: {expected_radial_contraction * 1e3:.6f} mm")
print(f"FEM radial contraction:        {radial_contraction * 1e3:.6f} mm")
print(f"Radial error:                  {radial_error_pct:.4f}%")

results = pd.DataFrame([{
    "pressure_psi": PRESSURE_PSI,
    "pressure_MPa": P / 1e6,
    "young_modulus_GPa": E / 1e9,
    "poisson_ratio": nu,
    "bulk_modulus_GPa": bulk_modulus / 1e9,
    "mesh_elements": mesh.nelements,
    "fem_axial_shortening_mm": axial_shortening * 1e3,
    "analytical_axial_shortening_mm": expected_axial_shortening * 1e3,
    "axial_error_pct": shortening_error_pct,
    "fem_radial_contraction_mm": radial_contraction * 1e3,
    "analytical_radial_contraction_mm": expected_radial_contraction * 1e3,
    "radial_error_pct": radial_error_pct,
}])

results.to_csv(
    OUTPUT_DIR / "fem_elastic_benchmark.csv",
    index=False,
)

print(f"\nResults saved to: {OUTPUT_DIR / 'fem_elastic_benchmark.csv'}")
