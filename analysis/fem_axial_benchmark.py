
from pathlib import Path

import numpy as np
import pandas as pd

from skfem import (
    MeshTri,
    Basis,
    FacetBasis,
    ElementVector,
    ElementTriP1,
    BilinearForm,
    LinearForm,
    asm,
    condense,
    solve,
)


# ---------------------------------------------------------
# 1. Geometry and benchmark parameters
# ---------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

R = 0.01885       # Sample radius, m
L = 0.06818       # Sample length, m
SIGMA = 1.0e6     # Axial compressive stress, Pa

NR = 20
NZ = 40

E_VALUES_GPA = [
    5.622, 8.433, 11.244, 14.055, 16.866
]

NU_VALUES = [0.15, 0.191, 0.25, 0.30, 0.35]


# ---------------------------------------------------------
# 2. Axisymmetric mesh
# ---------------------------------------------------------

mesh = MeshTri.init_tensor(
    np.linspace(0.0, R, NR + 1),
    np.linspace(0.0, L, NZ + 1),
)

element = ElementVector(ElementTriP1())
basis = Basis(mesh, element)

axis_facets = mesh.facets_satisfying(
    lambda x: np.isclose(x[0], 0.0)
)

bottom_facets = mesh.facets_satisfying(
    lambda x: np.isclose(x[1], 0.0)
)

top_facets = mesh.facets_satisfying(
    lambda x: np.isclose(x[1], L)
)

top_basis = FacetBasis(
    mesh, element, facets=top_facets
)


# ---------------------------------------------------------
# 3. Linear-elastic axisymmetric stiffness
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
        w.lam * div_u * div_v
        + 2.0 * w.mu * strain_product
    ) * w.x[0]


# ---------------------------------------------------------
# 4. Axial traction on the top face
# ---------------------------------------------------------

@LinearForm
def axial_load(v, w):
    # Compression acts downward on the top face.
    return -w.sigma * v[1] * w.x[0]


force = asm(
    axial_load,
    top_basis,
    sigma=SIGMA,
)


# ---------------------------------------------------------
# 5. Boundary conditions
# ---------------------------------------------------------

# Radial displacement is zero on the symmetry axis.
radial_axis_dofs = basis.get_dofs(
    facets=axis_facets
).all(["u^1"])

# Axial displacement is zero across the bottom face.
bottom_axial_dofs = basis.get_dofs(
    facets=bottom_facets
).all(["u^2"])

D = np.unique(
    np.concatenate([
        radial_axis_dofs,
        bottom_axial_dofs,
    ])
)

# Output locations
top_axial_dofs = basis.get_dofs(
    facets=top_facets
).all(["u^2"])

side_facets = mesh.facets_satisfying(
    lambda x: np.isclose(x[0], R)
)

side_radial_dofs = basis.get_dofs(
    facets=side_facets
).all(["u^1"])


# ---------------------------------------------------------
# 6. Parameter sweep and analytical comparison
# ---------------------------------------------------------

rows = []

for E_GPa in E_VALUES_GPA:
    E = E_GPa * 1e9

    for nu in NU_VALUES:
        lam = E * nu / (
            (1.0 + nu) * (1.0 - 2.0 * nu)
        )
        mu = E / (2.0 * (1.0 + nu))

        K = asm(
            elasticity,
            basis,
            lam=lam,
            mu=mu,
        )

        displacement = solve(
            *condense(K, force, D=D)
        )

        if not np.all(np.isfinite(displacement)):
            raise RuntimeError(
                f"Invalid solution for E={E_GPa}, nu={nu}"
            )

        # Positive values represent shortening/expansion.
        axial_shortening = -float(
            np.mean(displacement[top_axial_dofs])
        )

        radial_expansion = float(
            np.mean(displacement[side_radial_dofs])
        )

        # Analytical solution for a homogeneous,
        # isotropic, linearly elastic cylinder under
        # uniform uniaxial compression with free sides.
        axial_exact = SIGMA * L / E
        radial_exact = nu * SIGMA * R / E

        axial_error_pct = (
            abs(axial_shortening - axial_exact)
            / abs(axial_exact) * 100.0
        )

        radial_error_pct = (
            abs(radial_expansion - radial_exact)
            / abs(radial_exact) * 100.0
        )

        rows.append({
            "young_modulus_GPa": E_GPa,
            "poisson_ratio": nu,
            "axial_stress_MPa": SIGMA / 1e6,
            "axial_shortening_FEM_mm":
                axial_shortening * 1000.0,
            "axial_shortening_exact_mm":
                axial_exact * 1000.0,
            "axial_error_pct": axial_error_pct,
            "radial_expansion_FEM_mm":
                radial_expansion * 1000.0,
            "radial_expansion_exact_mm":
                radial_exact * 1000.0,
            "radial_error_pct": radial_error_pct,
        })


# ---------------------------------------------------------
# 7. Export and report
# ---------------------------------------------------------

results = pd.DataFrame(rows)

output_file = OUTPUT_DIR / "fem_axial_benchmark.csv"
results.to_csv(output_file, index=False)

print("\nAXIAL FEM BENCHMARK")
print("-" * 72)
print(f"Axial compressive stress: {SIGMA / 1e6:.2f} MPa")
print(f"Mesh: {NR} x {NZ} subdivisions")
print(f"Parameter combinations: {len(results)}")
print(
    "Maximum axial analytical error: "
    f"{results['axial_error_pct'].max():.6f}%"
)
print(
    "Maximum radial analytical error: "
    f"{results['radial_error_pct'].max():.6f}%"
)

print("\nResults for the reference parameters:")
reference = results[
    np.isclose(results["young_modulus_GPa"], 11.244)
    & np.isclose(results["poisson_ratio"], 0.191)
]

print(reference.to_string(index=False))
print(f"\nSaved: {output_file}")

