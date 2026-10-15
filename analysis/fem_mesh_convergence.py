
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

OUTPUT_FILE = OUTPUT_DIR / "fem_mesh_convergence.csv"
SUMMARY_FILE = OUTPUT_DIR / "fem_mesh_convergence_summary.csv"

# Illustrative geometry and elastic properties.
# These are not calibrated to B9.
R = 0.01885
L = 0.06818
E = 11.244e9
nu = 0.191
PSI_TO_PA = 6894.757293

# Compare increasing mesh resolutions.
MESH_LEVELS = [
    (10, 20),
    (20, 40),
    (40, 80),
    (80, 160),
]

TEST_PRESSURES_PSI = [800.0, 3600.0, 5700.0]

if E <= 0 or not (-1.0 < nu < 0.5):
    raise ValueError("Invalid elastic properties.")

if R <= 0 or L <= 0:
    raise ValueError("Invalid cylinder geometry.")

mu = E / (2.0 * (1.0 + nu))
lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

# =========================================================
# 2. FEM forms
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
# 3. Analytical solution for a homogeneous cylinder
# =========================================================

def analytical_solution(pressure_psi):
    """
    Analytical displacement for a homogeneous, isotropic,
    linearly elastic cylinder under uniform external pressure
    on the curved wall and both end faces.

    Returns positive shortening/contraction in mm.
    """
    pressure_pa = pressure_psi * PSI_TO_PA

    radial_contraction_m = (
        pressure_pa * R * (1.0 - 2.0 * nu) / E
    )

    axial_shortening_m = (
        pressure_pa * L * (1.0 - 2.0 * nu) / E
    )

    return (
        axial_shortening_m * 1000.0,
        radial_contraction_m * 1000.0,
    )


# =========================================================
# 4. Solve one mesh
# =========================================================

def run_mesh(nr, nz, pressure_psi):
    mesh = MeshTri.init_tensor(
        np.linspace(0.0, R, nr + 1),
        np.linspace(0.0, L, nz + 1),
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

    force = asm(
        pressure_load,
        fbasis,
        p=pressure_psi * PSI_TO_PA,
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

    u = solve(*condense(K, force, D=D))

    if not np.isfinite(u).all():
        raise RuntimeError(
            f"Non-finite solution for mesh {nr} x {nz}"
        )

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

    top_mean = float(np.mean(u[top_dofs]))
    bottom_mean = float(np.mean(u[bottom_dofs]))
    side_mean = float(np.mean(u[side_dofs]))

    axial_mm = -(top_mean - bottom_mean) * 1000.0
    radial_mm = -side_mean * 1000.0

    return {
        "NR": nr,
        "NZ": nz,
        "mesh_elements": mesh.nelements,
        "mesh_nodes": mesh.nvertices,
        "pressure_psi": pressure_psi,
        "fem_axial_shortening_mm": axial_mm,
        "fem_radial_contraction_mm": radial_mm,
    }


# =========================================================
# 5. Run all mesh levels and pressure cases
# =========================================================

rows = []

for pressure_psi in TEST_PRESSURES_PSI:
    analytical_axial, analytical_radial = (
        analytical_solution(pressure_psi)
    )

    for nr, nz in MESH_LEVELS:
        result = run_mesh(nr, nz, pressure_psi)

        result["analytical_axial_mm"] = analytical_axial
        result["analytical_radial_mm"] = analytical_radial

        result["axial_abs_error_mm"] = abs(
            result["fem_axial_shortening_mm"]
            - analytical_axial
        )

        result["radial_abs_error_mm"] = abs(
            result["fem_radial_contraction_mm"]
            - analytical_radial
        )

        result["axial_relative_error_pct"] = (
            100.0 * result["axial_abs_error_mm"]
            / abs(analytical_axial)
        )

        result["radial_relative_error_pct"] = (
            100.0 * result["radial_abs_error_mm"]
            / abs(analytical_radial)
        )

        rows.append(result)

results = pd.DataFrame(rows)
results.to_csv(OUTPUT_FILE, index=False)

# =========================================================
# 6. Mesh-to-mesh changes
# =========================================================

summary_rows = []

for pressure_psi, group in results.groupby("pressure_psi"):
    group = group.sort_values("mesh_elements").reset_index(drop=True)

    for i, row in group.iterrows():
        if i == 0:
            axial_change_pct = np.nan
            radial_change_pct = np.nan
        else:
            prev = group.iloc[i - 1]

            axial_change_pct = (
                100.0
                * abs(
                    row["fem_axial_shortening_mm"]
                    - prev["fem_axial_shortening_mm"]
                )
                / abs(row["fem_axial_shortening_mm"])
            )

            radial_change_pct = (
                100.0
                * abs(
                    row["fem_radial_contraction_mm"]
                    - prev["fem_radial_contraction_mm"]
                )
                / abs(row["fem_radial_contraction_mm"])
            )

        summary_rows.append({
            "pressure_psi": pressure_psi,
            "NR": row["NR"],
            "NZ": row["NZ"],
            "mesh_elements": row["mesh_elements"],
            "axial_relative_error_pct":
                row["axial_relative_error_pct"],
            "radial_relative_error_pct":
                row["radial_relative_error_pct"],
            "axial_change_from_previous_mesh_pct":
                axial_change_pct,
            "radial_change_from_previous_mesh_pct":
                radial_change_pct,
        })

summary = pd.DataFrame(summary_rows)
summary.to_csv(SUMMARY_FILE, index=False)

# =========================================================
# 7. Report
# =========================================================

print("=" * 76)
print("FEM MESH CONVERGENCE TEST")
print("=" * 76)
print(f"Young's modulus (illustrative): {E / 1e9:.3f} GPa")
print(f"Poisson ratio (illustrative): {nu:.3f}")
print(f"Mesh levels: {len(MESH_LEVELS)}")
print(f"Pressure cases: {len(TEST_PRESSURES_PSI)}")
print(f"Total FEM solves: {len(results)}")

print("\nRESULTS")
print(
    results[
        [
            "pressure_psi",
            "NR",
            "NZ",
            "mesh_elements",
            "fem_axial_shortening_mm",
            "analytical_axial_mm",
            "axial_relative_error_pct",
            "fem_radial_contraction_mm",
            "analytical_radial_mm",
            "radial_relative_error_pct",
        ]
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:.8f}",
    )
)

print("\nMESH-TO-MESH CHANGES")
print(
    summary.to_string(
        index=False,
        float_format=lambda x: (
            "NaN" if np.isnan(x) else f"{x:.6f}"
        ),
    )
)

print(f"\nDetailed results: {OUTPUT_FILE}")
print(f"Summary: {SUMMARY_FILE}")

print(
    "\nNOTE: This checks numerical convergence for the homogeneous "
    "linear-elastic benchmark. It does not validate the history-dependent "
    "rock model or calibrate elastic properties to B9."
)
