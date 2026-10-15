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

ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "experimental_data.csv"
OUTPUT_DIR = ROOT / "figures"
OUTPUT_DIR.mkdir(exist_ok=True)

# Illustrative geometry and elastic parameters.
R = 0.01885
L = 0.06818
E = 11.244e9
nu = 0.191

NR = 20
NZ = 40
PSI_TO_PA = 6894.757293

mu = E / (2.0 * (1.0 + nu))
lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
K_bulk = E / (3.0 * (1.0 - 2.0 * nu))

if E <= 0 or not (-1.0 < nu < 0.5):
    raise ValueError("Invalid elastic properties.")


# ---------------------------------------------------------
# Load matched B9 data
# ---------------------------------------------------------

data = pd.read_csv(DATA_FILE)
b9 = data[data["sample"].eq("B9")]

initial = (
    b9[b9["stage"].eq("initial_loading")]
    .set_index("pressure_psi")
)

loaded = (
    b9[b9["stage"].eq("7000psi_loading")]
    .set_index("pressure_psi")
)

pressures = sorted(
    set(initial.index).intersection(set(loaded.index))
)

if not pressures:
    raise ValueError("No matched pressure levels found.")


# ---------------------------------------------------------
# Axisymmetric FEM mesh
# ---------------------------------------------------------

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

top_facets = mesh.facets_satisfying(
    lambda x: np.isclose(x[1], L)
)
bottom_facets = mesh.facets_satisfying(
    lambda x: np.isclose(x[1], 0.0)
)
side_facets = mesh.facets_satisfying(
    lambda x: np.isclose(x[0], R)
)

fbasis = FacetBasis(mesh, element, facets=loaded_facets)


# ---------------------------------------------------------
# Weak forms
# ---------------------------------------------------------

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
    return 3.0 * K_bulk * eps_star * div_v * w.x[0]


# Assemble pressure-independent stiffness once.
K = asm(elasticity, basis)

radial_axis_dofs = basis.get_dofs(
    facets=axis_facets
).all(["u^1"])

bottom_axis = basis.get_dofs(
    lambda x: np.isclose(x[0], 0.0)
    & np.isclose(x[1], 0.0)
).all(["u^2"])

D = np.unique(np.concatenate([radial_axis_dofs, bottom_axis]))

top_dofs = basis.get_dofs(
    facets=top_facets
).all(["u^2"])

bottom_dofs = basis.get_dofs(
    facets=bottom_facets
).all(["u^2"])

side_radial_dofs = basis.get_dofs(
    facets=side_facets
).all(["u^1"])


# ---------------------------------------------------------
# Pressure sweep
# ---------------------------------------------------------

rows = []

print("B9 HISTORY-DEPENDENT FEM PRESSURE SWEEP")
print("=" * 65)

for pressure_psi in pressures:
    pressure_pa = float(pressure_psi) * PSI_TO_PA

    phi_initial = (
        float(initial.loc[pressure_psi, "porosity_pct"]) / 100.0
    )
    phi_loaded = (
        float(loaded.loc[pressure_psi, "porosity_pct"]) / 100.0
    )

    # Effective volume-change proxy inferred from porosity.
    volume_ratio = (1.0 - phi_initial) / (1.0 - phi_loaded)
    eps_v_history = volume_ratio - 1.0

    # Isotropic eigenstrain approximation.
    eps_star = eps_v_history / 3.0

    f_pressure = asm(
        pressure_load, fbasis, p=pressure_pa
    )
    f_history = asm(history_load, basis)

    # Combined pressure and history-strain contributions.
    u = solve(*condense(K, f_pressure + f_history, D=D))

    if not np.all(np.isfinite(u)):
        raise RuntimeError(
            f"Invalid FEM solution at {pressure_psi} psi."
        )

    top_mean = float(np.mean(u[top_dofs]))
    bottom_mean = float(np.mean(u[bottom_dofs]))

    axial_history_mm = abs(top_mean - bottom_mean) * 1000.0
    radial_history_mm = abs(
        float(np.mean(u[side_radial_dofs]))
    ) * 1000.0

    axial_elastic_mm = (
        pressure_pa * L / (3.0 * K_bulk) * 1000.0
    )
    radial_elastic_mm = (
        pressure_pa * R / (3.0 * K_bulk) * 1000.0
    )

    rows.append({
        "pressure_psi": pressure_psi,
        "initial_porosity_pct": phi_initial * 100.0,
        "post_loading_porosity_pct": phi_loaded * 100.0,
        "inferred_volume_change_pct": eps_v_history * 100.0,
        "eigenstrain_per_direction": eps_star,
        "elastic_axial_shortening_mm": axial_elastic_mm,
        "history_axial_shortening_mm": axial_history_mm,
        "elastic_radial_contraction_mm": radial_elastic_mm,
        "history_radial_contraction_mm": radial_history_mm,
    })

results = pd.DataFrame(rows)

output_file = OUTPUT_DIR / "fem_pressure_sweep.csv"
results.to_csv(output_file, index=False)

print(results.round(6).to_string(index=False))
print(f"\nSaved: {output_file}")
print("\nSweep completed.")
