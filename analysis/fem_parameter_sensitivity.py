
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skfem import asm

# Make the existing model importable when run from the repository root.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))

import fem_history_fem as base


OUTPUT_DIR = ROOT / "figures"
OUTPUT_DIR.mkdir(exist_ok=True)

# Baseline parameters are illustrative, not calibrated B9 properties.
E_BASE = 11.244e9
NU_BASE = 0.191

# Explore E from 50% to 150% of baseline.
E_VALUES_GPA = [
    0.50 * E_BASE / 1e9,
    0.75 * E_BASE / 1e9,
    1.00 * E_BASE / 1e9,
    1.25 * E_BASE / 1e9,
    1.50 * E_BASE / 1e9,
]

# Include the baseline Poisson ratio and a plausible exploratory range.
NU_VALUES = sorted(set([0.15, NU_BASE, 0.25, 0.30, 0.35]))

CASES = [
    "pressure_only",
    "history_5000_contribution",
    "history_7000_increment_contribution",
    "combined_5000_scenario",
    "combined_7000_scenario",
]


def solve_at_parameters(E_pa, nu):
    """Reassemble and solve the FEM system for one E, nu pair."""
    if E_pa <= 0 or not (-1.0 < nu < 0.5):
        raise ValueError(f"Invalid parameters: E={E_pa}, nu={nu}")

    # Update parameters used by the original weak forms.
    base.E = float(E_pa)
    base.nu = float(nu)
    base.mu = base.E / (2.0 * (1.0 + base.nu))
    base.lam = (
        base.E * base.nu
        / ((1.0 + base.nu) * (1.0 - 2.0 * base.nu))
    )
    base.K_bulk = base.E / (3.0 * (1.0 - 2.0 * base.nu))

    # Rebuild stiffness and all load vectors.
    K = asm(base.elasticity, base.basis)

    f_pressure = asm(
        base.pressure_load, base.fbasis, p=base.P
    )
    f_history_5000 = asm(
        base.history_load,
        base.basis,
        eps_star=base.eps_star_5000,
    )
    f_history_increment_7000 = asm(
        base.history_load,
        base.basis,
        eps_star=base.eps_star_increment_7000,
    )

    def solve_force(force):
        from skfem import condense, solve

        u = solve(*condense(K, force, D=base.D))
        if not np.all(np.isfinite(u)):
            raise RuntimeError("Non-finite FEM solution.")
        return u

    u_pressure = solve_force(f_pressure)
    u_history_5000 = solve_force(f_history_5000)
    u_history_increment = solve_force(f_history_increment_7000)

    solutions = {
        "pressure_only": u_pressure,
        "history_5000_contribution": u_history_5000,
        "history_7000_increment_contribution": u_history_increment,
        "combined_5000_scenario": u_pressure + u_history_5000,
        "combined_7000_scenario": (
            u_pressure + u_history_5000 + u_history_increment
        ),
    }

    rows = []
    for case, u in solutions.items():
        axial_mm, radial_mm = base.extract_deformation(u)
        rows.append({
            "case": case,
            "young_modulus_GPa_assumed": E_pa / 1e9,
            "poisson_ratio_assumed": nu,
            "axial_shortening_mm": axial_mm,
            "radial_contraction_mm": radial_mm,
        })

    return rows


def main():
    rows = []

    for E_gpa in E_VALUES_GPA:
        for nu in NU_VALUES:
            rows.extend(solve_at_parameters(E_gpa * 1e9, nu))

    results = pd.DataFrame(rows)

    # Add changes relative to the baseline E and nu for each case.
    baseline_mask = (
        np.isclose(results["young_modulus_GPa_assumed"], E_BASE / 1e9)
        & np.isclose(results["poisson_ratio_assumed"], NU_BASE)
    )
    baseline = results.loc[baseline_mask].set_index("case")

    if len(baseline) != len(CASES):
        raise RuntimeError(
            "Baseline parameter combination was not found in sensitivity grid."
        )

    results["axial_change_vs_baseline_pct"] = results.apply(
        lambda r: 100.0 * (
            r["axial_shortening_mm"]
            / baseline.loc[r["case"], "axial_shortening_mm"] - 1.0
        ),
        axis=1,
    )
    results["radial_change_vs_baseline_pct"] = results.apply(
        lambda r: 100.0 * (
            r["radial_contraction_mm"]
            / baseline.loc[r["case"], "radial_contraction_mm"] - 1.0
        ),
        axis=1,
    )

    csv_path = OUTPUT_DIR / "fem_parameter_sensitivity.csv"
    results.to_csv(csv_path, index=False)

    # Heatmaps for the combined 7000-stage scenario.
    combined = results[
        results["case"].eq("combined_7000_scenario")
    ]

    for column, title, filename, cbar_label in [
        (
            "axial_shortening_mm",
            "Axial shortening sensitivity",
            "fem_sensitivity_axial.png",
            "Axial shortening (mm)",
        ),
        (
            "radial_contraction_mm",
            "Radial contraction sensitivity",
            "fem_sensitivity_radial.png",
            "Radial contraction (mm)",
        ),
    ]:
        pivot = combined.pivot(
            index="poisson_ratio_assumed",
            columns="young_modulus_GPa_assumed",
            values=column,
        ).sort_index()

        fig, ax = plt.subplots(figsize=(8, 5))
        im = ax.imshow(
            pivot.to_numpy(),
            origin="lower",
            aspect="auto",
            interpolation="nearest",
        )
        ax.set_xticks(np.arange(len(pivot.columns)))
        ax.set_xticklabels(
            [f"{x:.3f}" for x in pivot.columns]
        )
        ax.set_yticks(np.arange(len(pivot.index)))
        ax.set_yticklabels(
            [f"{x:.3f}" for x in pivot.index]
        )
        ax.set_xlabel("Assumed Young's modulus E (GPa)")
        ax.set_ylabel("Assumed Poisson's ratio")
        ax.set_title(title + "\nCombined 7000-stage scenario")
        cb = fig.colorbar(im, ax=ax)
        cb.set_label(cbar_label)
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / filename, dpi=200)
        plt.close(fig)

    print("\nFEM PARAMETER SENSITIVITY COMPLETE")
    print(f"Parameter combinations: {len(E_VALUES_GPA) * len(NU_VALUES)}")
    print(f"CSV saved: {csv_path}")
    print(f"Axial plot: {OUTPUT_DIR / 'fem_sensitivity_axial.png'}")
    print(f"Radial plot: {OUTPUT_DIR / 'fem_sensitivity_radial.png'}")

    print("\nCombined 7000-stage scenario:")
    print(
        combined[
            [
                "young_modulus_GPa_assumed",
                "poisson_ratio_assumed",
                "axial_shortening_mm",
                "radial_contraction_mm",
                "axial_change_vs_baseline_pct",
                "radial_change_vs_baseline_pct",
            ]
        ].to_string(index=False, float_format=lambda x: f"{x:.5f}")
    )

    print(
        "\nLIMITATION: This is sensitivity analysis, not calibration "
        "or independent mechanical validation of sample B9."
    )


if __name__ == "__main__":
    main()
