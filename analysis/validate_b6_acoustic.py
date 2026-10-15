
from pathlib import Path

import pandas as pd


# Project root: parent of the analysis directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_FILE = PROJECT_ROOT / "data" / "b6_acoustic_elastic_properties.csv"
OUTPUT_FILE = PROJECT_ROOT / "analysis" / "b6_validation_results.csv"


def calculate_properties(row):
    vp = float(row["vp_m_s"])
    vs = float(row["vs_m_s"])

    nu_reported = float(row["poisson_ratio_reported"])
    young_reported = float(row["young_modulus_GPa_reported"])
    shear_reported = float(row["shear_modulus_GPa_reported"])

    # Poisson's ratio calculated from P- and S-wave velocities
    nu_calculated = (
        (vp**2 - 2 * vs**2)
        / (2 * (vp**2 - vs**2))
    )

    # Young's modulus calculated from reported G and nu
    young_from_reported = 2 * shear_reported * (1 + nu_reported)

    # Young's modulus using the Poisson's ratio calculated from velocities
    young_from_velocities = 2 * shear_reported * (1 + nu_calculated)

    # Density implied by reported G and measured Vs.
    # This is not an independent density measurement.
    implied_density_kg_m3 = (
        shear_reported * 1e9 / vs**2
    )

    return pd.Series({
        "poisson_ratio_calculated": nu_calculated,
        "poisson_ratio_difference": (
            nu_calculated - nu_reported
        ),
        "young_from_reported_G_nu_GPa": young_from_reported,
        "young_from_velocity_nu_GPa": young_from_velocities,
        "young_modulus_difference_GPa": (
            young_from_reported - young_reported
        ),
        "density_implied_kg_m3": implied_density_kg_m3,
    })


def main():
    df = pd.read_csv(INPUT_FILE)

    results = pd.concat(
        [df, df.apply(calculate_properties, axis=1)],
        axis=1,
    )

    results.to_csv(OUTPUT_FILE, index=False)

    columns_to_show = [
        "sample",
        "stage",
        "poisson_ratio_reported",
        "poisson_ratio_calculated",
        "young_modulus_GPa_reported",
        "young_from_reported_G_nu_GPa",
        "shear_modulus_GPa_reported",
        "density_implied_kg_m3",
    ]

    print("\nB6 acoustic and elastic-property validation\n")
    print(results[columns_to_show].round(6).to_string(index=False))
    print(f"\nResults saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
