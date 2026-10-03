import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Project paths
ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "experimental_data.csv"
FIGURES_DIR = ROOT / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

# Load data
df = pd.read_csv(DATA_FILE)

# Select sample B9
b9 = df[df["sample"] == "B9"].copy()

# ---------------------------------------------------------
# 1. Porosity response
# ---------------------------------------------------------

plt.figure(figsize=(9, 6))

for stage in b9["stage"].unique():
    subset = b9[b9["stage"] == stage]

    if stage == "initial_loading":
        label = "Initial loading"
    elif stage == "5000psi_loading":
        label = "After 5000 psi loading"
    elif stage == "7000psi_loading":
        label = "After 7000 psi loading"
    elif stage == "rest_after_5000psi":
        label = f"Recovery after 5000 psi ({subset['rest_time_hr'].iloc[0]} h)"
    else:
        label = f"Recovery after 7000 psi ({subset['rest_time_hr'].iloc[0]} h)"

    plt.plot(
        subset["pressure_psi"],
        subset["porosity_pct"],
        marker="o",
        linewidth=1.5,
        label=label
    )

plt.xlabel("Hydrostatic Pressure (psi)")
plt.ylabel("Porosity (%)")
plt.title("B9: Porosity Response and Recovery")
plt.grid(True)
plt.legend(fontsize=8)
plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "B9_porosity_hysteresis.png",
    dpi=300
)

plt.show()


# ---------------------------------------------------------
# 2. Permeability response
# ---------------------------------------------------------

plt.figure(figsize=(9, 6))

for stage in b9["stage"].unique():
    subset = b9[b9["stage"] == stage]

    if stage == "initial_loading":
        label = "Initial loading"
    elif stage == "5000psi_loading":
        label = "After 5000 psi loading"
    elif stage == "7000psi_loading":
        label = "After 7000 psi loading"
    elif stage == "rest_after_5000psi":
        label = f"Recovery after 5000 psi ({subset['rest_time_hr'].iloc[0]} h)"
    else:
        label = f"Recovery after 7000 psi ({subset['rest_time_hr'].iloc[0]} h)"

    plt.plot(
        subset["pressure_psi"],
        subset["permeability_mD"],
        marker="o",
        linewidth=1.5,
        label=label
    )

plt.xlabel("Hydrostatic Pressure (psi)")
plt.ylabel("Permeability (mD)")
plt.title("B9: Permeability Response and Recovery")
plt.grid(True)
plt.legend(fontsize=8)
plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "B9_permeability_hysteresis.png",
    dpi=300
)

plt.show()


# ---------------------------------------------------------
# 3. Residual porosity after 7000 psi loading
# ---------------------------------------------------------

initial = b9[b9["stage"] == "initial_loading"]
high_pressure = b9[b9["stage"] == "7000psi_loading"]

initial_5700 = initial[initial["pressure_psi"] == 5700]["porosity_pct"].iloc[0]
high_5700 = high_pressure[high_pressure["pressure_psi"] == 5700]["porosity_pct"].iloc[0]

porosity_reduction = (
    (initial_5700 - high_5700)
    / initial_5700
    * 100
)

print("\nB9 Residual Porosity Analysis")
print("--------------------------------")
print(f"Initial porosity at 5700 psi: {initial_5700:.3f}%")
print(f"Porosity after 7000 psi loading: {high_5700:.3f}%")
print(f"Porosity reduction: {porosity_reduction:.2f}%")
# ---------------------------------------------------------
# 4. Recovery and irreversible porosity change
# ---------------------------------------------------------

rest_stages = b9[
    b9["stage"].isin(["rest_after_7000psi"])
].copy()

print("\nB9 Porosity Recovery Analysis")
print("--------------------------------")

for rest_time in sorted(rest_stages["rest_time_hr"].unique()):

    recovered = rest_stages[
        rest_stages["rest_time_hr"] == rest_time
    ]

    recovered_5700 = recovered[
        recovered["pressure_psi"] == 5700
    ]["porosity_pct"].iloc[0]

    recovery_percent = (
        (recovered_5700 - high_5700)
        / (initial_5700 - high_5700)
        * 100
    )

    residual_loss = (
        (initial_5700 - recovered_5700)
        / initial_5700
        * 100
    )

    print(f"\nRest time: {rest_time} h")
    print(f"Recovered porosity at 5700 psi: {recovered_5700:.3f}%")
    print(f"Recovery: {recovery_percent:.2f}%")
    print(f"Residual porosity loss: {residual_loss:.2f}%")
    
# ---------------------------------------------------------
# 5. Permeability recovery and irreversible change
# ---------------------------------------------------------

initial_perm_5700 = initial[
    initial["pressure_psi"] == 5700
]["permeability_mD"].iloc[0]

high_perm_5700 = high_pressure[
    high_pressure["pressure_psi"] == 5700
]["permeability_mD"].iloc[0]

permeability_reduction = (
    (initial_perm_5700 - high_perm_5700)
    / initial_perm_5700
    * 100
)

print("\nB9 Residual Permeability Analysis")
print("--------------------------------")
print(f"Initial permeability at 5700 psi: {initial_perm_5700:.3f} mD")
print(
    f"Permeability after 7000 psi loading: "
    f"{high_perm_5700:.3f} mD"
)
print(f"Permeability reduction: {permeability_reduction:.2f}%")

print("\nB9 Permeability Recovery Analysis")
print("--------------------------------")

for rest_time in sorted(rest_stages["rest_time_hr"].unique()):

    recovered = rest_stages[
        rest_stages["rest_time_hr"] == rest_time
    ]

    recovered_perm_5700 = recovered[
        recovered["pressure_psi"] == 5700
    ]["permeability_mD"].iloc[0]

    recovery_percent = (
        (recovered_perm_5700 - high_perm_5700)
        / (initial_perm_5700 - high_perm_5700)
        * 100
    )

    residual_loss = (
        (initial_perm_5700 - recovered_perm_5700)
        / initial_perm_5700
        * 100
    )

    print(f"\nRest time: {rest_time} h")
    print(
        f"Recovered permeability at 5700 psi: "
        f"{recovered_perm_5700:.3f} mD"
    )
    print(f"Recovery: {recovery_percent:.2f}%")
    print(f"Residual permeability loss: {residual_loss:.2f}%")
    # ---------------------------------------------------------
# 6. Recovery comparison: Porosity vs Permeability
# ---------------------------------------------------------

recovery_times = []
porosity_recovery = []
permeability_recovery = []
porosity_residual_loss = []
permeability_residual_loss = []

for rest_time in sorted(rest_stages["rest_time_hr"].unique()):

    recovered = rest_stages[
        rest_stages["rest_time_hr"] == rest_time
    ]

    recovered_por = recovered[
        recovered["pressure_psi"] == 5700
    ]["porosity_pct"].iloc[0]

    recovered_perm = recovered[
        recovered["pressure_psi"] == 5700
    ]["permeability_mD"].iloc[0]

    por_recovery = (
        (recovered_por - high_5700)
        / (initial_5700 - high_5700)
        * 100
    )

    perm_recovery = (
        (recovered_perm - high_perm_5700)
        / (initial_perm_5700 - high_perm_5700)
        * 100
    )

    por_residual = (
        (initial_5700 - recovered_por)
        / initial_5700
        * 100
    )

    perm_residual = (
        (initial_perm_5700 - recovered_perm)
        / initial_perm_5700
        * 100
    )

    recovery_times.append(rest_time)
    porosity_recovery.append(por_recovery)
    permeability_recovery.append(perm_recovery)
    porosity_residual_loss.append(por_residual)
    permeability_residual_loss.append(perm_residual)


# ---------------------------------------------------------
# 6A. Recovery comparison
# ---------------------------------------------------------

plt.figure(figsize=(9, 6))

plt.plot(
    recovery_times,
    porosity_recovery,
    marker="o",
    linewidth=2,
    label="Porosity recovery"
)

plt.plot(
    recovery_times,
    permeability_recovery,
    marker="o",
    linewidth=2,
    label="Permeability recovery"
)

plt.xlabel("Rest Time (h)")
plt.ylabel("Recovery (%)")
plt.title("B9: Porosity and Permeability Recovery")
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "B9_recovery_comparison.png",
    dpi=300
)

plt.show()


# ---------------------------------------------------------
# 6B. Residual loss comparison
# ---------------------------------------------------------

plt.figure(figsize=(9, 6))

plt.plot(
    recovery_times,
    porosity_residual_loss,
    marker="o",
    linewidth=2,
    label="Residual porosity loss"
)

plt.plot(
    recovery_times,
    permeability_residual_loss,
    marker="o",
    linewidth=2,
    label="Residual permeability loss"
)

plt.xlabel("Rest Time (h)")
plt.ylabel("Residual Loss Relative to Initial Value (%)")
plt.title("B9: Residual Petrophysical Property Loss")
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "B9_residual_loss_comparison.png",
    dpi=300
)

plt.show()
# ---------------------------------------------------------
# 7. Irreversibility Index
# ---------------------------------------------------------

porosity_ii = porosity_residual_loss[-1]
permeability_ii = permeability_residual_loss[-1]

print("\nB9 Irreversibility Index")
print("--------------------------------")
print(
    f"Porosity irreversibility index "
    f"(730 h): {porosity_ii:.2f}%"
)
print(
    f"Permeability irreversibility index "
    f"(730 h): {permeability_ii:.2f}%"
)