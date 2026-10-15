# Pressure-History-Dependent Petrophysical Hysteresis in Carbonate Reservoir Rocks

## Overview

This repository contains a computational extension of experimental research conducted during my MSc in Petroleum Engineering. It investigates pressure-dependent changes, hysteresis, recovery, and residual loss in the petrophysical properties of carbonate reservoir rocks.

The project combines experimental-data analysis, constitutive and recovery-law fitting, temporal model comparison, and exploratory finite element method (FEM) simulations.

**The numerical models are research extensions of laboratory observations. They do not replace the original experiments, and their predictions should not be interpreted as independently validated physical results unless explicitly demonstrated.**

## Research Background

**MSc in Petroleum Engineering, Drilling and Production**
Islamic Azad University, Science and Research Branch, Tehran, Iran
Thesis completed in Fall 1396 (2017)

The underlying experimental research investigated hysteresis in reservoir-rock petrophysical properties under changes in hydrostatic pressure and drilling mud weight. The research involved carbonate limestone core samples from oil and gas reservoirs in southwest Iran, in collaboration with the Research Institute of Petroleum Industry (RIPI).

The experimental program included CMS measurements, CT imaging, SEM/EDAX characterization, acoustic measurements, and selected mechanical tests.

## Research Objectives

The computational work aims to:

- Quantify pressure-dependent changes in porosity and permeability.
- Evaluate the effects of prolonged high-pressure exposure.
- Characterize partial recovery and residual property loss during relaxation.
- Compare alternative recovery laws and temporal models.
- Investigate constitutive relationships between pressure history and petrophysical properties.
- Develop and examine FEM-based representations of pressure-dependent rock behavior.
- Assess model sensitivity and identify the additional validation needed for reliable physical interpretation.

## Experimental Data

### B9 petrophysical dataset

`data/experimental_data.csv` contains the experimental dataset used for the principal B9 analyses.

The dataset includes measurements associated with:

- Initial hydrostatic loading from 800 to 5700 psi
- Prolonged loading at 5000 psi
- Recovery following 5000 psi loading
- Prolonged loading at 7000 psi
- Recovery following 7000 psi loading

The recorded variables include sample identifier, pressure, porosity, permeability, pore volume, experimental stage, and recovery time where applicable.

### B6 acoustic and elastic properties

`data/b6_acoustic_elastic_properties.csv` contains a separate dataset used in the acoustic and elastic-property analysis.

The two datasets serve different analytical purposes and should not be treated as interchangeable.

See `data/README.md` for data descriptions and provenance.

## Key Experimental Findings: Sample B9

At the comparison pressure of 5700 psi, the recorded porosity decreased from 13.632% before the prolonged 7000 psi loading sequence to 12.535% afterward, corresponding to an 8.05% relative reduction.

Permeability decreased from 0.822 mD to 0.445 mD, a relative reduction of 45.86%.

The recovery observations show that porosity and permeability did not recover to the same extent.

| Recovery time | Porosity recovered | Residual porosity loss | Permeability recovered | Residual permeability loss |
|---|---:|---:|---:|---:|
| 96 h | 34.09% | 5.30% | 29.97% | 32.12% |
| 197 h | 34.28% | 5.29% | 33.69% | 30.41% |
| 730 h | 33.91% | 5.32% | 38.20% | 28.35% |

These results describe the available B9 observations and should not be generalized to all carbonate reservoir rocks.

## Computational Workflow

### 1. Experimental-data analysis

`analysis/pressure_response.py`

Processes the experimental data and generates plots and quantitative comparisons of pressure-dependent porosity and permeability changes.

### 2. Constitutive and recovery models

The analysis scripts explore relationships between petrophysical properties, pressure history, and recovery behavior. They include:

- `analysis/fit_petrophysical_constitutive_law.py`
- `analysis/fit_recovery_law.py`
- `analysis/fit_anchored_recovery_model.py`
- `analysis/history_state_model.py`
- `analysis/internal_state_model.py`
- `analysis/compare_recovery_models.py`
- `analysis/compare_recovery_models_fair_holdout.py`
- `analysis/compare_temporal_models.py`

Associated CSV files contain model coefficients, predictions, comparison metrics, and selected holdout results.

### 3. FEM-Based Numerical Investigation

The FEM workflow explores pressure-dependent mechanical behavior, pressure-history effects, recovery inputs, time-dependent responses, and numerical sensitivity in an axisymmetric representation of a carbonate rock sample.

Representative scripts include:

- `analysis/fem_model.py`
- `analysis/fem_hysteresis_model.py`
- `analysis/fem_history_fem.py`
- `analysis/fem_pressure_sweep.py`
- `analysis/fem_history_time_sweep.py`
- `analysis/fem_mesh_convergence.py`
- `analysis/fem_independent_validation.py`
- `analysis/fem_parameter_sensitivity.py`
- `analysis/fem_axial_benchmark.py`

#### Parameter Sensitivity

`analysis/fem_parameter_sensitivity.py` examines the response to selected combinations of Young's modulus and Poisson's ratio. The assumed material parameters are illustrative and are not independently calibrated B9 properties.

Under the current uniform hydrostatic loading and isotropic eigenstrain assumptions, the computed response is governed primarily by the bulk modulus. Different combinations of Young's modulus and Poisson's ratio that produce the same bulk modulus yield the same displacement response in the tested cases. The sensitivity results therefore do not establish independent sensitivity to these two parameters under this loading configuration.

Outputs:

- `figures/fem_parameter_sensitivity.csv`
- `figures/fem_sensitivity_axial.png`
- `figures/fem_sensitivity_radial.png`

#### Axial Analytical Benchmark

`analysis/fem_axial_benchmark.py` tests the axisymmetric linear-elastic implementation under uniform axial compression with a traction-free lateral surface. FEM displacement results are compared with analytical predictions for axial shortening and Poisson-related radial expansion across 25 combinations of assumed elastic parameters.

For the tested mesh and benchmark setup, the maximum reported relative errors for both axial and radial displacement are approximately \(10^{-11}\%\), effectively zero at the reported numerical precision. This demonstrates agreement with the analytical solution for this specific homogeneous, isotropic, linear-elastic benchmark. It does not validate the pressure-history constitutive assumptions or the porosity-derived eigenstrains used in the separate history model.

Output:

- `figures/fem_axial_benchmark.csv`

#### Interpretation and Validation Status

The FEM workflow uses illustrative elastic parameters where independent sample-specific calibration is unavailable. History-related isotropic eigenstrains are inferred from porosity changes under a constant-grain-volume assumption; they are not directly measured mechanical strains. Predicted displacements must therefore be interpreted as model outputs rather than experimentally verified deformation measurements.

Numerical convergence, agreement with an analytical benchmark, verification of equilibrium and boundary conditions, and independent physical validation are distinct requirements. Success in one does not establish the others. In particular, the axial benchmark checks the numerical implementation for a separate loading case; it does not independently validate the history-dependent constitutive law.

Further work should examine the model's equilibrium and boundary-condition checks, establish defensible sample-specific elastic parameters, and compare history-dependent displacement predictions with independent mechanical measurements where available.

## Repository Structure

The repository is organized into the following main directories and files:

```text
petrophysical-hysteresis-pressure-analysis/
|-- analysis/
|   |-- pressure_response.py
|   |-- fit_petrophysical_constitutive_law.py
|   |-- fit_recovery_law.py
|   |-- fit_anchored_recovery_model.py
|   |-- history_state_model.py
|   |-- internal_state_model.py
|   |-- compare_recovery_models.py
|   |-- compare_recovery_models_fair_holdout.py
|   |-- compare_temporal_models.py
|   |-- fem_model.py
|   |-- fem_hysteresis_model.py
|   |-- fem_history_fem.py
|   |-- fem_pressure_sweep.py
|   |   |-- fem_history_time_sweep.py
|   |-- fem_mesh_convergence.py
|   |-- fem_independent_validation.py
|   |-- fem_parameter_sensitivity.py
|   `-- fem_axial_benchmark.py
|-- data/
|   |-- experimental_data.csv
|   |-- b6_acoustic_elastic_properties.csv
|   `-- README.md
|-- figures/
|-- methodology/
|   `-- experimental_workflow.md
|-- reports/
|   `-- recovery_model_selection.md
|-- requirements.txt
|-- .gitignore
`-- README.md
```

The `analysis/` directory contains analysis scripts and associated numerical outputs. The `data/` directory contains the experimental datasets and their documentation. The `figures/` directory stores visualizations and numerical-result files, while `reports/` contains written analyses of model comparisons and limitations.

## Installation

Python 3.11 or a compatible environment is recommended, subject to verification of the installed package versions.

Install the required libraries with:

```bash
python -m pip install -r requirements.txt
```

The dependencies include NumPy, Pandas, SciPy, Matplotlib, and scikit-fem.

## Reproducing the Initial B9 Analysis

From the repository root, run:

```bash
python analysis/pressure_response.py
```

Other scripts perform separate modeling and comparison tasks. Their input paths, outputs, and execution requirements should be checked before running them individually. Not every script is intended to run as part of a single end-to-end workflow.

## Limitations and Current Development

- The experimental dataset is limited to the measurements available for the corresponding samples and stages.
- Model fitting and holdout comparisons do not, by themselves, establish generalization to other reservoir rocks.
- Numerical FEM outputs depend on the selected geometry, material assumptions, constitutive relations, boundary conditions, and loading implementation.
- Mesh sensitivity and analytical benchmarks are useful numerical checks but are not substitutes for experimental validation.
- FEM independent validation remains a work in progress.
- The numerical workflow and documentation will be refined as the models and validation tests are completed.

## Research Direction

The longer-term objective is to develop a reproducible framework connecting experimental petrophysical hysteresis, pressure-history-dependent constitutive behavior, time-dependent recovery, and numerical rock-mechanics modeling.

The repository is intended to document the development process, preserve analytical results, and make the assumptions and limitations of the work transparent.
