# Experimental Data

This directory contains the experimental data used for the computational analysis of petrophysical responses to hydrostatic pressure.

## Data Source

The dataset is derived from experimental measurements conducted as part of an MSc research project in Petroleum Engineering.

The original experimental work investigated changes in petrophysical properties of carbonate reservoir-rock samples under controlled hydrostatic pressure conditions.

The present GitHub repository provides a post-hoc computational analysis of selected experimental data.

## Current Dataset

The current `experimental_data.csv` file contains measurements for sample **B9** during the initial hydrostatic-pressure loading stage.

The data include measurements at hydrostatic pressures ranging from **800 to 5700 psi**.

## Data Columns

| Column            | Description                                             | Unit  |
| ----------------- | ------------------------------------------------------- | ----- |
| `sample`          | Rock sample identifier                                  | —     |
| `pressure_psi`    | Applied hydrostatic pressure                            | psi   |
| `porosity_pct`    | Measured porosity                                       | %     |
| `permeability_mD` | Measured permeability                                   | mD    |
| `pore_volume_cc`  | Measured pore volume                                    | cm³   |
| `stage`           | Experimental pressure/loading stage                     | —     |
| `rest_time_hr`    | Rest or relaxation time associated with the measurement | hours |

## Sample B9

Sample B9 is a carbonate reservoir-rock sample investigated during the experimental program.

The current dataset represents the initial pressure-loading response before incorporating the subsequent long-term loading and recovery stages.

## Experimental Stages

The complete experimental program included multiple stages of pressure loading and relaxation/recovery.

These stages are progressively being incorporated into the computational analysis, including:

* Initial hydrostatic pressure loading
* Long-term loading at elevated pressure
* Pressure release and relaxation
* Recovery measurements after different rest periods
* Additional high-pressure loading
* Comparison of loading and recovery paths

## Data Integrity

The values in this repository are intended to preserve the experimental measurements as recorded in the original research material.

No synthetic measurements have been added to the experimental dataset.

Where the original experimental record is incomplete or unavailable, the corresponding value will remain unidentified rather than being estimated without documentation.

## Computational Use

The data are analyzed using Python to investigate:

* Porosity response to hydrostatic pressure
* Permeability response to hydrostatic pressure
* Pressure-dependent petrophysical changes
* Recovery after pressure release
* Irreversible changes in petrophysical properties
* Petrophysical hysteresis

The analysis scripts are located in the `analysis/` directory, while generated figures are stored in `figures/`.

## Data Provenance

The experimental data originate from the author's MSc research project in Petroleum Engineering.

This repository is a computational extension of the original experimental research and is intended to improve reproducibility, visualization, and quantitative interpretation of the experimental results.
