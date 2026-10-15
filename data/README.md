# Experimental Data

## Data provenance

The data originate from experimental measurements associated with an MSc
research project in Petroleum Engineering. The laboratory research examined
pressure-dependent petrophysical behavior in carbonate reservoir-rock samples
from oil and gas reservoirs in southwestern Iran.

This repository provides a post-hoc computational analysis of selected
experimental measurements. The Python workflow did not generate the original
laboratory measurements.

## Current datasets

### B9 petrophysical data

`experimental_data.csv` contains 72 rows for sample B9, covering five stages:

| Stage | Rows | Description |
|---|---:|---|
| `initial_loading` | 8 | Initial hydrostatic loading from 800 to 5700 psi |
| `5000psi_loading` | 8 | Measurements associated with the prolonged 5000 psi stage |
| `rest_after_5000psi` | 24 | Recovery measurements at three recorded rest times |
| `7000psi_loading` | 8 | Measurements associated with the prolonged 7000 psi stage |
| `rest_after_7000psi` | 24 | Recovery measurements at three recorded rest times |

Each recovery stage contains eight pressure observations at each of three
recorded rest times. These are repeated observations across pressure conditions,
not independent rock samples.

The recorded rest times are 26, 294, and 381 hours after the 5000 psi stage,
and 96, 197, and 730 hours after the 7000 psi stage.

### B6 acoustic and elastic data

`b6_acoustic_elastic_properties.csv` is a separate dataset used for acoustic
and elastic-property analysis. It must not be treated as part of the B9
petrophysical dataset.

## B9 data columns

| Column | Description | Unit |
|---|---|---|
| `sample` | Rock sample identifier | Dimensionless |
| `pressure_psi` | Applied hydrostatic pressure | psi |
| `porosity_pct` | Measured porosity | % |
| `permeability_mD` | Measured permeability | mD |
| `pore_volume_cc` | Recorded pore volume | cm^3 |
| `stage` | Experimental loading or recovery stage | Dimensionless |
| `rest_time_hr` | Recorded rest or recovery time, where applicable | hours |

## Data integrity and interpretation

The CSV files are intended to preserve the values available in the source
research records. Derived model predictions and synthetic numerical results
are stored separately from the experimental input data.

Model outputs must not be interpreted as laboratory measurements. Any missing,
derived, or assumed values should be identified and documented by the relevant
analysis script.

The recovery analyses compare measurements with the corresponding loading-stage
values at matching pressure. The loading-stage anchor is a reference value,
not a separate recovery measurement at time zero.

## Computational use

The datasets support analysis of pressure-dependent porosity and permeability,
recovery after prolonged loading, residual property changes, petrophysical
hysteresis, and selected acoustic, elastic, and numerical-modeling questions.

Analysis scripts are located in `analysis/`, figures and numerical-result
files in `figures/`, and supporting workflow documentation in `methodology/`.
