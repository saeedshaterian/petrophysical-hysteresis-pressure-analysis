# Experimental Workflow

## 1. Research Objective

This project investigates the evolution and residual changes of reservoir-rock petrophysical properties under increasing hydrostatic pressure.

The main focus is on the pressure-dependent behavior of:

* Porosity
* Permeability
* Pore volume

Particular attention is given to the changes that remain after prolonged pressure loading and subsequent relaxation periods.

## 2. Rock Samples

The experimental study was conducted on carbonate reservoir-rock core plugs collected from oil- and gas-field reservoirs in southwestern Iran.

The thesis included several core samples, including samples designated A2, A3, B6, and B9.

The present computational repository initially focuses on **Sample B9**, for which pressure-dependent CMS measurements are being digitized and analyzed.

## 3. Experimental Characterization

The experimental investigation included several rock-physics and petrophysical measurements, including:

* Porosity and permeability measurements using CMS
* Computed tomography (CT)
* Scanning electron microscopy (SEM)
* Energy-dispersive X-ray analysis (EDX/EDAX)
* Acoustic velocity measurements
* Uniaxial compressive strength testing for selected samples
* Brazilian tensile strength testing for selected samples

These measurements were used to characterize the rock before and after pressure-induced changes.

## 4. Hydrostatic Pressure Loading

The CMS measurements were performed at a sequence of increasing hydrostatic pressure levels.

For Sample B9, the initial pressure sequence analyzed in this repository is:

800, 1500, 2200, 2900, 3600, 4300, 5000, and 5700 psi.

At each pressure level, petrophysical properties were measured.

The initial pressure-response dataset therefore provides a baseline for evaluating the effect of increasing hydrostatic pressure on porosity and permeability.

## 5. Long-Term Pressure Loading

Following the initial pressure measurements, the sample was subjected to prolonged hydrostatic loading.

For Sample B9, the experimental sequence included:

* Prolonged loading at 5000 psi
* Relaxation/rest periods
* Prolonged loading at 7000 psi
* Additional relaxation/rest periods

The measurements after these loading and relaxation stages are important for distinguishing immediate pressure-dependent changes from residual or partially recoverable changes.

## 6. Relaxation and Recovery

After prolonged pressure loading, the sample was allowed to rest for specified periods before petrophysical properties were measured again.

This procedure makes it possible to investigate the degree of recovery of the rock properties after pressure removal.

In this computational analysis, the response during relaxation is treated as an important component of the petrophysical hysteresis assessment.

## 7. Petrophysical Hysteresis Concept

The term **petrophysical hysteresis** is used in this project to describe the difference between the original pressure-dependent response and the response observed after pressure loading and subsequent relaxation.

The analysis distinguishes between:

* Pressure-induced changes
* Recoverable changes during relaxation
* Residual changes remaining after relaxation

A property that does not return to its previous value after relaxation indicates a residual change associated with the pressure history of the sample.

## 8. Computational Analysis

The original experimental measurements are being converted into a structured dataset for computational analysis.

Python is used to:

1. Load the experimental data.
2. Organize measurements according to sample, pressure, and experimental stage.
3. Plot porosity versus hydrostatic pressure.
4. Plot permeability versus hydrostatic pressure.
5. Compare measurements before and after prolonged pressure loading.
6. Quantify recovery during relaxation.
7. Evaluate residual changes associated with pressure history.

The computational work represents a **post-hoc analytical extension of the experimental thesis work**. It does not imply that the original laboratory experiments were performed using Python.

## 9. Current Analysis Scope

The first computational stage focuses on Sample B9 and its initial pressure-response data.

Subsequent stages will incorporate the prolonged 5000 psi and 7000 psi loading periods and the corresponding relaxation measurements.

The final analysis will be used to investigate the relationship between hydrostatic pressure, petrophysical property changes, and partial recovery after pressure relaxation.

## 10. Data Provenance

The experimental values used in this repository are derived from the author's MSc thesis experimental measurements.

The repository is being developed as a computational research extension for reproducible analysis, visualization, and interpretation of the experimental dataset.

Where only selected or digitized measurements are included in the repository, the dataset should not be interpreted as a complete reproduction of the original laboratory database.
