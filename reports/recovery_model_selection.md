# Empirical Recovery Model Comparison

## Objective

This analysis compares alternative empirical models for predicting the
recovery of porosity and permeability following hydrostatic loading of
carbonate rock sample B9. Predictions are anchored to the corresponding
loading-stage measurement at the same pressure.

The comparison evaluates whether adding a pressure interaction to a
log-time recovery model improves held-out prediction performance.

## Experimental data

The input dataset contains 72 rows, including 48 recovery observations.
The recovery observations comprise two stages:

- `rest_after_5000psi`: 26, 294, and 381 hours.
- `rest_after_7000psi`: 96, 197, and 730 hours.

Each stage contains eight pressure observations at each of the three
recorded recovery times.

## Models compared

1. **Anchor baseline (`anchor_baseline`)**
   Predicts no change from the matching loading-stage measurement.

2. **Time-only model (`time_only`)**
   Models the recovery change as a function of log time.

3. **Time-pressure model (`time_pressure`)**
   Adds an interaction between normalized pressure and log time.

The models are empirical descriptions of the observed data. They are not
mechanistic constitutive laws.

## Validation approach

Leave-one-observed-recovery-time-level-out validation was used separately
for each recovery stage. In each fold, all eight observations at one
recovery time were held out, and the model was fitted using the other
two observed recovery-time levels.

Metrics were pooled over the held-out predictions for each stage,
variable, and model. RMSE and MAE quantify prediction error; lower values
are preferable. R-squared is reported as a complementary metric.

This design tests prediction at an omitted observed time level within the
available dataset. It does not constitute validation on independent rock
samples, independent pressure levels, or times outside the observed range.

## Pooled holdout results

### Recovery after the 5000 psi stage

| Variable | Model | N | RMSE | MAE | R-squared |
|---|---|---:|---:|---:|---:|
| Porosity (percentage points) | time_pressure | 24 | 0.018090 | 0.014951 | 0.985305 |
| Porosity (percentage points) | time_only | 24 | 0.019282 | 0.015989 | 0.983306 |
| Porosity (percentage points) | anchor_baseline | 24 | 0.050532 | 0.049417 | 0.885339 |
| Permeability (mD) | time_pressure | 24 | 0.089448 | 0.081103 | -0.619734 |
| Permeability (mD) | time_only | 24 | 0.089483 | 0.081150 | -0.620972 |
| Permeability (mD) | anchor_baseline | 24 | 0.093183 | 0.074833 | -0.757791 |

For porosity, the time-pressure model gives a modest improvement over the
time-only model. For permeability, the difference in RMSE between the two
time-dependent models is negligible, and both have higher MAE than the
anchor baseline. The negative R-squared values indicate weak predictive
performance relative to the held-out mean for permeability in this stage.

### Recovery after the 7000 psi stage

| Variable | Model | N | RMSE | MAE | R-squared |
|---|---|---:|---:|---:|---:|
| Porosity (percentage points) | time_only | 24 | 0.132194 | 0.114916 | 0.323361 |
| Porosity (percentage points) | time_pressure | 24 | 0.133653 | 0.114916 | 0.308347 |
| Porosity (percentage points) | anchor_baseline | 24 | 0.387483 | 0.385763 | -4.813511 |
| Permeability (mD) | time_only | 24 | 0.037528 | 0.030346 | 0.443168 |
| Permeability (mD) | time_pressure | 24 | 0.038433 | 0.030856 | 0.415974 |
| Permeability (mD) | anchor_baseline | 24 | 0.154386 | 0.152625 | -8.423939 |

For both variables, the time-only model slightly outperforms the
time-pressure model. Both time-dependent models improve substantially
over the anchor baseline in this stage.

## Model-selection decision

The time-only model is provisionally selected as the primary empirical
recovery model because it is simpler and performs comparably to, or
slightly better than, the time-pressure model across most comparisons.

The time-pressure model is retained as a sensitivity and comparison
model. Its modest porosity advantage after the 5000 psi stage does not
demonstrate a consistent benefit from adding pressure dependence.

Permeability predictions following the 5000 psi stage require particular
caution: both time-dependent models have negative held-out R-squared
values and higher MAE than the anchor baseline.

## Limitations

- Only three observed recovery-time levels are available per stage.
  Model-selection conclusions are therefore preliminary.
- The folds reuse the same rock sample and pressure levels. The results
  do not establish generalization to other samples or pressure ranges.
- The validation design evaluates omitted observed time levels; it does
  not test extrapolation beyond the recorded recovery times.
- Recovery time zero is represented by the loading-stage anchor, not by
  a separate measured recovery observation.
- The models describe empirical relationships and do not establish a
  physical recovery mechanism or a validated constitutive law.
- The available comparison does not establish statistical significance
  or quantify uncertainty from independent experimental replicates.

## Reproducibility

Run the scripts from the repository root:

```powershell
python .\analysis\fit_anchored_recovery_model.py
python .\analysis\compare_recovery_model_complexity.py
```

The scripts generate prediction, metric, and coefficient CSV files in
the `analysis` directory.

## Conclusion

The observed results support using a parsimonious time-only model as the
provisional main empirical model, while retaining the time-pressure
model for comparison. Further recovery-time measurements and validation
on independent rock samples are needed before claiming broader
predictive capability.
