# Patient-level validation on MIMIC-IV

Cohort: 25062 patients in the loader output; eligible 9530 (excluded: {'too_few_early': 9703, 'no_future': 4117, 'egfr0_out_of_domain': 1712}). Split: temporal (test = anchor_year_group 2017 - 2019, 2020 - 2022); 8672 training and 858 test patients. eGFR thinned to one median value per 90 days; observed window = first 2 years after the confirmed index.

Measurement-noise model learned from the training patients (27782 short-interval pairs): Student-t, scale 5.26 mL/min/1.73 m², 3.3 degrees of freedom (SD 8.38).

## Horizon 1y after the last observed visit (n = 811 patient-visits)

| Method | RMSE | MAE | Bias | 90 % coverage | CRPS |
|---|---|---|---|---|---|
| Population parameters | 17.31 | 12.68 | +2.24 | — | — |
| Least squares, rate only | 14.46 | 10.47 | -0.75 | — | — |
| MLP ensemble (current app) | 14.92 | 11.06 | +1.17 | 72 % | 8.29 |
| NPE, uniform prior | 14.22 | 10.32 | -3.88 | 82 % | 7.44 |
| NPE, reweighted to the learned prior | 13.22 | 9.46 | +0.19 | 83 % | 7.03 |
| **NPE, retrained on the learned prior** | 13.27 | 9.50 | -0.03 | 83 % | 7.06 |
| **Hybrid: NPE + learned residual correction** | 12.47 | 9.10 | +1.43 | 87 % | 6.58 |
| Per-patient OLS slope | 20.31 | 13.80 | -2.29 | 72 % | 10.37 |
| Mixed-model slope (empirical Bayes) | 15.64 | 11.14 | -1.18 | 75 % | 8.44 |
| Gradient boosting (trained on MIMIC) | 12.69 | 9.27 | +1.37 | 85 % | 6.73 |

## Horizon 3y after the last observed visit (n = 411 patient-visits)

| Method | RMSE | MAE | Bias | 90 % coverage | CRPS |
|---|---|---|---|---|---|
| Population parameters | 19.72 | 15.09 | +0.44 | — | — |
| Least squares, rate only | 19.07 | 14.26 | -3.44 | — | — |
| MLP ensemble (current app) | 18.53 | 13.88 | -0.29 | 65 % | 10.65 |
| NPE, uniform prior | 20.88 | 15.66 | -7.93 | 76 % | 11.23 |
| NPE, reweighted to the learned prior | 18.26 | 13.23 | +0.09 | 75 % | 10.02 |
| **NPE, retrained on the learned prior** | 18.70 | 13.37 | -0.49 | 74 % | 10.20 |
| **Hybrid: NPE + learned residual correction** | 17.12 | 12.32 | +2.69 | 84 % | 9.06 |
| Per-patient OLS slope | 37.34 | 26.28 | -3.20 | 64 % | 20.44 |
| Mixed-model slope (empirical Bayes) | 27.33 | 19.67 | -1.99 | 70 % | 15.11 |
| Gradient boosting (trained on MIMIC) | 17.13 | 12.41 | +2.58 | 85 % | 9.07 |

## Horizon 5y after the last observed visit (n = 39 patient-visits)

| Method | RMSE | MAE | Bias | 90 % coverage | CRPS |
|---|---|---|---|---|---|
| Population parameters | 19.85 | 16.47 | -4.46 | — | — |
| Least squares, rate only | 19.08 | 14.93 | -4.97 | — | — |
| MLP ensemble (current app) | 21.75 | 16.10 | -2.69 | 69 % | 12.28 |
| NPE, uniform prior | 19.70 | 16.91 | -9.43 | 77 % | 11.20 |
| NPE, reweighted to the learned prior | 16.08 | 12.39 | +1.34 | 74 % | 9.31 |
| **NPE, retrained on the learned prior** | 16.11 | 12.45 | +0.88 | 74 % | 9.44 |
| **Hybrid: NPE + learned residual correction** | 15.51 | 12.24 | +2.03 | 85 % | 8.77 |
| Per-patient OLS slope | 47.22 | 30.55 | +5.96 | 72 % | 25.35 |
| Mixed-model slope (empirical Bayes) | 26.81 | 19.68 | +3.81 | 74 % | 14.66 |
| Gradient boosting (trained on MIMIC) | 15.85 | 12.30 | +1.63 | 85 % | 8.84 |

## All horizons pooled (n = 1261 patient-visits)

| Method | RMSE | MAE | Bias | 90 % coverage | CRPS |
|---|---|---|---|---|---|
| Population parameters | 18.21 | 13.59 | +1.44 | — | — |
| Least squares, rate only | 16.25 | 11.84 | -1.76 | — | — |
| MLP ensemble (current app) | 16.42 | 12.14 | +0.58 | 70 % | 9.18 |
| NPE, uniform prior | 16.86 | 12.26 | -5.37 | 80 % | 8.79 |
| NPE, reweighted to the learned prior | 15.13 | 10.78 | +0.19 | 80 % | 8.07 |
| **NPE, retrained on the learned prior** | 15.34 | 10.85 | -0.15 | 79 % | 8.16 |
| **Hybrid: NPE + learned residual correction** | 14.24 | 10.25 | +1.86 | 86 % | 7.45 |
| Per-patient OLS slope | 28.09 | 18.39 | -2.33 | 70 % | 14.12 |
| Mixed-model slope (empirical Bayes) | 20.57 | 14.18 | -1.29 | 73 % | 10.80 |
| Gradient boosting (trained on MIMIC) | 14.39 | 10.39 | +1.77 | 85 % | 7.56 |

## RMSE difference versus NPE, retrained on the learned prior (patient bootstrap, 95 % CI)

Positive = the method is worse than the reference.

| Method | ΔRMSE 95 % CI |
|---|---|
| Population parameters | +1.77 to +3.94 |
| Least squares, rate only | +0.07 to +1.64 |
| MLP ensemble (current app) | +0.24 to +1.86 |
| NPE, uniform prior | +1.05 to +1.96 |
| NPE, reweighted to the learned prior | -0.47 to +0.05 |
| Hybrid: NPE + learned residual correction | -1.66 to -0.57 |
| Per-patient OLS slope | +10.19 to +15.38 |
| Mixed-model slope (empirical Bayes) | +4.06 to +6.44 |
| Gradient boosting (trained on MIMIC) | -1.60 to -0.39 |

## RMSE difference versus Gradient boosting (trained on MIMIC) (patient bootstrap, 95 % CI)

Positive = the method is worse than the reference.

| Method | ΔRMSE 95 % CI |
|---|---|
| Population parameters | +2.90 to +4.83 |
| Least squares, rate only | +1.19 to +2.57 |
| MLP ensemble (current app) | +1.38 to +2.77 |
| NPE, uniform prior | +1.65 to +3.28 |
| NPE, reweighted to the learned prior | +0.28 to +1.27 |
| NPE, retrained on the learned prior | +0.39 to +1.60 |
| Hybrid: NPE + learned residual correction | -0.40 to +0.11 |
| Per-patient OLS slope | +11.01 to +16.31 |
| Mixed-model slope (empirical Bayes) | +4.83 to +7.39 |

## Event: eGFR < 30 at a held-out visit (patients ≥ 30 at the end of the window)

| Method | Visits | Events | AUC | Brier |
|---|---|---|---|---|
| MLP ensemble (current app) | 1157 | 57 | 0.887 | 0.040 |
| NPE, uniform prior | 1157 | 57 | 0.869 | 0.060 |
| NPE, reweighted to the learned prior | 1157 | 57 | 0.855 | 0.047 |
| NPE, retrained on the learned prior | 1157 | 57 | 0.858 | 0.050 |
| Hybrid: NPE + learned residual correction | 1157 | 57 | 0.891 | 0.039 |
| Per-patient OLS slope | 1157 | 57 | 0.797 | 0.090 |
| Mixed-model slope (empirical Bayes) | 1157 | 57 | 0.851 | 0.064 |
| Gradient boosting (trained on MIMIC) | 1157 | 57 | 0.889 | 0.038 |

## Population prior learned from the training patients (empirical Bayes)

K = 2 component(s), chosen by held-out marginal likelihood ({1: 0.162, 2: 0.225}). Importance-sampling efficiency of the reweighted stage-1 draws on the test patients: 11 % (the retrained stage-2 network needs no weights).

| Component | Weight | Mean q | SD q | Median injury rate | SD log rate |
|---|---|---|---|---|---|
| 1 | 0.87 | 1.38 | 0.18 | 0.40× | 0.21 |
| 2 | 0.13 | 1.73 | 0.59 | 4.23× | 0.16 |
