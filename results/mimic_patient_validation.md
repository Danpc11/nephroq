# Patient-level validation on MIMIC-IV

Cohort: 25062 patients in the loader output; eligible 9517 (excluded: {'index_not_confirmed_in_window': 3806, 'too_few_early': 5866, 'no_future': 4156, 'egfr0_out_of_domain': 1717}). Split: temporal (test = anchor_year_group 2017 - 2019, 2020 - 2022); 8663 training and 854 test patients. eGFR thinned to one median value per 90 days; observed window = first 2 years after the confirmed index.

## Cohort selection

Exclusions in order of application: {'index_not_confirmed_in_window': 3806, 'too_few_early': 5866, 'no_future': 4156, 'egfr0_out_of_domain': 1717}. Covariates come from the observed window only; values missing there or imputed by the loader were replaced by training-patient medians ({'a1c': 6.8, 'uacr0': 16.7, 'sbp': 131.0}; values imputed: {'a1c': 4606, 'uacr0': 7731, 'sbp': 4275}).

| Group | Patients | Mean age at index | Female | Median first eGFR | Median follow-up (y) |
|---|---|---|---|---|---|
| Included | 9517 | 64.7 | 51 % | 76.6 | 7.1 |
| Excluded | 15545 | 63.6 | 45 % | 80.3 | 2.3 |

Measurement-noise model learned from the training patients (28203 short-interval pairs): Student-t, scale 2.77 mL/min/1.73 m², 2.5 degrees of freedom (SD 6.19).

## Horizon 1y after the last observed visit (n = 807 patient-visits)

| Method | RMSE | MAE | Bias | 90 % coverage | CRPS |
|---|---|---|---|---|---|
| Population parameters | 17.48 | 12.70 | +2.90 | — | — |
| Least squares, rate only | 14.77 | 10.63 | -0.05 | — | — |
| MLP ensemble (current app) | 15.21 | 11.24 | +1.80 | 61 % | 8.80 |
| NPE, uniform prior | 15.20 | 10.89 | -4.29 | 66 % | 8.21 |
| NPE, reweighted to the learned prior | 14.08 | 10.11 | -1.92 | 67 % | 7.72 |
| **NPE, retrained on the learned prior** | 13.96 | 10.00 | -1.53 | 68 % | 7.63 |
| **Hybrid: NPE + learned residual correction** | 12.69 | 9.31 | +1.09 | 87 % | 6.73 |
| Per-patient OLS slope | 20.40 | 13.86 | -2.38 | 71 % | 10.43 |
| Mixed-model slope (empirical Bayes) | 15.74 | 11.20 | -1.31 | 74 % | 8.53 |
| Gradient boosting (trained on MIMIC) | 12.66 | 9.29 | +1.19 | 86 % | 6.73 |

## Horizon 3y after the last observed visit (n = 407 patient-visits)

| Method | RMSE | MAE | Bias | 90 % coverage | CRPS |
|---|---|---|---|---|---|
| Population parameters | 19.78 | 14.99 | +1.82 | — | — |
| Least squares, rate only | 18.78 | 14.04 | -2.15 | — | — |
| MLP ensemble (current app) | 18.57 | 13.91 | +0.86 | 56 % | 11.04 |
| NPE, uniform prior | 22.92 | 16.87 | -8.52 | 62 % | 12.66 |
| NPE, reweighted to the learned prior | 20.71 | 14.81 | -3.45 | 65 % | 11.54 |
| **NPE, retrained on the learned prior** | 20.92 | 14.84 | -3.32 | 64 % | 11.63 |
| **Hybrid: NPE + learned residual correction** | 17.08 | 12.35 | +2.56 | 85 % | 9.08 |
| Per-patient OLS slope | 37.33 | 26.35 | -3.36 | 64 % | 20.50 |
| Mixed-model slope (empirical Bayes) | 27.23 | 19.66 | -2.13 | 71 % | 15.09 |
| Gradient boosting (trained on MIMIC) | 17.28 | 12.39 | +2.43 | 86 % | 9.10 |

## Horizon 5y after the last observed visit (n = 38 patient-visits)

| Method | RMSE | MAE | Bias | 90 % coverage | CRPS |
|---|---|---|---|---|---|
| Population parameters | 19.97 | 16.32 | -2.85 | — | — |
| Least squares, rate only | 18.74 | 14.07 | -4.37 | — | — |
| MLP ensemble (current app) | 21.85 | 16.15 | -1.78 | 55 % | 12.86 |
| NPE, uniform prior | 21.04 | 16.55 | -8.42 | 55 % | 12.09 |
| NPE, reweighted to the learned prior | 18.48 | 12.85 | -1.67 | 68 % | 10.43 |
| **NPE, retrained on the learned prior** | 17.17 | 12.16 | -0.76 | 74 % | 9.83 |
| **Hybrid: NPE + learned residual correction** | 15.29 | 12.18 | +1.71 | 84 % | 8.62 |
| Per-patient OLS slope | 47.69 | 30.87 | +5.62 | 71 % | 25.57 |
| Mixed-model slope (empirical Bayes) | 26.65 | 19.57 | +3.36 | 74 % | 14.57 |
| Gradient boosting (trained on MIMIC) | 15.56 | 12.05 | +0.87 | 84 % | 8.64 |

## All horizons pooled (n = 1252 patient-visits)

| Method | RMSE | MAE | Bias | 90 % coverage | CRPS |
|---|---|---|---|---|---|
| Population parameters | 18.34 | 13.55 | +2.37 | — | — |
| Least squares, rate only | 16.31 | 11.85 | -0.86 | — | — |
| MLP ensemble (current app) | 16.60 | 12.26 | +1.39 | 59 % | 9.65 |
| NPE, uniform prior | 18.25 | 13.00 | -5.79 | 64 % | 9.78 |
| NPE, reweighted to the learned prior | 16.66 | 11.72 | -2.41 | 66 % | 9.04 |
| **NPE, retrained on the learned prior** | 16.64 | 11.64 | -2.09 | 67 % | 9.00 |
| **Hybrid: NPE + learned residual correction** | 14.34 | 10.39 | +1.59 | 86 % | 7.55 |
| Per-patient OLS slope | 28.11 | 18.44 | -2.46 | 69 % | 14.16 |
| Mixed-model slope (empirical Bayes) | 20.55 | 14.20 | -1.43 | 73 % | 10.84 |
| Gradient boosting (trained on MIMIC) | 14.42 | 10.38 | +1.58 | 86 % | 7.56 |

## RMSE difference versus NPE, retrained on the learned prior (patient bootstrap, 95 % CI)

Positive = the method is worse than the reference.

| Method | ΔRMSE 95 % CI |
|---|---|
| Population parameters | +0.43 to +2.98 |
| Least squares, rate only | -1.28 to +0.68 |
| MLP ensemble (current app) | -1.02 to +0.96 |
| NPE, uniform prior | +1.18 to +2.11 |
| NPE, reweighted to the learned prior | -0.29 to +0.39 |
| Hybrid: NPE + learned residual correction | -3.07 to -1.54 |
| Per-patient OLS slope | +8.98 to +14.23 |
| Mixed-model slope (empirical Bayes) | +2.77 to +5.22 |
| Gradient boosting (trained on MIMIC) | -3.03 to -1.42 |

## RMSE difference versus Gradient boosting (trained on MIMIC) (patient bootstrap, 95 % CI)

Positive = the method is worse than the reference.

| Method | ΔRMSE 95 % CI |
|---|---|
| Population parameters | +3.00 to +4.86 |
| Least squares, rate only | +1.18 to +2.65 |
| MLP ensemble (current app) | +1.55 to +2.88 |
| NPE, uniform prior | +2.79 to +4.89 |
| NPE, reweighted to the learned prior | +1.43 to +3.05 |
| NPE, retrained on the learned prior | +1.42 to +3.03 |
| Hybrid: NPE + learned residual correction | -0.29 to +0.14 |
| Per-patient OLS slope | +11.06 to +16.63 |
| Mixed-model slope (empirical Bayes) | +4.90 to +7.46 |

## Event: eGFR < 30 at a held-out visit (patients ≥ 30 at the end of the window)

| Method | Visits | Events | AUC | Brier |
|---|---|---|---|---|
| MLP ensemble (current app) | 1146 | 56 | 0.888 | 0.040 |
| NPE, uniform prior | 1146 | 56 | 0.844 | 0.074 |
| NPE, reweighted to the learned prior | 1146 | 56 | 0.837 | 0.061 |
| NPE, retrained on the learned prior | 1146 | 56 | 0.846 | 0.061 |
| Hybrid: NPE + learned residual correction | 1146 | 56 | 0.882 | 0.039 |
| Per-patient OLS slope | 1146 | 56 | 0.793 | 0.090 |
| Mixed-model slope (empirical Bayes) | 1146 | 56 | 0.847 | 0.063 |
| Gradient boosting (trained on MIMIC) | 1146 | 56 | 0.881 | 0.038 |

## Population prior learned from the training patients (empirical Bayes)

K = 2 component(s), chosen by held-out marginal likelihood ({1: 0.102, 2: 0.197}). Importance-sampling efficiency of the reweighted stage-1 draws on the test patients: 15 % (the retrained stage-2 network needs no weights).

| Component | Weight | Mean q | SD q | Median injury rate | SD log rate |
|---|---|---|---|---|---|
| 1 | 0.69 | 1.37 | 0.33 | 0.30× | 0.07 |
| 2 | 0.31 | 1.89 | 0.60 | 2.89× | 0.76 |

## Provenance

NephroQ 0.17.0, commit 1240ee1b1191; model configuration hash 264920ef6830; cohort content hash 0x539227cf463ac5b7; seed 0; network weights {'stage1': 'ae82cf9d6796', 'stage2': 'c19656354da7'}; Python 3.12.7, numpy 2.5.3, scikit-learn 1.9.1, torch 2.15.0+cpu.