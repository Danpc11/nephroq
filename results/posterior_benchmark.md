# Per-patient inference benchmark

300 held-out virtual patients drawn from the NPE prior (q ~ U(0.8, 3.5); injury rate ~ log-U(0.25, 6); true baseline eGFR ~ U(20, 110)); 2-9 visits over 0.5-8 years; measurement noise SD 6 mL/min/1.73 m2. Target: latent eGFR 5 years after the last visit.

## Forecast

| Method | RMSE | 90% interval coverage | 90% interval width | CRPS | ms / patient |
|---|---|---|---|---|---|
| Population parameters | 24.87 | — | — | — | — |
| MLP ensemble (current app) | 12.46 | 33% | 7.17 | 7.11 | 2.66 |
| Least squares, rate only | 11.29 | — | — | — | 5.47 |
| Least squares, rate + eGFR0 | 10.39 | — | — | — | 171.57 |
| **NPE (posterior predictive)** | 9.47 | 91% | 20.74 | 3.63 | 61.30 |
| NPE with q fixed (ablation) | 11.87 | 83% | 24.89 | 5.49 | — |
| Exact posterior (grid reference) | 9.35 | 92% | 21.32 | 3.59 | 962.17 |

## Parameters (posterior of the NPE vs the exact grid posterior)

| Parameter | Prior SD | NPE: RMSE of posterior mean | Exact: RMSE | NPE: contraction | Exact: contraction | NPE 90% coverage | SBC p (NPE) | SBC p (exact) |
|---|---|---|---|---|---|---|---|---|
| q | 0.779 | 0.674 | 0.671 | 0.21 | 0.12 | 89% | 0.10 | 0.08 |
| log_scale | 0.917 | 0.543 | 0.550 | 0.68 | 0.64 | 90% | 0.13 | 0.38 |
| egfr0 | 25.981 | 4.374 | 4.350 | 0.97 | 0.97 | 88% | 0.05 | 0.69 |

Contraction = 1 − (mean posterior variance / prior variance): 0 means the data teach nothing about the parameter, 1 means it is pinned down. SBC p is a chi-square test of uniform simulation-based-calibration ranks (10 bins); p > 0.05 is consistent with a calibrated posterior.
