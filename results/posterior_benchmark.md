# Per-patient inference benchmark

300 held-out virtual patients drawn from the NPE prior (q ~ U(0.8, 3.5); injury rate ~ log-U(0.25, 3); true baseline eGFR ~ U(20, 110)); 2-9 visits over 0.5-8 years; measurement noise SD 6 mL/min/1.73 m2. Target: latent eGFR 5 years after the last visit.

## Forecast

| Method | RMSE | 90% interval coverage | 90% interval width | CRPS | ms / patient |
|---|---|---|---|---|---|
| Population parameters | 20.52 | — | — | — | — |
| MLP ensemble (current app) | 13.33 | 18% | 7.13 | 9.12 | 2.73 |
| Least squares, rate only | 10.60 | — | — | — | 5.04 |
| Least squares, rate + eGFR0 | 10.09 | — | — | — | 133.32 |
| **NPE (posterior predictive)** | 8.91 | 93% | 24.17 | 4.09 | 46.07 |
| NPE with q fixed (ablation) | 12.07 | 68% | 25.24 | 6.53 | — |
| Exact posterior (grid reference) | 8.73 | 95% | 24.72 | 4.03 | 896.33 |

## Parameters (posterior of the NPE vs the exact grid posterior)

| Parameter | Prior SD | NPE: RMSE of posterior mean | Exact: RMSE | NPE: contraction | Exact: contraction | NPE 90% coverage | SBC p (NPE) | SBC p (exact) |
|---|---|---|---|---|---|---|---|---|
| q | 0.779 | 0.664 | 0.660 | 0.22 | 0.14 | 89% | 0.12 | 0.06 |
| log_scale | 0.717 | 0.489 | 0.493 | 0.55 | 0.51 | 91% | 0.04 | 0.53 |
| egfr0 | 25.981 | 4.072 | 4.067 | 0.98 | 0.97 | 88% | 0.26 | 0.33 |

Contraction = 1 − (mean posterior variance / prior variance): 0 means the data teach nothing about the parameter, 1 means it is pinned down. SBC p is a chi-square test of uniform simulation-based-calibration ranks (10 bins); p > 0.05 is consistent with a calibrated posterior.
