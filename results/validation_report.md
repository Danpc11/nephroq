# NephroQ — Validation Report

**Data source:** synthetic (demonstration)
**Patients:** 198

## Calibrated population parameters
- Feedback exponent **q = 1.64**
- Hyperfiltration k_hf = 0.0131
- Insult weights (A1c, UACR, SBP) = [0.0159, 0.0, 0.0094]

## Fit
- R² (observed vs predicted) = **0.97**

## Forecast validation (fit the past -> predict the future)
- Patients evaluated: 176
- Forecast RMSE, **mechanistic model = 3.90** mL/min/1.73m²
- Forecast RMSE, linear extrapolation = 4.22 mL/min/1.73m²
- The model beats the line in **57%** of patients
- Correlation of predicted vs observed final eGFR: **r = 0.98**

> NOTE: synthetic data. Real validation requires patient data.
