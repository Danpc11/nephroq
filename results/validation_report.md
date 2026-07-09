# Validation Report — Digital Twin Diabetes -> CKD

**Data source:** synthetic (demonstration)
**Patients:** 195

## Calibrated population parameters
- Feedback exponent **q = 1.33**
- Hyperfiltration k_hf = 0.0175
- Insult weights (A1c, UACR, SBP) = [0.0124, 0.0205, 0.0113]

## Fit
- R² (observed vs predicted) = **0.98**

## Forecast validation (fit the past -> predict the future)
- Patients evaluated: 180
- Forecast RMSE, **mechanistic model = 4.03** mL/min/1.73m²
- Forecast RMSE, linear extrapolation = 4.30 mL/min/1.73m²
- The model beats the line in **55%** of patients
- Correlation of predicted vs observed final eGFR: **r = 0.98**

> NOTE: synthetic data. Real validation requires patient data.
