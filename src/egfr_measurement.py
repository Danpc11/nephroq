"""
================================================================================
eGFR MEASUREMENT MODEL  ·  CKD-EPI 2021 equations (race-free)
================================================================================
The twin does NOT observe the filtration rate directly: it observes eGFR,
CALCULATED from blood biomarkers (creatinine and/or cystatin C) + age + sex.

Three assays, from lower to higher precision (and lower measurement noise):
  1) eGFR_cr      : creatinine only            -> higher noise
  2) eGFR_cys     : cystatin C only              -> medium noise
  3) eGFR_cr_cys  : creatinine + cystatin (best) -> lower noise  <- recommended

Lower noise => the feedback exponent q is identified much better
(see noise_identifiability.py: q = 1.78±0.15 -> 1.61±0.11 -> 1.54±0.03).

Units: Scr in mg/dL ; Scys in mg/L ; age in years ; sex='F' or 'M'.
================================================================================
"""
import numpy as np

_FLOOR = 1e-3   # numerical floor for lab values entering CKD-EPI power terms
                # (0**negative_power raises ZeroDivisionError; real lab errors
                # occasionally report 0 or near-0 for creatinine/cystatin)

def egfr_cr(scr, age, sex):
    """CKD-EPI 2021 creatinine (race-free)."""
    k, a = (0.7, -0.241) if sex == 'F' else (0.9, -0.302)
    r = max(scr, _FLOOR) / k
    e = 142 * min(r, 1)**a * max(r, 1)**(-1.200) * 0.9938**age
    return e * (1.012 if sex == 'F' else 1.0)

def egfr_cys(scys, age, sex):
    """CKD-EPI 2021 cystatin C."""
    r = max(scys, _FLOOR) / 0.8
    e = 133 * min(r, 1)**(-0.499) * max(r, 1)**(-1.328) * 0.996**age
    return e * (0.932 if sex == 'F' else 1.0)

def egfr_cr_cys(scr, scys, age, sex):
    """CKD-EPI 2021 creatinine + cystatin C (the most precise)."""
    k, a = (0.7, -0.219) if sex == 'F' else (0.9, -0.144)
    rc = max(scr, _FLOOR) / k; ry = max(scys, _FLOOR) / 0.8
    e = (135 * min(rc, 1)**a * max(rc, 1)**(-0.544)
         * min(ry, 1)**(-0.323) * max(ry, 1)**(-0.778) * 0.9961**age)
    return e * (0.963 if sex == 'F' else 1.0)

# Measurement noise (eGFR sigma) by assay. INDICATIVE VALUES: calibrate to the
# local lab's coefficient of variation + biological variability.
NOISE_BY_ASSAY = {
    "creatinine":            3.5,
    "cystatin":              2.6,
    "creatinine+cystatin":   1.8,
}

if __name__ == "__main__":
    # example: same person, three assays
    print("Woman, 60y, Scr=1.2 mg/dL, Scys=1.3 mg/L:")
    print(f"  eGFR_cr      = {egfr_cr(1.2,60,'F'):.1f}")
    print(f"  eGFR_cys     = {egfr_cys(1.3,60,'F'):.1f}")
    print(f"  eGFR_cr_cys  = {egfr_cr_cys(1.2,1.3,60,'F'):.1f}")
