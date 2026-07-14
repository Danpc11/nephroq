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

Units: Scr in mg/dL ; Scys in mg/L ; age in years ; sex='F' or 'M'.
================================================================================
"""
import numpy as np

_FLOOR = 1e-3   # numerical floor for lab values entering CKD-EPI power terms
                # (0**negative_power raises ZeroDivisionError; real lab errors
                # occasionally report 0 or near-0 for creatinine/cystatin)

def _female(sex):
    """Accepts 'F'/'M' (scalar or array) or a boolean mask."""
    a = np.asarray(sex)
    if a.dtype == bool:
        return a
    return np.char.upper(a.astype(str)) == "F"


def egfr_cr(scr, age, sex):
    """
    CKD-EPI 2021 creatinine (race-free). Scalars OR numpy arrays.

    eGFR = 142 * min(Scr/k,1)^a * max(Scr/k,1)^-1.200 * 0.9938^age * 1.012 [if F]
      k = 0.7 (F) / 0.9 (M);   a = -0.241 (F) / -0.302 (M)

    Vectorised with np.where so it can be applied to a whole cohort column at once
    instead of being called row by row (the scalar path raised the classic
    "truth value of an array is ambiguous" ValueError). ~20x faster on 1.5M rows,
    though in absolute terms that is ~2s -- it is a correctness/ergonomics win far
    more than a performance one.

    Coefficients verified against NKF / NIDDK / Inker NEJM 2021. Note in
    particular a = -0.241/-0.302 here, which are the CREATININE-ONLY values; the
    creatinine-cystatin equation uses different ones (see egfr_cr_cys).
    """
    female = _female(sex)
    scr = np.asarray(scr, dtype=float)
    age = np.asarray(age, dtype=float)

    k = np.where(female, 0.7, 0.9)
    a = np.where(female, -0.241, -0.302)
    r = np.maximum(scr, _FLOOR) / k

    e = (142.0 * np.minimum(r, 1.0) ** a * np.maximum(r, 1.0) ** (-1.200)
         * 0.9938 ** age * np.where(female, 1.012, 1.0))
    return float(e) if np.isscalar(scr) or e.ndim == 0 else e


def egfr_cys(scys, age, sex):
    """
    CKD-EPI 2012 cystatin C. Scalars OR numpy arrays.

    eGFR = 133 * min(Scys/0.8,1)^-0.499 * max(Scys/0.8,1)^-1.328 * 0.996^age
           * 0.932 [if F]

    The leading constant is 133 and the age base is 0.996 -- NOT 135 / 0.9946,
    which belong to the creatinine-cystatin equation and are a common
    transcription error. Verified against NKF, NIDDK and Inker NEJM 2012.
    """
    female = _female(sex)
    scys = np.asarray(scys, dtype=float)
    age = np.asarray(age, dtype=float)

    r = np.maximum(scys, _FLOOR) / 0.8
    e = (133.0 * np.minimum(r, 1.0) ** (-0.499) * np.maximum(r, 1.0) ** (-1.328)
         * 0.996 ** age * np.where(female, 0.932, 1.0))
    return float(e) if np.isscalar(scys) or e.ndim == 0 else e


def egfr_cr_cys(scr, scys, age, sex):
    """
    CKD-EPI 2021 creatinine + cystatin C (the most precise). Scalars OR arrays.

    eGFR = 135 * min(Scr/k,1)^a * max(Scr/k,1)^-0.544
               * min(Scys/0.8,1)^-0.323 * max(Scys/0.8,1)^-0.778
               * 0.9961^age * 0.963 [if F]
      k = 0.7 (F) / 0.9 (M);   a = -0.219 (F) / **-0.144 (M)**

    The male exponent is -0.144, NOT -0.291. Verified against NKF, NIDDK,
    Medscape/QxMD and Inker NEJM 2021.
    """
    female = _female(sex)
    scr = np.asarray(scr, dtype=float)
    scys = np.asarray(scys, dtype=float)
    age = np.asarray(age, dtype=float)

    k = np.where(female, 0.7, 0.9)
    a = np.where(female, -0.219, -0.144)
    rc = np.maximum(scr, _FLOOR) / k
    ry = np.maximum(scys, _FLOOR) / 0.8

    e = (135.0 * np.minimum(rc, 1.0) ** a * np.maximum(rc, 1.0) ** (-0.544)
         * np.minimum(ry, 1.0) ** (-0.323) * np.maximum(ry, 1.0) ** (-0.778)
         * 0.9961 ** age * np.where(female, 0.963, 1.0))
    return float(e) if np.isscalar(scr) or e.ndim == 0 else e

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
