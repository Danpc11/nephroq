"""Unit tests for the eGFR measurement model (isolated component)."""
import sys, os

import numpy as np
import pytest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from egfr_measurement import egfr_cr, egfr_cys, egfr_cr_cys

def test_egfr_physiological_range():
    for scr in [0.6, 1.0, 1.5, 2.5, 4.0]:
        e = egfr_cr(scr, 55, 'M')
        assert 0 < e < 160, f"eGFR out of physiological range: {e}"

def test_egfr_decreases_with_creatinine():
    e1 = egfr_cr(0.8, 55, 'M'); e2 = egfr_cr(2.0, 55, 'M')
    assert e2 < e1, "eGFR must drop as creatinine rises"

def test_egfr_decreases_with_age():
    e_young = egfr_cr(1.0, 30, 'F'); e_old = egfr_cr(1.0, 75, 'F')
    assert e_old < e_young, "eGFR must drop with age, at equal creatinine"

def test_egfr_combined_between_the_two():
    scr, scys, age, sex = 1.2, 1.3, 60, 'F'
    e_cr  = egfr_cr(scr, age, sex)
    e_cys = egfr_cys(scys, age, sex)
    e_comb = egfr_cr_cys(scr, scys, age, sex)
    lo, hi = sorted([e_cr, e_cys])
    assert lo - 15 <= e_comb <= hi + 15, "the combined estimate should be close to both individual ones"

def test_egfr_reproducible():
    """Same input -> same output, exactly (basic requirement of a lab system)."""
    a = egfr_cr(1.1, 50, 'M'); b = egfr_cr(1.1, 50, 'M')
    assert a == b


def test_historical_creatinine_must_use_the_age_at_the_time():
    """
    A creatinine drawn N years ago has to be converted with the age the patient HAD
    THEN. Using today's age understates the historical eGFR, which flattens the
    apparent decline -- a systematic bias that grows with the length of the history.
    """
    from egfr_measurement import egfr_cr

    creat, age_today, sex = 1.05, 58.0, "F"
    for years_ago in (3.0, 10.0, 20.0):
        wrong = egfr_cr(creat, age_today, sex)                       # today's age
        right = egfr_cr(creat, max(age_today - years_ago, 18.0), sex)  # age at the time
        assert right > wrong, "using today's age must understate the historical eGFR"

    # and the bias must GROW with the length of the history
    bias_3 = egfr_cr(creat, age_today - 3, sex) - egfr_cr(creat, age_today, sex)
    bias_20 = egfr_cr(creat, age_today - 20, sex) - egfr_cr(creat, age_today, sex)
    assert bias_20 > bias_3


def test_ckd_epi_coefficients_are_the_published_ones():
    """
    GUARD AGAINST A PLAUSIBLE-LOOKING "FIX".

    A code review proposed rewriting these equations with two coefficients that
    are WRONG, and both errors are the kind that look right:

      * eGFRcys with a leading 135 and age base 0.9946  -> those belong to the
        creatinine-cystatin equation. eGFRcys is 133 and 0.996.
      * eGFRcr-cys with a male alpha of -0.291          -> the correct value is
        -0.144. (-0.302 is the male alpha of the CREATININE-ONLY equation, and
        -0.291 is close enough to it to pass a glance.)

    Verified against NKF, NIDDK, and Inker et al. NEJM 2021/2012. These assertions
    pin the exact published equations, so a well-meaning "vectorisation refactor"
    cannot silently change the numbers a clinician sees.
    """
    # eGFRcr, female, Scr = 0.5 (below kappa): 142 * (0.5/0.7)^-0.241 * 0.9938^50 * 1.012
    expect = 142.0 * (0.5 / 0.7) ** (-0.241) * 0.9938 ** 50 * 1.012
    assert egfr_cr(0.5, 50, "F") == pytest.approx(expect, rel=1e-9)

    # eGFRcys: leading constant 133, age base 0.996 -- NOT 135 / 0.9946
    expect = 133.0 * (1.2 / 0.8) ** (-1.328) * 0.996 ** 60 * 0.932
    assert egfr_cys(1.2, 60, "F") == pytest.approx(expect, rel=1e-9)

    # eGFRcr-cys, MALE: alpha = -0.144, NOT -0.291
    expect = (135.0 * (0.5 / 0.9) ** (-0.144) * (1.2 / 0.8) ** (-0.778)
              * 0.9961 ** 60)
    assert egfr_cr_cys(0.5, 1.2, 60, "M") == pytest.approx(expect, rel=1e-9)


def test_egfr_equations_accept_arrays_and_agree_with_the_scalar_path():
    """They are applied to whole cohort columns, so they must vectorise -- and
    vectorising must not change a single number."""
    scr = np.array([0.6, 1.2, 2.4])
    age = np.array([40.0, 60.0, 80.0])
    sex = np.array(["F", "M", "F"])

    vec = egfr_cr(scr, age, sex)
    assert vec.shape == (3,)
    for i in range(3):
        assert vec[i] == pytest.approx(egfr_cr(float(scr[i]), float(age[i]), str(sex[i])),
                                       rel=1e-12)
