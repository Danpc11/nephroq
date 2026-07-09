"""Unit tests for the eGFR measurement model (isolated component)."""
import sys, os
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
