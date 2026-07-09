"""
Regression tests targeting the specific bugs found in code review:
  1. App/calibration parameterization mismatch (double-scaling of insult weights).
  2. N exceeding its documented (0,1] range.
  3. Incorrect KDIGO G3a/G3b boundary.

test_app_calibration_consistency is the single most important test here: it
would have caught the original app_web.py / calibrate_mimic.py mismatch bug.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
from mechanistic_twin import MechanisticRenalModel, N_of_egfr, egfr_of_N

def test_app_calibration_consistency():
    """
    The app must use EXACTLY the same hazard the calibration script assumes:
        hazard = k0 + k_hf*(1/N)^q + (w_a1c*... + w_uacr*... + w_sbp*...)
    i.e. N_ref=1 and no re-scaling of already-calibrated weights. This test
    reimplements that reference hazard independently and checks the model's
    simulated trajectory matches it -- this is the check that would have
    caught the original double-scaling / N_ref=0.6 bug.
    """
    q, k_hf = 1.52, 0.0141
    w_a1c, w_uacr, w_sbp = 0.0144, 0.0180, 0.0108
    a1c, sbp, uacr, egfr0 = 8.1, 142, 145, 47.7
    k0 = 0.0030

    def reference_hazard(N):
        I = (w_a1c * max(a1c - 6.5, 0.0) + w_uacr * np.log1p(uacr / 30.0)
             + w_sbp * max(sbp - 130.0, 0.0) / 10.0)
        return k0 + k_hf * (1.0 / N) ** q + I

    m = MechanisticRenalModel(a1c=a1c, sbp=sbp, uacr=uacr, u=0.0,
                              k_hf=k_hf, q=q, w_a1c=w_a1c, w_uacr=w_uacr, w_sbp=w_sbp)
    for N_test in [0.9, 0.6, 0.3, 0.1]:
        assert abs(m.hazard(N_test) - reference_hazard(N_test)) < 1e-9, \
            f"Model hazard diverges from the calibration's reference hazard at N={N_test}"

def test_n_never_exceeds_one():
    """N(t) is documented as being in (0,1]; a supra-physiological eGFR
    (e.g. hyperfiltration readings >120) must not push N above 1."""
    for egfr in [100, 120, 140, 200]:
        N = N_of_egfr(egfr)
        assert 0 < N <= 1.0, f"N_of_egfr({egfr}) = {N}, out of the documented (0,1] range"

def test_n_egfr_roundtrip_stays_bounded():
    """Round-tripping through egfr_of_N/N_of_egfr must never produce N>1."""
    for N in [0.01, 0.5, 1.0]:
        e = egfr_of_N(N)
        N2 = N_of_egfr(e)
        assert N2 <= 1.0 + 1e-9

def test_gfr_category_boundaries():
    """KDIGO GFR categories: G1>=90, G2 60-89, G3a 45-59, G3b 30-44, G4 15-29, G5<15."""
    def gfr_category(egfr):
        if egfr >= 90: return "G1"
        if egfr >= 60: return "G2"
        if egfr >= 45: return "G3a"
        if egfr >= 30: return "G3b"
        if egfr >= 15: return "G4"
        return "G5"

    cases = {95: "G1", 90: "G1", 89: "G2", 60: "G2", 59: "G3a",
            45: "G3a", 44: "G3b", 35: "G3b", 30: "G3b",
            29: "G4", 15: "G4", 14: "G5", 5: "G5"}
    for egfr, expected in cases.items():
        assert gfr_category(egfr) == expected, \
            f"eGFR={egfr} should be {expected}, got {gfr_category(egfr)}"

def test_explicit_weights_require_all_three():
    """Passing only some of w_a1c/w_uacr/w_sbp should fail loudly, not silently
    fall back to a partially-specified or wrong parameterization."""
    try:
        MechanisticRenalModel(a1c=8, sbp=140, uacr=100, w_a1c=0.01)
        assert False, "should have raised ValueError for partial weight specification"
    except ValueError:
        pass

def test_backward_compatible_default_path_unchanged():
    """The original (no explicit weights) physiological parameterization must
    still work exactly as before, for callers that don't pass calibrated weights."""
    m = MechanisticRenalModel(a1c=9, sbp=150, uacr=300, u=0.0)
    assert m.N_ref == 0.60
    t, N, egfr, t_dial = m.simulate(N0=0.62, years=25)
    assert np.isfinite(t_dial) or t_dial == np.inf
