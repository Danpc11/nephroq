"""Tests for the model in model_core (v2: saturating hyperfiltration +
endogenous albuminuria)."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import model_core as core


def test_hyperfiltration_saturates():
    """The hazard must stay BOUNDED as nephrons vanish. An unbounded power law
    diverges and over-predicts decline in advanced CKD."""
    h = [core.hyperfiltration_hazard_v2(N, core.TRIAL_CALIBRATION_V2["k_hf"],
                                        core.TRIAL_CALIBRATION_V2["q"], s_sat=3.5)
         for N in (0.8, 0.4, 0.2, 0.05, 0.01)]
    assert all(b >= a for a, b in zip(h, h[1:]))          # monotone
    ceiling = core.TRIAL_CALIBRATION_V2["k_hf"] * 3.5 ** core.TRIAL_CALIBRATION_V2["q"]
    assert h[-1] <= ceiling * 1.01                        # bounded


def test_saturation_ceiling_is_read_at_call_time():
    """s_sat must be honoured when passed explicitly (a default argument would
    bind once at import and silently ignore any override)."""
    kw = dict(N=0.1, k_hf=core.TRIAL_CALIBRATION_V2["k_hf"],
              q=core.TRIAL_CALIBRATION_V2["q"])
    assert core.hyperfiltration_hazard_v2(s_sat=2.0, **kw) < \
           core.hyperfiltration_hazard_v2(s_sat=8.0, **kw)


def test_albuminuria_is_an_endogenous_output():
    """UACR must rise as nephrons are lost, and fall under treatment by the
    calibrated ~29% (SGLT2i trials published 31-35%)."""
    kw = dict(egfr0=47.7, a1c=8.1, uacr0=145.0, sbp=142.0, years=15)
    _, _, ua_untreated, _ = core.simulate_trajectory_v2(u=0.0, **kw)
    _, _, ua_treated, _ = core.simulate_trajectory_v2(u=1.0, **kw)
    assert ua_untreated[-1] > ua_untreated[0]
    drop = 1.0 - ua_treated[0] / ua_untreated[0]
    assert 0.25 < drop < 0.35
    assert ua_treated[-1] < ua_untreated[-1]


def test_treatment_slows_decline():
    """Use an advanced patient so the untreated arm actually crosses the threshold
    within the horizon; otherwise both times are inf and the comparison is vacuous."""
    kw = dict(egfr0=30.0, a1c=9.0, uacr0=1500.0, sbp=160.0, years=15)
    _, e_untreated, _, td_untreated = core.simulate_trajectory_v2(u=0.0, **kw)
    _, e_treated, _, td_treated = core.simulate_trajectory_v2(u=1.0, **kw)
    assert e_treated[-1] > e_untreated[-1]
    assert np.isfinite(td_untreated), "untreated patient should reach the threshold"
    assert td_treated > td_untreated


def test_worse_markers_mean_faster_decline():
    base = dict(egfr0=60.0, a1c=7.0, uacr0=50.0, sbp=125.0, u=0.0, years=15)
    _, e_ok, _, _ = core.simulate_trajectory_v2(**base)
    _, e_bad, _, _ = core.simulate_trajectory_v2(**{**base, "a1c": 10.0,
                                                    "uacr0": 1500.0, "sbp": 165.0})
    assert e_bad[-1] < e_ok[-1]


def test_predict_egfr_at_v2_matches_the_trajectory():
    """The calibration predictor must agree with the canonical simulator --
    there is only ONE model, and no second integrator allowed to drift."""
    p = core.TRIAL_CALIBRATION_V2
    t, egfr, _, _ = core.simulate_trajectory_v2(50.0, 8.0, 300.0, 140.0, u=0.0,
                                                p=p, years=8)
    q = np.array([1.0, 3.0, 6.0])
    pred = core.predict_egfr_at_v2(50.0, 8.0, 300.0, 140.0, 0.0, p, q, years=8)
    assert np.allclose(pred, np.interp(q, t, egfr), atol=0.5)


def test_gfr_category():
    assert core.gfr_category(95) == "G1"
    assert core.gfr_category(47) == "G3a"
    assert core.gfr_category(10) == "G5"
