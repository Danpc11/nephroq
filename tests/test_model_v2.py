"""Tests for the v2 structural changes (saturating hyperfiltration + endogenous
albuminuria). These lock the STRUCTURE, not the numbers."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import model_v2 as v2


def test_hyperfiltration_saturates():
    """The v2 hazard must remain BOUNDED as nephrons vanish; v1's power law
    diverged, which is why it over-predicted decline in advanced CKD."""
    kw = dict(uacr_t=300.0, a1c=8.0, sbp=140.0, u=0.0, k_hf=v2.K_HF, w=v2.W,
              eff_met=0.0, eff_hf=0.0, s_sat=3.0)
    h = [v2.hazard_v2(N, **kw) for N in (0.8, 0.4, 0.2, 0.05, 0.01)]
    assert all(b >= a for a, b in zip(h, h[1:]))      # still monotone increasing
    # bounded: the hyperfiltration part cannot exceed k_hf * s_sat**q
    ceiling = v2.K0 + v2.K_HF * 3.0 ** v2.Q + 10.0    # + generous insult allowance
    assert h[-1] < ceiling


def test_saturation_is_not_a_default_arg_trap():
    """s_sat must be read at CALL time, so overriding it actually changes the
    hazard (a default argument would bind once at import and silently ignore it)."""
    kw = dict(N=0.1, uacr_t=300.0, a1c=8.0, sbp=140.0, u=0.0, k_hf=v2.K_HF,
              w=v2.W, eff_met=0.0, eff_hf=0.0)
    assert v2.hazard_v2(s_sat=2.0, **kw) < v2.hazard_v2(s_sat=8.0, **kw)


def test_albuminuria_is_endogenous():
    """UACR must RISE as nephrons are lost (it is a readout of hyperfiltration),
    and the drug must lower it. v1 could express neither."""
    N0 = 0.6
    healthy = v2.uacr_of_state(N0, N0, 300.0, u=0.0, eff_alb=0.3)
    progressed = v2.uacr_of_state(0.3, N0, 300.0, u=0.0, eff_alb=0.3)
    treated = v2.uacr_of_state(N0, N0, 300.0, u=1.0, eff_alb=0.3)
    assert progressed > healthy          # albuminuria tracks nephron loss
    assert treated < healthy             # drug lowers albuminuria
    assert treated == pytest.approx(healthy * 0.7)


def test_treatment_still_slows_decline():
    args = dict(egfr0=45.0, a1c=8.0, uacr0=900.0, sbp=140.0, k_hf=v2.K_HF, w=v2.W,
                eff_met=0.4, eff_hf=0.3, eff_alb=0.3, years=2.4)
    _, e_u, _ = v2.simulate_v2(u=0.0, **args)
    _, e_t, _ = v2.simulate_v2(u=1.0, **args)
    assert e_t[-1] > e_u[-1]


def test_uacr_reduction_measured_against_pretreatment_baseline():
    """Regression test for a real bug: the drug effect is applied at t=0, so
    dividing by the post-drug ua[0] cancels it and reports ~0% reduction. The
    ratio must be taken against the PRE-treatment baseline."""
    spec = v2.it.TRIALS["CREDENCE"]
    out = v2.trial_arms_v2(spec, v2.K_HF, v2.W, eff_met=0.3, eff_hf=0.2,
                           eff_alb=0.30, n=40, seed=3)
    # a 30% direct effect must show up as a ~30% placebo-subtracted reduction
    assert 20.0 < out["uacr_reduction_pct"] < 40.0


# --- v2 integrated into model_core (the single source of truth) ---------------

def test_model_core_v2_is_available_and_v1_is_intact():
    """v2 must live in model_core alongside v1 -- calibrate_mimic.py still needs
    v1, and feeding v1 parameters into the v2 structure would be invalid."""
    import model_core as core
    assert hasattr(core, "simulate_trajectory_v2")
    assert hasattr(core, "simulate_trajectory")          # v1 untouched
    assert core.TRIAL_CALIBRATION_V2["s_sat"] == core.S_SAT


def test_model_core_v2_albuminuria_is_an_output():
    """UACR must be returned as a trajectory: rising as nephrons are lost, and
    dropping immediately under treatment by the calibrated ~29%."""
    import model_core as core
    kw = dict(egfr0=47.7, a1c=8.1, uacr0=145.0, sbp=142.0, years=15)
    _, _, ua_u, _ = core.simulate_trajectory_v2(u=0.0, **kw)
    _, _, ua_t, _ = core.simulate_trajectory_v2(u=1.0, **kw)
    assert ua_u[-1] > ua_u[0]                       # albuminuria worsens untreated
    drop = 1.0 - ua_t[0] / ua_u[0]
    assert 0.25 < drop < 0.35                       # published SGLT2i effect: 31-35%
    assert ua_t[-1] < ua_u[-1]


def test_model_core_v2_is_slower_than_v1():
    """The whole point of the trial anchoring: v1 declined ~2x too fast versus
    real placebo arms, so v2 must reach the threshold LATER for the same patient."""
    import model_core as core
    from mechanistic_twin import MechanisticRenalModel, N_of_egfr
    _, _, _, td_v2 = core.simulate_trajectory_v2(egfr0=47.7, a1c=8.1, uacr0=145.0,
                                                 sbp=142.0, u=0.0, years=15)
    m = MechanisticRenalModel(a1c=8.1, sbp=142, uacr=145, u=0.0,
                              k_hf=0.0141, q=1.52, w_a1c=0.0144, w_uacr=0.0180, w_sbp=0.0108)
    _, _, _, td_v1 = m.simulate(N_of_egfr(47.7), years=15)
    assert td_v2 > td_v1
