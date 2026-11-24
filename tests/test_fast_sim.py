"""The vectorized simulator must reproduce the reference ODE solver, because the
neural posterior is trained on it and the exact grid posterior is computed with it."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import model_core as core
import personalize as pz
import fast_sim


def test_matches_the_lsoda_reference_across_the_prior():
    rng = np.random.default_rng(0)
    t = np.array([0.0, 0.4, 1.7, 3.0, 6.5, 10.0, 13.0])
    worst = 0.0
    for _ in range(25):
        q = rng.uniform(0.8, 3.5)
        s = float(np.exp(rng.uniform(np.log(0.25), np.log(3.0))))
        e0, a1c = rng.uniform(20, 110), rng.uniform(5.5, 12)
        uacr0, sbp = float(np.exp(rng.uniform(np.log(5), np.log(3000)))), rng.uniform(105, 185)
        ref = core.predict_egfr_at_v2(e0, a1c, uacr0, sbp, 0.0, pz.patient_params(q, s), t, years=13.0)
        out = fast_sim.simulate_egfr(q, s, e0, a1c, uacr0, sbp, t)
        worst = max(worst, float(np.max(np.abs(out - ref))))
    assert worst < 0.01          # mL/min/1.73m2


def test_per_patient_query_times_and_batch_shapes():
    q = np.array([1.2, 2.5])
    times = np.array([[0.0, 1.0, 2.0], [0.0, 3.0, 6.0]])
    out = fast_sim.simulate_egfr(q, 1.0, 60.0, 8.0, 100.0, 140.0, times)
    assert out.shape == (2, 3)
    assert out[0, 0] == pytest.approx(60.0, abs=1e-6)
    solo = fast_sim.simulate_egfr(2.5, 1.0, 60.0, 8.0, 100.0, 140.0, [0.0, 3.0, 6.0])
    assert np.allclose(out[1], solo[0], atol=1e-6)


def test_injury_rate_scales_the_albuminuric_term_too():
    """A higher personal injury rate must accelerate decline even when the
    albuminuric pathway dominates (this caught a Round 16 regression in
    patient_params, which scaled the unused w_uacr instead of k_alb)."""
    p1 = pz.patient_params(1.52, 1.0)
    p2 = pz.patient_params(1.52, 2.0)
    assert p2["k_alb"] == pytest.approx(2 * core._k_alb_of(core.TRIAL_CALIBRATION_V2))
    base = dict(egfr0=80.0, a1c=6.5, uacr0=2000.0, sbp=130.0)
    e1 = fast_sim.simulate_egfr(1.52, 1.0, *base.values(), [5.0])[0, 0]
    e2 = fast_sim.simulate_egfr(1.52, 2.0, *base.values(), [5.0])[0, 0]
    assert e2 < e1


def _base_with(**kw):
    p = dict(core.TRIAL_CALIBRATION_V2)
    p.update(kw)
    p.pop("w_uacr", None)
    return p


def _scaled(p, q, s):
    d = dict(p, q=q)
    for k in ("k_hf", "w_a1c", "w_sbp"):
        d[k] = p[k] * s
    d["k_alb"] = core._k_alb_of(p) * s
    return d


@pytest.mark.parametrize("calib", [
    {}, {"k_hf": 0.004, "k0": 0.006}, {"s_sat": 2.5, "beta": 1.4, "k_alb": 0.03},
])
@pytest.mark.parametrize("u", [0.0, 1.0])
def test_matches_lsoda_for_other_calibrations_and_under_treatment(calib, u):
    """Audit P0.5: the fast simulator integrates exactly the configured model,
    not a hidden copy of the trial calibration, with and without treatment."""
    p = _base_with(**calib)
    t = np.array([0.0, 1.0, 3.0, 7.0, 12.0])
    rng = np.random.default_rng(5)
    for _ in range(6):
        q, s = rng.uniform(0.8, 3.5), float(np.exp(rng.uniform(-1.3, 1.7)))
        e0, a1c = rng.uniform(25, 105), rng.uniform(6, 11)
        uacr, sbp = float(np.exp(rng.uniform(np.log(10), np.log(2500)))), rng.uniform(110, 175)
        ref = core.predict_egfr_at_v2(e0, a1c, uacr, sbp, u, _scaled(p, q, s), t, years=12.0)
        out = fast_sim.simulate_egfr(q, s, e0, a1c, uacr, sbp, t, params=p, u=u)[0]
        assert np.max(np.abs(out - ref)) < 0.01


def test_config_hash_identifies_the_model():
    h0 = core.config_hash()
    assert core.config_hash(dict(core.TRIAL_CALIBRATION_V2)) == h0
    assert core.config_hash(_base_with(k_hf=0.011)) != h0
    assert core.config_hash(_base_with(s_sat=3.0)) != h0
