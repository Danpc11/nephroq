"""Audit P0.1-P0.3: load convention, eGFR domain, and when the decline accelerates."""
import os
import sys
import warnings

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import model_core as core
import personalize as pz
import fast_sim


def test_load_convention_is_an_exact_reparametrization():
    """s = 1/N with (q, S_SAT, beta) equals s' = N^(alpha-1) with
    (q/(1-alpha), S_SAT^(1-alpha), beta/(1-alpha)): same model, different q."""
    a = core.ALPHA
    N = np.linspace(0.01, 1.0, 400)
    s, sp = 1.0 / N, N ** (a - 1.0)
    for q in (0.9, 1.52, 2.9):
        old = s ** q / (1 + (s / core.S_SAT) ** q)
        qp, ssp = q / (1 - a), core.S_SAT ** (1 - a)
        new = sp ** qp / (1 + (sp / ssp) ** qp)
        assert np.allclose(old, new, rtol=1e-12)
    beta = core.BETA
    assert np.allclose((s / s[0]) ** beta, (sp / sp[0]) ** (beta / (1 - a)), rtol=1e-12)


def test_second_derivative_matches_the_elasticity_formula():
    """G'' = ALPHA^2 G h^2 (1 + eps) against finite differences of a trajectory
    integrated with fixed covariates."""
    for q, uacr in [(1.52, 300.0), (2.9, 30.0)]:
        p = pz.patient_params(q, 1.0)
        t = np.linspace(0.0, 8.0, 1601)
        G = fast_sim.simulate_egfr(q, 1.0, 90.0, 8.0, uacr, 140.0, t, params=p, dt=0.005)[0]
        dt = t[1] - t[0]
        g2_fd = (G[2:] - 2 * G[1:-1] + G[:-2]) / dt ** 2
        h, eps = core.hazard_elasticity(G[1:-1], 90.0, 8.0, uacr, 140.0, 0.0, p)
        g2 = core.ALPHA ** 2 * G[1:-1] * h ** 2 * (1 + eps)
        sel = slice(20, -20)
        assert np.allclose(g2_fd[sel], g2[sel], rtol=2e-2, atol=2e-3)


def test_trial_calibration_never_accelerates_and_nothing_collapses():
    """With q = 1.52 the decline decelerates at every eGFR (eps > -1), and the
    hazard stays bounded (no finite-time singularity) for every q."""
    G = np.linspace(8.0, 115.0, 300)
    for uacr in (10.0, 30.0, 300.0, 3000.0):
        _, eps = core.hazard_elasticity(G, 115.0, 8.0, uacr, 140.0, 0.0, core.TRIAL_CALIBRATION_V2)
        assert np.all(eps > -1.0)
        for q in (0.8, 2.0, 3.5):
            h, _ = core.hazard_elasticity(G, 115.0, 12.0, uacr, 185.0, 0.0, pz.patient_params(q, 6.0))
            assert np.all(np.isfinite(h)) and h.max() < core.HAZARD_CAP


def test_acceleration_requires_a_large_exponent_and_stops_below_saturation():
    G = np.linspace(12.0, 110.0, 300)
    _, eps = core.hazard_elasticity(G, 110.0, 8.0, 30.0, 140.0, 0.0, pz.patient_params(2.9, 1.0))
    acc = G[eps < -1]
    assert acc.size and acc.min() > 30.0        # accelerating region lies above ~eGFR 36
    assert np.all(eps[G < 25.0] > -1.0)         # decelerates below saturation


def test_egfr_domain_is_reported_not_silently_truncated():
    f = core.egfr_domain_flags([45.0, 118.0, 135.0])
    assert f["n_above"] == 1 and not f["in_domain"]
    assert core.egfr_domain_flags([20.0, 60.0])["in_domain"]
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        out = pz.personalize([0.0, 1.0, 2.0], [130.0, 110.0, 95.0], 8.0, 100.0, 140.0,
                             estimator=pz.train(n_train=300, save=False))
    assert out["domain_warning"] and any("above" in str(x.message) for x in w)
