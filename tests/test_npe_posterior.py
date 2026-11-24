"""Contract tests for the amortized neural posterior. Skipped when the optional
dependencies (torch, zuko) are not installed."""
import os
import sys

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("zuko")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import npe_posterior as npe


def test_unbounded_transform_round_trips_inside_the_prior():
    rng = np.random.default_rng(0)
    th = npe.sample_prior(500, rng)
    assert np.allclose(npe.from_unbounded(npe.to_unbounded(th)), th, atol=1e-6)


def test_schedules_start_at_zero_and_respect_visit_and_span_ranges():
    rng = np.random.default_rng(1)
    times, mask = npe.sample_schedules(300, rng)
    n = mask.sum(1)
    assert n.min() >= npe.VISITS_RANGE[0] and n.max() <= npe.VISITS_RANGE[1]
    assert np.all(times[:, 0] == 0.0)
    span = np.array([t[m][-1] for t, m in zip(times, mask)])
    assert span.min() >= npe.SPAN_RANGE[0] and span.max() <= npe.SPAN_RANGE[1]
    assert np.all(np.diff(times, axis=1) >= 0)


def test_posterior_draws_stay_inside_the_prior_box():
    net, _ = npe.train(n_train=3000, epochs=1, save=False, verbose=False)
    th = npe.posterior_samples(net, [0.0, 1.0, 2.5], [70.0, 64.0, 58.0], 8.0, 300.0, 145.0, n=200)
    assert th.shape == (200, 3)
    assert np.all(th >= npe.PRIOR_LO - 1e-9) and np.all(th <= npe.PRIOR_HI + 1e-9)


def test_exact_grid_posterior_concentrates_on_the_truth_with_rich_data():
    """Sanity check of the reference: with many precise visits the exact
    posterior must recover the injury rate and the baseline eGFR."""
    import fast_sim
    q, ls, e0 = 1.8, np.log(1.6), 72.0
    t = np.linspace(0, 6, 9)
    e = fast_sim.simulate_egfr(q, np.exp(ls), e0, 8.0, 300.0, 145.0, t)[0]
    pts, w, _, _ = npe.grid_posterior(t, e, 8.0, 300.0, 145.0, noise_sd=1.0)
    assert w.sum() == pytest.approx(1.0)
    mean = (w[:, None] * pts).sum(0)
    assert mean[1] == pytest.approx(ls, abs=0.15)
    assert mean[2] == pytest.approx(e0, abs=2.0)


def test_weights_from_another_model_configuration_are_refused(tmp_path, monkeypatch):
    """Audit P0.5: an artifact records the configuration hash it was trained on."""
    import model_core as core
    path = str(tmp_path / "w.pt")
    npe.train(n_train=1500, epochs=1, save=True, verbose=False, path=path)
    assert npe.load(path).sim_config_hash == core.config_hash()
    monkeypatch.setattr(core, "config_hash", lambda p=None: "another-model")
    with pytest.raises(ValueError):
        npe.load(path)
