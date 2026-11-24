"""End-to-end test of the patient-level validation on a SYNTHETIC cohort in the
mimic_loader output format (no real data in the repository). Skipped without the
optional NPE dependencies."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")
pytest.importorskip("zuko")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import fast_sim
import npe_posterior as npe
import personalize as pz
import validate_mimic_patients as vm


def _synthetic_cohort(n=80, seed=0):
    rng = np.random.default_rng(seed)
    rows, groups = [], []
    for i in range(n):
        q, s = rng.uniform(0.9, 3.0), float(np.exp(rng.uniform(-1, 1)))
        e0, a1c = rng.uniform(30, 100), rng.uniform(6, 11)
        uacr, sbp = float(np.exp(rng.uniform(np.log(10), np.log(2000)))), rng.uniform(115, 170)
        t = np.sort(np.concatenate([[0.0], rng.uniform(0, 7, rng.integers(8, 25))]))
        e = fast_sim.simulate_egfr(q, s, e0, a1c, uacr, sbp, t)[0] + rng.normal(0, 8, len(t))
        for tt, ee in zip(t, e):
            rows.append(dict(patient_id=f"p{i}", time_years=tt, egfr=ee,
                             hba1c=a1c, uacr=uacr, sbp=sbp))
        groups.append(dict(subject_id=f"p{i}",
                           anchor_year_group=["2008 - 2010", "2011 - 2013", "2014 - 2016"][i % 3]))
    return pd.DataFrame(rows), pd.DataFrame(groups)


def test_thinning_keeps_one_median_per_window():
    t, e = vm.thin(np.array([0.0, 0.01, 0.02, 0.5]), np.array([50.0, 70.0, 60.0, 40.0]))
    assert len(t) == 2 and e[0] == pytest.approx(60.0)


def test_pipeline_runs_and_reports_only_aggregates(tmp_path):
    cohort, groups = _synthetic_cohort()
    pfile = tmp_path / "patients.csv.gz"
    groups.to_csv(pfile, index=False)
    mlp = pz.train(n_train=600, save=False)
    df, tables, ev, ci, meta = vm.run(cohort, str(pfile), noise_model="t", mlp=mlp, n_boot=50,
                                      verbose=False,
                                      npe_kwargs=dict(n_train=3000, epochs=1, save=False))
    assert meta["split"].startswith("temporal")
    assert set(df.method) == set(vm.METHODS)
    for m in vm.METHODS:
        assert np.isfinite(tables["all"][m]["rmse"])
    assert 0.0 <= tables["all"]["npe_pop"]["cov90"] <= 1.0
    assert set(ci) == {"npe_pop", "gbm"}
    assert meta["noise"]["df"] is not None and meta["noise"]["scale"] > 0
    text = vm.report(tables, ev, ci, meta)
    assert "NPE" in text
    assert not any(f"p{i}" in text for i in range(80))   # no patient ids


def test_no_held_out_visit_leaks_into_the_observed_window():
    cohort, _ = _synthetic_cohort(n=40, seed=3)
    patients, _ = vm.build_patients(cohort)
    assert patients
    for p in patients:
        assert p["t"][0] == 0.0 and p["t"][-1] <= vm.EARLY_YEARS
        for t_f, _ in p["future"].values():
            assert t_f - p["t"][-1] > vm.HORIZONS["1y"][0]


def test_population_prior_recovers_two_regimes_from_posterior_draws():
    """Empirical Bayes over NPE-style draws must recover a known two-regime
    population (the structure Round 17 found in MIMIC-IV)."""
    import population_prior as pp
    rng = np.random.default_rng(0)
    N, J = 1200, 300
    reg = rng.uniform(size=N) < 0.7
    qt = np.where(reg, 0.8 + 2.7 * rng.beta(2, 12, N), 0.8 + 2.7 * rng.beta(12, 3, N))
    st = np.clip(rng.normal(np.where(reg, -0.3, 0.4), 0.35), np.log(0.25) + 0.01, np.log(3) - 0.01)
    z = npe.to_unbounded(np.column_stack([qt, st, np.full(N, 60.0)]))
    zh = z + rng.normal(0, [0.8, 0.5, 0.0], (N, 3))
    draws = [npe.from_unbounded(zz + rng.normal(0, [0.8, 0.5, 0.0], (J, 3))) for zz in zh]
    pr = pp.fit(draws, K=2)
    lo, hi = pr.summary()
    assert lo["weight"] == pytest.approx(0.7, abs=0.08)
    assert lo["q_mean"] < 1.5 < 2.6 < hi["q_mean"]
    w, ess = pp.reweight(draws[0], pr)
    assert w.sum() == pytest.approx(1.0) and 0.0 < ess <= 1.0


def test_noise_model_recovers_a_known_student_t():
    """The noise fit uses only short-interval differences in the observed window."""
    rng = np.random.default_rng(4)
    train = []
    for _ in range(600):
        t = np.sort(rng.uniform(0, 2, 8)); t[0] = 0.0
        train.append(dict(t=t, e=60.0 + 5.0 * rng.standard_t(4, len(t))))
    scale, df, n = vm.estimate_noise(train)
    assert n > 500
    assert scale == pytest.approx(5.0, rel=0.15)
    assert 2.5 <= df <= 8.0


def test_stage_two_network_is_trained_under_the_learned_prior():
    import population_prior as pp
    pr = pp.PopulationPrior(np.array([0.9, 0.1]), np.array([[3.0, 20.0], [15.0, 3.0]]),
                            np.array([[-0.5, 0.3], [1.0, 0.3]]))
    th = npe.sample_prior(4000, np.random.default_rng(0), pr)
    assert th[:, 0].mean() < 1.6                      # mostly the low-q regime
    assert np.all(th >= npe.PRIOR_LO) and np.all(th <= npe.PRIOR_HI)
    net = npe.load_or_train(noise_sd=7.0, noise_df=4.0, prior=pr, save=False, verbose=False,
                            n_train=2000, epochs=1)
    assert net.prior is pr and net.noise_df == 4.0
