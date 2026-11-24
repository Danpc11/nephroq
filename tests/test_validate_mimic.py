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
    assert df == pytest.approx(4.0, rel=0.35)


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


def test_post_origin_data_cannot_change_anything_before_the_origin():
    """Audit P0.4: rewrite every value after the forecast origin (eGFR and the
    covariate columns) and the observed window, the covariates and the
    eligibility must be identical, except the held-out targets themselves."""
    cohort, _ = _synthetic_cohort(n=60, seed=9)
    cohort["hba1c_imputed"] = False
    late = cohort.time_years > vm.EARLY_YEARS
    altered = cohort.copy()
    altered.loc[late, "egfr"] = 5.0 + 100.0 * np.random.default_rng(0).uniform(size=late.sum())
    for c in ("hba1c", "uacr", "sbp"):
        altered.loc[late, c] = altered.loc[late, c] * 3.0
    a, ea = vm.build_patients(cohort)
    b, eb = vm.build_patients(altered)
    assert [p["pid"] for p in a] == [p["pid"] for p in b] and ea == eb
    pb = {p["pid"]: p for p in b}
    for p in a:
        if p["pid"] in pb:
            q = pb[p["pid"]]
            assert np.array_equal(p["t"], q["t"]) and np.array_equal(p["e"], q["e"])
            for k in ("a1c", "uacr0", "sbp"):
                assert p[k] == q[k] or (np.isnan(p[k]) and np.isnan(q[k]))


def test_a_window_straddling_the_origin_does_not_leak():
    """The case that leaked before cutting first: draws just before and just
    after the origin fall in the same 90-day bin."""
    t = np.array([0.0, 0.5, 1.0, 1.975, 1.98, 1.99, 2.2])
    e = np.array([60.0, 60, 60, 50, 50, 20, 20])
    rows = [dict(patient_id="x", time_years=a, egfr=b, hba1c=8.0, uacr=100.0, sbp=140.0)
            for a, b in zip(t, e)]
    rows += [dict(patient_id="x", time_years=3.2, egfr=40.0, hba1c=8.0, uacr=100.0, sbp=140.0)]
    import pandas as pd
    p0 = vm.build_patients(pd.DataFrame(rows))[0][0]
    rows[6]["egfr"] = 999.0                     # the draw after the origin
    p1 = vm.build_patients(pd.DataFrame(rows))[0][0]
    assert np.array_equal(p0["e"], p1["e"])


def test_covariates_are_imputed_with_training_medians_only():
    train = [dict(a1c=7.0, uacr0=50.0, sbp=130.0), dict(a1c=9.0, uacr0=np.nan, sbp=150.0)]
    test = [dict(a1c=np.nan, uacr0=np.nan, sbp=np.nan)]
    med, n = vm.impute_covariates(train, test)
    assert med["a1c"] == 8.0 and med["uacr0"] == 50.0 and med["sbp"] == 140.0
    assert test[0]["a1c"] == 8.0 and n["uacr0"] == 2


def test_prior_density_is_minus_infinity_outside_its_support_and_integrates_to_one():
    """Audit P1.4."""
    import population_prior as pp
    pr = pp.PopulationPrior(np.array([0.7, 0.3]), np.array([[2.0, 9.0], [8.0, 2.5]]),
                            np.array([[-0.6, 0.4], [1.0, 0.5]]))
    assert np.isneginf(pr.logpdf(pp.Q_HI + 0.01, 0.0))
    assert np.isneginf(pr.logpdf(1.5, pp.S_LO - 0.01))
    assert np.isneginf(pr.logpdf(np.nan, 0.0))
    q = np.linspace(pp.Q_LO, pp.Q_HI, 801)
    s = np.linspace(pp.S_LO, pp.S_HI, 801)
    Q, S = np.meshgrid(q, s, indexing="ij")
    trap = getattr(np, "trapezoid", None) or np.trapz
    mass = trap(trap(np.exp(pr.logpdf(Q, S)), s, axis=1), q)
    assert mass == pytest.approx(1.0, abs=5e-3)
