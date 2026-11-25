"""Regression tests for the methodological fixes of Round 29: history reduction,
Student-t grid likelihood, between-visit deviation, cross-fitted interval
recalibration, bootstrap CIs of every metric, mixture ordering and stability,
blood-only lab items, the least-squares comparator's box and the selection
flow with deaths."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import fast_sim
import npe_posterior as npe
import personalize as pz


def test_long_histories_keep_the_first_visit_as_the_time_origin():
    t = np.linspace(0.0, 6.0, 25)
    e = 80.0 - 2.0 * t
    tr, er = npe.reduce_history(t, e)
    assert len(tr) == npe.MAX_VISITS
    assert tr[0] == 0.0 and er[0] == 80.0          # eGFR0 and t = 0 still the first visit
    assert tr[-1] == 6.0 and np.all(np.diff(tr) > 0)
    t2, e2 = npe.reduce_history(t[:5], e[:5])
    assert np.array_equal(t2, t[:5])


def test_training_schedules_cover_every_history_length_the_network_accepts():
    rng = np.random.default_rng(0)
    _, mask = npe.sample_schedules(20000, rng)
    assert mask.sum(1).max() == npe.MAX_VISITS


def test_injury_rate_box_reaches_ageing_only_decline():
    """Round 28: the stable MIMIC regime sat against the old 0.25x floor."""
    assert np.exp(npe.PRIOR_LO[1]) <= 0.05
    t = np.array([0.0, 5.0])
    e = fast_sim.simulate_egfr(npe.Q_POP, float(np.exp(npe.PRIOR_LO[1])), 75.0, 6.8, 17.0, 131.0, t)[0]
    assert (e[0] - e[1]) / 5.0 < 0.3                 # < 0.3 mL/min/1.73 m2 per year


def test_grid_posterior_student_t_likelihood_resists_an_outlier():
    t = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
    true = fast_sim.simulate_egfr(1.5, 1.0, 70.0, 8.0, 100.0, 140.0, t)[0]
    e = true.copy(); e[2] -= 40.0                       # one acute spike
    kw = dict(nq=10, ns=20, ne=41, noise_sd=5.0)
    pts_g, w_g, _, _ = npe.grid_posterior(t, e, 8.0, 100.0, 140.0, **kw)
    pts_t, w_t, _, _ = npe.grid_posterior(t, e, 8.0, 100.0, 140.0, noise_df=3.0, **kw)
    e0_g, e0_t = np.sum(w_g * pts_g[:, 2]), np.sum(w_t * pts_t[:, 2])
    assert abs(e0_t - 70.0) < 0.5 * abs(e0_g - 70.0)    # the spike drags the Gaussian fit


def test_least_squares_comparator_searches_the_requested_box():
    t = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
    e = fast_sim.simulate_egfr(npe.Q_POP, 5.0, 70.0, 9.0, 800.0, 160.0, t)[0]
    s_app = pz.least_squares_scale_only(t, e, 9.0, 800.0, 160.0)
    s_box = pz.least_squares_scale_only(t, e, 9.0, 800.0, 160.0, bounds=(0.02, 6.0))
    assert s_app <= pz.SCALE_RANGE[1] + 1e-9
    assert s_box == pytest.approx(5.0, rel=0.05)


def test_lab_items_are_restricted_to_blood():
    import mimic_loader as ml
    d = pd.DataFrame(dict(itemid=pd.array([1, 2, 3, 4, 5], dtype="Int64"),
                          label=["Creatinine", "Creatinine", "Creatinine, Urine",
                                 "Creatinine, Ascites", "Creatinine, Serum"],
                          fluid=["Blood", "Pleural", "Urine", "Ascites", "Blood"]))
    ids = ml.find_itemids(d, ml.LABEL_KEYWORDS["creatinine"],
                          exclude=["albumin", "ratio", "clearance"] + ml.NON_BLOOD_LABEL_WORDS,
                          fluid="Blood")
    assert ids == [1, 5]
    no_fluid = d.drop(columns="fluid")
    ids2 = ml.find_itemids(no_fluid, ml.LABEL_KEYWORDS["creatinine"],
                           exclude=ml.NON_BLOOD_LABEL_WORDS, fluid="Blood")
    assert 3 not in ids2 and 4 not in ids2


# ---- pieces of the validation pipeline that need no network ---------------------
import validate_mimic_patients as vm  # noqa: E402


def test_between_visit_deviation_is_recovered():
    rng = np.random.default_rng(1)
    v0, v1, scale, df = 9.0, 16.0, 2.5, 3.0
    fc = []
    for _ in range(3000):
        k = rng.integers(1, 3)
        dt = np.sort(rng.uniform(0.6, 5.0, k))
        mu = rng.uniform(30, 90, k)
        lat = mu[None, :] + rng.normal(0, 1.0, (100, k))            # posterior spread
        y = (mu + rng.normal(0, 1.0, k) + scale * rng.standard_t(df, k)
             + rng.normal(0, np.sqrt(v0 + v1 * dt)))
        fc.append(dict(tf=dt, lat=lat, y=y, dt=dt))
    dev = vm.fit_deviation(fc, (scale, df))
    assert dev["v1"] == pytest.approx(v1, rel=0.25)
    assert dev["v0"] + 3 * dev["v1"] == pytest.approx(v0 + 3 * v1, rel=0.15)
    assert dev["loglik_gain_per_visit"] > 0


def test_cross_fitted_recalibration_restores_nominal_coverage():
    rng = np.random.default_rng(2)
    n = 3000
    X = rng.uniform(-2, 2, (n, 3))
    y = X[:, 0] * 3 + rng.standard_t(3, n) * (1 + np.abs(X[:, 1]))
    models = vm._fit_boosting(X[:2000], y[:2000], seed=0)
    mu, sd = vm._gbm_mu_sd(models, X[2000:])
    draws = vm._calibrated_draws(mu, sd, models["z"], rng, n=2000)
    lo, hi = np.quantile(draws, [0.05, 0.95], axis=0)
    cov = np.mean((y[2000:] >= lo) & (y[2000:] <= hi))
    assert cov == pytest.approx(0.90, abs=0.04)


def test_bootstrap_reports_every_metric_against_each_reference():
    rng = np.random.default_rng(3)
    rows = []
    for i in range(150):
        y = rng.uniform(15, 80)
        for m in ("npe_dev", "gbm", "ols"):
            mu = y + rng.normal(0, {"npe_dev": 5, "gbm": 4, "ols": 9}[m])
            r = dict(pid=f"p{i}", horizon="1y", method=m, y=y, mu=mu, at_risk=True,
                     event=y < 30, ess=np.nan, crps=abs(mu - y) * 0.6,
                     p_event=float(1 / (1 + np.exp(mu - 30))))
            for lv in vm.LEVELS:
                r[f"hit_{lv:g}"] = bool(abs(mu - y) < 8)
            rows.append(r)
    tables, ev, ci = vm.summarize(pd.DataFrame(rows), n_boot=200)
    assert set(ci) == {"npe_dev", "gbm"}
    d = ci["gbm"]["ols"]
    assert set(d) == set(vm.BOOT_METRICS)
    assert d["rmse"][0] > 0                              # OLS clearly worse than GBM
    assert d["auc"][1] < 0.05


def test_selection_flow_reports_deaths_by_exclusion_reason(tmp_path):
    cohort = pd.DataFrame(dict(patient_id=["a", "a", "b", "b", "c"],
                               time_years=[0.0, 3.0, 0.0, 1.0, 0.0],
                               egfr=[60.0, 55.0, 40.0, 38.0, 70.0]))
    reasons = {"a": "included", "b": "no_future", "c": "too_few_early"}
    pf = tmp_path / "patients.csv.gz"
    pd.DataFrame(dict(subject_id=["a", "b", "c"], anchor_year_group=["x"] * 3,
                      dod=[None, "2150-01-01", None])).to_csv(pf, index=False)
    rows = vm.selection_table(cohort, reasons, str(pf))
    by = {r["group"]: r for r in rows}
    assert by["Included"]["died"] == 0.0
    assert by[vm.REASON_LABELS["no_future"]]["died"] == 1.0


def test_mixture_components_are_ordered_by_rate_and_bootstrap_is_stable():
    import population_prior as pp
    rng = np.random.default_rng(5)
    N, J = 600, 120
    reg = rng.uniform(size=N) < 0.6
    qt = 0.8 + 2.7 * rng.beta(4, 4, N)
    st = np.where(reg, rng.normal(-2.0, 0.3, N), rng.normal(0.8, 0.3, N))
    st = np.clip(st, pp.S_LO + 0.05, pp.S_HI - 0.05)
    draws = [np.column_stack([np.clip(q + rng.normal(0, 0.5, J), pp.Q_LO, pp.Q_HI),
                              np.clip(s + rng.normal(0, 0.25, J), pp.S_LO, pp.S_HI),
                              np.full(J, 60.0)]) for q, s in zip(qt, st)]
    pr = pp.fit(draws, K=2)
    a, b = pr.summary()
    assert a["rate_median"] < b["rate_median"]
    assert a["weight"] == pytest.approx(0.6, abs=0.08)
    stab = pp.bootstrap_stability(draws, 2, n_boot=4, n_sub=40)
    w = np.array(stab["weight"])
    assert w.shape == (2, 3) and np.all(w[:, 0] <= w[:, 2])
    assert stab["rate_median"][0][2] < stab["rate_median"][1][0]   # intervals do not overlap
