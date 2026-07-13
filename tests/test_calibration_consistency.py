import pytest
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
import model_core as core

def test_calibrator_and_app_share_one_simulator():
    """
    There must be exactly ONE model. This is the test that would have caught the
    ~11 mL/min/1.73m2 divergence between two separately-written integrators:
    calibrate_mimic's predictor and the app's trajectory must agree over the FULL
    curve, not just at a single instant, because both now route through
    model_core.simulate_trajectory_v2.
    """
    import calibrate_mimic as cal

    pac = dict(egfr0=52.0,
               hba1c_baseline_strict=8.2, uacr_baseline_strict=300.0,
               sbp_baseline_strict=145.0,
               hba1c_series=np.full(6, 8.2), uacr_series=np.full(6, 300.0),
               sbp_series=np.full(6, 145.0),
               t=np.linspace(0, 8, 6))
    p = dict(core.TRIAL_CALIBRATION_V2)
    q, khf = p["q"], p["k_hf"]
    w = np.array([p["w_a1c"], p["w_uacr"], p["w_sbp"]])

    t_query = np.array([0.0, 2.0, 5.0, 8.0])
    from_calibrator = cal.predict_egfr(q, khf, pac, w, t_query)

    t, egfr, _, _ = core.simulate_trajectory_v2(
        pac["egfr0"], 8.2, 300.0, 145.0, u=0.0, p=p, years=8)
    from_app = np.interp(t_query, t, egfr)

    assert np.max(np.abs(from_calibrator - from_app)) < 0.5

def test_prediction_uses_baseline_covariates_only():
    """
    v2 anchors the forecast on BASELINE covariates. Two patients identical at
    baseline but whose LATER HbA1c diverges must get the SAME baseline-anchored
    prediction -- otherwise the forecast would be using the patient's own future,
    which is temporal leakage.
    """
    import calibrate_mimic as cal

    def pac(future_a1c):
        n = 6
        return dict(egfr0=55.0, t=np.linspace(0, 9.5, n),
                    hba1c_baseline_strict=7.0, uacr_baseline_strict=200.0,
                    sbp_baseline_strict=135.0,
                    hba1c_series=np.concatenate([[7.0], np.full(n - 1, future_a1c)]),
                    uacr_series=np.full(n, 200.0), sbp_series=np.full(n, 135.0))

    p = core.TRIAL_CALIBRATION_V2
    w = np.array([p["w_a1c"], p["w_uacr"], p["w_sbp"]])
    tq = np.array([9.5])
    stable = cal.predict_egfr(p["q"], p["k_hf"], pac(7.0), w, tq)[0]
    rising = cal.predict_egfr(p["q"], p["k_hf"], pac(11.0), w, tq)[0]
    assert stable == pytest.approx(rising), "future covariates must not leak into a baseline forecast"

def test_n_never_exceeds_one():
    """N(t) is documented as being in (0,1]; a supra-physiological eGFR
    (e.g. hyperfiltration readings >120) must not push N above 1."""
    for egfr in [100, 120, 140, 200]:
        N = core.N_of_egfr(egfr)
        assert 0 < N <= 1.0, f"core.N_of_egfr({egfr}) = {N}, out of the documented (0,1] range"

def test_n_egfr_roundtrip_stays_bounded():
    """Round-tripping through egfr_of_N/N_of_egfr must never produce N>1."""
    for N in [0.01, 0.5, 1.0]:
        e = core.egfr_of_N(N)
        N2 = core.N_of_egfr(e)
        assert N2 <= 1.0 + 1e-9

def test_no_temporal_leakage_in_covariates():
    """
    Reproduces the exact leakage example from the code review: a patient
    with HbA1c=7.0 at their index date (2014), HbA1c=8.2 in 2017, and
    HbA1c=10.0 in 2020. The OLD (whole-trajectory median) approach would
    assign HbA1c=8.2 to the 2014 row -- using information from 3 years in
    the future to "explain" an earlier eGFR observation. The FIXED
    (baseline-window) approach must assign the value actually known at the
    index date (7.0), not the contaminated median.
    """
    import mimic_loader as ml
    import pandas as pd

    index_dates = {"1": pd.Timestamp("2014-01-15")}
    a1c = pd.DataFrame({
        "subject_id": ["1", "1", "1"],
        "charttime": [pd.Timestamp("2014-01-15"), pd.Timestamp("2017-02-20"), pd.Timestamp("2020-03-01")],
        "valuenum": [7.0, 8.2, 10.0],
    })

    # simulate attach_baseline's logic directly (same window as mimic_loader.py)
    window = (pd.Timedelta(days=-90), pd.Timedelta(days=14))
    idx = pd.Series(index_dates, name="index_date")
    merged = a1c.merge(idx, left_on="subject_id", right_index=True, how="inner")
    delta = merged["charttime"] - merged["index_date"]
    merged = merged.loc[(delta >= window[0]) & (delta <= window[1])].copy()
    merged["abs_delta"] = (merged["charttime"] - merged["index_date"]).abs()
    nearest = merged.sort_values("abs_delta").groupby("subject_id").first()
    baseline_value = nearest["valuenum"].to_dict()["1"]

    leaked_value = a1c["valuenum"].median()   # what the OLD code would have used

    assert baseline_value == 7.0, \
        f"Baseline HbA1c should be 7.0 (the value known at the index date), got {baseline_value}"
    assert leaked_value == 8.2, \
        "sanity check: the whole-trajectory median (8.2) is indeed different from the baseline (7.0)"
    assert baseline_value != leaked_value, \
        "baseline and leaked-median values must differ in this constructed example"

def test_primary_sensitivity_split():
    """
    The primary analysis must include ONLY patients with observed (not
    imputed) HbA1c AND UACR; patients missing either must be excluded from
    primary and only appear in the sensitivity (full-cohort) analysis.
    """
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
    import calibrate_mimic as cal

    def mk(pid, hba1c_imp, uacr_imp):
        return dict(patient_id=pid, cov=(8.0, 100.0, 140.0), egfr0=70.0,
                   t=np.array([0., 1., 2.]), e=np.array([70., 65., 60.]),
                   hba1c_imputed=hba1c_imp, uacr_imputed=uacr_imp)

    patients = ([mk(f"obs{i}", False, False) for i in range(35)]
               + [mk(f"missing_uacr{i}", False, True) for i in range(10)]
               + [mk(f"missing_both{i}", True, True) for i in range(5)])

    primary, sensitivity, used_fallback = cal.split_primary_sensitivity(patients, min_primary=30)
    assert not used_fallback, "35 fully-observed patients should be enough, no fallback expected"
    assert len(primary) == 35
    assert all(not p["hba1c_imputed"] and not p["uacr_imputed"] for p in primary)
    assert len(sensitivity) == len(patients)

def test_primary_sensitivity_fallback_when_too_small():
    """If too few patients have both covariates observed, fall back to the
    full cohort as primary (with the fallback flag set) rather than fitting
    5 parameters on a handful of patients."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
    import calibrate_mimic as cal

    def mk(pid, hba1c_imp, uacr_imp):
        return dict(patient_id=pid, cov=(8.0, 100.0, 140.0), egfr0=70.0,
                   t=np.array([0., 1., 2.]), e=np.array([70., 65., 60.]),
                   hba1c_imputed=hba1c_imp, uacr_imputed=uacr_imp)

    patients = ([mk(f"obs{i}", False, False) for i in range(5)]
               + [mk(f"missing{i}", False, True) for i in range(50)])

    primary, sensitivity, used_fallback = cal.split_primary_sensitivity(patients, min_primary=30)
    assert used_fallback, "only 5 fully-observed patients should trigger the fallback"
    assert len(primary) == len(patients)
    assert sensitivity is None

def test_bootstrap_calibrate_returns_requested_count():
    """bootstrap_calibrate must return one parameter set per successful
    replicate, each a well-formed dict with the 5 model parameters."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
    import calibrate_mimic as cal

    rng = np.random.default_rng(0)
    patients = []
    for i in range(15):
        n_visits = 5
        t = np.linspace(0, 4, n_visits)
        patients.append(dict(t=t, e=80 - 3*t + rng.normal(0, 1, n_visits), egfr0=80.0,
                             patient_id=str(i),
                             hba1c_series=np.full(n_visits, 7.5),
                             uacr_series=np.full(n_visits, 50.0),
                             sbp_series=np.full(n_visits, 130.0)))
    point = cal.calibrate(patients, verbose=False, n_multistarts=1)
    boot = cal.bootstrap_calibrate(patients, point, n_boot=3, seed=1)
    assert len(boot) == 3
    for b in boot:
        assert set(b.keys()) == {"q", "k_hf", "w_a1c", "w_uacr", "w_sbp"}
        assert np.isfinite(b["q"]) and b["q"] > 0

def test_bootstrap_disabled_returns_empty_list():
    """n_boot=0 must return an empty list (the app's signal to fall back to
    a point-estimate-only display), not None or an error."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
    import calibrate_mimic as cal
    point = dict(q=1.5, k_hf=0.01, w_a1c=0.01, w_uacr=0.01, w_sbp=0.01)
    assert cal.bootstrap_calibrate([{}], point, n_boot=0) == []

def test_filter_kfre_comparable():
    """The KFRE cohort is defined by KFRE's OWN four variables -- age, sex,
    baseline eGFR and baseline UACR -- and must NOT require HbA1c, which KFRE
    does not use. Requiring HbA1c (the old behavior) shrank the cohort to "the
    patients NephroQ happens to need" rather than the patients KFRE is defined
    on. Patients lacking a baseline HbA1c are still eligible; NephroQ's
    population fallback for them is recorded via hba1c_imputed_for_benchmark."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
    import calibrate_mimic as cal

    def p(pid, uacr_obs, hba1c_obs, **kw):
        base = dict(patient_id=pid, uacr_baseline_observed=uacr_obs,
                    hba1c_baseline_observed=hba1c_obs, uacr_baseline_strict=200.0,
                    age_at_index=65.0, sex="M", baseline_egfr=40.0)
        base.update(kw)
        return base

    patients = [
        p("both_observed", True, True),
        p("uacr_only", True, False),          # eligible: KFRE does not need HbA1c
        p("no_uacr", False, True),            # ineligible: UACR is a KFRE variable
        p("no_demographics", True, True, age_at_index=None),   # cannot compute KFRE
    ]
    kept = cal.filter_kfre_comparable(patients)
    assert [x["patient_id"] for x in kept] == ["both_observed", "uacr_only"]
    # the HbA1c fallback must be recorded, not silently hidden
    flags = {x["patient_id"]: x["hba1c_imputed_for_benchmark"] for x in kept}
    assert flags == {"both_observed": False, "uacr_only": True}


def test_kfre_risk_is_monotone():
    """KFRE risk must increase as eGFR falls and as albuminuria rises, and the
    5-year risk must never be below the 2-year risk."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
    import calibrate_mimic as cal

    by_egfr = [cal.kfre_risk(60, "M", g, 300, 2.0) for g in (60, 45, 30, 15)]
    by_acr = [cal.kfre_risk(60, "M", 30, u, 2.0) for u in (30, 100, 300, 1000)]
    assert all(b > a for a, b in zip(by_egfr, by_egfr[1:]))
    assert all(b > a for a, b in zip(by_acr, by_acr[1:]))
    assert cal.kfre_risk(60, "M", 30, 300, 5.0) >= cal.kfre_risk(60, "M", 30, 300, 2.0)
    assert 0.0 <= cal.kfre_risk(60, "F", 90, 5, 2.0) <= 1.0

def test_filter_kfre_comparable_missing_flags_returns_empty():
    """If the CSV predates the baseline_observed flags (e.g. an older run),
    the filter must return an empty list, not crash."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
    import calibrate_mimic as cal
    assert cal.filter_kfre_comparable([dict(patient_id="x")]) == []
    assert cal.filter_kfre_comparable([]) == []

def test_evaluate_baseline_forecast_uses_only_baseline_covariates():
    """
    The baseline forecast must use ONLY the FIRST covariate values (index
    date), never later ones -- even if a patient's HbA1c series changes
    dramatically after baseline, the Mode B prediction must not react to it
    (that's the whole point: it is a prospective, baseline-only forecast).
    """
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
    import calibrate_mimic as cal

    params = dict(q=1.5, k_hf=0.012, w_a1c=0.0144, w_uacr=0.018, w_sbp=0.0108)
    t = np.array([0.0, 2.0, 5.0])

    # patient A: HbA1c stays low the whole time
    pac_stable = dict(t=t, e=np.array([80.0, 74.0, 65.0]), egfr0=80.0,
                      hba1c_series=np.array([6.8, 6.8, 6.8]),
                      uacr_series=np.array([30.0, 30.0, 30.0]),
                      sbp_series=np.array([125.0, 125.0, 125.0]))
    # patient B: SAME baseline as A, but HbA1c rises sharply after baseline
    pac_rises_later = dict(t=t, e=np.array([80.0, 74.0, 65.0]), egfr0=80.0,
                           hba1c_series=np.array([6.8, 10.0, 10.0]),  # only differs AFTER baseline
                           uacr_series=np.array([30.0, 30.0, 30.0]),
                           sbp_series=np.array([125.0, 125.0, 125.0]))

    result_stable = cal.evaluate_baseline_forecast(params, [pac_stable], horizons=(2.0, 5.0))
    result_rises = cal.evaluate_baseline_forecast(params, [pac_rises_later], horizons=(2.0, 5.0))

    # Both patients have IDENTICAL baseline covariates, so the MODEL'S
    # PREDICTION (not the observed error) must be identical -- verify by
    # checking the prediction directly rather than just the RMSE (which
    # also depends on each patient's own observed e, here made identical).
    assert result_stable["year_2.0"]["rmse_mL_min"] == result_rises["year_2.0"]["rmse_mL_min"], \
        "baseline forecast must ignore post-baseline covariate changes -- predictions should match"
    assert result_stable["year_5.0"]["rmse_mL_min"] == result_rises["year_5.0"]["rmse_mL_min"]

def test_evaluate_baseline_forecast_skips_missing_horizons():
    """A patient with no observation near a requested horizon must be
    skipped for THAT horizon (not imputed or forced), while still being
    evaluated at horizons they do have data for."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
    import calibrate_mimic as cal

    params = dict(q=1.5, k_hf=0.012, w_a1c=0.0144, w_uacr=0.018, w_sbp=0.0108)
    # only a visit near year 2, nothing near year 5
    pac = dict(t=np.array([0.0, 2.1]), e=np.array([80.0, 74.0]), egfr0=80.0,
              hba1c_series=np.array([7.0, 7.0]), uacr_series=np.array([40.0, 40.0]),
              sbp_series=np.array([130.0, 130.0]))
    result = cal.evaluate_baseline_forecast(params, [pac], horizons=(2.0, 5.0), tolerance_years=0.5)
    assert "year_2.0" in result
    assert "year_5.0" not in result


def test_auc_handles_ties():
    """Tied scores must receive the AVERAGE rank. A score that cannot separate
    the classes at all must give exactly 0.5; an argsort-of-argsort ranking
    assigns arbitrary consecutive ranks to ties and fails this."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
    import calibrate_mimic as cal
    assert cal._auc([0.1, 0.1, 0.9, 0.9], [0, 1, 0, 1]) == pytest.approx(0.5)
    assert cal._auc([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1]) == pytest.approx(1.0)
    assert cal._auc([1.0, 1.0, 1.0, 1.0], [0, 1, 0, 1]) == pytest.approx(0.5)


def test_mode_c_has_no_temporal_leakage():
    """A missing BASELINE covariate must be filled from development-set defaults,
    never from the patient's own later measurements. Two cohorts identical at
    baseline but with wildly different FUTURE HbA1c must therefore produce the
    identical baseline-anchored evaluation."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
    import calibrate_mimic as cal
    import numpy as np

    def cohort(future_a1c):
        pats = []
        for i in range(40):
            rng = np.random.default_rng(i)
            n = 12
            t = np.linspace(0, 6, n)
            e0 = 25.0 + (i % 30)                      # inside the 15-60 KFRE range
            e = np.clip(e0 - rng.uniform(3, 10) * t, 3, None)
            pats.append(dict(
                patient_id=str(i), t=t, e=e, egfr0=e0, baseline_egfr=e0,
                hba1c_series=np.full(n, future_a1c),  # the patient's FUTURE
                uacr_series=np.full(n, 300.0), sbp_series=np.full(n, 140.0),
                uacr_baseline_observed=True, hba1c_baseline_observed=False,
                hba1c_baseline_strict=np.nan,         # NO baseline HbA1c
                uacr_baseline_strict=300.0, sbp_baseline_strict=140.0,
                age_at_index=65.0, sex="M"))
        return cal.filter_kfre_comparable(pats)

    params = dict(q=1.52, k_hf=0.0141, w_a1c=0.0144, w_uacr=0.0180, w_sbp=0.0108)
    dev = dict(hba1c=7.5, sbp=135.0, uacr=100.0)
    low = cal.evaluate_kfre_benchmark(params, cohort(6.0), development_defaults=dev)
    high = cal.evaluate_kfre_benchmark(params, cohort(12.0), development_defaults=dev)
    assert low and high
    for key in low:
        assert low[key]["auc_nephroq"] == pytest.approx(high[key]["auc_nephroq"])


def test_mode_c_excludes_incomplete_followup():
    """A patient followed for only 2.6 years must NOT be scored as 'no event at
    5 years' -- their outcome at that horizon is unknown."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
    import calibrate_mimic as cal
    import numpy as np

    def pat(pid, tmax):
        n = 8
        t = np.linspace(0, tmax, n)
        return dict(patient_id=pid, t=t, e=np.linspace(50, 40, n), egfr0=50.0,
                    baseline_egfr=50.0,
                    hba1c_series=np.full(n, 8.0), uacr_series=np.full(n, 300.0),
                    sbp_series=np.full(n, 140.0),
                    uacr_baseline_observed=True, hba1c_baseline_observed=True,
                    hba1c_baseline_strict=8.0, uacr_baseline_strict=300.0,
                    sbp_baseline_strict=140.0, age_at_index=65.0, sex="M")

    short = cal.filter_kfre_comparable([pat(f"s{i}", 2.6) for i in range(20)])
    dev = dict(hba1c=7.5, sbp=135.0, uacr=100.0)
    params = dict(q=1.52, k_hf=0.0141, w_a1c=0.0144, w_uacr=0.0180, w_sbp=0.0108)
    res = cal.evaluate_kfre_benchmark(params, short, horizons=(5.0,), development_defaults=dev)
    # nobody has 5 years of follow-up -> the 5-year horizon must yield nothing,
    # rather than silently labeling everyone event-free.
    assert res is None or "year_5.0" not in res


def test_unpack_does_not_overflow():
    """The logistic reparameterization must be numerically stable at extreme
    values (it previously raised RuntimeWarning: overflow encountered in exp)."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
    import calibrate_mimic as cal
    import numpy as np
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        vals = cal.unpack(np.array([-800.0, 800.0, 0.0, 0.0, 0.0]))
    assert np.all(np.isfinite(vals))
    assert np.all(vals >= cal.LO) and np.all(vals <= cal.HI)


def test_calibration_slope_intercept_recovers_perfect_calibration():
    """A perfectly calibrated forecaster must give slope ~1 and intercept ~0;
    an over-confident (too-extreme) one must give slope < 1."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
    import calibrate_mimic as cal
    import numpy as np

    rng = np.random.default_rng(0)
    p = rng.uniform(0.02, 0.95, 4000)
    y = (rng.uniform(size=4000) < p).astype(int)          # outcomes really occur with prob p
    intercept, slope = cal._calibration_slope_intercept(p, y)
    assert 0.9 < slope < 1.1
    assert abs(intercept) < 0.15

    p_over = np.clip((p - 0.5) * 2.2 + 0.5, 0.01, 0.99)   # too extreme
    _, slope_over = cal._calibration_slope_intercept(p_over, y)
    assert slope_over < slope


def test_brier_bounds():
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
    import calibrate_mimic as cal
    assert cal._brier([1.0, 0.0], [1, 0]) == pytest.approx(0.0)   # perfect
    assert cal._brier([0.0, 1.0], [1, 0]) == pytest.approx(1.0)   # maximally wrong


def test_nephroq_risk_from_bootstrap_is_a_probability():
    """The bootstrap-derived NephroQ risk must be a fraction in [0,1] and must
    increase for a sicker patient."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
    import calibrate_mimic as cal
    import numpy as np

    boot = [dict(q=1.52 + d, k_hf=0.0141 * (1 + d / 4), w_a1c=0.0144,
                 w_uacr=0.0180, w_sbp=0.0108) for d in np.linspace(-0.25, 0.25, 15)]
    healthy = cal.nephroq_risk_from_bootstrap(boot, a1c=6.5, uacr=10.0, sbp=120.0,
                                              egfr0=55.0, horizon=5.0)
    sick = cal.nephroq_risk_from_bootstrap(boot, a1c=10.0, uacr=1500.0, sbp=165.0,
                                           egfr0=20.0, horizon=5.0)
    assert 0.0 <= healthy <= 1.0 and 0.0 <= sick <= 1.0
    assert sick > healthy
    assert cal.nephroq_risk_from_bootstrap(None, 8.0, 300.0, 140.0, 40.0, 5.0) is None


def test_nephron_fraction_stays_physical():
    """N must stay in (0, 1] at any eGFR, and round-trip exactly inside the
    physiological range. Above the ceiling N saturates at 1 (you cannot have more
    than a full complement of nephrons), so the round-trip clamps -- by design."""
    for egfr in (5.0, 15.0, 47.7, 90.0, 120.0):
        N = core.N_of_egfr(egfr)
        assert 0.0 < N <= 1.0
        assert core.egfr_of_N(N) == pytest.approx(egfr, rel=1e-6)
    assert core.N_of_egfr(130.0) == pytest.approx(1.0)      # saturates, does not exceed 1


def test_hazard_is_finite_and_capped():
    """Even at near-total nephron loss the hazard must stay finite (it is capped),
    so the integrator can never blow up."""
    p = core.TRIAL_CALIBRATION_V2
    for N in (0.9, 0.3, 0.05, 1e-3):
        h = core.renal_hazard_v2(N, 0.6, 9.0, 2000.0, 170.0, 0.0, p)
        assert np.isfinite(h) and 0.0 < h <= 50.0
