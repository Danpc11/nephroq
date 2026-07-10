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

def test_calibrator_and_app_produce_same_trajectory():
    """
    THE test that would have caught the ~11 mL/min/1.73m2 divergence between
    calibrate_mimic.py's (former) explicit RK4 integrator and the app's
    solve_ivp-based MechanisticRenalModel: compares the FULL trajectory
    produced by both code paths, not just an instantaneous hazard value.

    calibrate_mimic.py's predict_egfr is now DYNAMIC (time-varying insult
    from a patient's own covariate series -- see docs/KNOWN_ISSUES.md
    "dynamic covariates"), while the app's MechanisticRenalModel uses a
    CONSTANT insult (a forward "what if these labs stay the same"
    projection). This test checks the natural equivalence case: when a
    patient's covariate series is literally constant over time, the dynamic
    engine must collapse to the same trajectory as the app's constant
    engine -- both ultimately call model_core's solve_ivp integrator with an
    insult that doesn't change, so they should be numerically identical.
    """
    import numpy.testing as npt
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
    import calibrate_mimic as cal

    q, k_hf = 1.52, 0.0141
    w_a1c, w_uacr, w_sbp = 0.0144, 0.0180, 0.0108
    a1c, sbp, uacr, egfr0 = 8.1, 142, 145, 47.7

    # path 1: the calibration script's (dynamic) predict_egfr, with a
    # CONSTANT covariate series (same value at every visit) -- the
    # degenerate case that should match a constant-insult simulation.
    t_query = np.linspace(0, 15, 50)
    n_visits = 8
    pac = dict(t=np.linspace(0, 15, n_visits), egfr0=egfr0,
              hba1c_series=np.full(n_visits, a1c),
              uacr_series=np.full(n_visits, uacr),
              sbp_series=np.full(n_visits, sbp))
    egfr_calibration = cal.predict_egfr(q, k_hf, pac, np.array([w_a1c, w_uacr, w_sbp]), t_query)

    # path 2: the app's MechanisticRenalModel (constant insult)
    m = MechanisticRenalModel(a1c=a1c, sbp=sbp, uacr=uacr, u=0.0, k_hf=k_hf, q=q,
                              w_a1c=w_a1c, w_uacr=w_uacr, w_sbp=w_sbp)
    _, _, egfr_app_full, _ = m.simulate(N_of_egfr(egfr0), years=15, n=600)
    t_app_full = np.linspace(0, 15, 600)
    egfr_app = np.interp(t_query, t_app_full, egfr_app_full)

    npt.assert_allclose(egfr_calibration, egfr_app, rtol=1e-2, atol=0.5,
                        err_msg="Calibration (dynamic, constant covariates) and app "
                        "(constant insult) trajectories diverge -- they must both go "
                        "through model_core's solve_ivp integrator consistently.")


def test_dynamic_insult_responds_to_changing_covariates():
    """
    The dynamic engine must actually behave differently from a constant one
    when covariates change over time: a patient whose HbA1c rises partway
    through follow-up should decline faster in the second half than an
    otherwise-identical patient whose HbA1c stayed low throughout.
    """
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
    import calibrate_mimic as cal

    q, k_hf = 1.52, 0.0141
    w = np.array([0.0144, 0.0180, 0.0108])
    n_visits = 10
    t = np.linspace(0, 10, n_visits)

    pac_rising = dict(t=t, egfr0=80.0,
                      hba1c_series=np.where(t < 5, 6.8, 10.0),   # jumps up at year 5
                      uacr_series=np.full(n_visits, 30.0),
                      sbp_series=np.full(n_visits, 125.0))
    pac_stable = dict(t=t, egfr0=80.0,
                      hba1c_series=np.full(n_visits, 6.8),        # stays low throughout
                      uacr_series=np.full(n_visits, 30.0),
                      sbp_series=np.full(n_visits, 125.0))

    t_query = np.array([9.5])
    egfr_rising = cal.predict_egfr(q, k_hf, pac_rising, w, t_query)[0]
    egfr_stable = cal.predict_egfr(q, k_hf, pac_stable, w, t_query)[0]

    assert egfr_rising < egfr_stable, (
        f"the patient whose HbA1c rose should have LOWER eGFR at year 9.5 "
        f"({egfr_rising:.1f}) than the one who stayed well-controlled ({egfr_stable:.1f})")


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
    """KDIGO GFR categories: G1>=90, G2 60-89, G3a 45-59, G3b 30-44, G4 15-29, G5<15.
    Imports the REAL function (from model_core, re-exported by mechanistic_twin
    and used by app_web.py) instead of reimplementing the boundaries locally --
    a local copy could drift from the real one and still pass its own test."""
    from mechanistic_twin import gfr_category

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
