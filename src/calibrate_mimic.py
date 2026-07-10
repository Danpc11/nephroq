"""
================================================================================
CALIBRATION WITH LOCAL MIMIC-IV  ·  100% on your machine, uploads nothing
================================================================================
Takes your LOCAL copy of MIMIC-IV (hosp module), builds the cohort of
diabetic patients with eGFR trajectories (via mimic_loader.py), calibrates
the mechanistic model, and writes A SINGLE FILE:

    calibration/mimic_calibration.json

That JSON contains only AGGREGATE PARAMETERS (q, k_hf, weights) -- never
patient data. It is not pushed to git (see .gitignore). It is the file that:
  - the web app (app_web.py) uses automatically if present, as the
    research/demo calibration -- but ONLY if quality_status == "pass"
    (see docs/KNOWN_ISSUES.md for what that means and why).
  - you can share "upon reasonable request" in the publication, consistent
    with the MIMIC-IV license (see docs/MIMIC_COMPLIANCE.md) -- the
    aggregate parameters are not PHI, but you control who receives them.

METHODOLOGY NOTES (see docs/KNOWN_ISSUES.md for full detail):
  - Uses the SAME model_core simulator as the app (no more duplicated,
    silently-diverging integrators).
  - Covariates (HbA1c/UACR/SBP) are still a per-patient scalar (median),
    NOT a proper baseline-at-index-date or time-varying covariate. This is
    a known simplification -- see docs/KNOWN_ISSUES.md item on temporal
    covariate handling.
  - Observations are weighted so that no single heavily-monitored patient
    dominates the fit (each patient contributes ~equal total weight,
    regardless of how many creatinine measurements they have).
  - A held-out test split (by patient) is evaluated but NOT used to choose
    q ranges, filters, or weights -- see the "holdout_*" fields.
  - --chronic-only is an outcome-selected subset (conditions on observed
    future decline) and must NOT be used as the primary validation cohort
    for comparisons like NephroQ-vs-KFRE. It is a secondary, mechanistic
    sanity check only.

USAGE:
    python calibrate_mimic.py --mimic-dir /path/to/your/mimic-iv/hosp

Requires: numpy, pandas, scipy (already in requirements.txt). No network needed.
================================================================================
"""
import argparse, json, os, subprocess, sys, datetime, time
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from mimic_loader import main as build_mimic_csv  # reuses the already-tested loader
import model_core as core

# ---- SINGLE canonical model implementation -- see docs/KNOWN_ISSUES.md for why
# this used to be a second, independent simulator that silently diverged from
# the app's (mechanistic_twin.py) by up to ~11 mL/min/1.73m2 near collapse.
G_MAX, ALPHA, N_FLOOR, K0_FIX = core.G_MAX, core.ALPHA, core.N_FLOOR, core.K0_DEFAULT
N_of_egfr = core.N_of_egfr
egfr_of_N = core.egfr_of_N

def insult(cov, w):
    a1c, uacr, sbp = cov
    return core.metabolic_hazard(a1c, uacr, sbp, w[0], w[1], w[2])

def predict_egfr(q, khf, cov, w, t_query, egfr0):
    """Delegates to model_core.predict_egfr_at -- the SAME solve_ivp-based
    integrator MechanisticRenalModel.simulate() uses in the app."""
    N0 = N_of_egfr(egfr0)
    return core.predict_egfr_at(K0_FIX, khf, q, 1.0, insult(cov, w), N0, t_query)


def load_cohort(csv_path):
    """
    Returns (patients, missingness) where missingness is the fraction of
    patients whose HbA1c/UACR/SBP were fully imputed (never actually
    measured) rather than observed -- important context for how much to
    trust w_uacr etc. (see docs/KNOWN_ISSUES.md).
    """
    df = pd.read_csv(csv_path)
    has_flags = {"hba1c_imputed", "uacr_imputed", "sbp_imputed"}.issubset(df.columns)
    patients = []
    n_imputed = dict(hba1c=0, uacr=0, sbp=0)
    for pid, g in df.groupby("patient_id"):
        g = g.sort_values("time_years")
        if len(g) < 3:
            continue
        cov = (float(g["hba1c"].median()), float(g["uacr"].median()), float(g["sbp"].median()))
        pat = dict(cov=cov, egfr0=float(g["egfr"].iloc[0]),
                  t=g["time_years"].values.astype(float),
                  e=g["egfr"].values.astype(float), patient_id=str(pid))
        if has_flags:
            pat["hba1c_imputed"] = bool(g["hba1c_imputed"].iloc[0])
            pat["uacr_imputed"]  = bool(g["uacr_imputed"].iloc[0])
            pat["sbp_imputed"]   = bool(g["sbp_imputed"].iloc[0])
            for k in ("hba1c", "uacr", "sbp"):
                if pat[f"{k}_imputed"]:
                    n_imputed[k] += 1
        patients.append(pat)
    n = len(patients)
    missingness = {k: round(n_imputed[k]/n, 3) for k in n_imputed} if (has_flags and n) else None
    return patients, missingness


LO = np.array([0.5, 1e-4, 1e-4, 1e-4, 1e-4])
HI = np.array([3.0, 0.06, 0.06, 0.06, 0.06])
def unpack(p): return LO + (HI - LO) / (1 + np.exp(-p))
def pack(th):
    z = np.clip((th - LO) / (HI - LO), 1e-4, 1 - 1e-4)
    return np.log(z / (1 - z))


def split_train_test(patients, test_frac=0.3, seed=42):
    """Splits by PATIENT (never by row) so no patient's data leaks across
    the split. The test set is not used anywhere in fitting or model
    selection -- only for the held-out metrics reported at the end."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(patients))
    n_test = int(round(test_frac * len(patients)))
    test_idx, train_idx = set(idx[:n_test].tolist()), set(idx[n_test:].tolist())
    train = [patients[i] for i in sorted(train_idx)]
    test  = [patients[i] for i in sorted(test_idx)]
    return train, test


def evaluate_holdout(params, patients, noise_sd=3.5):
    """Unweighted RMSE/chi2 of the ALREADY-FITTED params on a patient set
    that was not used for fitting. No parameters are adjusted here."""
    q, khf, w = params["q"], params["k_hf"], np.array([params["w_a1c"], params["w_uacr"], params["w_sbp"]])
    res = []
    for pac in patients:
        pred = predict_egfr(q, khf, pac["cov"], w, pac["t"], pac["egfr0"])
        res.append((pred - pac["e"]) / noise_sd)
    if not res:
        return None
    r = np.concatenate(res)
    r = r[np.isfinite(r)]
    n_obs = len(r)
    chi2_n = float(np.mean(r**2))
    rmse = float(np.sqrt(chi2_n) * noise_sd)
    return dict(n_patients=len(patients), n_obs=n_obs, chi2_per_n=chi2_n, rmse_mL_min=rmse)


def calibrate(patients, noise_sd=3.5, seed=0, max_patients=None, verbose=True):
    """
    max_patients: if set, randomly subsample the cohort (fixed seed, for
    reproducibility) before fitting. 5 parameters need nowhere near tens of
    thousands of patients to be well identified -- subsampling to a few
    thousand gives essentially the same precision at a fraction of the
    per-iteration cost.

    Observations are weighted by 1/sqrt(n_i) PER PATIENT (in addition to
    1/noise_sd) so that a patient with 100 creatinine measurements does not
    dominate the objective over one with 4 -- both are one person. This
    matters a lot in MIMIC-IV, where critically ill patients can have labs
    drawn many times a day (see docs/KNOWN_ISSUES.md).
    """
    if max_patients and len(patients) > max_patients:
        rng_sub = np.random.default_rng(seed)
        idx = rng_sub.choice(len(patients), size=max_patients, replace=False)
        patients = [patients[i] for i in idx]
        if verbose:
            print(f"      Subsampled to {max_patients} patients for fitting "
                 f"(seed={seed}, for speed -- statistically sufficient for 5 parameters).")

    def residuals(p):
        q, khf, wa, wu, wb = unpack(p)
        w = np.array([wa, wu, wb])
        r = []
        for pac in patients:
            n_i = max(len(pac["t"]), 1)
            per_patient_scale = noise_sd * np.sqrt(n_i)   # equalizes total per-patient weight
            r.append((predict_egfr(q, khf, pac["cov"], w, pac["t"], pac["egfr0"]) - pac["e"]) / per_patient_scale)
        r = np.concatenate(r)
        return np.where(np.isfinite(r), r, 100.0)

    rng = np.random.default_rng(seed)
    base = np.array([1.5, 0.012, 0.014, 0.018, 0.011])
    best = None
    for s in range(5):
        t0 = time.time()
        p_init = pack(base) if s == 0 else pack(np.clip(base*rng.uniform(0.5, 1.8, 5), LO*1.01, HI*0.99))
        sol = least_squares(residuals, p_init, method="trf", max_nfev=3000)
        dt = time.time() - t0
        if verbose:
            q_s, khf_s, *_ = unpack(sol.x)
            print(f"      [fit {s+1}/5] {dt:5.1f}s  cost={sol.cost:.1f}  "
                 f"nfev={sol.nfev}  q={q_s:.2f}  k_hf={khf_s:.4f}"
                 f"{'  <- best so far' if best is None or sol.cost < best.cost else ''}")
        if best is None or sol.cost < best.cost:
            best = sol

    q, khf, wa, wu, wb = unpack(best.x)
    params = dict(q=float(q), k_hf=float(khf), w_a1c=float(wa), w_uacr=float(wu), w_sbp=float(wb))

    # Report UNWEIGHTED rmse/chi2 (on the fitting set) for an interpretable
    # number in the original units -- the weighting above only affects which
    # parameters get chosen, not how the fit is subsequently described.
    fit_eval = evaluate_holdout(params, patients, noise_sd=noise_sd)
    result = dict(params)
    result.update(n_patients=fit_eval["n_patients"], n_obs=fit_eval["n_obs"],
                  chi2_per_n=fit_eval["chi2_per_n"], rmse_mL_min=fit_eval["rmse_mL_min"])
    return result


def split_primary_sensitivity(patients, min_primary=30):
    """
    PRIMARY analysis: only patients with OBSERVED HbA1c and UACR (not
    population-median imputed) -- the reviewer's recommendation, since
    treating an imputed value with the same weight as a measured one can
    bias w_uacr/w_a1c toward whatever the imputation constant was, rather
    than a real population effect.

    SENSITIVITY analysis: the full cohort (imputation included), fit
    separately and reported for comparison, never as the "active" result.

    Falls back to using the full cohort as primary (with a flag) if the
    observed-only subset is too small to fit 5 parameters reliably.

    Returns (primary_patients, sensitivity_patients_or_None, used_fallback).
    """
    has_flags = all(("hba1c_imputed" in p and "uacr_imputed" in p) for p in patients) if patients else False
    if not has_flags:
        return patients, None, False   # no flags available (e.g. non-MIMIC CSV) -- can't split

    observed = [p for p in patients
               if not p.get("hba1c_imputed", True) and not p.get("uacr_imputed", True)]
    if len(observed) >= min_primary:
        return observed, patients, False
    else:
        return patients, None, True   # fallback: primary cohort too small, used full (imputed) cohort


def diagnose_cohort(patients, noise_sd=3.5):
    """
    Cheap, per-patient diagnostics computed from trajectories ALREADY in
    memory (no need to re-read labevents.csv.gz). Flags whether the cohort
    looks like smooth chronic decline (what the mechanistic model assumes)
    or acute, fluctuating trajectories (common in a hospital/critical-care
    cohort like MIMIC-IV) -- the latter will make a global population fit
    converge to a boundary/degenerate q with a huge chi2/n, even though
    nothing is wrong with the code.
    """
    net_decline = 0
    volatilities = []
    obs_counts = []
    for pat in patients:
        t, e = pat["t"], pat["e"]
        if len(t) < 3:
            continue
        obs_counts.append(len(t))
        slope = np.polyfit(t, e, 1)[0]
        if slope < 0:
            net_decline += 1
        resid = e - np.polyval(np.polyfit(t, e, 1), t)
        volatilities.append(np.std(resid))
    n = len(patients)
    frac_declining = net_decline / n if n else 0.0
    median_volatility = float(np.median(volatilities)) if volatilities else float("nan")
    max_obs = int(np.max(obs_counts)) if obs_counts else 0
    median_obs = float(np.median(obs_counts)) if obs_counts else 0.0
    print(f"\n[diagnostics] Patients with net-declining eGFR trend: "
          f"{net_decline}/{n} ({100*frac_declining:.0f}%)")
    print(f"[diagnostics] Median within-patient volatility (residual std around "
          f"a straight line): {median_volatility:.1f} mL/min/1.73m² "
          f"(assumed measurement noise: {noise_sd})")
    print(f"[diagnostics] Observations per patient: median={median_obs:.0f}, max={max_obs} "
          f"(per-patient weighting is applied during fitting -- see docs/KNOWN_ISSUES.md)")
    if median_volatility > 3 * noise_sd:
        print("[diagnostics] WARNING: within-patient volatility is much larger than the "
              "assumed measurement noise -- trajectories look ACUTE/fluctuating rather "
              "than smooth chronic decline. A single global fit will likely converge to a "
              "degenerate q (e.g. stuck at the lower bound) with a very high chi2/n. "
              "Consider --chronic-only, or use the hierarchical model instead of a flat fit.")
    if frac_declining < 0.6:
        print(f"[diagnostics] WARNING: only {100*frac_declining:.0f}% of patients show a net "
              "declining trend -- the mechanistic model (monotonic decline only) structurally "
              "cannot represent the rest. Consider --chronic-only to fit on the subset it can "
              "represent.")
    return dict(n_patients=n, frac_net_declining=round(frac_declining, 3),
               median_volatility_mL_min=round(median_volatility, 2) if volatilities else None,
               median_obs_per_patient=median_obs, max_obs_per_patient=max_obs)


def filter_chronic_like(patients, max_volatility_ratio=2.5, noise_sd=3.5):
    """
    Optional stricter cohort: keep only patients with a net-declining trend
    AND within-patient volatility not too far above measurement noise --
    i.e. trajectories the monotonic mechanistic model can plausibly represent.

    IMPORTANT (outcome-selection bias): this filter looks at each patient's
    FULL observed trajectory (including their "future" relative to any
    index date) to decide whether to keep them. That means a fit on this
    subset CANNOT be used to claim the model predicts decline -- it was
    selected for having already declined smoothly. Use this only as a
    secondary mechanistic sanity check (does the model fit well on
    textbook-like chronic trajectories?), never as the primary cohort for
    comparing predictive performance against KFRE or any other baseline.
    """
    kept = []
    for pat in patients:
        t, e = pat["t"], pat["e"]
        if len(t) < 3:
            continue
        slope = np.polyfit(t, e, 1)[0]
        resid = e - np.polyval(np.polyfit(t, e, 1), t)
        vol = np.std(resid)
        if slope < 0 and vol <= max_volatility_ratio * noise_sd:
            kept.append(pat)
    return kept


def assess_quality(result, diagnostics):
    """
    Formal accept/warn gate the APP checks before silently trusting a MIMIC
    calibration as its active parameters (see app_web.py). Returns
    (status, reasons).
    """
    reasons = []
    if result["q"] <= LO[0] + 1e-6 or result["q"] >= HI[0] - 1e-6:
        reasons.append("q_at_bound")
    if result["chi2_per_n"] > 5:
        reasons.append("high_chi2_per_observation")
    if diagnostics.get("frac_net_declining", 1.0) < 0.6:
        reasons.append("majority_nondeclining_cohort")
    if diagnostics.get("median_volatility_mL_min", 0) and diagnostics["median_volatility_mL_min"] > 3 * 3.5:
        reasons.append("high_within_patient_volatility")
    if result["n_patients"] < 30:
        reasons.append("small_cohort")
    if result.get("primary_analysis", {}).get("used_fallback_to_full_cohort"):
        reasons.append("primary_cohort_too_small_used_full_imputed_cohort")
    status = "pass" if not reasons else "warning"
    return status, reasons


def main():
    ap = argparse.ArgumentParser(description="Calibrates the twin with your LOCAL MIMIC-IV copy.")
    ap.add_argument("--mimic-dir", default=None, help="Path to the hosp/ folder of your local MIMIC-IV copy (not needed with --from-csv)")
    ap.add_argument("--out", default=os.path.join(HERE, "..", "calibration", "mimic_calibration.json"))
    ap.add_argument("--mimic-version", default="3.1")
    ap.add_argument("--min-span-days", type=int, default=180)
    ap.add_argument("--min-points", type=int, default=4)
    ap.add_argument("--chronic-only", action="store_true",
                    help="SECONDARY mechanistic analysis only (outcome-selected -- see "
                         "filter_chronic_like docstring). Do not use as the primary cohort "
                         "for predictive comparisons against KFRE or similar.")
    ap.add_argument("--max-patients", type=int, default=None,
                    help="Randomly subsample the (post-filter) TRAINING cohort to at most "
                         "this many patients before fitting, for speed. Fixed seed.")
    ap.add_argument("--keep-csv", action="store_true",
                    help="Do not delete the intermediate per-patient CSV after fitting. "
                         "Reuse it with --from-csv to skip re-reading labevents.csv.gz.")
    ap.add_argument("--from-csv", default=None,
                    help="Skip rebuilding the cohort from raw MIMIC-IV and calibrate "
                         "directly from a CSV previously produced with --keep-csv.")
    ap.add_argument("--test-frac", type=float, default=0.3,
                    help="Fraction of patients held out (by patient, not by row) for the "
                         "reported holdout metrics. Never used to choose parameters/filters.")
    ap.add_argument("--include-imputed", action="store_true",
                    help="Skip the primary(observed-only)/sensitivity(imputed-included) split "
                         "and fit the full cohort directly as before. Useful for quick "
                         "iteration; not recommended for a result you plan to report.")
    a = ap.parse_args()
    if not a.from_csv and not a.mimic_dir:
        ap.error("--mimic-dir is required unless --from-csv is given.")

    tmp_csv = a.from_csv if a.from_csv else os.path.join(HERE, "..", "data", "_mimic_tmp.csv")
    os.makedirs(os.path.dirname(os.path.abspath(tmp_csv)), exist_ok=True)

    if a.from_csv:
        print(f"[1/3] Skipping MIMIC-IV rebuild -- calibrating directly from {a.from_csv}")
    else:
        print("[1/3] Building the cohort from local MIMIC-IV (never leaves your machine)...")
        build_mimic_csv(a.mimic_dir, tmp_csv, a.min_span_days, a.min_points)

    print("\n[2/3] Calibrating the mechanistic model...")
    patients, missingness = load_cohort(tmp_csv)
    if len(patients) < 5:
        print(f"Only {len(patients)} patients with a usable trajectory -- insufficient, aborting.")
        if not a.keep_csv and not a.from_csv:
            os.remove(tmp_csv)
        return

    diagnostics = diagnose_cohort(patients)
    if missingness:
        print(f"[diagnostics] Missingness (fraction of patients with a fully-imputed value): "
             f"hba1c={missingness['hba1c']:.0%}  uacr={missingness['uacr']:.0%}  "
             f"sbp={missingness['sbp']:.0%}")
        if missingness["uacr"] > 0.5:
            print("[diagnostics] WARNING: UACR is imputed for the majority of patients -- "
                 "w_uacr is likely poorly identified. Consider it exploratory, not a robust "
                 "population estimate, until a cohort with better UACR coverage is used.")

    if a.chronic_only:
        before = len(patients)
        patients = filter_chronic_like(patients)
        print(f"[diagnostics] --chronic-only: kept {len(patients)}/{before} patients "
             f"(SECONDARY, outcome-selected subset -- see --chronic-only help text).")
        if len(patients) < 5:
            print("Too few patients remain after --chronic-only filtering -- aborting.")
            if not a.keep_csv and not a.from_csv:
                os.remove(tmp_csv)
            return

    if a.include_imputed:
        primary_patients, sensitivity_patients, used_fallback = patients, None, False
        print("[diagnostics] --include-imputed: skipping the primary/sensitivity split "
             "(fitting the full cohort directly, as before).")
    else:
        primary_patients, sensitivity_patients, used_fallback = split_primary_sensitivity(patients)
        if used_fallback:
            print(f"[diagnostics] WARNING: fewer than 30 patients have BOTH HbA1c and UACR "
                 f"observed (not imputed) -- falling back to the full cohort "
                 f"(n={len(patients)}) as the primary analysis. Treat weights with caution.")
        elif sensitivity_patients is not None:
            print(f"[diagnostics] Primary analysis: {len(primary_patients)}/{len(patients)} patients "
                 f"with OBSERVED HbA1c and UACR (not imputed). Sensitivity analysis (full "
                 f"cohort, imputation included) will be fit separately and reported for comparison.")

    train, test = split_train_test(primary_patients, test_frac=a.test_frac)
    print(f"[diagnostics] Train/test split by patient: {len(train)} train, {len(test)} held-out "
         f"(held-out set is NOT used to choose parameters/filters).")
    if len(train) > 3000 and not a.max_patients:
        print(f"[diagnostics] NOTE: {len(train)} training patients with the accurate (solve_ivp) "
             "engine may take a long time to fit. Consider --max-patients 2000-3000 for a much "
             "faster fit with essentially the same precision for 5 parameters.")

    print("\n      --- PRIMARY analysis ---")
    result = calibrate(train, max_patients=a.max_patients)
    result["diagnostics"] = diagnostics
    result["missingness"] = missingness
    result["chronic_only_filter"] = bool(a.chronic_only)
    result["max_patients_subsample"] = a.max_patients
    result["primary_analysis"] = dict(
        observed_covariates_only=not (a.include_imputed or used_fallback),
        used_fallback_to_full_cohort=used_fallback,
        n_patients_available=len(patients),
    )

    if sensitivity_patients is not None and len(sensitivity_patients) > len(primary_patients):
        print("\n      --- SENSITIVITY analysis (full cohort, imputation included) ---")
        sens_train, sens_test = split_train_test(sensitivity_patients, test_frac=a.test_frac)
        sens_result = calibrate(sens_train, max_patients=a.max_patients, seed=1)
        sens_holdout = evaluate_holdout(sens_result, sens_test)
        result["sensitivity_analysis"] = dict(
            q=sens_result["q"], k_hf=sens_result["k_hf"],
            w_a1c=sens_result["w_a1c"], w_uacr=sens_result["w_uacr"], w_sbp=sens_result["w_sbp"],
            n_patients=sens_result["n_patients"], chi2_per_n=sens_result["chi2_per_n"],
            rmse_mL_min=sens_result["rmse_mL_min"],
            holdout=sens_holdout,
        )
        print(f"      Sensitivity: q={sens_result['q']:.2f}  k_hf={sens_result['k_hf']:.4f}  "
             f"w_uacr={sens_result['w_uacr']:.4f}  (n={sens_result['n_patients']}) "
             f"-- compare against the primary result printed below.")

    holdout = evaluate_holdout(result, test)
    if holdout:
        result["holdout"] = holdout
        print(f"\n[diagnostics] Held-out (n={holdout['n_patients']} patients): "
             f"chi2/n={holdout['chi2_per_n']:.2f}  rmse={holdout['rmse_mL_min']:.1f} mL/min")

    quality_status, quality_reasons = assess_quality(result, diagnostics)
    result["quality_status"] = quality_status
    result["quality_reasons"] = quality_reasons
    result["validated"] = False   # never auto-set to True; externally-validated is a human judgment

    result["source"] = "MIMIC-IV (calibrated locally, not redistributed -- see docs/MIMIC_COMPLIANCE.md)"
    result["mimic_version"] = a.mimic_version
    result["calibration_date"] = datetime.datetime.now().isoformat(timespec="seconds")
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=HERE,
                                capture_output=True, text=True).stdout.strip()
        result["code_commit"] = commit or "unknown"
    except Exception:
        result["code_commit"] = "unknown"

    out_path = os.path.abspath(a.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    if not a.keep_csv and not a.from_csv:
        os.remove(tmp_csv)
    elif a.keep_csv:
        print(f"\n      (kept intermediate CSV at {tmp_csv} -- excluded from git; "
             f"reuse with --from-csv {tmp_csv} to skip rebuilding next time)")

    print(f"\n[3/3] Saved: {out_path}")
    print(f"       q={result['q']:.2f}  k_hf={result['k_hf']:.4f}  "
         f"n_patients={result['n_patients']}  chi2/n={result['chi2_per_n']:.2f}  "
         f"quality={quality_status}")
    if result["primary_analysis"]["observed_covariates_only"]:
        print(f"       (PRIMARY analysis: observed HbA1c+UACR only, "
             f"{result['n_patients']}/{result['primary_analysis']['n_patients_available']} patients)")
    if "sensitivity_analysis" in result:
        s = result["sensitivity_analysis"]
        print(f"       Sensitivity (full/imputed cohort, n={s['n_patients']}): "
             f"q={s['q']:.2f}  k_hf={s['k_hf']:.4f}  w_uacr={s['w_uacr']:.4f}")
        rel_diff_q = abs(s["q"] - result["q"]) / max(result["q"], 1e-6)
        if rel_diff_q > 0.25:
            print(f"       WARNING: primary and sensitivity q differ by {100*rel_diff_q:.0f}% -- "
                 "the imputed covariates are likely materially affecting the fit. Trust the "
                 "primary (observed-only) result, but investigate before publishing either.")

    if quality_reasons:
        print(f"\nWARNING: quality_status='warning' -- reasons: {quality_reasons}. "
             "The app will display this warning if it loads this calibration. "
             "See docs/KNOWN_ISSUES.md before treating this as a trustworthy calibration.")

    print("\nThis JSON is NOT pushed to git (see .gitignore). It is the file that:")
    print("  - the web app uses automatically as the research/demo calibration.")
    print("  - you can share 'upon reasonable request' in the publication.")
    print("  - see calibration/README.md for the handling policy of this file.")
    print("  - carries a 'Research-use calibration -- not externally validated' label "
         "regardless of quality_status: MIMIC-IV data does not by itself make this a "
         "validated clinical tool.")

if __name__ == "__main__":
    main()
