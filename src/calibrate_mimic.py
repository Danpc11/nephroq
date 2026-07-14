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
    (see the README (Limitations) for what that means and why).
  - you can share "upon reasonable request" in the publication, consistent
    with the MIMIC-IV license (see docs/MIMIC_COMPLIANCE.md) -- the
    aggregate parameters are not PHI, but you control who receives them.

METHODOLOGY NOTES (see the README (Limitations) for full detail):
  - Uses the SAME model_core simulator as the app (no more duplicated,
    silently-diverging integrators).
  - Covariates (HbA1c/UACR/SBP) are TIME-VARYING series (one value per visit,
    from mimic_loader.py's three-tier model: per-visit measurement > patient
    baseline > population imputation). What is still NOT enforced is a strict
    baseline-at-index-date definition, and the index date itself is simply the
    first available creatinine -- with no AKI exclusion. See the README (Limitations)
    ("temporal covariate handling" and "index date is not an AKI-free baseline").
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
from scipy.special import expit
from scipy.stats import rankdata

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from mimic_loader import main as build_mimic_csv  # reuses the already-tested loader
import model_core as core

# ---- SINGLE canonical model implementation -- for why
# this used to be a second, independent simulator that silently diverged from
# the app's (model_core.py) by up to ~11 mL/min/1.73m2 near collapse.
G_MAX, ALPHA, N_FLOOR, K0_FIX = core.G_MAX, core.ALPHA, core.N_FLOOR, core.K0_DEFAULT
N_of_egfr = core.N_of_egfr
egfr_of_N = core.egfr_of_N

def predict_egfr(q, khf, pac, w, t_query):
    """
    v2 predictor: saturating hyperfiltration + ENDOGENOUS albuminuria.

    Only BASELINE covariates are used. In v2 the UACR trajectory is an OUTPUT of
    the model, so per-visit UACR values are not fed in -- which also removes the
    dependence on imputed albuminuria, the least reliable input in any hospital
    dataset. Missing baseline covariates fall back to the patient's first
    observed value, never to a future one.
    """
    p = dict(core.TRIAL_CALIBRATION_V2)
    p.update(q=q, k_hf=khf, w_a1c=w[0], w_uacr=w[1], w_sbp=w[2])

    def _base(key, series):
        v = pac.get(key)
        if v is not None and np.isfinite(v):
            return float(v)
        arr = np.asarray(pac[series], dtype=float)
        arr = arr[np.isfinite(arr)]
        return float(arr[0]) if len(arr) else np.nan

    a1c = _base("hba1c_baseline_strict", "hba1c_series")
    uacr0 = _base("uacr_baseline_strict", "uacr_series")
    sbp = _base("sbp_baseline_strict", "sbp_series")
    return core.predict_egfr_at_v2(pac["egfr0"], a1c, uacr0, sbp, 0.0, p, t_query)


def _residual_chunk(patients_chunk, q, khf, w, noise_sd):
    """Residuals for one chunk of patients. Module-level (not a closure) so that
    it can be pickled and shipped to worker processes."""
    out = []
    for pac in patients_chunk:
        n_i = max(len(pac["t"]), 1)
        per_patient_scale = noise_sd * np.sqrt(n_i)
        out.append((predict_egfr(q, khf, pac, w, pac["t"]) - pac["e"]) / per_patient_scale)
    return np.concatenate(out) if out else np.array([])


def load_cohort(csv_path):
    """
    Returns (patients, missingness). Each patient carries their FULL
    HbA1c/UACR/SBP time series (hba1c_series, uacr_series, sbp_series,
    aligned with 't'), not a single per-patient median -- this is what
    lets the model use a time-varying insult (see predict_egfr above).
    `cov` (the old median-based triple) is kept for reference/backward
    compatibility but is not used by the fitting path anymore.

    missingness is the fraction of patients with NO real (non-imputed)
    measurement anywhere in their trajectory for a covariate -- important
    context for how much to trust w_uacr etc..
    """
    df = pd.read_csv(csv_path)
    has_flags = {"hba1c_imputed", "uacr_imputed", "sbp_imputed"}.issubset(df.columns)
    has_baseline_flags = {"hba1c_baseline_observed", "uacr_baseline_observed",
                          "sbp_baseline_observed"}.issubset(df.columns)
    patients = []
    n_imputed = dict(hba1c=0, uacr=0, sbp=0)
    for pid, g in df.groupby("patient_id"):
        g = g.sort_values("time_years")
        if len(g) < 3:
            continue
        cov = (float(g["hba1c"].median()), float(g["uacr"].median()), float(g["sbp"].median()))
        pat = dict(cov=cov, egfr0=float(g["egfr"].iloc[0]),
                  t=g["time_years"].values.astype(float),
                  e=g["egfr"].values.astype(float), patient_id=str(pid),
                  hba1c_series=g["hba1c"].values.astype(float),
                  uacr_series=g["uacr"].values.astype(float),
                  sbp_series=g["sbp"].values.astype(float))
        # KFRE demographics (MODE C). Exported by mimic_loader; absent in older
        # CSVs / non-MIMIC sources, in which case the KFRE benchmark is skipped.
        for col in ("age_at_index", "sex", "baseline_egfr"):
            if col in g.columns:
                v = g[col].iloc[0]
                pat[col] = (float(v) if col != "sex" else str(v)) if pd.notna(v) else None
        if has_flags:
            # patient-level flag = ALL rows imputed (no real measurement anywhere
            # in the trajectory) -- with time-varying covariates, a patient with
            # SOME real measurements has meaningful signal even if not every row.
            pat["hba1c_imputed"] = bool(g["hba1c_imputed"].all())
            pat["uacr_imputed"]  = bool(g["uacr_imputed"].all())
            pat["sbp_imputed"]   = bool(g["sbp_imputed"].all())
            for k in ("hba1c", "uacr", "sbp"):
                if pat[f"{k}_imputed"]:
                    n_imputed[k] += 1
        if has_baseline_flags:
            # STRICT, backward-only flags (see mimic_loader.py's
            # flag_baseline_observed): whether a REAL measurement exists at
            # or before the patient's index date -- used for the
            # KFRE-comparable baseline-forecast cohort, distinct from the
            # more permissive *_imputed flags above.
            pat["hba1c_baseline_observed"] = bool(g["hba1c_baseline_observed"].iloc[0])
            pat["uacr_baseline_observed"]  = bool(g["uacr_baseline_observed"].iloc[0])
            pat["sbp_baseline_observed"]   = bool(g["sbp_baseline_observed"].iloc[0])
            # STRICT baseline VALUES (not just the flag) -- what
            # evaluate_baseline_forecast must use, since hba1c_series[0]
            # could still come from the more permissive dynamic/baseline
            # tiers (a small forward tolerance).
            if "hba1c_baseline_strict" in g.columns:
                pat["hba1c_baseline_strict"] = float(g["hba1c_baseline_strict"].iloc[0])
                pat["uacr_baseline_strict"]  = float(g["uacr_baseline_strict"].iloc[0])
                sbp_strict = g["sbp_baseline_strict"].iloc[0]
                pat["sbp_baseline_strict"] = float(sbp_strict) if pd.notna(sbp_strict) else None
        patients.append(pat)
    n = len(patients)
    missingness = {k: round(n_imputed[k]/n, 3) for k in n_imputed} if (has_flags and n) else None
    return patients, missingness


LO = np.array([0.5, 1e-4, 1e-4, 1e-4, 1e-4])
HI = np.array([3.0, 0.06, 0.06, 0.06, 0.06])
# expit is the numerically stable logistic: 1/(1+exp(-p)) overflows in exp for
# very negative p (harmless here, but it emits RuntimeWarnings during fitting).
def unpack(p): return LO + (HI - LO) * expit(p)
def pack(th):
    z = np.clip((th - LO) / (HI - LO), 1e-4, 1 - 1e-4)
    return np.log(z / (1 - z))


def cross_validate(patients, k=5, seed=42, noise_sd=8.7, n_jobs=1, max_patients=None,
                   verbose=True):
    """
    K-fold cross-validation, split BY PATIENT.

    The point is NOT a slightly better RMSE estimate. It is a STABILITY /
    IDENTIFIABILITY check on the parameters themselves:

      - If q swings wildly from fold to fold (say 1.1 -> 2.4), then q is not
        identifiable from this cohort, and any single point estimate of it --
        however tight its bootstrap interval looks -- is an artifact of which
        patients happened to land in the training set.
      - If the parameters are stable but the out-of-fold RMSE is poor, the model
        is consistently wrong rather than unstable, which is a different problem
        and calls for a different fix.

    A bootstrap resamples the SAME patients and so cannot see this: it measures
    sampling noise around one fit. K-fold refits the model on genuinely different
    subsets, which is what exposes an unidentifiable parameter.

    Returns per-fold parameters, their spread, and out-of-fold prediction error.
    """
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(patients))
    folds = [idx[i::k] for i in range(k)]

    rows = []
    for f in range(k):
        test_idx = set(folds[f].tolist())
        train = [patients[i] for i in range(len(patients)) if i not in test_idx]
        test = [patients[i] for i in sorted(test_idx)]
        if len(train) < 10 or len(test) < 3:
            continue

        fit = calibrate(train, noise_sd=noise_sd, seed=seed + f, verbose=False,
                        n_jobs=n_jobs, max_patients=max_patients, n_multistarts=3)

        # out-of-fold error: these patients were never seen during this fit
        w = np.array([fit["w_a1c"], fit["w_uacr"], fit["w_sbp"]])
        err = []
        for pac in test:
            pred = predict_egfr(fit["q"], fit["k_hf"], pac, w, pac["t"])
            err.append(pred - pac["e"])
        err = np.concatenate(err) if err else np.array([np.nan])

        rows.append(dict(fold=f, n_train=len(train), n_test=len(test),
                         q=fit["q"], k_hf=fit["k_hf"],
                         w_a1c=fit["w_a1c"], w_uacr=fit["w_uacr"], w_sbp=fit["w_sbp"],
                         oof_rmse=float(np.sqrt(np.nanmean(err ** 2))),
                         oof_mae=float(np.nanmean(np.abs(err)))))
        if verbose:
            print(f"      fold {f + 1}/{k}: q={fit['q']:.3f}  k_hf={fit['k_hf']:.5f}  "
                  f"out-of-fold RMSE={rows[-1]['oof_rmse']:.2f} mL/min  "
                  f"(train {len(train)}, test {len(test)})")

    if not rows:
        return None

    def spread(key):
        v = np.array([r[key] for r in rows], dtype=float)
        mean = float(v.mean())
        sd = float(v.std(ddof=1)) if len(v) > 1 else 0.0
        # coefficient of variation: how much the estimate moves relative to itself
        cv = abs(sd / mean) if mean != 0 else np.inf
        return dict(mean=mean, sd=sd, cv=cv, min=float(v.min()), max=float(v.max()))

    summary = {key: spread(key) for key in ("q", "k_hf", "w_a1c", "w_uacr", "w_sbp",
                                            "oof_rmse")}

    # Two ways a parameter can fail to be identifiable, and they look OPPOSITE:
    #
    #   (a) it SWINGS across folds        -> large coefficient of variation
    #   (b) it is PINNED AT A BOUND       -> CV of exactly 0, which naively reads
    #                                        as perfect stability
    #
    # (b) is the dangerous one. If the data carry no information about a
    # parameter, the optimizer slams it into the boundary in every fold, and a
    # spread-based check reports it as rock-solid. A parameter sitting on its
    # bound is degenerate, not identified. Both are flagged.
    KEYS = ("q", "k_hf", "w_a1c", "w_uacr", "w_sbp")
    bound_lo = dict(zip(KEYS, LO))
    bound_hi = dict(zip(KEYS, HI))

    at_bound = {}
    for j, key in enumerate(KEYS):
        v = np.array([r[key] for r in rows], dtype=float)
        span = bound_hi[key] - bound_lo[key]
        tol = 0.01 * span                       # within 1% of a bound counts as pinned
        n_pinned = int(np.sum((v <= bound_lo[key] + tol) | (v >= bound_hi[key] - tol)))
        if n_pinned >= max(2, len(v) // 2):     # pinned in at least half the folds
            at_bound[key] = n_pinned

    unstable = [key for key in KEYS
                if summary[key]["cv"] > 0.20 or key in at_bound]

    if verbose:
        print(f"\n      {'parameter':<10}{'mean':>10}{'sd':>10}{'CV':>8}{'min':>10}{'max':>10}")
        for key in KEYS:
            st = summary[key]
            if key in at_bound:
                flag = f"  <-- PINNED AT BOUND in {at_bound[key]}/{len(rows)} folds"
            elif key in unstable:
                flag = "  <-- UNSTABLE (swings across folds)"
            else:
                flag = ""
            print(f"      {key:<10}{st['mean']:10.4f}{st['sd']:10.4f}{st['cv']:8.2f}"
                  f"{st['min']:10.4f}{st['max']:10.4f}{flag}")
        print(f"      out-of-fold RMSE: {summary['oof_rmse']['mean']:.2f} "
              f"+/- {summary['oof_rmse']['sd']:.2f} mL/min")
        if unstable:
            print(f"\n      >>> NOT IDENTIFIABLE from this cohort: {', '.join(unstable)}.")
            print("          A bootstrap CANNOT see this -- it resamples the SAME patients, so")
            print("          it measures noise around one fit, not whether a different set of")
            print("          patients would have given a different answer. Do not report these")
            print("          as estimates; fix them from an external anchor, or report a range.")
            if at_bound:
                print(f"\n          NOTE -- {', '.join(at_bound)} sat ON a bound in most folds.")
                print("          That yields a spread of ~0, which naively LOOKS like perfect")
                print("          stability. It is the opposite: the data carry no information")
                print("          about that parameter, so the optimizer simply slams into the")
                print("          boundary every time. This is the failure mode most likely to")
                print("          be mistaken for a good result.")
        else:
            print("\n      >>> All parameters stable across folds (CV <= 20%).")

    return dict(k=k, folds=rows, summary=summary, unstable=unstable, at_bound=at_bound,
                interpretation="K-fold refits on genuinely different patient subsets. A "
                               "parameter with a large across-fold CV is not identifiable "
                               "from this cohort, regardless of its bootstrap interval.")


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


def bootstrap_calibrate(patients, point_estimate, n_boot=15, max_patients=None, seed=100, n_jobs=1):
    """
    Patient-level bootstrap for uncertainty quantification: resample
    patients WITH REPLACEMENT (same size as the training set) n_boot times,
    refit on each resample (1 multistart, seeded at the point estimate --
    a bootstrap resample is similar data, so this converges fast), and
    return the list of fitted parameter sets.

    This runs ONCE, offline, during calibration -- the app then just
    RE-SIMULATES (cheap) a patient's projection under each of these
    parameter sets to get a parameter-uncertainty band, instead of running any
    fitting at request time. See the README (Limitations) "uncertainty
    intervals" and app_web.py.
    """
    if n_boot <= 0:
        return []
    init = np.array([point_estimate["q"], point_estimate["k_hf"], point_estimate["w_a1c"],
                     point_estimate["w_uacr"], point_estimate["w_sbp"]])
    boot_params, failures = [], []
    n = len(patients)
    rng = np.random.default_rng(seed)
    print(f"      Bootstrap ({n_boot} resamples, patient-level, for the app's uncertainty band)...")
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)   # resample WITH replacement
        resample = [patients[i] for i in idx]
        t0 = time.time()
        try:
            r = calibrate(resample, max_patients=max_patients, seed=seed + b, n_jobs=n_jobs,
                          verbose=False, n_multistarts=1, init_guess=init)
            boot_params.append(dict(q=r["q"], k_hf=r["k_hf"],
                                    w_a1c=r["w_a1c"], w_uacr=r["w_uacr"], w_sbp=r["w_sbp"]))
        except Exception as e:
            # A failed replicate used to be printed and forgotten. If 12 of 15 fail,
            # the JSON would silently carry a 3-replicate "uncertainty band" and
            # nobody would know. The count is now recorded and surfaced.
            failures.append(f"replicate {b + 1}: {type(e).__name__}: {e}")
            print(f"      (bootstrap replicate {b+1} failed, skipped: {e})")
        if (b + 1) % 5 == 0:
            print(f"        ...{b+1}/{n_boot} bootstrap replicates done "
                 f"({time.time()-t0:.1f}s for the last one)")

    if failures:
        print(f"      WARNING: only {len(boot_params)}/{n_boot} bootstrap replicates "
              f"succeeded. The uncertainty band is based on the successful ones only.")
    return dict(params=boot_params, n_requested=int(n_boot),
                n_successful=len(boot_params), n_failed=len(failures),
                failures=failures[:10])


def filter_kfre_comparable(patients):
    """
    Cohort for the MODE C head-to-head benchmark against KFRE.

    KFRE is a 4-variable baseline model: age, sex, eGFR, UACR at an index date.
    It does NOT use HbA1c. So the eligibility criteria here are exactly those
    four, all known at/before the index date -- previously this function also
    required an observed baseline HbA1c, which needlessly shrank the cohort and
    made it "the cohort NephroQ happens to need" rather than "the cohort KFRE
    is defined on".

    NephroQ additionally needs HbA1c. Both models are therefore scored on
    EXACTLY these same patients, and NephroQ's handling of a missing baseline
    HbA1c (population-median fallback) is recorded per patient in
    `hba1c_imputed_for_benchmark`, so the comparison stays like-for-like and the
    handling is documented rather than hidden.
    """
    if not patients or "uacr_baseline_observed" not in patients[0]:
        return []   # baseline flags not available (e.g. older CSV / non-MIMIC source)
    out = []
    for p in patients:
        if not p.get("uacr_baseline_observed"):
            continue
        if p.get("age_at_index") is None or p.get("sex") is None:
            continue          # KFRE demographics not exported (older CSV)
        if not np.isfinite(p.get("baseline_egfr") or np.nan):
            continue
        u = p.get("uacr_baseline_strict")
        if u is None or not np.isfinite(u) or u <= 0:
            continue
        p = dict(p)
        p["hba1c_imputed_for_benchmark"] = not bool(p.get("hba1c_baseline_observed"))
        out.append(p)
    return out


# ------------------------------------------------------------------------------
# MODE C -- direct KFRE benchmark
# ------------------------------------------------------------------------------
# 4-variable Kidney Failure Risk Equation (Tangri et al., JAMA 2011; the
# North-American-calibrated baseline survivals are used below). Predicts the
# probability of TREATED kidney failure (dialysis or transplant) within 2 and 5
# years from age, sex, eGFR and UACR at an index date.
#
# !! VERIFY BEFORE PUBLICATION: these coefficients and baseline survivals are
# transcribed from the published equation and are NOT independently validated
# here. Check them against the source paper before reporting any number.
KFRE_COEF = dict(age=-0.2201, male=0.2467, egfr=-0.5567, log_acr=0.4510)
KFRE_CENTER = dict(age=7.036, male=0.5642, egfr=7.222, log_acr=5.137)
KFRE_S0 = {2.0: 0.9832, 5.0: 0.9365}   # baseline survival at 2 and 5 years


def kfre_risk(age, sex, egfr, uacr_mg_g, horizon_years=2.0):
    """Probability of treated kidney failure within `horizon_years`.
    uacr in mg/g; sex 'M'/'F'; egfr in mL/min/1.73m2."""
    if horizon_years not in KFRE_S0:
        raise ValueError(f"KFRE baseline survival only defined for {list(KFRE_S0)}")
    male = 1.0 if str(sex).upper().startswith("M") else 0.0
    acr = max(float(uacr_mg_g), 1e-6)
    xb = (KFRE_COEF["age"]     * (age / 10.0        - KFRE_CENTER["age"]) +
          KFRE_COEF["male"]    * (male              - KFRE_CENTER["male"]) +
          KFRE_COEF["egfr"]    * (egfr / 5.0        - KFRE_CENTER["egfr"]) +
          KFRE_COEF["log_acr"] * (np.log(acr)       - KFRE_CENTER["log_acr"]))
    return float(1.0 - KFRE_S0[horizon_years] ** np.exp(xb))


def _auc(scores, labels):
    """AUC via the Mann-Whitney rank statistic.

    Uses rankdata(method="average") so that TIED scores share the average rank.
    An argsort-of-argsort assigns tied scores arbitrary consecutive ranks, which
    biases the statistic -- and ties are common here (rounded risks, patients
    with near-identical trajectories, and the floor/ceiling of the eGFR
    projection). With correct tie handling, a score that cannot separate the
    classes at all returns exactly 0.5.
    """
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    pos_mask = labels == 1
    n_pos, n_neg = int(pos_mask.sum()), int((~pos_mask).sum())
    if n_pos == 0 or n_neg == 0:
        return None
    ranks = rankdata(scores, method="average")
    r_pos = ranks[pos_mask].sum()
    return float((r_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def compute_development_defaults(patients):
    """
    Population fallback values for covariates that are MISSING AT BASELINE,
    computed ONLY from the development (training) set's BASELINE values.

    This exists to prevent temporal leakage. The obvious-looking fallback --
    "use the median of this patient's own series" -- is WRONG for any
    baseline-anchored evaluation, because that series contains measurements
    taken AFTER the index date. A prediction that is supposed to be made at
    baseline would then be using the patient's own future. These defaults are
    stored in the calibration JSON so the app and any evaluation reuse exactly
    the same numbers.
    """
    def _med(key):
        vals = [p.get(key) for p in patients]
        vals = [float(v) for v in vals if v is not None and np.isfinite(v)]
        return float(np.median(vals)) if vals else None
    return dict(hba1c=_med("hba1c_baseline_strict"),
                sbp=_med("sbp_baseline_strict"),
                uacr=_med("uacr_baseline_strict"))


def _brier(probs, labels):
    """Mean squared error of a probabilistic forecast. Lower is better."""
    p = np.asarray(probs, dtype=float)
    y = np.asarray(labels, dtype=float)
    return float(np.mean((p - y) ** 2))


def _calibration_slope_intercept(probs, labels, max_iter=100, tol=1e-8):
    """
    Logistic recalibration: fit  y ~ intercept + slope * logit(p).

    A perfectly calibrated model gives slope = 1 and intercept = 0.
    slope < 1 => predictions are too extreme (overfitted / over-dispersed).
    intercept != 0 => calibration-in-the-large is off (systematically high/low).

    IRLS in plain numpy (no sklearn dependency). Returns (intercept, slope) or
    (None, None) if the fit is degenerate (e.g. no outcome variation).
    """
    p = np.clip(np.asarray(probs, dtype=float), 1e-6, 1 - 1e-6)
    y = np.asarray(labels, dtype=float)
    if len(np.unique(y)) < 2:
        return None, None
    x = np.log(p / (1 - p))                      # logit of the predicted risk
    if np.allclose(x, x[0]):
        return None, None                        # no spread in predictions
    X = np.column_stack([np.ones_like(x), x])
    beta = np.zeros(2)
    for _ in range(max_iter):
        eta = X @ beta
        mu = 1.0 / (1.0 + np.exp(-eta))
        W = np.clip(mu * (1 - mu), 1e-10, None)
        z = eta + (y - mu) / W
        try:
            beta_new = np.linalg.solve((X * W[:, None]).T @ X, (X * W[:, None]).T @ z)
        except np.linalg.LinAlgError:
            return None, None
        if np.max(np.abs(beta_new - beta)) < tol:
            beta = beta_new
            break
        beta = beta_new
    return float(beta[0]), float(beta[1])


def nephroq_risk_from_bootstrap(bootstrap_params, a1c, uacr, sbp, egfr0, horizon):
    """
    Turn NephroQ's mechanistic projection into an actual PROBABILITY, so it can
    be compared with KFRE on more than rank-based metrics (AUC).

        P_NQ(T <= h) = (1/B) * sum_b  1[ eGFR_b(h) < 15 ]

    i.e. the fraction of bootstrap parameter resamples under which this patient
    crosses the threshold by the horizon. This unlocks Brier score and
    calibration slope/intercept, which a bare score (-eGFR) cannot support.

    CAVEAT: this propagates ONLY calibration-parameter uncertainty (the same
    limitation as the app's uncertainty band --). It
    does not include measurement noise, individual random effects, or unknown
    future covariates, so it will tend to be UNDER-dispersed (too confident).
    Read the calibration slope with that in mind.
    """
    if not bootstrap_params:
        return None
    crossed = 0
    for bp in bootstrap_params:
        p = dict(core.TRIAL_CALIBRATION_V2)
        p.update(q=bp["q"], k_hf=bp["k_hf"], w_a1c=bp["w_a1c"],
                 w_uacr=bp["w_uacr"], w_sbp=bp["w_sbp"])
        eg = core.predict_egfr_at_v2(egfr0, a1c, uacr, sbp, 0.0, p,
                                     np.array([horizon]))[0]
        crossed += int(eg < core.DIALYSIS_eGFR)
    return crossed / len(bootstrap_params)


def evaluate_kfre_benchmark(params, patients, horizons=(2.0, 5.0),
                            development_defaults=None,
                            min_followup_tolerance=0.25,
                            egfr_range=(15.0, 60.0),
                            bootstrap_params=None):
    """
    MODE C -- EXPLORATORY KFRE SCORE COMPARISON (using an eGFR<15 proxy outcome).

    Deliberately NOT called a "KFRE validation": KFRE is defined on TREATED
    kidney failure (dialysis/transplant), and the outcome used here is a PROXY
    (observed eGFR<15 within the horizon). The question actually answered is
    "which score ranks patients better for reaching a modeled eGFR<15
    threshold", which is related to, but not the same as, KFRE's target.

    Both models are scored on the SAME patients and compared by discrimination:
      - KFRE     : P(treated kidney failure by horizon) from age/sex/eGFR/UACR.
      - NephroQ  : mechanistic projection from baseline covariates held constant.
                   If `bootstrap_params` are supplied, NephroQ is turned into an
                   actual PROBABILITY -- the fraction of bootstrap parameter
                   resamples under which the patient crosses eGFR<15 by the
                   horizon -- which makes Brier score and calibration
                   slope/intercept computable alongside AUC. Without them it
                   falls back to a bare score (-eGFR at horizon), which supports
                   AUC only. NOTE the probability propagates ONLY parameter
                   uncertainty, so it is likely UNDER-dispersed (over-confident);
                   read the calibration slope accordingly.

    NO TEMPORAL LEAKAGE: a missing baseline covariate is filled from
    `development_defaults` (medians of BASELINE values in the training set),
    never from the patient's own later measurements.

    FOLLOW-UP: a patient is only scored at horizon h if they were actually
    followed to (approximately) h. Otherwise a patient with 2.6 years of
    follow-up would be silently labeled "no event at 5 years", which is
    outcome misclassification. The methodologically correct treatment is
    survival analysis with censoring (time-dependent AUC, IPCW, Brier, and --
    with real KRT -- death as a competing risk); this simple complete-follow-up
    restriction is a stopgap, not a substitute.
    """
    if not patients:
        return None
    if development_defaults is None:
        # Fall back to the cohort's own BASELINE medians (still never a
        # patient's own future), but this should normally be passed in from
        # the training set.
        development_defaults = compute_development_defaults(patients)
    q, khf = params["q"], params["k_hf"]
    w = np.array([params["w_a1c"], params["w_uacr"], params["w_sbp"]])
    out = {}
    for h in horizons:
        if h not in KFRE_S0:
            continue
        kfre_scores, nq_scores, nq_probs, labels = [], [], [], []
        n_hba1c_imputed = 0
        n_excluded_followup = 0
        n_excluded_egfr = 0
        for pac in patients:
            t, e = pac["t"], pac["e"]
            egfr0 = float(pac["baseline_egfr"])
            if egfr_range is not None and not (egfr_range[0] <= egfr0 < egfr_range[1]):
                n_excluded_egfr += 1
                continue      # incident-prediction cohort (default G3a-G4): a
                              # patient already below the threshold is in the very
                              # state we are trying to predict.
            if t.max() < (h - min_followup_tolerance):
                n_excluded_followup += 1
                continue      # incomplete follow-up: we do NOT know their outcome at h
            within = t <= h
            if not within.any():
                continue
            label = int(np.nanmin(e[within]) < core.DIALYSIS_eGFR)

            # Baseline covariates ONLY. Missing -> development-set default.
            a1c = pac.get("hba1c_baseline_strict")
            if a1c is None or not np.isfinite(a1c):
                a1c = development_defaults.get("hba1c")
                n_hba1c_imputed += 1
            sbp_b = pac.get("sbp_baseline_strict")
            if sbp_b is None or not np.isfinite(sbp_b):
                sbp_b = development_defaults.get("sbp")
            if a1c is None or sbp_b is None:
                continue      # no usable baseline value and no development default
            uacr_b = float(pac["uacr_baseline_strict"])

            p = dict(core.TRIAL_CALIBRATION_V2)
            p.update(q=q, k_hf=khf, w_a1c=w[0], w_uacr=w[1], w_sbp=w[2])
            egfr_h = core.predict_egfr_at_v2(egfr0, float(a1c), uacr_b, float(sbp_b),
                                             0.0, p, np.array([h]))[0]
            nq_scores.append(-float(egfr_h))     # lower projected eGFR = higher risk
            kfre_scores.append(kfre_risk(pac["age_at_index"], pac["sex"], egfr0, uacr_b, h))
            labels.append(label)
            # PROBABILISTIC NephroQ risk (needs the bootstrap parameter sets)
            nq_probs.append(nephroq_risk_from_bootstrap(
                bootstrap_params, float(a1c), uacr_b, float(sbp_b), egfr0, h))

        if len(labels) < 10 or len(set(labels)) < 2:
            continue    # AUC undefined / meaningless

        n_scored = len(labels)
        entry = dict(
            n_patients=n_scored,
            n_events=int(sum(labels)),
            auc_kfre=_auc(kfre_scores, labels),
            auc_nephroq=_auc(nq_scores, labels),
            n_hba1c_imputed_for_nephroq=n_hba1c_imputed,
            frac_hba1c_baseline_observed=round(1 - n_hba1c_imputed / n_scored, 3),
            frac_hba1c_imputed_from_development_cohort=round(n_hba1c_imputed / n_scored, 3),
            n_excluded_incomplete_followup=n_excluded_followup,
            n_excluded_egfr_out_of_range=n_excluded_egfr,
            egfr_range=list(egfr_range) if egfr_range else None,
            outcome="PROXY: observed eGFR<15 within horizon (NOT treated kidney failure)",
            interpretation="exploratory score comparison; not a KFRE validation",
        )

        # KFRE is already a probability -> it always gets the probabilistic metrics.
        entry["brier_kfre"] = _brier(kfre_scores, labels)
        ci, sl = _calibration_slope_intercept(kfre_scores, labels)
        entry["calibration_intercept_kfre"], entry["calibration_slope_kfre"] = ci, sl
        # !! DO NOT REPORT KFRE's Brier/calibration AGAINST THE PROXY OUTCOME AS A
        # FAIR COMPARISON. KFRE is calibrated for TREATED KIDNEY FAILURE, a much
        # RARER event than "observed eGFR<15". Scoring it against the common proxy
        # penalises it for predicting the right (rarer) thing: its Brier looks bad
        # and its calibration intercept blows up, purely because of the outcome
        # mismatch -- not because it is a worse model. AUC (rank-based) is not
        # affected by this base-rate mismatch and remains the only fair headline
        # comparison until a real KRT outcome is available.
        entry["kfre_absolute_risk_metrics_are_not_interpretable"] = (
            "KFRE is calibrated for treated kidney failure; scoring its Brier/calibration "
            "against the eGFR<15 proxy (a much more common event) is an outcome-base-rate "
            "mismatch. Use AUC for the head-to-head; do NOT report KFRE Brier/calibration "
            "as evidence that NephroQ is better calibrated.")

        # NephroQ only becomes a probability if bootstrap replicates were supplied.
        if bootstrap_params and all(p is not None for p in nq_probs):
            entry["nephroq_output"] = ("probability P(eGFR<15 by horizon), from the fraction of "
                                       "bootstrap parameter resamples crossing the threshold")
            entry["brier_nephroq"] = _brier(nq_probs, labels)
            ci, sl = _calibration_slope_intercept(nq_probs, labels)
            entry["calibration_intercept_nephroq"], entry["calibration_slope_nephroq"] = ci, sl
            # AUC on the probability agrees with AUC on the score up to ties; report both.
            entry["auc_nephroq_prob"] = _auc(nq_probs, labels)
        else:
            entry["nephroq_output"] = ("score (-eGFR at horizon), NOT a probability -- "
                                       "run with --n-bootstrap > 0 to get Brier/calibration")
        out[f"year_{h}"] = entry
    return out or None


def evaluate_baseline_forecast(params, patients, horizons=(2.0, 5.0), tolerance_years=0.5):
    """
    MODE B (prospective baseline forecast) --
    "three evaluation modes". Uses ONLY each patient's BASELINE (first
    observed) HbA1c/UACR/SBP, held CONSTANT from the index date forward
    (via model_core's constant-insult engine, NOT the dynamic one), and
    predicts eGFR at fixed horizons.

    IMPORTANT -- this is NOT a KFRE benchmark. KFRE predicts the PROBABILITY of
    treated kidney failure (dialysis/transplant) by 2 and 5 years from
    age+sex+eGFR+UACR. Mode B answers a different question: how close is the
    predicted eGFR(2y)/eGFR(5y) to the observed eGFR. Both are useful, but they
    are not interchangeable, and Mode B must never be reported as "KFRE-
    comparable". The actual head-to-head risk benchmark is MODE C
    (kfre_risk() / evaluate_kfre_benchmark() below).

    Unlike evaluate_holdout() above (MODE A, dynamic
    reconstruction), which uses each patient's full observed covariate
    history and therefore measures a different, easier task (reconstructing
    a trajectory given knowledge of how it evolved, not forecasting it
    from baseline alone).

    Only meaningful on a filter_kfre_comparable() cohort -- patients whose
    baseline covariates were actually observed, not imputed.
    """
    q, khf, w = params["q"], params["k_hf"], np.array([params["w_a1c"], params["w_uacr"], params["w_sbp"]])
    per_horizon = {h: [] for h in horizons}
    n_used = 0
    for pac in patients:
        t, e = pac["t"], pac["e"]
        if len(t) < 1:
            continue
        # STRICT baseline values (backward-only match at/before index date)
        # -- NOT hba1c_series[0]/uacr_series[0]/sbp_series[0], which come
        # from the more permissive dynamic/baseline-window tiers and could
        # still carry a small amount of forward-looking information. Falls
        # back to series[0] only if strict values weren't computed (e.g. an
        # older CSV without them) -- shouldn't happen when this function is
        # only ever called on a filter_kfre_comparable() cohort.
        a1c0 = pac.get("hba1c_baseline_strict", pac["hba1c_series"][0])
        uacr0 = pac.get("uacr_baseline_strict", pac["uacr_series"][0])
        sbp0 = pac.get("sbp_baseline_strict") or pac["sbp_series"][0]
        p_v2 = dict(core.TRIAL_CALIBRATION_V2)
        p_v2.update(q=q, k_hf=khf, w_a1c=w[0], w_uacr=w[1], w_sbp=w[2])
        used_patient = False
        for h in horizons:
            idx_near = int(np.argmin(np.abs(t - h)))
            if abs(t[idx_near] - h) > tolerance_years:
                continue   # no observation near this horizon for this patient -- skip, don't impute
            pred = core.predict_egfr_at_v2(pac["egfr0"], a1c0, uacr0, sbp0, 0.0, p_v2,
                                           np.array([h]))[0]
            per_horizon[h].append(float((pred - e[idx_near]) ** 2))
            used_patient = True
        if used_patient:
            n_used += 1
    out = {}
    for h in horizons:
        errs = per_horizon[h]
        if errs:
            out[f"year_{h}"] = dict(n_patients=len(errs), rmse_mL_min=round(float(np.sqrt(np.mean(errs))), 2))
    out["n_patients_evaluated"] = n_used
    out["n_patients_available"] = len(patients)
    return out


def evaluate_holdout(params, patients, noise_sd=3.5):
    """
    MODE A (dynamic reconstruction) -- "three
    evaluation modes". Unweighted RMSE/chi2 of the ALREADY-FITTED params on
    a patient set that was not used for fitting. Uses each patient's FULL
    observed covariate history (via predict_egfr's dynamic insult), so this
    measures how well the model reconstructs a trajectory GIVEN the actual
    exposure history -- not a prospective forecast from baseline alone.
    NOT directly comparable to KFRE; see evaluate_baseline_forecast (MODE B)
    for that. No parameters are adjusted here.
    """
    q, khf, w = params["q"], params["k_hf"], np.array([params["w_a1c"], params["w_uacr"], params["w_sbp"]])
    res = []
    for pac in patients:
        pred = predict_egfr(q, khf, pac, w, pac["t"])
        res.append((pred - pac["e"]) / noise_sd)
    if not res:
        return None
    r = np.concatenate(res)
    r = r[np.isfinite(r)]
    n_obs = len(r)
    chi2_n = float(np.mean(r**2))
    rmse = float(np.sqrt(chi2_n) * noise_sd)
    return dict(n_patients=len(patients), n_obs=n_obs, chi2_per_n=chi2_n, rmse_mL_min=rmse)


def calibrate(patients, noise_sd=3.5, seed=0, max_patients=None, verbose=True, n_jobs=1,
              n_multistarts=5, init_guess=None):
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
    drawn many times a day.

    n_multistarts / init_guess: used to make bootstrap refits (see
    bootstrap_calibrate below) cheap -- 1 multistart, seeded at the primary
    fit's optimum, converges fast since a bootstrap resample is similar data.
    """
    if max_patients and len(patients) > max_patients:
        rng_sub = np.random.default_rng(seed)
        idx = rng_sub.choice(len(patients), size=max_patients, replace=False)
        patients = [patients[i] for i in idx]
        if verbose:
            print(f"      Subsampled to {max_patients} patients for fitting "
                 f"(seed={seed}, for speed -- statistically sufficient for 5 parameters).")

    # PARALLELISM. Patients are independent, so the residual vector splits cleanly
    # across cores. The patients are chunked ONCE (not re-sent per call) and the
    # chunks are kept in the SAME ORDER, so the concatenated residual vector is
    # bit-for-bit what the serial loop produced -- the optimizer cannot tell the
    # difference. Parallelism here speeds up everything downstream: every
    # multistart, every Jacobian step, every bootstrap replicate.
    n_jobs_eff = 1 if n_jobs in (0, 1) else n_jobs
    chunks = None
    if n_jobs_eff != 1:
        try:
            from joblib import Parallel, delayed
            n_workers = os.cpu_count() if n_jobs_eff in (-1, None) else n_jobs_eff
            n_workers = max(1, min(int(n_workers or 1), len(patients)))
            if n_workers > 1:
                chunks = [patients[i::n_workers] for i in range(n_workers)]
                # keep a flat index so the parallel result can be reassembled in
                # the ORIGINAL patient order
                order = np.concatenate([np.arange(len(patients))[i::n_workers]
                                        for i in range(n_workers)])
                inv = np.argsort(order)
                sizes = np.array([max(len(p["t"]), 1) for p in patients])
                pool = Parallel(n_jobs=n_workers, backend="loky", batch_size=1)
        except Exception:
            chunks = None      # joblib unavailable -> fall back to serial

    def residuals(p):
        q, khf, wa, wu, wb = unpack(p)
        w = np.array([wa, wu, wb])
        if chunks is None:
            r = [_residual_chunk([pac], q, khf, w, noise_sd) for pac in patients]
        else:
            parts = pool(delayed(_residual_chunk)(c, q, khf, w, noise_sd) for c in chunks)
            # regroup per patient, then restore the original order
            per_patient, k = [], 0
            for c, part in zip(chunks, parts):
                j = 0
                for pac in c:
                    n_i = max(len(pac["t"]), 1)
                    per_patient.append(part[j:j + n_i]); j += n_i
            r = [per_patient[i] for i in inv]
        r = np.concatenate(r)
        return np.where(np.isfinite(r), r, 100.0)

    rng = np.random.default_rng(seed)
    base = init_guess if init_guess is not None else np.array([1.5, 0.012, 0.014, 0.018, 0.011])

    # Robust-loss scale for soft_l1: calibrate f_scale to the SPREAD of the
    # residuals at the initial guess (robust MAD estimate). soft_l1 then
    # progressively downweights observations whose standardized residual sits
    # well beyond the bulk -- i.e. acute AKI-type spikes, common in a hospital
    # cohort like MIMIC-IV -- instead of letting a handful of them dominate the
    # least-squares objective. (Because residuals are also weighted 1/sqrt(n_i)
    # per patient, a single global f_scale maps to a slightly different raw-error
    # threshold per patient; that is an accepted approximation.) See
    # the README (Limitations) "acute-event contamination".
    r0 = residuals(pack(base))
    mad = float(np.median(np.abs(r0 - np.median(r0))))
    f_scale = max(1.4826 * mad, 1e-6)

    best = None
    for s in range(n_multistarts):
        t0 = time.time()
        p_init = pack(base) if s == 0 else pack(np.clip(base*rng.uniform(0.5, 1.8, 5), LO*1.01, HI*0.99))
        # x_scale="jac" is essential here: the logit reparameterization gives the
        # Jacobian columns very different magnitudes (the q column is ~60x steeper
        # than the k_hf / weight columns near the initial guess), so with the
        # default x_scale=1 TRF's scaled-gradient test (gtol) can trigger on the
        # FIRST evaluation and return the initial guess almost unchanged -- a
        # "frozen" optimizer that produces round-number parameters and a bootstrap
        # with ~0 variance. Scaling by the Jacobian column norms removes that
        # disparity. loss="soft_l1" adds robustness to acute spikes (f_scale above).
        sol = least_squares(residuals, p_init, method="trf", max_nfev=3000,
                            x_scale="jac", loss="soft_l1", f_scale=f_scale,
                            xtol=1e-10, ftol=1e-10, gtol=1e-12)
        dt = time.time() - t0
        if verbose:
            q_s, khf_s, *_ = unpack(sol.x)
            moved = float(np.max(np.abs(unpack(sol.x) - base)))
            print(f"      [fit {s+1}/{n_multistarts}] {dt:5.1f}s  cost={sol.cost:.1f}  "
                 f"nfev={sol.nfev}  status={sol.status}  q={q_s:.3f}  k_hf={khf_s:.4f}  "
                 f"|Δparam|max={moved:.3g}"
                 f"{'  <- best so far' if best is None or sol.cost < best.cost else ''}")
        if best is None or sol.cost < best.cost:
            best = sol

    if verbose and best is not None:
        # Explicit freeze check: if the optimum still equals the initial guess to
        # working precision, the optimizer did not move -- the fit is not trustworthy.
        moved_best = float(np.max(np.abs(unpack(best.x) - base)))
        msg = (best.message or "").strip()
        print(f"      [fit] best status={best.status} ({msg}); nfev={best.nfev}; "
             f"|Δparam|max from x0 = {moved_best:.3g}"
             f"{'   *** WARNING: optimizer did not move off the initial guess ***' if moved_best < 1e-6 else ''}")

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
          f"(per-patient weighting is applied during fitting --)")
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

    Now also checks HOLDOUT performance (both evaluation modes), not just
    training-fit metrics -- a calibration can look fine on the data it was
    fit to and still generalize poorly.
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

    holdout_a = result.get("holdout_dynamic_reconstruction")
    if holdout_a:
        if holdout_a["chi2_per_n"] > 2 * max(result["chi2_per_n"], 0.1):
            reasons.append("holdout_much_worse_than_training")   # possible overfitting
        if holdout_a["chi2_per_n"] > 5:
            reasons.append("high_holdout_chi2")

    baseline_fc = result.get("holdout_baseline_forecast")
    if not baseline_fc:
        reasons.append("no_baseline_forecast_evaluation")   # can't yet compare to KFRE
    else:
        for h in (2.0, 5.0):
            key = f"year_{h}"
            if key in baseline_fc and baseline_fc[key]["rmse_mL_min"] > 15:
                reasons.append(f"poor_baseline_forecast_accuracy_{key}")

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
    ap.add_argument("--kfre-egfr-min", type=float, default=15.0,
                    help="Lower bound of baseline eGFR for the MODE C cohort. Default 15 "
                         "excludes patients already at/below the threshold being predicted.")
    ap.add_argument("--kfre-egfr-max", type=float, default=60.0,
                    help="Upper bound of baseline eGFR for the MODE C cohort (default 60 = "
                         "an incident G3a-G4 prediction cohort).")
    ap.add_argument("--cv-folds", type=int, default=0,
                    help="Run K-fold cross-validation (by patient) after the primary fit. "
                         "This is a STABILITY check: a parameter that swings across folds is "
                         "not identifiable from the cohort, which a bootstrap cannot reveal "
                         "because it resamples the same patients. 5 is a reasonable value; 0 "
                         "disables it.")
    ap.add_argument("--n-jobs", type=int, default=1,
                    help="Parallel workers for the residual evaluation. Patients are "
                         "independent, so this scales close to linearly with cores. -1 uses "
                         "every core. Results are IDENTICAL to the serial run (the residual "
                         "vector is reassembled in the original patient order).")
    ap.add_argument("--n-bootstrap", type=int, default=15,
                    help="Number of patient-level bootstrap resamples for the app's "
                         "parameter-uncertainty band. 0 disables (app falls back to a point "
                         "estimate only). Each replicate is a cheap 1-multistart refit "
                         "seeded at the point estimate. SIZING: the default 15 is a PIPELINE "
                         "SMOKE TEST, not a result -- use 100-200 for a preliminary analysis "
                         "and 500-1000 for a number you intend to publish.")
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
    # LONGITUDINAL RESIDUAL SCALE (NOT "measurement noise").
    # This is the median within-patient dispersion around each patient's own
    # straight-line trend. It is deliberately NOT called a measurement-noise
    # estimate, because it conflates several sources: analytical error, biological
    # variability, genuine clinical change, AKI episodes, irregular sampling, eGFR
    # equation error, and the structural error of the straight line used as the
    # reference. It is a residual SCALE, useful for making chi2/n interpretable
    # (chi2/n = (rmse/scale)^2) -- reporting chi2/n against the 3.5 mL/min
    # instrument-only floor inflates it several-fold for no good reason. Floored
    # at 3.5 so we never claim more precision than the instrument itself.
    #
    _mv = diagnostics.get("median_volatility_mL_min")
    resid_scale = float(max(_mv, 3.5)) if (_mv and np.isfinite(_mv)) else 3.5
    print(f"[diagnostics] Empirical longitudinal residual scale = {resid_scale:.1f} mL/min "
         f"(within-patient dispersion; NOT pure measurement noise) used for chi2/n reporting "
         f"and the quality gate. The fitted parameters are unaffected by this choice "
         f"(a global residual scale does not move the optimum); only the reported chi2/n is.")
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
    result = calibrate(train, max_patients=a.max_patients, noise_sd=resid_scale, n_jobs=a.n_jobs)
    result["diagnostics"] = diagnostics
    result["resid_scaleirical"] = resid_scale
    result["missingness"] = missingness
    result["chronic_only_filter"] = bool(a.chronic_only)
    result["max_patients_subsample"] = a.max_patients
    result["primary_analysis"] = dict(
        observed_covariates_only=not (a.include_imputed or used_fallback),
        used_fallback_to_full_cohort=used_fallback,
        n_patients_available=len(patients),
    )

    # Population fallbacks for MISSING BASELINE covariates, computed from the
    # TRAINING set's baseline values only. Stored in the JSON so evaluation and
    # the app reuse identical numbers, and so we never fill a baseline gap with
    # the patient's own future measurements (temporal leakage).
    result["development_defaults"] = compute_development_defaults(train)
    print(f"[diagnostics] Development defaults (training-set BASELINE medians, used to fill "
         f"missing baseline covariates without temporal leakage): "
         f"{result['development_defaults']}")

    if a.cv_folds > 1:
        print(f"[diagnostics] {a.cv_folds}-fold cross-validation (by patient) -- "
              f"parameter STABILITY, not just accuracy:")
        result["cross_validation"] = cross_validate(
            train, k=a.cv_folds, noise_sd=resid_scale, n_jobs=a.n_jobs,
            max_patients=a.max_patients)

    if a.n_bootstrap > 0:
        _boot = bootstrap_calibrate(train, result, n_boot=a.n_bootstrap, n_jobs=a.n_jobs,
                                    max_patients=a.max_patients)
        result["bootstrap_params"] = _boot["params"]
        # Surfaced in the JSON: a band built from 3 surviving replicates out of 15 is
        # NOT the band that was asked for, and the reader must be able to see that.
        result["bootstrap_diagnostics"] = dict(
            n_requested=_boot["n_requested"], n_successful=_boot["n_successful"],
            n_failed=_boot["n_failed"], failures=_boot["failures"])
        print(f"      Bootstrap done: {_boot['n_successful']}/{_boot['n_requested']} "
              f"replicates succeeded.")

    if sensitivity_patients is not None and len(sensitivity_patients) > len(primary_patients):
        print("\n      --- SENSITIVITY analysis (full cohort, imputation included) ---")
        sens_train, sens_test = split_train_test(sensitivity_patients, test_frac=a.test_frac)
        sens_result = calibrate(sens_train, max_patients=a.max_patients, seed=1, noise_sd=resid_scale,
                                n_jobs=a.n_jobs)
        sens_holdout = evaluate_holdout(sens_result, sens_test, noise_sd=resid_scale)
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

    holdout = evaluate_holdout(result, test, noise_sd=resid_scale)
    if holdout:
        result["holdout_dynamic_reconstruction"] = holdout
        print(f"\n[diagnostics] Held-out, MODE A (dynamic reconstruction, uses full covariate "
             f"history -- NOT comparable to KFRE): n={holdout['n_patients']} patients, "
             f"chi2/n={holdout['chi2_per_n']:.2f}  rmse={holdout['rmse_mL_min']:.1f} mL/min")

    kfre_test = filter_kfre_comparable(test)
    if kfre_test:
        baseline_forecast = evaluate_baseline_forecast(result, kfre_test)
        result["holdout_baseline_forecast"] = baseline_forecast
        print(f"[diagnostics] Held-out, MODE B (baseline eGFR forecast, baseline covariates "
             f"held constant -- an eGFR-accuracy metric, NOT a KFRE benchmark; see MODE C): "
             f"{kfre_test and len(kfre_test)}/{len(test)} held-out patients are KFRE-eligible.")
        for h in (2.0, 5.0):
            key = f"year_{h}"
            if key in baseline_forecast:
                print(f"      Year {h}: n={baseline_forecast[key]['n_patients']}  "
                     f"rmse={baseline_forecast[key]['rmse_mL_min']:.1f} mL/min")

        # ---- MODE C: direct head-to-head against KFRE (same patients) ----
        kfre_bench = evaluate_kfre_benchmark(
            result, kfre_test,
            development_defaults=result["development_defaults"],
            egfr_range=(a.kfre_egfr_min, a.kfre_egfr_max),
            bootstrap_params=result.get("bootstrap_params"))
        result["holdout_kfre_benchmark"] = kfre_bench
        if kfre_bench:
            print("[diagnostics] Held-out, MODE C (EXPLORATORY KFRE SCORE COMPARISON -- "
                  "discrimination on the SAME patients; NOT a KFRE validation):")
            for key, v in kfre_bench.items():
                print(f"      --- {key} : n={v['n_patients']} ({v['n_events']} events) ---")
                print(f"          baseline HbA1c observed {100*v['frac_hba1c_baseline_observed']:.0f}%  |  "
                     f"imputed from development cohort {100*v['frac_hba1c_imputed_from_development_cohort']:.0f}%")
                print(f"          excluded: {v['n_excluded_incomplete_followup']} incomplete follow-up, "
                     f"{v['n_excluded_egfr_out_of_range']} eGFR outside {v['egfr_range']}")
                def _f(x):
                    return "  --  " if x is None else f"{x:6.3f}"
                print(f"          {'metric':<22}{'NephroQ':>9}{'KFRE':>9}")
                print(f"          {'AUC':<22}{_f(v.get('auc_nephroq_prob', v['auc_nephroq'])):>9}{_f(v['auc_kfre']):>9}")
                print(f"          {'Brier':<22}{_f(v.get('brier_nephroq')):>9}{_f(v.get('brier_kfre')):>9}")
                print(f"          {'calibration slope':<22}{_f(v.get('calibration_slope_nephroq')):>9}"
                     f"{_f(v.get('calibration_slope_kfre')):>9}")
                print(f"          {'calibration intercept':<22}{_f(v.get('calibration_intercept_nephroq')):>9}"
                     f"{_f(v.get('calibration_intercept_kfre')):>9}")
                print(f"          NephroQ output: {v['nephroq_output']}")
            print("      NOTE: outcome is a PROXY (observed eGFR<15 within horizon), NOT treated "
                  "kidney failure. Report this as an 'exploratory KFRE score comparison', never "
                  "as a KFRE validation.")
            print("      !! WARNING -- KFRE's Brier and calibration numbers above are NOT a fair "
                  "comparison. KFRE predicts TREATED KIDNEY FAILURE (rare); the proxy outcome "
                  "here is eGFR<15 (common). That base-rate mismatch inflates KFRE's Brier and "
                  "its calibration intercept no matter how good the model is. Use AUC for the "
                  "head-to-head. Absolute-risk metrics only become meaningful once a real KRT "
                  "outcome is extracted from MIMIC procedure/ICD codes.")
            print("      !! NephroQ's probability propagates ONLY parameter uncertainty, so it is "
                  "under-dispersed and its calibration slope will read << 1. That is a property "
                  "of the uncertainty model, not (necessarily) of the mechanism.")
        else:
            print("[diagnostics] MODE C (KFRE benchmark) skipped -- too few eligible patients "
                  "or no outcome variation in the held-out set.")
    else:
        print("[diagnostics] MODE B (baseline forecast, KFRE-comparable) skipped -- no "
             "held-out patients have a strictly observed baseline HbA1c AND UACR. See "
             "the README (Limitations) 'three evaluation modes'.")
        result["holdout_baseline_forecast"] = None

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
             "See the README (Limitations) before treating this as a trustworthy calibration.")

    print("\nThis JSON is NOT pushed to git (see .gitignore). It is the file that:")
    print("  - the web app uses automatically as the research/demo calibration.")
    print("  - you can share 'upon reasonable request' in the publication.")
    print("  - see calibration/README.md for the handling policy of this file.")
    print("  - carries a 'Research-use calibration -- not externally validated' label "
         "regardless of quality_status: MIMIC-IV data does not by itself make this a "
         "validated clinical tool.")

if __name__ == "__main__":
    main()
