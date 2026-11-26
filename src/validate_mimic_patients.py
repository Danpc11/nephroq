"""
================================================================================
PATIENT-LEVEL VALIDATION ON REAL TRAJECTORIES (MIMIC-IV)
================================================================================
Implements the protocol fixed in external_validation.validate_patient_level, on
the MIMIC-IV cohort built by mimic_loader.py:

  1. Each patient's creatinine-based eGFR series (confirmed index, see the
     loader) is thinned to one value per window (median of the draws in each
     90-day window), which damps inpatient spikes and repeated same-day draws.
  2. The first EARLY_YEARS (default 2) after the index are the observed window;
     every method personalizes on it and nothing later.
  3. Held-out visits after the window are scored at three horizons measured from
     the last observed visit: ~1 y (0.5-2], ~3 y (2-4], ~5 y (4-6.5]; one visit
     per patient per horizon (the one closest to the horizon centre), so no
     patient dominates.
  4. Split: by anchor_year_group (temporal: the most recent group(s) form the
     test set) when patients.csv.gz is given, otherwise a random 30 % of patients.
  5. What is learned from the TRAINING patients (and nothing from the test ones):
        noise model     Student-t fitted to short-interval eGFR differences
        stage 1 NPE     neural posterior under the uniform prior box, trained on
                        simulations with that noise model
        population      empirical-Bayes prior over (q, log injury rate) from the
        prior           stage-1 draws (population_prior.py), K = 1 or 2
        stage 2 NPE     neural posterior retrained on simulations drawn from the
                        learned population prior (full sample quality, no
                        importance weights)
        hybrid          gradient boosting on the RESIDUALS of the stage-2
                        mechanistic forecast
        LMM-EB, GBM     the two statistical baselines
  6. Methods scored on identical held-out visits:
        population      q at population value, injury rate 1 (no personalization)
        last value      persistence: the last observed (thinned) eGFR, with the
                        training patients' empirical change by horizon as its
                        predictive distribution
        LS rate only    one-parameter least squares on the injury rate (same
                        injury-rate range as the NPE prior box)
        MLP ensemble    the app's current personalizer
        NPE, uniform    stage-1 posterior predictive
        NPE, reweighted stage-1 draws importance-reweighted to the learned prior
        NPE, pop. prior stage-2 posterior predictive (measurement noise only)
        NPE + deviation stage-2 posterior predictive plus a between-visit
                        deviation whose variance grows with the time since the
                        last observation, fitted on the training patients'
                        held-out visits     <- mechanistic reference
        Hybrid          stage-2 forecast + learned residual correction
        OLS slope       per-patient straight line through the early window
        Theil-Sen       robust per-patient slope (median of pairwise slopes)
        LMM-EB slope    empirical-Bayes random-slope model
        GBM             gradient boosting from the early window to the future
     Predictive distributions of the NPE methods include the fitted measurement
     noise, because the targets are observations. The intervals of GBM and the
     hybrid are recalibrated with cross-fitted (out-of-fold) standardized
     residuals, because quantile models scored on their own training data are
     over-confident.
  7. Scores: RMSE, MAE, bias, 90 % interval coverage, CRPS by horizon; AUC and
     Brier for eGFR < 30 at a held-out visit (patients >= 30 at the end of the
     window); patient-bootstrap 95 % CIs of the differences in RMSE, MAE, CRPS,
     90 % coverage and AUC against the mechanistic reference and against GBM.
  8. Diagnostics: identifiability of (q, injury rate) on the test patients
     (posterior contraction and q-rate correlation), bootstrap stability of the
     learned mixture, and the selection flow (including recorded deaths) by
     exclusion reason.

DATA PROTECTION (PhysioNet Credentialed Health Data License 1.5.0): run this on
the machine that holds MIMIC. It writes per-patient rows to a *.csv in the output
directory (results/, or --out-dir, or $NEPHROQ_OUT_DIR), which must stay on that
machine. Only the AGGREGATE report (mimic_patient_validation.md / .png) may be
shared or published.

Run (trains and caches both networks on first use, ~30 min on a CPU node):
    python mimic_loader.py --mimic-dir <mimic>/hosp --out ../data/_mimic_cohort.tsv
    python validate_mimic_patients.py --cohort ../data/_mimic_cohort.tsv \\
        --patients <mimic>/hosp/patients.csv.gz
================================================================================
"""
import argparse
import os

import numpy as np
import pandas as pd
from scipy import special, stats

import fast_sim
import personalize as pz
import npe_posterior as npe
import population_prior as popprior
import outputs

HERE = os.path.dirname(os.path.abspath(__file__))

EARLY_YEARS = 2.0
WINDOW_DAYS = 90
HORIZONS = {"1y": (0.5, 2.0, 1.0), "3y": (2.0, 4.0, 3.0), "5y": (4.0, 6.5, 5.0)}
LEVELS = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95])
COV_RANGES = dict(a1c=(5.5, 12.0), uacr0=(5.0, 3000.0), sbp=(105.0, 185.0))
METHODS = ["population", "last", "ols", "theil_sen", "lmm_eb", "ls_rate", "mlp", "npe",
           "npe_eb", "npe_pop", "npe_dev", "hybrid", "gbm"]
REFERENCES = ["npe_dev", "gbm"]      # bootstrap differences are reported against these
HIGHLIGHT = ("npe_dev", "hybrid")
LABELS = {"population": "Population parameters", "last": "Last observed value",
          "ls_rate": "Least squares, rate only",
          "mlp": "MLP ensemble (current app)", "npe": "NPE, uniform prior",
          "npe_eb": "NPE, reweighted to the learned prior",
          "npe_pop": "NPE, retrained on the learned prior",
          "npe_dev": "NPE, learned prior + between-visit deviation",
          "hybrid": "Hybrid: NPE + learned residual correction",
          "ols": "Per-patient OLS slope", "theil_sen": "Per-patient Theil-Sen slope",
          "lmm_eb": "Mixed-model slope (empirical Bayes)",
          "gbm": "Gradient boosting (trained on MIMIC)"}
N_PRED_DRAWS = 1000                  # predictive draws per test patient
NOISE_DF_MIN = 2.5                   # lower limit of the fitted Student-t degrees of freedom


# ==============================================================================
# Cohort -> per-patient early window and held-out visits
# ==============================================================================
COVARIATES = {"a1c": ("hba1c", "hba1c_baseline_strict", "hba1c_imputed"),
              "uacr0": ("uacr", "uacr_baseline_strict", "uacr_imputed"),
              "sbp": ("sbp", "sbp_baseline_strict", "sbp_imputed")}


def _baseline_covariate(g_early, col, strict_col, imputed_col):
    """
    Baseline covariate from the OBSERVED WINDOW ONLY (audit P0.4): the strict,
    backward-only value at the index if the loader recorded one; otherwise the
    first row's value unless the loader imputed it (its imputation used the whole
    cohort, test patients included). Returns NaN when neither exists; NaNs are
    imputed later with TRAINING medians only.
    """
    if strict_col in g_early and g_early[strict_col].notna().any():
        return float(g_early[strict_col].dropna().iloc[0])
    if col in g_early and len(g_early):
        row = g_early.iloc[0]
        imputed = bool(row.get(imputed_col, False)) if imputed_col in g_early else False
        if not imputed and np.isfinite(row[col]):
            return float(row[col])
    return np.nan


def thin(t, e, window_days=WINDOW_DAYS):
    """One value per window: the median time and the median eGFR of the draws in it."""
    if len(t) == 0:
        return np.asarray(t, float), np.asarray(e, float)
    b = np.floor(np.asarray(t) * 365.25 / window_days).astype(int)
    df = pd.DataFrame(dict(b=b, t=t, e=e)).groupby("b").median()
    return df.t.to_numpy(), df.e.to_numpy()


def _index_confirmed_in_window(t, e, confirm_days=90, max_recovery=0.30):
    """KDIGO-style confirmation of the index eGFR using ONLY the observed window:
    a later value at >= 90 days must not have recovered by more than 30 %. The
    loader may have confirmed the index with a value from after the forecast
    origin; this re-check makes the cohort definition use only data available
    at the origin."""
    later = np.flatnonzero(t - t[0] >= confirm_days / 365.25)
    if later.size == 0:
        return False
    j = later[0]
    return bool((e[j] - e[0]) / e[0] <= max_recovery)


def build_patients(cohort, early_years=EARLY_YEARS, window_days=WINDOW_DAYS,
                   reasons=None):
    """
    Returns a list of dicts, one per eligible patient, and a dict of exclusion
    counts.

    Temporal integrity (audit P0.4): each trajectory is CUT at the forecast
    origin (EARLY_YEARS after the index) BEFORE anything is computed, and the two
    sides are thinned separately, so no observed-window value can contain a draw
    from after the origin. The index confirmation and every covariate use the
    observed window only. Changing or deleting any data after the origin leaves
    the observed window, the covariates and every prediction unchanged
    (tests/test_validate_mimic.py).

    Eligible: index confirmable within the window, >= 3 thinned visits spanning
    >= 0.75 y inside it, >= 1 held-out visit inside a scoring horizon, and a first
    eGFR inside the model's validated range (20-110). If `reasons` is a dict,
    it receives patient id -> exclusion reason ("included" for eligible ones).
    """
    reasons = {} if reasons is None else reasons
    out = []
    excl = dict(index_not_confirmed_in_window=0, too_few_early=0, no_future=0,
                egfr0_out_of_domain=0)
    for pid, g in cohort.groupby("patient_id", sort=False):
        g = g.sort_values("time_years")
        tt = g.time_years.to_numpy(float)
        g_early, g_fut = g[tt <= early_years], g[tt > early_years]
        te_raw, ee_raw = g_early.time_years.to_numpy(float), g_early.egfr.to_numpy(float)
        if len(te_raw) == 0 or not _index_confirmed_in_window(te_raw, ee_raw):
            excl["index_not_confirmed_in_window"] += 1
            reasons[str(pid)] = "index_not_confirmed_in_window"
            continue
        te, ee = thin(te_raw, ee_raw, window_days)
        if len(te) < pz.MIN_VISITS or te[-1] - te[0] < pz.MIN_SPAN_YEARS:
            excl["too_few_early"] += 1
            reasons[str(pid)] = "too_few_early"
            continue
        if not (npe.PRIOR_LO[2] <= ee[0] <= npe.PRIOR_HI[2]):
            excl["egfr0_out_of_domain"] += 1
            reasons[str(pid)] = "egfr0_out_of_domain"
            continue
        tf, ef = thin(g_fut.time_years.to_numpy(float), g_fut.egfr.to_numpy(float), window_days)
        t0 = te[0]
        te, tf = te - t0, tf - t0
        picks = {}
        for h, (lo, hi, centre) in HORIZONS.items():
            dt = tf - te[-1]
            ok = (dt > lo) & (dt <= hi)
            if ok.any():
                j = np.flatnonzero(ok)[np.argmin(np.abs(dt[ok] - centre))]
                picks[h] = (float(tf[j]), float(ef[j]))
        if not picks:
            excl["no_future"] += 1
            reasons[str(pid)] = "no_future"
            continue
        cov = {k: _baseline_covariate(g_early, *cols) for k, cols in COVARIATES.items()}
        out.append(dict(pid=str(pid), t=te, e=ee, future=picks, **cov))
        reasons[str(pid)] = "included"
    return out, excl


def impute_covariates(train, test):
    """Fill missing covariates with medians of the TRAINING patients' observed
    values, then clip to the model's covariate ranges. Returns the medians and the
    number of values imputed per covariate (train + test)."""
    med, n_imp = {}, {}
    for k, (lo, hi) in COV_RANGES.items():
        vals = np.array([p[k] for p in train if np.isfinite(p[k])])
        med[k] = float(np.median(vals)) if len(vals) else float(np.median([lo, hi]))
        n_imp[k] = 0
        for p in train + test:
            if not np.isfinite(p[k]):
                p[k] = med[k]
                n_imp[k] += 1
            p[k] = float(np.clip(p[k], lo, hi))
    return med, n_imp


def split_patients(patients, patients_csv=None, seed=0, test_frac=0.3):
    """Temporal split by anchor_year_group if available, else random."""
    ids = [p["pid"] for p in patients]
    if patients_csv and os.path.exists(patients_csv):
        pt = pd.read_csv(patients_csv, usecols=["subject_id", "anchor_year_group"],
                         dtype={"subject_id": str})
        grp = dict(zip(pt.subject_id, pt.anchor_year_group))
        groups = sorted({grp.get(i) for i in ids if grp.get(i) is not None})
        if len(groups) >= 2:
            test_groups = {groups[-1]}
            if np.mean([grp.get(i) in test_groups for i in ids]) < 0.15 and len(groups) >= 3:
                test_groups.add(groups[-2])
            is_test = np.array([grp.get(i) in test_groups for i in ids])
            return is_test, f"temporal (test = anchor_year_group {', '.join(sorted(test_groups))})"
    rng = np.random.default_rng(seed)
    is_test = rng.uniform(size=len(ids)) < test_frac
    return is_test, f"random ({int(100 * test_frac)} % of patients)"


# ==============================================================================
# Learned from the TRAINING patients
# ==============================================================================
def estimate_noise(train, max_gap=0.5, n_sim=100_000, seed=0):
    """
    Student-t measurement-noise model from consecutive thinned values less than
    `max_gap` years apart in the training patients' observed windows. Over such
    short gaps true decline is ~1 mL/min, so the difference of two values is
    essentially the difference of two independent noise terms.

    The difference of two Student-t variables is NOT Student-t (audit P1.6), so
    the parameters are fitted by matching the empirical quantiles of the
    differences to those of simulated differences sigma*(t1 - t2), t_i ~ t(nu),
    with common random numbers (fixed uniforms through the t quantile function),
    which makes the objective smooth. Returns (scale, df, n_pairs).
    """
    from scipy import optimize
    d = []
    for p in train:
        dt, de = np.diff(p["t"]), np.diff(p["e"])
        d.extend(de[dt <= max_gap])
    d = np.asarray(d, float)
    levels = np.concatenate([[0.005, 0.01, 0.025], np.linspace(0.05, 0.95, 19),
                             [0.975, 0.99, 0.995]])
    emp = np.quantile(d, levels)
    rng = np.random.default_rng(seed)
    u1, u2 = rng.uniform(1e-6, 1 - 1e-6, (2, n_sim))
    w = 1.0 / np.maximum(np.abs(emp), 1.0)            # relative fit across the range

    def loss(x):
        sigma, nu = np.exp(x[0]), NOISE_DF_MIN + np.exp(x[1])   # df bounded inside the fit
        sim = sigma * (stats.t.ppf(u1, nu) - stats.t.ppf(u2, nu))
        return float(np.sum((w * (np.quantile(sim, levels) - emp)) ** 2))

    # start from the moment-based guess under a Gaussian
    x0 = [np.log(max(np.std(d) / np.sqrt(2.0), 1e-3)), np.log(3.0)]
    res = optimize.minimize(loss, x0, method="Nelder-Mead",
                            options=dict(xatol=1e-3, fatol=1e-6, maxiter=400))
    scale, df = float(np.exp(res.x[0])), float(NOISE_DF_MIN + np.exp(res.x[1]))
    return scale, float(np.clip(df, NOISE_DF_MIN, 30.0)), int(len(d))


def _noise_sd_equivalent(scale, df):
    """SD of the fitted noise (Student-t has SD = scale * sqrt(df / (df - 2)))."""
    return scale if df is None else scale * np.sqrt(df / (df - 2.0))


def fit_population_prior(train, net, n_draws=300, seed=0, ks=(1, 2, 3), n_boot=30,
                         verbose=True):
    """Empirical-Bayes prior from the training patients' stage-1 NPE draws. K is
    chosen by the held-out marginal likelihood on 20 % of the training patients,
    then the prior is refitted on all of them. Its stability is measured by
    refitting on patient bootstrap resamples (population_prior.bootstrap_stability)."""
    draws = [npe.posterior_samples(net, p["t"], p["e"], p["a1c"], p["uacr0"], p["sbp"],
                                   n=n_draws, seed=10_000 + i) for i, p in enumerate(train)]
    rng = np.random.default_rng(seed)
    hold = rng.uniform(size=len(draws)) < 0.2
    fit_d = [d for d, h in zip(draws, hold) if not h]
    val_d = [d for d, h in zip(draws, hold) if h]
    lu = popprior._uniform_logpdf()
    best_k, best_ll, val_ll = 1, -np.inf, {}
    for K in ks:
        pr = popprior.fit(fit_d, K=K, seed=seed)
        lr = np.stack([pr.logpdf(d[:, 0], d[:, 1]) - lu for d in val_d])
        ll = float(np.mean(special.logsumexp(lr, axis=1) - np.log(lr.shape[1])))
        val_ll[K] = ll
        if ll > best_ll:
            best_k, best_ll = K, ll
    prior = popprior.fit(draws, K=best_k, seed=seed)
    stab = (popprior.bootstrap_stability(draws, best_k, n_boot=n_boot, seed=seed, ref=prior)
            if n_boot else None)
    prior.extra.update(K=best_k, heldout_loglik_per_patient=val_ll, stability=stab,
                       n_boot=n_boot)
    if verbose:
        print(f"Population prior: K={best_k} (held-out log-lik/patient {val_ll})")
    return prior


def prior_marginal_sd(prior=None, n=200_000, seed=0):
    """SD of q and of log injury rate under the uniform box (prior=None) or a
    learned prior; the denominators of posterior contraction."""
    th = npe.sample_prior(n, np.random.default_rng(seed), prior)
    return th[:, 0].std(), th[:, 1].std()


def _ols(t, e):
    n = len(t)
    tb = t.mean()
    sxx = np.sum((t - tb) ** 2)
    b = np.sum((t - tb) * (e - e.mean())) / sxx
    a = e.mean() - b * tb
    resid = e - (a + b * t)
    s2 = np.sum(resid ** 2) / max(n - 2, 1)
    return a, b, s2, tb, sxx, n


def fit_lmm_eb(train):
    """Random-slope model by moments: population slope mu, between-patient slope
    variance tau2, pooled residual variance sigma2."""
    bs, s2s = [], []
    for p in train:
        _, b, s2, _, _, _ = _ols(p["t"], p["e"])
        bs.append(b); s2s.append(s2)
    bs = np.array(bs)
    sigma2 = float(np.median(s2s))
    se_pool = sigma2 / np.array([np.sum((p["t"] - p["t"].mean()) ** 2) for p in train])
    tau2 = float(max(np.var(bs) - np.mean(se_pool), 1e-3))
    w = 1.0 / (se_pool + tau2)
    mu = float(np.sum(w * bs) / np.sum(w))
    return dict(mu=mu, tau2=tau2, sigma2=sigma2)


def fit_persistence(train):
    """Empirical change (observed - last observed value) by horizon in the
    training patients: the predictive distribution of the last-value forecast."""
    out = {h: [] for h in HORIZONS}
    for p in train:
        for h, (_, y) in p["future"].items():
            out[h].append(y - p["e"][-1])
    return {h: np.asarray(v, float) for h, v in out.items()}


def _window_features(p, t_f):
    f = pz.features(p["t"], p["e"], p["a1c"], p["uacr0"], p["sbp"])
    return np.concatenate([f, [t_f - p["t"][-1], t_f]])


def _fit_boosting(X, y, seed=0, folds=5):
    """Mean and 5 % / 95 % quantile gradient boosting, plus the out-of-fold
    standardized residuals z = (y - mean) / sd used to recalibrate the predictive
    distribution (split-conformal style: quantile models scored on their own
    training rows are over-confident)."""
    from sklearn.ensemble import HistGradientBoostingRegressor

    def fit(Xf, yf):
        m = {}
        for name, kw in [("mean", dict(loss="squared_error")),
                         ("q05", dict(loss="quantile", quantile=0.05)),
                         ("q95", dict(loss="quantile", quantile=0.95))]:
            m[name] = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05,
                                                    random_state=seed, **kw).fit(Xf, yf)
        return m

    X, y = np.asarray(X), np.asarray(y)
    z = np.full(len(y), np.nan)
    fold = np.random.default_rng(seed).integers(0, folds, len(y))
    for k in range(folds):
        tr, va = fold != k, fold == k
        if va.sum() == 0 or tr.sum() < 10:
            continue
        m = fit(X[tr], y[tr])
        mu, sd = _gbm_mu_sd(m, X[va])
        z[va] = (y[va] - mu) / sd
    models = fit(X, y)
    models["z"] = np.sort(z[np.isfinite(z)])
    return models


def _gbm_mu_sd(models, X):
    q05, q95 = models["q05"].predict(X), models["q95"].predict(X)
    return models["mean"].predict(X), np.maximum((q95 - q05) / (2 * 1.645), 1e-3)


def _calibrated_draws(mu, sd, z, rng, n=N_PRED_DRAWS):
    """Predictive draws mu + sd * z with z resampled from out-of-fold standardized
    residuals (falls back to a standard normal if there are none)."""
    zz = rng.choice(z, size=n) if len(z) else rng.standard_normal(n)
    return mu[None, :] + sd[None, :] * zz[:, None]


def fit_gbm(train, seed=0):
    X, y = [], []
    for p in train:
        for t_f, e_f in p["future"].values():
            X.append(_window_features(p, t_f)); y.append(e_f)
    return _fit_boosting(np.array(X), np.array(y), seed)


def _mechanistic_forecast(net, p, tf, n_draws, seed):
    post = npe.posterior_samples(net, p["t"], p["e"], p["a1c"], p["uacr0"], p["sbp"],
                                 n=n_draws, seed=seed)
    lat = npe.predictive(post, p["a1c"], p["uacr0"], p["sbp"], tf)
    return post, lat


def train_forecasts(train, net, n_draws=200):
    """Stage-2 latent forecasts at every training patient's held-out visits. The
    networks never see these visits (they read only the observed windows), so the
    forecasts are genuinely out of sample for the deviation and hybrid fits."""
    fc = []
    for i, p in enumerate(train):
        keys = sorted(p["future"])
        tf = np.array([p["future"][k][0] for k in keys])
        _, lat = _mechanistic_forecast(net, p, tf, n_draws, 20_000 + i)
        fc.append(dict(tf=tf, lat=lat.astype(np.float32),
                       y=np.array([p["future"][k][1] for k in keys]), dt=tf - p["t"][-1]))
    return fc


def deviation_var(dev, dt):
    """Variance of the between-visit deviation at `dt` years after the last
    observation: v0 + v1 * dt."""
    return dev["v0"] + dev["v1"] * np.asarray(dt, float)


def fit_deviation(fc, noise, n_mix=200, seed=0):
    """
    Explicit between-visit deviation (audit P1.6). Short-interval pairs measure
    assay and day-to-day variation; over years the observed eGFR also wanders
    (intercurrent illness, acute kidney injury and recovery, changes in muscle
    mass and treatment) in ways the monotone mechanistic state cannot follow. The
    observation model for a held-out visit is

        y = latent(t) + measurement noise + d,   d ~ N(0, v0 + v1 * dt),

    with dt the time since the last observed visit. (v0, v1) are fitted by
    maximum likelihood on the training patients' held-out visits, using the
    stage-2 latent draws and the Student-t noise written as a scale mixture of
    normals (Rao-Blackwellized Monte Carlo, smooth in v0 and v1).
    """
    from scipy import optimize
    scale, df = noise
    rng = np.random.default_rng(seed)
    lat = np.concatenate([f["lat"].T for f in fc]).astype(float)        # (n_visits, J)
    y = np.concatenate([f["y"] for f in fc])
    dt = np.concatenate([f["dt"] for f in fc])
    J = min(lat.shape[1], n_mix)
    lat = lat[:, :J]
    if df is None:
        meas = np.full(J, scale ** 2)
    else:
        meas = scale ** 2 / rng.gamma(df / 2.0, 2.0 / df, J)           # t as N(0, s^2/lambda)
    r2 = (y[:, None] - lat) ** 2

    def nll(x):
        v = np.exp(x[0]) + np.exp(x[1]) * dt
        var = meas[None, :] + v[:, None]
        ll = special.logsumexp(-0.5 * r2 / var - 0.5 * np.log(2 * np.pi * var), axis=1) - np.log(J)
        return -float(np.sum(ll))

    best = None
    for x0 in ([np.log(4.0), np.log(4.0)], [np.log(25.0), np.log(10.0)], [np.log(1.0), np.log(40.0)]):
        res = optimize.minimize(nll, x0, method="Nelder-Mead",
                                options=dict(xatol=1e-3, fatol=1e-3, maxiter=400))
        if best is None or res.fun < best.fun:
            best = res
    v0, v1 = float(np.exp(best.x[0])), float(np.exp(best.x[1]))
    nll0 = nll([np.log(1e-6), np.log(1e-6)])
    return dict(v0=v0, v1=v1, n_visits=int(len(y)),
                loglik_gain_per_visit=float((nll0 - best.fun) / len(y)))


def _hybrid_features(p, t_f, mu, sd):
    return np.concatenate([_window_features(p, t_f), [mu, sd]])


def fit_hybrid(train, fc, seed=0):
    """Gradient boosting on the residuals (observed - mechanistic posterior mean)
    of the stage-2 forecast, with the forecast and its spread as extra inputs."""
    X, r = [], []
    for p, f in zip(train, fc):
        mu, sd = f["lat"].mean(0), f["lat"].std(0)
        for j in range(len(f["tf"])):
            X.append(_hybrid_features(p, f["tf"][j], mu[j], sd[j]))
            r.append(f["y"][j] - mu[j])
    return _fit_boosting(np.array(X), np.array(r), seed)


# ==============================================================================
# Predictions for one test patient
# ==============================================================================
def _posterior_shape(post, sd_ref):
    """Identifiability diagnostics of one posterior over (q, log rate)."""
    c = np.corrcoef(post[:, 0], post[:, 1])[0, 1] if post[:, 0].std() > 0 and post[:, 1].std() > 0 else np.nan
    return dict(sd_q=float(post[:, 0].std()), sd_s=float(post[:, 1].std()), corr=float(c),
                contr_q=float(1 - post[:, 0].var() / sd_ref[0] ** 2),
                contr_s=float(1 - post[:, 1].var() / sd_ref[1] ** 2))


def predict_patient(p, nets, mlp, lmm, gbm, hybrid, prior, noise, dev, persist, prior_sd,
                    n_draws=N_PRED_DRAWS, seed=0):
    """Returns (keys, tf, {method: dict(mu, sd|None, draws|None)}, diagnostics)."""
    keys = sorted(p["future"])
    tf = np.array([p["future"][k][0] for k in keys])
    cov = (p["a1c"], p["uacr0"], p["sbp"])
    t, e = p["t"], p["e"]
    q_pop = npe.Q_POP
    scale, df = noise
    sd_eq = _noise_sd_equivalent(scale, df)
    rng = np.random.default_rng(seed)
    box = (float(np.exp(npe.PRIOR_LO[1])), float(np.exp(npe.PRIOR_HI[1])))
    out = {}

    def sim(q, s, e0):
        return fast_sim.simulate_egfr(q, s, e0, *cov, tf)

    def noisy(lat):
        return lat + npe.observation_noise(lat.shape, rng, scale, df)

    out["population"] = dict(mu=sim(q_pop, 1.0, e[0])[0], sd=None, draws=None)
    pers = np.stack([e[-1] + rng.choice(persist[k], size=n_draws) if len(persist[k])
                     else np.full(n_draws, e[-1]) for k in keys], axis=1)
    out["last"] = dict(mu=np.full(len(tf), e[-1]), sd=None, draws=pers)
    s1 = pz.least_squares_scale_only(t, e, *cov, q=q_pop, bounds=box)
    out["ls_rate"] = dict(mu=sim(q_pop, s1, e[0])[0], sd=None, draws=None)

    feats = pz.features(t, e, *cov)[None]
    member = np.array([np.exp(m.predict(mlp.scaler.transform(feats))[0, 1]) for m in mlp.models])
    _, _, s_m, _ = mlp.predict(feats)
    f_mem = sim(q_pop, member, e[0])
    out["mlp"] = dict(mu=sim(q_pop, float(s_m[0]), e[0])[0],
                      sd=np.sqrt(f_mem.std(0) ** 2 + sd_eq ** 2), draws=None)

    post1, lat1 = _mechanistic_forecast(nets["stage1"], p, tf, n_draws, seed)
    out["npe"] = dict(mu=lat1.mean(0), sd=None, draws=noisy(lat1))
    w, ess = popprior.reweight(post1, prior)
    lat_rw = lat1[rng.choice(len(post1), size=len(post1), p=w)]
    out["npe_eb"] = dict(mu=lat_rw.mean(0), sd=None, draws=noisy(lat_rw), ess=ess)

    post2, lat2 = _mechanistic_forecast(nets["stage2"], p, tf, n_draws, seed)
    mu2, sd2 = lat2.mean(0), lat2.std(0)
    out["npe_pop"] = dict(mu=mu2, sd=None, draws=noisy(lat2))
    dvar = deviation_var(dev, tf - t[-1])
    out["npe_dev"] = dict(mu=mu2, sd=None,
                          draws=noisy(lat2) + rng.standard_normal(lat2.shape) * np.sqrt(dvar)[None, :])

    Xh = np.array([_hybrid_features(p, tf[j], mu2[j], sd2[j]) for j in range(len(tf))])
    hm, hs = _gbm_mu_sd(hybrid, Xh)
    out["hybrid"] = dict(mu=mu2 + hm, sd=None, draws=_calibrated_draws(mu2 + hm, hs, hybrid["z"], rng))

    a, b, s2, tb, sxx, n = _ols(t, e)
    out["ols"] = dict(mu=a + b * tf, sd=np.sqrt(s2 * (1 + 1 / n + (tf - tb) ** 2 / sxx)), draws=None)
    ts = stats.theilslopes(e, t)
    out["theil_sen"] = dict(mu=ts[1] + ts[0] * tf, sd=None, draws=None)

    se2 = lmm["sigma2"] / sxx
    wl = lmm["tau2"] / (lmm["tau2"] + se2)
    b_eb = wl * b + (1 - wl) * lmm["mu"]
    var_eb = lmm["sigma2"] / n + (tf - tb) ** 2 * wl * se2 + lmm["sigma2"]
    out["lmm_eb"] = dict(mu=e.mean() + b_eb * (tf - tb), sd=np.sqrt(var_eb), draws=None)

    X = np.array([_window_features(p, x) for x in tf])
    gm, gs = _gbm_mu_sd(gbm, X)
    out["gbm"] = dict(mu=gm, sd=None, draws=_calibrated_draws(gm, gs, gbm["z"], rng))

    diag = dict(stage1=_posterior_shape(post1, prior_sd["uniform"]),
                stage2=_posterior_shape(post2, prior_sd["learned"]))
    return keys, tf, out, diag


# ==============================================================================
# Scoring
# ==============================================================================
def _crps_normal(mu, sd, y):
    z = (y - mu) / sd
    return sd * (z * (2 * stats.norm.cdf(z) - 1) + 2 * stats.norm.pdf(z) - 1 / np.sqrt(np.pi))


def _crps_draws(x, y):
    x = np.sort(x); n = len(x)
    return np.mean(np.abs(x - y)) - np.sum(x * (2 * np.arange(1, n + 1) - n - 1)) / (n * n)


def score_rows(test, nets, mlp, lmm, gbm, hybrid, prior, noise, dev, persist, prior_sd,
               verbose=True):
    """One row per (patient, horizon, method), and one diagnostics row per patient."""
    rows, diags = [], []
    for i, p in enumerate(test):
        keys, tf, pred, diag = predict_patient(p, nets, mlp, lmm, gbm, hybrid, prior, noise,
                                               dev, persist, prior_sd, seed=i)
        diags.append({f"{st}_{k}": v for st, d in diag.items() for k, v in d.items()})
        e_end = p["e"][-1]
        for j, h in enumerate(keys):
            y = p["future"][h][1]
            for m in METHODS:
                d = pred[m]
                r = dict(pid=p["pid"], horizon=h, method=m, y=y, mu=float(d["mu"][j]),
                         at_risk=bool(e_end >= 30.0), event=bool(y < 30.0),
                         ess=d.get("ess", np.nan))
                if d["draws"] is not None:
                    x = d["draws"][:, j]
                    r["crps"] = _crps_draws(x, y)
                    r["p_event"] = float(np.mean(x < 30.0))
                    for lv in LEVELS:
                        lo, hi = np.quantile(x, [0.5 - lv / 2, 0.5 + lv / 2])
                        r[f"hit_{lv:g}"] = bool(lo <= y <= hi)
                elif d["sd"] is not None:
                    sd = float(d["sd"][j])
                    r["crps"] = float(_crps_normal(r["mu"], sd, y))
                    r["p_event"] = float(stats.norm.cdf((30.0 - r["mu"]) / sd))
                    for lv in LEVELS:
                        z = stats.norm.ppf(0.5 + lv / 2)
                        r[f"hit_{lv:g}"] = bool(abs(y - r["mu"]) <= z * sd)
                rows.append(r)
        if verbose and (i + 1) % 200 == 0:
            print(f"  {i + 1}/{len(test)} test patients")
    return pd.DataFrame(rows), pd.DataFrame(diags)


def identifiability_summary(diag):
    """Aggregate identifiability of (q, log injury rate) over the test patients."""
    out = {}
    for st in ("stage1", "stage2"):
        out[st] = dict(contr_q=float(np.nanmedian(diag[f"{st}_contr_q"])),
                       contr_s=float(np.nanmedian(diag[f"{st}_contr_s"])),
                       sd_q=float(np.nanmedian(diag[f"{st}_sd_q"])),
                       sd_s=float(np.nanmedian(diag[f"{st}_sd_s"])),
                       corr=float(np.nanmedian(diag[f"{st}_corr"])),
                       frac_strong_corr=float(np.nanmean(np.abs(diag[f"{st}_corr"]) > 0.5)))
    return out


def _auc(y, p, w=None):
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y)
    if w is not None:
        keep = w > 0
        y, p, w = y[keep], np.asarray(p)[keep], w[keep]
    return float(roc_auc_score(y, p, sample_weight=w)) if 0 < y.sum() < len(y) else np.nan


BOOT_METRICS = ("rmse", "mae", "crps", "cov90", "auc")


def summarize(df, n_boot=1000, seed=0):
    """Aggregate tables (no patient-level content), and patient-bootstrap 95 %
    CIs of the differences (method - reference) in RMSE, MAE, CRPS, 90 %
    coverage and AUC. Patients are resampled with all their visits."""
    def block(d):
        err = d.mu - d.y
        res = dict(n=len(d), rmse=float(np.sqrt(np.mean(err ** 2))),
                   mae=float(np.mean(np.abs(err))), bias=float(np.mean(err)))
        res["cov90"] = float(d["hit_0.9"].mean()) if "hit_0.9" in d and d["hit_0.9"].notna().all() else np.nan
        res["crps"] = float(d.crps.mean()) if "crps" in d and d.crps.notna().all() else np.nan
        return res

    methods = [m for m in METHODS if m in set(df.method)]
    tables = {}
    for h in list(HORIZONS) + ["all"]:
        sub = df if h == "all" else df[df.horizon == h]
        tables[h] = {m: block(sub[sub.method == m]) for m in methods}

    ev = {}
    risk = df[df.at_risk]
    for m in methods:
        d = risk[risk.method == m]
        if "p_event" in d and d.p_event.notna().all() and len(d):
            ev[m] = dict(n=len(d), events=int(d.event.sum()),
                         auc=_auc(d.event.to_numpy(), d.p_event.to_numpy()),
                         brier=float(np.mean((d.p_event - d.event) ** 2)))

    # per-patient sufficient statistics, so a bootstrap resample is a weighted sum
    rng = np.random.default_rng(seed)
    pids = np.sort(df.pid.unique())
    idx = {pid: k for k, pid in enumerate(pids)}
    P = len(pids)
    stat, cnt = {}, None
    for m in methods:
        d = df[df.method == m]
        k = d.pid.map(idx).to_numpy()
        err = (d.mu - d.y).to_numpy()
        st = dict(se=np.bincount(k, weights=err ** 2, minlength=P),
                  ae=np.bincount(k, weights=np.abs(err), minlength=P))
        if "crps" in d and d.crps.notna().all():
            st["crps"] = np.bincount(k, weights=d.crps.to_numpy(float), minlength=P)
        if "hit_0.9" in d and d["hit_0.9"].notna().all():
            st["hit"] = np.bincount(k, weights=d["hit_0.9"].to_numpy(float), minlength=P)
        stat[m] = st
        if cnt is None:
            cnt = np.bincount(k, minlength=P).astype(float)
    ev_rows = {}
    for m in ev:
        d = risk[risk.method == m]
        ev_rows[m] = (d.pid.map(idx).to_numpy(), d.event.to_numpy(), d.p_event.to_numpy())

    def metrics(w):
        n_tot = np.sum(w * cnt)
        out = {}
        for m, st in stat.items():
            r = dict(rmse=np.sqrt(np.sum(w * st["se"]) / n_tot), mae=np.sum(w * st["ae"]) / n_tot)
            if "crps" in st:
                r["crps"] = np.sum(w * st["crps"]) / n_tot
            if "hit" in st:
                r["cov90"] = np.sum(w * st["hit"]) / n_tot
            if m in ev_rows:
                k, y, pe = ev_rows[m]
                r["auc"] = _auc(y, pe, w[k].astype(float))
            out[m] = r
        return out

    refs = [r for r in REFERENCES if r in methods]
    boot = {r: {m: {x: [] for x in BOOT_METRICS} for m in methods if m != r} for r in refs}
    for _ in range(n_boot):
        w = np.bincount(rng.integers(0, P, P), minlength=P)
        mt = metrics(w)
        for r in refs:
            for m in boot[r]:
                for x in BOOT_METRICS:
                    if x in mt[m] and x in mt[r]:
                        boot[r][m][x].append(mt[m][x] - mt[r][x])
    ci = {r: {m: {x: (float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5)))
                  for x, v in d.items() if len(v)}
              for m, d in boot[r].items()} for r in refs}
    return tables, ev, ci


def _weights_hash(net):
    import hashlib
    h = hashlib.sha256()
    for k, v in sorted(net.state_dict().items()):
        h.update(k.encode()); h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()[:12]


def provenance(cohort, nets, seed):
    """What is needed to trace every number in the report (audit P0.6/P1.11)."""
    import subprocess
    import sys
    import platform
    def git(*args):
        try:
            return subprocess.run(["git", *args], cwd=HERE, capture_output=True, text=True,
                                  timeout=10).stdout.strip()
        except Exception:
            return ""
    sha = git("rev-parse", "HEAD") or "unknown"
    dirty = bool(git("status", "--porcelain", "--untracked-files=no"))
    cohort_hash = hex(int(pd.util.hash_pandas_object(cohort, index=False).sum()) & (2**64 - 1))
    import sklearn
    import torch
    return dict(commit=sha[:12] + (" (uncommitted changes)" if dirty else ""),
                version=_project_version(), cohort_hash=cohort_hash,
                sim_config_hash=npe.core.config_hash(), seed=seed,
                weights={k: _weights_hash(v) for k, v in nets.items()},
                python=platform.python_version(), numpy=np.__version__,
                sklearn=sklearn.__version__, torch=torch.__version__)


def _project_version():
    try:
        import re
        txt = open(os.path.join(HERE, "..", "pyproject.toml"), encoding="utf8").read()
        return re.search(r'^version\s*=\s*"([^"]+)"', txt, re.M).group(1)
    except Exception:
        return "unknown"


REASON_LABELS = {"included": "Included",
                 "index_not_confirmed_in_window": "Excluded: index not confirmable in the window",
                 "too_few_early": "Excluded: too few visits in the window",
                 "egfr0_out_of_domain": "Excluded: first eGFR outside 20-110",
                 "no_future": "Excluded: no held-out visit in a horizon"}


def selection_table(cohort, reasons, patients_csv=None):
    """Included patients and each exclusion group at the index (aggregate; audit
    P1.5), with the fraction with a recorded death when patients.csv.gz gives
    `dod`: patients who die early leave no held-out visit, so the flow shows how
    much informative censoring the selection carries."""
    first = cohort.sort_values("time_years").groupby("patient_id").first()
    first.index = first.index.astype(str)
    first["follow_up"] = cohort.groupby("patient_id").time_years.max().set_axis(first.index)
    first["reason"] = first.index.map(lambda i: reasons.get(i, "not_evaluated"))
    died = None
    if patients_csv and os.path.exists(patients_csv):
        head = pd.read_csv(patients_csv, nrows=0).columns
        if "dod" in head:
            pt = pd.read_csv(patients_csv, usecols=["subject_id", "dod"], dtype={"subject_id": str})
            died = first.index.map(dict(zip(pt.subject_id, pt.dod.notna())))
    if died is not None:
        first["died"] = np.asarray(died, dtype=float)
    rows = []
    for key, name in REASON_LABELS.items():
        d = first[first.reason == key]
        if not len(d):
            continue
        r = dict(group=name, n=len(d), follow_up=float(d.follow_up.median()),
                 egfr=float(d.egfr.median()))
        if "age_at_index" in d:
            r["age"] = float(d.age_at_index.mean())
        if "sex" in d:
            r["female"] = float((d.sex == "F").mean())
        if "died" in d:
            r["died"] = float(np.nanmean(d.died))
        rows.append(r)
    return rows


def _ci(x, scale=1.0, fmt="{:+.2f}"):
    if x is None:
        return "—"
    lo, hi = x
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return "—"
    return f"{fmt.format(scale * lo)} to {fmt.format(scale * hi)}"


def report(tables, ev, ci, meta):
    L = ["# Patient-level validation on MIMIC-IV\n"]
    L.append(f"Cohort: {meta['n_cohort']} patients in the loader output; eligible "
             f"{meta['n_eligible']} (excluded: {meta['excl']}). Split: {meta['split']}; "
             f"{meta['n_train']} training and {meta['n_test']} test patients. eGFR thinned "
             f"to one median value per {WINDOW_DAYS} days; observed window = first "
             f"{EARLY_YEARS:g} years after the confirmed index.\n")
    sel = meta.get("selection")
    if sel:
        L.append("## Cohort selection\n")
        L.append(f"Exclusions in order of application: {meta['excl']}. Covariates come "
                 f"from the observed window only; values missing there or imputed by the "
                 f"loader were replaced by training-patient medians "
                 f"({ {k: round(v, 2) for k, v in meta['cov_medians'].items()} }; "
                 f"values imputed: {meta['n_imputed']}).\n")
        has_d = any("died" in r for r in sel)
        L.append("| Group | Patients | Mean age at index | Female | Median first eGFR | "
                 "Median follow-up (y) |" + (" Recorded death |" if has_d else ""))
        L.append("|---|---|---|---|---|---|" + ("---|" if has_d else ""))
        for r in sel:
            age = f"{r['age']:.1f}" if "age" in r else "—"
            fem = f"{100 * r['female']:.0f} %" if "female" in r else "—"
            dd = (f" {100 * r['died']:.0f} % |" if "died" in r else " — |") if has_d else ""
            L.append(f"| {r['group']} | {r['n']} | {age} | {fem} | {r['egfr']:.1f} | "
                     f"{r['follow_up']:.1f} |{dd}")
        L.append("")
    nz = meta["noise"]
    src = ("fixed by the user (sensitivity run)" if nz.get("fixed")
           else f"learned from the training patients ({nz['n_pairs']} short-interval pairs)")
    if nz.get("df") is not None:
        edge = (" The degrees of freedom sit at the lower limit of the fit (2.5): the "
                "short-interval differences are at least as heavy-tailed as the heaviest "
                "Student-t allowed, as expected when 90-day medians pool different numbers "
                "of draws."
                if nz["df"] < NOISE_DF_MIN + 0.05 else "")
        L.append(f"Measurement-noise model {src}: Student-t, scale {nz['scale']:.2f} "
                 f"mL/min/1.73 m², {nz['df']:.1f} degrees of freedom (SD {nz['sd']:.2f}).{edge}\n")
    else:
        L.append(f"Measurement-noise model {src}: Gaussian, SD {nz['scale']:.2f} mL/min/1.73 m².\n")
    dv = meta.get("deviation")
    if dv:
        L.append(f"Between-visit deviation learned from the training patients' held-out visits "
                 f"({dv['n_visits']} visits): variance {dv['v0']:.1f} + {dv['v1']:.1f} × "
                 f"(years since the last observation) (mL/min/1.73 m²)², i.e. SD "
                 f"{np.sqrt(dv['v0'] + dv['v1']):.1f} at 1 year and "
                 f"{np.sqrt(dv['v0'] + 3 * dv['v1']):.1f} at 3 years; log-likelihood gain "
                 f"{dv['loglik_gain_per_visit']:.2f} per visit over measurement noise alone.\n")
    for h in list(HORIZONS) + ["all"]:
        title = "All horizons pooled" if h == "all" else f"Horizon {h} after the last observed visit"
        n = next(iter(tables[h].values()))["n"]
        L.append(f"## {title} (n = {n} patient-visits)\n")
        L.append("| Method | RMSE | MAE | Bias | 90 % coverage | CRPS |")
        L.append("|---|---|---|---|---|---|")
        for m, r in tables[h].items():
            cov = "—" if np.isnan(r["cov90"]) else f"{100 * r['cov90']:.0f} %"
            crps = "—" if np.isnan(r["crps"]) else f"{r['crps']:.2f}"
            name = f"**{LABELS[m]}**" if m in HIGHLIGHT else LABELS[m]
            L.append(f"| {name} | {r['rmse']:.2f} | {r['mae']:.2f} | {r['bias']:+.2f} | {cov} | {crps} |")
        L.append("")
    for ref, rows in ci.items():
        L.append(f"## Differences versus {LABELS[ref]} (patient bootstrap, 95 % CI)\n")
        L.append("Method minus reference. Positive RMSE, MAE or CRPS = worse than the "
                 "reference; coverage in percentage points (nominal 90 %); positive AUC = better.\n")
        L.append("| Method | ΔRMSE | ΔMAE | ΔCRPS | Δcoverage 90 % (pp) | ΔAUC, eGFR < 30 |")
        L.append("|---|---|---|---|---|---|")
        for m, d in rows.items():
            L.append(f"| {LABELS[m]} | {_ci(d.get('rmse'))} | {_ci(d.get('mae'))} | "
                     f"{_ci(d.get('crps'))} | {_ci(d.get('cov90'), 100, '{:+.0f}')} | "
                     f"{_ci(d.get('auc'), 1, '{:+.3f}')} |")
        L.append("")
    L.append("## Event: eGFR < 30 at a held-out visit (patients ≥ 30 at the end of the window)\n")
    L.append("| Method | Visits | Events | AUC | Brier |")
    L.append("|---|---|---|---|---|")
    for m, r in ev.items():
        L.append(f"| {LABELS[m]} | {r['n']} | {r['events']} | {r['auc']:.3f} | {r['brier']:.3f} |")
    idf = meta.get("identifiability")
    if idf:
        L.append("\n## Identifiability of the patient parameters (test patients)\n")
        L.append("Posterior contraction = 1 − posterior variance / prior variance (median over "
                 "patients; 0 = the data add nothing, 1 = fully determined). Stage 1 is relative "
                 "to the uniform box, stage 2 to the learned population prior.\n")
        L.append("| Posterior | Contraction of q | Contraction of log rate | Median SD q | "
                 "Median SD log rate | Median corr(q, log rate) | Patients with abs corr > 0.5 |")
        L.append("|---|---|---|---|---|---|---|")
        for st, name in (("stage1", "Uniform prior (stage 1)"), ("stage2", "Learned prior (stage 2)")):
            r = idf[st]
            L.append(f"| {name} | {r['contr_q']:.2f} | {r['contr_s']:.2f} | {r['sd_q']:.2f} | "
                     f"{r['sd_s']:.2f} | {r['corr']:+.2f} | {100 * r['frac_strong_corr']:.0f} % |")
    pr = meta.get("prior")
    if pr is not None:
        L.append("\n## Population prior learned from the training patients (empirical Bayes)\n")
        L.append(f"K = {pr['K']} component(s), chosen by held-out marginal likelihood "
                 f"({pr['heldout']}). Prior box: q {npe.PRIOR_LO[0]:g}–{npe.PRIOR_HI[0]:g}, "
                 f"injury rate {np.exp(npe.PRIOR_LO[1]):.2g}–{np.exp(npe.PRIOR_HI[1]):.2g}× "
                 f"the population rate. Importance-sampling efficiency of the reweighted "
                 f"stage-1 draws on the test patients: {100 * meta['ess']:.0f} % (the "
                 f"retrained stage-2 network needs no weights).\n")
        stab = pr.get("stability")
        if stab:
            L.append(f"Components are ordered by injury rate. Brackets: 95 % interval over "
                     f"{pr['n_boot']} patient-bootstrap refits.\n")
        L.append("| Component | Weight | Mean q | SD q | Median injury rate | SD log rate |")
        L.append("|---|---|---|---|---|---|")
        for k, c in enumerate(pr["components"]):
            if stab:
                w = stab["weight"][k]; qm = stab["q_mean"][k]; rm = stab["rate_median"][k]
                L.append(f"| {k + 1} | {c['weight']:.2f} [{w[0]:.2f}–{w[2]:.2f}] | "
                         f"{c['q_mean']:.2f} [{qm[0]:.2f}–{qm[2]:.2f}] | {c['q_sd']:.2f} | "
                         f"{c['rate_median']:.2f}× [{rm[0]:.2f}–{rm[2]:.2f}] | {c['log_rate_sd']:.2f} |")
            else:
                L.append(f"| {k + 1} | {c['weight']:.2f} | {c['q_mean']:.2f} | {c['q_sd']:.2f} | "
                         f"{c['rate_median']:.2f}× | {c['log_rate_sd']:.2f} |")
        lo_edge = [c for c in pr["components"]
                   if np.log(c["rate_median"]) - npe.PRIOR_LO[1] < 2 * c["log_rate_sd"]]
        hi_edge = [c for c in pr["components"]
                   if npe.PRIOR_HI[1] - np.log(c["rate_median"]) < 2 * c["log_rate_sd"]]
        if lo_edge or hi_edge:
            L.append("\nWarning: a component lies within two SDs of an edge of the injury-rate "
                     "box; the box may be truncating that part of the population.")
    pv = meta.get("provenance")
    if pv:
        L.append("\n## Provenance\n")
        L.append(f"NephroQ {pv['version']}, commit {pv['commit']}; model configuration hash "
                 f"{pv['sim_config_hash']}; cohort content hash {pv['cohort_hash']}; seed "
                 f"{pv['seed']}; network weights {pv['weights']}; Python {pv['python']}, "
                 f"numpy {pv['numpy']}, scikit-learn {pv['sklearn']}, torch {pv['torch']}.")
    return "\n".join(L) + "\n"


def plot(df, tables, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 7, "axes.spines.top": False, "axes.spines.right": False,
                         "savefig.dpi": 300})
    methods = [m for m in METHODS if m in set(df.method)]
    col = {"population": "#999999", "last": "#bdbdbd", "ls_rate": "#56B4E9", "mlp": "#D55E00",
           "npe": "#9ecae1", "npe_eb": "#6baed6", "npe_pop": "#4292c6", "npe_dev": "#0072B2",
           "hybrid": "#000000", "ols": "#E69F00", "theil_sen": "#F0E442", "lmm_eb": "#CC79A7",
           "gbm": "#009E73"}
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.6))
    hs = list(HORIZONS)
    w = 0.85 / len(methods)
    for k, m in enumerate(methods):
        ax[0].bar(np.arange(len(hs)) + (k - len(methods) / 2 + 0.5) * w,
                  [tables[h][m]["rmse"] for h in hs], w, color=col[m], label=LABELS[m])
    ax[0].set_xticks(range(len(hs)), hs)
    ax[0].set_xlabel("horizon after the last observed visit")
    ax[0].set_ylabel("RMSE (mL/min/1.73 m²)")
    ax[0].set_title("a  forecast error on held-out visits", loc="left", fontsize=7)
    ax[0].legend(frameon=False, fontsize=4.5)
    for m in methods:
        d = df[df.method == m]
        if "hit_0.9" in d and d["hit_0.9"].notna().all():
            ax[1].plot(LEVELS, [d[f"hit_{lv:g}"].mean() for lv in LEVELS], "o-", ms=2.5, lw=1,
                       color=col[m], label=LABELS[m])
    ax[1].plot([0, 1], [0, 1], "k--", lw=0.6)
    ax[1].set_xlabel("nominal coverage")
    ax[1].set_ylabel("empirical coverage")
    ax[1].set_title("b  interval calibration", loc="left", fontsize=7)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def run(cohort, patients_csv=None, noise_model="t", noise_sd=8.7, mlp=None, seed=0,
        n_boot=1000, verbose=True, npe_kwargs=None, noise_fixed=None, prior_boot=30,
        train_draws=200):
    """
    Full pipeline. `noise_model` = "t" learns a Student-t noise model from the
    training patients; "gaussian" uses `noise_sd`. `noise_fixed` = (scale, df)
    skips the noise fit (sensitivity runs; df None = Gaussian). `npe_kwargs` is
    passed to the network training (e.g. dict(n_train=3000, epochs=1, save=False)
    in tests). `prior_boot` = bootstrap refits for the mixture-stability check.
    """
    npe_kwargs = dict(npe_kwargs or {})
    reasons = {}
    patients, excl = build_patients(cohort, reasons=reasons)
    if len(patients) < 20:
        raise ValueError(f"only {len(patients)} eligible patients; check the cohort")
    is_test, split = split_patients(patients, patients_csv, seed=seed)
    train = [p for p, z in zip(patients, is_test) if not z]
    test = [p for p, z in zip(patients, is_test) if z]
    cov_medians, n_imputed = impute_covariates(train, test)
    if verbose:
        print(f"{len(patients)} eligible patients; split {split}: "
              f"{len(train)} train / {len(test)} test; imputed (train medians) {n_imputed}")

    if noise_fixed is not None:
        scale, df_ = float(noise_fixed[0]), (None if noise_fixed[1] is None else float(noise_fixed[1]))
        n_pairs = 0
    elif noise_model == "t":
        scale, df_, n_pairs = estimate_noise(train)
        scale, df_ = round(scale, 2), round(df_, 1)      # stable cache keys
    else:
        scale, df_, n_pairs = float(noise_sd), None, 0
    noise = (scale, df_)
    if verbose:
        print(f"Noise model: scale {scale}, df {df_} ({n_pairs} pairs)")

    stage1 = npe.load_or_train(noise_sd=scale, noise_df=df_, verbose=verbose, **npe_kwargs)
    prior = fit_population_prior(train, stage1, seed=seed, n_boot=prior_boot, verbose=verbose)
    stage2 = npe.load_or_train(noise_sd=scale, noise_df=df_, prior=prior, verbose=verbose,
                               **npe_kwargs)
    nets = dict(stage1=stage1, stage2=stage2)
    if mlp is None:
        mlp = pz.get_estimator()
    lmm = fit_lmm_eb(train)
    persist = fit_persistence(train)
    gbm = fit_gbm(train, seed=seed)
    if verbose:
        print("Stage-2 forecasts at the training patients' held-out visits...")
    fc = train_forecasts(train, stage2, n_draws=train_draws)
    dev = fit_deviation(fc, noise, seed=seed)
    if verbose:
        print(f"Between-visit deviation: v0 {dev['v0']:.1f}, v1 {dev['v1']:.1f} per year")
        print("Fitting the hybrid residual model on the training patients...")
    hybrid = fit_hybrid(train, fc, seed=seed)
    prior_sd = dict(uniform=prior_marginal_sd(None), learned=prior_marginal_sd(prior))

    df, diag = score_rows(test, nets, mlp, lmm, gbm, hybrid, prior, noise, dev, persist,
                          prior_sd, verbose=verbose)
    tables, ev, ci = summarize(df, n_boot=n_boot, seed=seed)
    meta = dict(n_cohort=int(cohort.patient_id.nunique()), n_eligible=len(patients),
                excl=excl, split=split, n_train=len(train), n_test=len(test),
                cov_medians=cov_medians, n_imputed=n_imputed,
                noise=dict(scale=scale, df=df_, sd=_noise_sd_equivalent(scale, df_),
                           n_pairs=n_pairs, fixed=noise_fixed is not None),
                deviation=dev,
                identifiability=identifiability_summary(diag),
                prior=dict(K=prior.extra["K"], components=prior.summary(),
                           heldout={k: round(v, 3) for k, v in
                                    prior.extra["heldout_loglik_per_patient"].items()},
                           stability=prior.extra.get("stability"),
                           n_boot=prior.extra.get("n_boot")),
                ess=float(df.loc[df.method == "npe_eb", "ess"].mean()),
                selection=selection_table(cohort, reasons, patients_csv),
                provenance=provenance(cohort, nets, seed))
    return df, tables, ev, ci, meta


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Patient-level validation on MIMIC-IV")
    ap.add_argument("--cohort", default=os.path.join(HERE, "..", "data", "_mimic_cohort.tsv"))
    ap.add_argument("--patients", default=None, help="hosp/patients.csv.gz (temporal split)")
    ap.add_argument("--noise-model", choices=["t", "gaussian"], default="t")
    ap.add_argument("--noise-sd", type=float, default=8.7, help="only with --noise-model gaussian")
    ap.add_argument("--noise-scale", type=float, default=None,
                    help="fix the noise scale instead of fitting it (sensitivity runs)")
    ap.add_argument("--noise-df", type=float, default=None,
                    help="with --noise-scale: Student-t degrees of freedom (omit for Gaussian)")
    ap.add_argument("--prior-boot", type=int, default=30,
                    help="bootstrap refits for the mixture-stability check (0 = skip)")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--tag", default="",
                    help="suffix for the output files (e.g. _noise_v016 for a sensitivity run)")
    ap.add_argument("--out-dir", default=None,
                    help="output directory (default: $NEPHROQ_OUT_DIR, else results/)")
    ap.add_argument("--quick", action="store_true",
                    help="small networks and bootstrap, for CI smoke runs on synthetic data")
    ap.add_argument("--threads", type=int, default=None,
                    help="cores to use (default: $NEPHROQ_THREADS, else all available)")
    a = ap.parse_args()
    print(f"Using at most {outputs.limit_threads(a.threads)} cores")
    RESULTS = outputs.results_dir(a.out_dir)
    print(f"Writing outputs to {RESULTS}")
    cohort = pd.read_csv(a.cohort, sep=None, engine="python", dtype={"patient_id": str})
    kw = dict(npe_kwargs=dict(n_train=6000, epochs=2, save=False), n_boot=100, prior_boot=3) \
        if a.quick else dict(n_boot=a.n_boot, prior_boot=a.prior_boot)
    fixed = (a.noise_scale, a.noise_df) if a.noise_scale is not None else None
    df, tables, ev, ci, meta = run(cohort, a.patients, a.noise_model, a.noise_sd,
                                   noise_fixed=fixed, **kw)
    base = f"mimic_patient_validation{a.tag}"
    df.to_csv(os.path.join(RESULTS, base + "_rows.csv"), index=False)  # never share
    text = report(tables, ev, ci, meta)
    with open(os.path.join(RESULTS, base + ".md"), "w", encoding="utf8") as f:
        f.write(text)
    plot(df, tables, os.path.join(RESULTS, base + ".png"))
    print(text)
