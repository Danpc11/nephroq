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
        LS rate only    one-parameter least squares on the injury rate
        MLP ensemble    the app's current personalizer
        NPE, uniform    stage-1 posterior predictive
        NPE, reweighted stage-1 draws importance-reweighted to the learned prior
        NPE, pop. prior stage-2 posterior predictive   <- mechanistic reference
        Hybrid          stage-2 forecast + learned residual correction
        OLS slope       per-patient straight line through the early window
        LMM-EB slope    empirical-Bayes random-slope model
        GBM             gradient boosting from the early window to the future
     Predictive distributions of the NPE methods include the fitted measurement
     noise, because the targets are observations.
  7. Scores: RMSE, MAE, bias, 90 % interval coverage, CRPS by horizon; AUC and
     Brier for eGFR < 30 at a held-out visit (patients >= 30 at the end of the
     window); patient-bootstrap 95 % CIs of RMSE differences against the
     mechanistic reference and against GBM.

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
METHODS = ["population", "ls_rate", "mlp", "npe", "npe_eb", "npe_pop", "hybrid",
           "ols", "lmm_eb", "gbm"]
REFERENCES = ["npe_pop", "gbm"]      # bootstrap differences are reported against these
LABELS = {"population": "Population parameters", "ls_rate": "Least squares, rate only",
          "mlp": "MLP ensemble (current app)", "npe": "NPE, uniform prior",
          "npe_eb": "NPE, reweighted to the learned prior",
          "npe_pop": "NPE, retrained on the learned prior",
          "hybrid": "Hybrid: NPE + learned residual correction",
          "ols": "Per-patient OLS slope", "lmm_eb": "Mixed-model slope (empirical Bayes)",
          "gbm": "Gradient boosting (trained on MIMIC)"}


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


def build_patients(cohort, early_years=EARLY_YEARS, window_days=WINDOW_DAYS):
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
    eGFR inside the model's validated range (20-110).
    """
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
            continue
        te, ee = thin(te_raw, ee_raw, window_days)
        if len(te) < pz.MIN_VISITS or te[-1] - te[0] < pz.MIN_SPAN_YEARS:
            excl["too_few_early"] += 1
            continue
        if not (npe.PRIOR_LO[2] <= ee[0] <= npe.PRIOR_HI[2]):
            excl["egfr0_out_of_domain"] += 1
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
            continue
        cov = {k: _baseline_covariate(g_early, *cols) for k, cols in COVARIATES.items()}
        out.append(dict(pid=str(pid), t=te, e=ee, future=picks, **cov))
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
        sigma, nu = np.exp(x[0]), 2.05 + np.exp(x[1])
        sim = sigma * (stats.t.ppf(u1, nu) - stats.t.ppf(u2, nu))
        return float(np.sum((w * (np.quantile(sim, levels) - emp)) ** 2))

    # start from the moment-based guess under a Gaussian
    x0 = [np.log(max(np.std(d) / np.sqrt(2.0), 1e-3)), np.log(5.0)]
    res = optimize.minimize(loss, x0, method="Nelder-Mead",
                            options=dict(xatol=1e-3, fatol=1e-6, maxiter=400))
    scale, df = float(np.exp(res.x[0])), float(2.05 + np.exp(res.x[1]))
    return scale, float(np.clip(df, 2.5, 30.0)), int(len(d))


def _noise_sd_equivalent(scale, df):
    """SD of the fitted noise (Student-t has SD = scale * sqrt(df / (df - 2)))."""
    return scale if df is None else scale * np.sqrt(df / (df - 2.0))


def fit_population_prior(train, net, n_draws=300, seed=0, verbose=True):
    """Empirical-Bayes prior from the training patients' stage-1 NPE draws. K (1 or
    2 components) is chosen by the held-out marginal likelihood on 20 % of the
    training patients, then refitted on all of them."""
    draws = [npe.posterior_samples(net, p["t"], p["e"], p["a1c"], p["uacr0"], p["sbp"],
                                   n=n_draws, seed=10_000 + i) for i, p in enumerate(train)]
    rng = np.random.default_rng(seed)
    hold = rng.uniform(size=len(draws)) < 0.2
    fit_d = [d for d, h in zip(draws, hold) if not h]
    val_d = [d for d, h in zip(draws, hold) if h]
    lu = popprior._uniform_logpdf()
    best_k, best_ll, val_ll = 1, -np.inf, {}
    for K in (1, 2):
        pr = popprior.fit(fit_d, K=K, seed=seed)
        lr = np.stack([pr.logpdf(d[:, 0], d[:, 1]) - lu for d in val_d])
        ll = float(np.mean(special.logsumexp(lr, axis=1) - np.log(lr.shape[1])))
        val_ll[K] = ll
        if ll > best_ll:
            best_k, best_ll = K, ll
    prior = popprior.fit(draws, K=best_k, seed=seed)
    prior.extra.update(K=best_k, heldout_loglik_per_patient=val_ll)
    if verbose:
        print(f"Population prior: K={best_k} (held-out log-lik/patient {val_ll})")
    return prior


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
    w = 1.0 / (se_pool + np.var(bs))
    mu = float(np.sum(w * bs) / np.sum(w))
    tau2 = float(max(np.var(bs) - np.mean(se_pool), 1e-3))
    return dict(mu=mu, tau2=tau2, sigma2=sigma2)


def _window_features(p, t_f):
    f = pz.features(p["t"], p["e"], p["a1c"], p["uacr0"], p["sbp"])
    return np.concatenate([f, [t_f - p["t"][-1], t_f]])


def _fit_boosting(X, y, seed=0):
    from sklearn.ensemble import HistGradientBoostingRegressor
    models = {}
    for name, kw in [("mean", dict(loss="squared_error")),
                     ("q05", dict(loss="quantile", quantile=0.05)),
                     ("q95", dict(loss="quantile", quantile=0.95))]:
        models[name] = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05,
                                                     random_state=seed, **kw).fit(X, y)
    return models


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


def _hybrid_features(p, t_f, mu, sd):
    return np.concatenate([_window_features(p, t_f), [mu, sd]])


def fit_hybrid(train, net, n_draws=200, seed=0):
    """Gradient boosting on the residuals (observed - mechanistic posterior mean)
    of the stage-2 forecast, with the forecast and its spread as extra inputs."""
    X, r = [], []
    for i, p in enumerate(train):
        keys = sorted(p["future"])
        tf = np.array([p["future"][k][0] for k in keys])
        _, lat = _mechanistic_forecast(net, p, tf, n_draws, 20_000 + i)
        mu, sd = lat.mean(0), lat.std(0)
        for j, k in enumerate(keys):
            X.append(_hybrid_features(p, tf[j], mu[j], sd[j]))
            r.append(p["future"][k][1] - mu[j])
    return _fit_boosting(np.array(X), np.array(r), seed)


# ==============================================================================
# Predictions for one test patient
# ==============================================================================
def predict_patient(p, nets, mlp, lmm, gbm, hybrid, prior, noise, n_draws=1000, seed=0):
    """Returns (keys, tf, {method: dict(mu, sd|None, draws|None)})."""
    keys = sorted(p["future"])
    tf = np.array([p["future"][k][0] for k in keys])
    cov = (p["a1c"], p["uacr0"], p["sbp"])
    t, e = p["t"], p["e"]
    q_pop = npe.Q_POP
    scale, df = noise
    sd_eq = _noise_sd_equivalent(scale, df)
    rng = np.random.default_rng(seed)
    out = {}

    def sim(q, s, e0):
        return fast_sim.simulate_egfr(q, s, e0, *cov, tf)

    def noisy(lat):
        return lat + npe.observation_noise(lat.shape, rng, scale, df)

    out["population"] = dict(mu=sim(q_pop, 1.0, e[0])[0], sd=None, draws=None)
    s1 = pz.least_squares_scale_only(t, e, *cov, q=q_pop)
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

    _, lat2 = _mechanistic_forecast(nets["stage2"], p, tf, n_draws, seed)
    mu2, sd2 = lat2.mean(0), lat2.std(0)
    out["npe_pop"] = dict(mu=mu2, sd=None, draws=noisy(lat2))

    Xh = np.array([_hybrid_features(p, tf[j], mu2[j], sd2[j]) for j in range(len(tf))])
    h05, h95 = hybrid["q05"].predict(Xh), hybrid["q95"].predict(Xh)
    out["hybrid"] = dict(mu=mu2 + hybrid["mean"].predict(Xh),
                         sd=np.maximum((h95 - h05) / (2 * 1.645), 1e-3), draws=None)

    a, b, s2, tb, sxx, n = _ols(t, e)
    out["ols"] = dict(mu=a + b * tf, sd=np.sqrt(s2 * (1 + 1 / n + (tf - tb) ** 2 / sxx)), draws=None)

    se2 = lmm["sigma2"] / sxx
    wl = lmm["tau2"] / (lmm["tau2"] + se2)
    b_eb = wl * b + (1 - wl) * lmm["mu"]
    var_eb = lmm["sigma2"] / n + (tf - tb) ** 2 * wl * se2 + lmm["sigma2"]
    out["lmm_eb"] = dict(mu=e.mean() + b_eb * (tf - tb), sd=np.sqrt(var_eb), draws=None)

    X = np.array([_window_features(p, x) for x in tf])
    q05, q95 = gbm["q05"].predict(X), gbm["q95"].predict(X)
    out["gbm"] = dict(mu=gbm["mean"].predict(X),
                      sd=np.maximum((q95 - q05) / (2 * 1.645), 1e-3), draws=None)
    return keys, tf, out


# ==============================================================================
# Scoring
# ==============================================================================
def _crps_normal(mu, sd, y):
    z = (y - mu) / sd
    return sd * (z * (2 * stats.norm.cdf(z) - 1) + 2 * stats.norm.pdf(z) - 1 / np.sqrt(np.pi))


def _crps_draws(x, y):
    x = np.sort(x); n = len(x)
    return np.mean(np.abs(x - y)) - np.sum(x * (2 * np.arange(1, n + 1) - n - 1)) / (n * n)


def score_rows(test, nets, mlp, lmm, gbm, hybrid, prior, noise, verbose=True):
    """One row per (patient, horizon, method)."""
    rows = []
    for i, p in enumerate(test):
        keys, tf, pred = predict_patient(p, nets, mlp, lmm, gbm, hybrid, prior, noise, seed=i)
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
    return pd.DataFrame(rows)


def _auc(y, p):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, p)) if 0 < y.sum() < len(y) else np.nan


def summarize(df, n_boot=1000, seed=0):
    """Aggregate tables (no patient-level content)."""
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

    # patient bootstrap of pooled-RMSE differences: resample PATIENTS (their
    # visits travel together) and recompute every method's pooled RMSE
    rng = np.random.default_rng(seed)
    pids = np.sort(df.pid.unique())
    idx = {pid: k for k, pid in enumerate(pids)}
    sse, cnt = {}, None
    for m in methods:
        d = df[df.method == m]
        k = d.pid.map(idx).to_numpy()
        sse[m] = np.bincount(k, weights=(d.mu - d.y).to_numpy() ** 2, minlength=len(pids))
        if cnt is None:
            cnt = np.bincount(k, minlength=len(pids)).astype(float)
    refs = [r for r in REFERENCES if r in methods]
    boot = {r: {m: [] for m in methods if m != r} for r in refs}
    for _ in range(n_boot):
        w = np.bincount(rng.integers(0, len(pids), len(pids)), minlength=len(pids))
        n_tot = np.sum(w * cnt)
        rm = {m: np.sqrt(np.sum(w * sse[m]) / n_tot) for m in methods}
        for r in refs:
            for m in boot[r]:
                boot[r][m].append(rm[m] - rm[r])
    ci = {r: {m: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
              for m, v in boot[r].items()} for r in refs}
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


def selection_table(cohort, patients):
    """Included vs excluded patients at the index (aggregate; audit P1.5)."""
    inc = {p["pid"] for p in patients}
    first = cohort.sort_values("time_years").groupby("patient_id").first()
    span = cohort.groupby("patient_id").time_years.max()
    first["included"] = first.index.astype(str).isin(inc)
    first["follow_up"] = span
    rows = []
    for flag, name in [(True, "Included"), (False, "Excluded")]:
        d = first[first.included == flag]
        r = dict(group=name, n=len(d),
                 follow_up=float(d.follow_up.median()) if len(d) else np.nan,
                 egfr=float(d.egfr.median()) if len(d) else np.nan)
        if "age_at_index" in d:
            r["age"] = float(d.age_at_index.mean())
        if "sex" in d:
            r["female"] = float((d.sex == "F").mean())
        rows.append(r)
    return rows


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
        L.append("| Group | Patients | Mean age at index | Female | Median first eGFR | Median follow-up (y) |")
        L.append("|---|---|---|---|---|---|")
        for r in sel:
            age = f"{r['age']:.1f}" if "age" in r else "—"
            fem = f"{100 * r['female']:.0f} %" if "female" in r else "—"
            L.append(f"| {r['group']} | {r['n']} | {age} | {fem} | {r['egfr']:.1f} | {r['follow_up']:.1f} |")
        L.append("")
    nz = meta["noise"]
    if nz.get("df") is not None:
        L.append(f"Measurement-noise model learned from the training patients "
                 f"({nz['n_pairs']} short-interval pairs): Student-t, scale "
                 f"{nz['scale']:.2f} mL/min/1.73 m², {nz['df']:.1f} degrees of freedom "
                 f"(SD {nz['sd']:.2f}).\n")
    else:
        L.append(f"Measurement-noise model: Gaussian, SD {nz['scale']:.2f} mL/min/1.73 m².\n")
    for h in list(HORIZONS) + ["all"]:
        title = "All horizons pooled" if h == "all" else f"Horizon {h} after the last observed visit"
        n = next(iter(tables[h].values()))["n"]
        L.append(f"## {title} (n = {n} patient-visits)\n")
        L.append("| Method | RMSE | MAE | Bias | 90 % coverage | CRPS |")
        L.append("|---|---|---|---|---|---|")
        for m, r in tables[h].items():
            cov = "—" if np.isnan(r["cov90"]) else f"{100 * r['cov90']:.0f} %"
            crps = "—" if np.isnan(r["crps"]) else f"{r['crps']:.2f}"
            name = f"**{LABELS[m]}**" if m in ("npe_pop", "hybrid") else LABELS[m]
            L.append(f"| {name} | {r['rmse']:.2f} | {r['mae']:.2f} | {r['bias']:+.2f} | {cov} | {crps} |")
        L.append("")
    for ref, rows in ci.items():
        L.append(f"## RMSE difference versus {LABELS[ref]} (patient bootstrap, 95 % CI)\n")
        L.append("Positive = the method is worse than the reference.\n")
        L.append("| Method | ΔRMSE 95 % CI |")
        L.append("|---|---|")
        for m, (lo, hi) in rows.items():
            L.append(f"| {LABELS[m]} | {lo:+.2f} to {hi:+.2f} |")
        L.append("")
    L.append("## Event: eGFR < 30 at a held-out visit (patients ≥ 30 at the end of the window)\n")
    L.append("| Method | Visits | Events | AUC | Brier |")
    L.append("|---|---|---|---|---|")
    for m, r in ev.items():
        L.append(f"| {LABELS[m]} | {r['n']} | {r['events']} | {r['auc']:.3f} | {r['brier']:.3f} |")
    pr = meta.get("prior")
    if pr is not None:
        L.append("\n## Population prior learned from the training patients (empirical Bayes)\n")
        L.append(f"K = {pr['K']} component(s), chosen by held-out marginal likelihood "
                 f"({pr['heldout']}). Importance-sampling efficiency of the reweighted "
                 f"stage-1 draws on the test patients: {100 * meta['ess']:.0f} % (the "
                 f"retrained stage-2 network needs no weights).\n")
        L.append("| Component | Weight | Mean q | SD q | Median injury rate | SD log rate |")
        L.append("|---|---|---|---|---|---|")
        for k, c in enumerate(pr["components"]):
            L.append(f"| {k + 1} | {c['weight']:.2f} | {c['q_mean']:.2f} | {c['q_sd']:.2f} | "
                     f"{c['rate_median']:.2f}× | {c['log_rate_sd']:.2f} |")
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
    col = {"population": "#999999", "ls_rate": "#56B4E9", "mlp": "#D55E00", "npe": "#9ecae1",
           "npe_eb": "#4292c6", "npe_pop": "#0072B2", "hybrid": "#000000",
           "ols": "#E69F00", "lmm_eb": "#CC79A7", "gbm": "#009E73"}
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
        n_boot=1000, verbose=True, npe_kwargs=None):
    """
    Full pipeline. `noise_model` = "t" learns a Student-t noise model from the
    training patients; "gaussian" uses `noise_sd`. `npe_kwargs` is passed to the
    network training (e.g. dict(n_train=3000, epochs=1, save=False) in tests).
    """
    npe_kwargs = dict(npe_kwargs or {})
    patients, excl = build_patients(cohort)
    if len(patients) < 20:
        raise ValueError(f"only {len(patients)} eligible patients; check the cohort")
    is_test, split = split_patients(patients, patients_csv, seed=seed)
    train = [p for p, z in zip(patients, is_test) if not z]
    test = [p for p, z in zip(patients, is_test) if z]
    cov_medians, n_imputed = impute_covariates(train, test)
    if verbose:
        print(f"{len(patients)} eligible patients; split {split}: "
              f"{len(train)} train / {len(test)} test; imputed (train medians) {n_imputed}")

    if noise_model == "t":
        scale, df_, n_pairs = estimate_noise(train)
        scale, df_ = round(scale, 2), round(df_, 1)      # stable cache keys
    else:
        scale, df_, n_pairs = float(noise_sd), None, 0
    noise = (scale, df_)
    if verbose:
        print(f"Noise model: scale {scale}, df {df_} ({n_pairs} pairs)")

    stage1 = npe.load_or_train(noise_sd=scale, noise_df=df_, verbose=verbose, **npe_kwargs)
    prior = fit_population_prior(train, stage1, seed=seed, verbose=verbose)
    stage2 = npe.load_or_train(noise_sd=scale, noise_df=df_, prior=prior, verbose=verbose,
                               **npe_kwargs)
    nets = dict(stage1=stage1, stage2=stage2)
    if mlp is None:
        mlp = pz.get_estimator()
    lmm = fit_lmm_eb(train)
    gbm = fit_gbm(train, seed=seed)
    if verbose:
        print("Fitting the hybrid residual model on the training patients...")
    hybrid = fit_hybrid(train, stage2, seed=seed)

    df = score_rows(test, nets, mlp, lmm, gbm, hybrid, prior, noise, verbose=verbose)
    tables, ev, ci = summarize(df, n_boot=n_boot, seed=seed)
    meta = dict(n_cohort=int(cohort.patient_id.nunique()), n_eligible=len(patients),
                excl=excl, split=split, n_train=len(train), n_test=len(test),
                cov_medians=cov_medians, n_imputed=n_imputed,
                noise=dict(scale=scale, df=df_, sd=_noise_sd_equivalent(scale, df_),
                           n_pairs=n_pairs),
                prior=dict(K=prior.extra["K"], components=prior.summary(),
                           heldout={k: round(v, 3) for k, v in
                                    prior.extra["heldout_loglik_per_patient"].items()}),
                ess=float(df.loc[df.method == "npe_eb", "ess"].mean()),
                selection=selection_table(cohort, patients),
                provenance=provenance(cohort, nets, seed))
    return df, tables, ev, ci, meta


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Patient-level validation on MIMIC-IV")
    ap.add_argument("--cohort", default=os.path.join(HERE, "..", "data", "_mimic_cohort.tsv"))
    ap.add_argument("--patients", default=None, help="hosp/patients.csv.gz (temporal split)")
    ap.add_argument("--noise-model", choices=["t", "gaussian"], default="t")
    ap.add_argument("--noise-sd", type=float, default=8.7, help="only with --noise-model gaussian")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--out-dir", default=None,
                    help="output directory (default: $NEPHROQ_OUT_DIR, else results/)")
    ap.add_argument("--quick", action="store_true",
                    help="small networks and bootstrap, for CI smoke runs on synthetic data")
    a = ap.parse_args()
    RESULTS = outputs.results_dir(a.out_dir)
    print(f"Writing outputs to {RESULTS}")
    cohort = pd.read_csv(a.cohort, sep=None, engine="python", dtype={"patient_id": str})
    kw = dict(npe_kwargs=dict(n_train=6000, epochs=2, save=False), n_boot=100) if a.quick \
        else dict(n_boot=a.n_boot)
    df, tables, ev, ci, meta = run(cohort, a.patients, a.noise_model, a.noise_sd, **kw)
    df.to_csv(os.path.join(RESULTS, "mimic_patient_validation_rows.csv"), index=False)  # never share
    text = report(tables, ev, ci, meta)
    with open(os.path.join(RESULTS, "mimic_patient_validation.md"), "w", encoding="utf8") as f:
        f.write(text)
    plot(df, tables, os.path.join(RESULTS, "mimic_patient_validation.png"))
    print(text)
