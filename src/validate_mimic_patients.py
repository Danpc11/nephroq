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
  4. Methods:
        population      q at population value, injury rate 1 (no personalization)
        LS rate only    one-parameter least squares on the injury rate
        MLP ensemble    the app's current personalizer
        NPE             amortized neural posterior; posterior predictive
                        (+ measurement noise, since the targets are observations)
        NPE + EB prior  the same NPE draws, importance-reweighted to a population
                        prior over (q, log injury rate) learned from the TRAINING
                        patients by empirical Bayes (population_prior.py); the
                        network is not retrained
        OLS slope       per-patient straight line through the early window
        LMM-EB slope    empirical-Bayes shrinkage of the per-patient slope toward
                        the population slope (a random-slope mixed model fitted
                        by moments on the TRAINING patients)
        GBM             gradient boosting trained on the TRAINING patients to map
                        the early window + horizon to the future eGFR (quantile
                        loss for the 5/50/95 % bands)
     The mechanistic methods are not fitted on MIMIC at all, so the training
     split only feeds the two statistical baselines that need it.
  5. Split: by anchor_year_group (temporal: the most recent group is the test
     set) when patients.csv.gz is given, otherwise a random 30 % of patients.
  6. Scores: RMSE, MAE, bias, 90 % interval coverage, CRPS by horizon; for the
     event eGFR < 30 at a held-out visit (patients >= 30 at the end of the
     window), AUC and Brier score; patient-bootstrap 95 % CI of the RMSE
     difference between NPE and every other method.

DATA PROTECTION (PhysioNet Credentialed Health Data License 1.5.0): run this on
the machine that holds MIMIC. It writes per-patient rows to results/*.csv, which
is git-ignored and must stay there. Only the AGGREGATE report
(results/mimic_patient_validation.md / .png) may be shared or published.

Run:
    python mimic_loader.py --mimic-dir <mimic>/hosp --out ../data/_mimic_cohort.tsv
    python -c "import npe_posterior as n; n.train(noise_sd=8.7)"
    python validate_mimic_patients.py --cohort ../data/_mimic_cohort.tsv \\
        --patients <mimic>/hosp/patients.csv.gz --noise-sd 8.7
================================================================================
"""
import argparse
import os

import numpy as np
import pandas as pd
from scipy import stats

import fast_sim
import personalize as pz
import npe_posterior as npe
import population_prior as popprior

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "..", "results")

EARLY_YEARS = 2.0
WINDOW_DAYS = 90
HORIZONS = {"1y": (0.5, 2.0, 1.0), "3y": (2.0, 4.0, 3.0), "5y": (4.0, 6.5, 5.0)}
LEVELS = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95])
COV_RANGES = dict(a1c=(5.5, 12.0), uacr0=(5.0, 3000.0), sbp=(105.0, 185.0))
METHODS = ["population", "ls_rate", "mlp", "npe", "npe_eb", "ols", "lmm_eb", "gbm"]
REFERENCE = "npe_eb"          # bootstrap differences are reported against this
LABELS = {"population": "Population parameters", "ls_rate": "Least squares, rate only",
          "mlp": "MLP ensemble (current app)", "npe": "NPE, uniform prior",
          "npe_eb": "NPE + population prior (empirical Bayes)",
          "ols": "Per-patient OLS slope", "lmm_eb": "Mixed-model slope (empirical Bayes)",
          "gbm": "Gradient boosting (trained on MIMIC)"}


# ==============================================================================
# Cohort -> per-patient early window and held-out visits
# ==============================================================================
def _first_valid(g, cols):
    for c in cols:
        if c in g and g[c].notna().any():
            return float(g[c].dropna().iloc[0])
    return np.nan


def thin(t, e, window_days=WINDOW_DAYS):
    """One value per window: the median time and the median eGFR of the draws in it."""
    b = np.floor(np.asarray(t) * 365.25 / window_days).astype(int)
    df = pd.DataFrame(dict(b=b, t=t, e=e)).groupby("b").median()
    return df.t.to_numpy(), df.e.to_numpy()


def build_patients(cohort, early_years=EARLY_YEARS, window_days=WINDOW_DAYS):
    """
    Returns a list of dicts, one per eligible patient, and a dict of exclusion
    counts. Eligible: >= 3 thinned visits spanning >= 0.75 y inside the early
    window, >= 1 held-out visit inside a scoring horizon, and a first eGFR inside
    the model's validated range (20-110).
    """
    out, excl = [], dict(too_few_early=0, no_future=0, egfr0_out_of_domain=0)
    for pid, g in cohort.groupby("patient_id", sort=False):
        g = g.sort_values("time_years")
        t, e = thin(g.time_years.to_numpy(float), g.egfr.to_numpy(float), window_days)
        early = t <= early_years
        te, ee = t[early], e[early]
        if len(te) < pz.MIN_VISITS or te[-1] - te[0] < pz.MIN_SPAN_YEARS:
            excl["too_few_early"] += 1
            continue
        if not (npe.PRIOR_LO[2] <= ee[0] <= npe.PRIOR_HI[2]):
            excl["egfr0_out_of_domain"] += 1
            continue
        te, t_fut, e_fut = te - te[0], t[~early] - t[early][0], e[~early]
        picks = {}
        for h, (lo, hi, centre) in HORIZONS.items():
            dt = t_fut - te[-1]
            ok = (dt > lo) & (dt <= hi)
            if ok.any():
                j = np.flatnonzero(ok)[np.argmin(np.abs(dt[ok] - centre))]
                picks[h] = (float(t_fut[j]), float(e_fut[j]))
        if not picks:
            excl["no_future"] += 1
            continue
        cov = dict(
            a1c=_first_valid(g, ["hba1c_baseline_strict", "hba1c"]),
            uacr0=_first_valid(g, ["uacr_baseline_strict", "uacr"]),
            sbp=_first_valid(g, ["sbp_baseline_strict", "sbp"]))
        for k, (lo, hi) in COV_RANGES.items():
            cov[k] = float(np.clip(cov[k] if np.isfinite(cov[k]) else np.median([lo, hi]), lo, hi))
        out.append(dict(pid=str(pid), t=te, e=ee, future=picks, **cov))
    return out, excl


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
# Statistical baselines fitted on the training split
# ==============================================================================
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
    bs, ses, s2s = [], [], []
    for p in train:
        a, b, s2, tb, sxx, n = _ols(p["t"], p["e"])
        bs.append(b); s2s.append(s2); ses.append(s2 / sxx)
    bs, ses = np.array(bs), np.array(ses)
    sigma2 = float(np.median(s2s))
    se_pool = sigma2 / np.array([np.sum((p["t"] - p["t"].mean()) ** 2) for p in train])
    w = 1.0 / (se_pool + np.var(bs))
    mu = float(np.sum(w * bs) / np.sum(w))
    tau2 = float(max(np.var(bs) - np.mean(se_pool), 1e-3))
    return dict(mu=mu, tau2=tau2, sigma2=sigma2)


def _gbm_features(p, t_f):
    f = pz.features(p["t"], p["e"], p["a1c"], p["uacr0"], p["sbp"])
    return np.concatenate([f, [t_f - p["t"][-1], t_f]])


def fit_gbm(train, seed=0):
    from sklearn.ensemble import HistGradientBoostingRegressor
    X, y = [], []
    for p in train:
        for t_f, e_f in p["future"].values():
            X.append(_gbm_features(p, t_f)); y.append(e_f)
    X, y = np.array(X), np.array(y)
    models = {}
    for name, kw in [("mean", dict(loss="squared_error")),
                     ("q05", dict(loss="quantile", quantile=0.05)),
                     ("q95", dict(loss="quantile", quantile=0.95))]:
        models[name] = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05,
                                                     random_state=seed, **kw).fit(X, y)
    return models


# ==============================================================================
# Predictions for one patient
# ==============================================================================
def fit_population_prior(train, net, n_draws=300, seed=0, verbose=True):
    """Empirical-Bayes prior from the training patients' NPE draws. K (1 or 2
    components) is chosen by the held-out marginal likelihood on 20 % of the
    training patients, then refitted on all of them."""
    draws = [npe.posterior_samples(net, p["t"], p["e"], p["a1c"], p["uacr0"], p["sbp"],
                                   n=n_draws, seed=10_000 + i) for i, p in enumerate(train)]
    rng = np.random.default_rng(seed)
    hold = rng.uniform(size=len(draws)) < 0.2
    fit_d = [d for d, h in zip(draws, hold) if not h]
    val_d = [d for d, h in zip(draws, hold) if h]
    best_k, best_ll, val_ll = 1, -np.inf, {}
    lu = popprior._uniform_logpdf()
    from scipy import special as _sp
    for K in (1, 2):
        pr = popprior.fit(fit_d, K=K, seed=seed)
        lr = np.stack([pr.logpdf(d[:, 0], d[:, 1]) - lu for d in val_d])
        ll = float(np.mean(_sp.logsumexp(lr, axis=1) - np.log(lr.shape[1])))
        val_ll[K] = ll
        if ll > best_ll:
            best_k, best_ll = K, ll
    prior = popprior.fit(draws, K=best_k, seed=seed)
    prior.extra.update(K=best_k, heldout_loglik_per_patient=val_ll)
    if verbose:
        print(f"Population prior: K={best_k} (held-out log-lik/patient {val_ll})")
    return prior


def predict_patient(p, net, mlp, lmm, gbm, noise_sd, prior=None, n_draws=1000, seed=0):
    """Returns {method: dict(mu=array, sd=array|None, draws=array|None)} over the
    patient's scored future times (sorted by horizon key)."""
    keys = sorted(p["future"])
    tf = np.array([p["future"][k][0] for k in keys])
    cov = (p["a1c"], p["uacr0"], p["sbp"])
    t, e = p["t"], p["e"]
    q_pop = npe.Q_POP
    out = {}

    def sim(q, s, e0):
        return fast_sim.simulate_egfr(q, s, e0, *cov, tf)

    out["population"] = dict(mu=sim(q_pop, 1.0, e[0])[0], sd=None, draws=None)
    s1 = pz.least_squares_scale_only(t, e, *cov, q=q_pop)
    out["ls_rate"] = dict(mu=sim(q_pop, s1, e[0])[0], sd=None, draws=None)

    feats = pz.features(t, e, *cov)[None]
    member = np.array([np.exp(m.predict(mlp.scaler.transform(feats))[0, 1]) for m in mlp.models])
    _, _, s_m, _ = mlp.predict(feats)
    f_mem = sim(q_pop, member, e[0])
    out["mlp"] = dict(mu=sim(q_pop, float(s_m[0]), e[0])[0],
                      sd=np.sqrt(f_mem.std(0) ** 2 + noise_sd ** 2), draws=None)

    post = npe.posterior_samples(net, t, e, *cov, n=n_draws, seed=seed)
    lat = npe.predictive(post, *cov, tf)
    rng = np.random.default_rng(seed)
    obs = lat + rng.normal(0.0, noise_sd, lat.shape)
    out["npe"] = dict(mu=lat.mean(0), sd=None, draws=obs)
    if prior is not None:
        w, ess = popprior.reweight(post, prior)
        idx = rng.choice(len(post), size=len(post), p=w)
        lat_eb = lat[idx]
        out["npe_eb"] = dict(mu=lat_eb.mean(0), sd=None,
                             draws=lat_eb + rng.normal(0.0, noise_sd, lat_eb.shape), ess=ess)

    a, b, s2, tb, sxx, n = _ols(t, e)
    var_ols = s2 * (1 + 1 / n + (tf - tb) ** 2 / sxx)
    out["ols"] = dict(mu=a + b * tf, sd=np.sqrt(var_ols), draws=None)

    se2 = lmm["sigma2"] / sxx
    w = lmm["tau2"] / (lmm["tau2"] + se2)
    b_eb = w * b + (1 - w) * lmm["mu"]
    post_var = w * se2
    mu_eb = e.mean() + b_eb * (tf - tb)
    var_eb = lmm["sigma2"] / n + (tf - tb) ** 2 * post_var + lmm["sigma2"]
    out["lmm_eb"] = dict(mu=mu_eb, sd=np.sqrt(var_eb), draws=None)

    X = np.array([_gbm_features(p, x) for x in tf])
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


def score_rows(test, net, mlp, lmm, gbm, noise_sd, prior=None, verbose=True):
    """One row per (patient, horizon, method)."""
    rows = []
    methods = [m for m in METHODS if m != "npe_eb" or prior is not None]
    for i, p in enumerate(test):
        keys, tf, pred = predict_patient(p, net, mlp, lmm, gbm, noise_sd, prior=prior, seed=i)
        e_end = p["e"][-1]
        for j, h in enumerate(keys):
            y = p["future"][h][1]
            for m in methods:
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
    ref = REFERENCE if REFERENCE in methods else "npe"
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

    # patient bootstrap of RMSE(method) - RMSE(NPE), pooled over horizons:
    # resample PATIENTS (their visits travel together), recompute pooled RMSEs
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
    boot = {m: [] for m in methods if m != ref}
    for _ in range(n_boot):
        w = np.bincount(rng.integers(0, len(pids), len(pids)), minlength=len(pids))
        n_tot = np.sum(w * cnt)
        r_npe = np.sqrt(np.sum(w * sse[ref]) / n_tot)
        for m in boot:
            boot[m].append(np.sqrt(np.sum(w * sse[m]) / n_tot) - r_npe)
    ci = {m: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) for m, v in boot.items()}
    ci["_reference"] = ref
    return tables, ev, ci


def report(tables, ev, ci, meta):
    L = ["# Patient-level validation on MIMIC-IV\n"]
    L.append(f"Cohort: {meta['n_cohort']} patients in the loader output; eligible "
             f"{meta['n_eligible']} (excluded: {meta['excl']}). Split: {meta['split']}; "
             f"{meta['n_train']} training and {meta['n_test']} test patients. eGFR thinned "
             f"to one median value per {WINDOW_DAYS} days; observed window = first "
             f"{EARLY_YEARS:g} years after the confirmed index. Measurement noise assumed "
             f"by the mechanistic methods: {meta['noise_sd']:g} mL/min/1.73 m².\n")
    for h in list(HORIZONS) + ["all"]:
        title = "All horizons pooled" if h == "all" else f"Horizon {h} after the last observed visit"
        n = next(iter(tables[h].values()))["n"]
        L.append(f"## {title} (n = {n} patient-visits)\n")
        L.append("| Method | RMSE | MAE | Bias | 90 % coverage | CRPS |")
        L.append("|---|---|---|---|---|---|")
        for m, r in tables[h].items():
            cov = "—" if np.isnan(r["cov90"]) else f"{100 * r['cov90']:.0f} %"
            crps = "—" if np.isnan(r["crps"]) else f"{r['crps']:.2f}"
            name = f"**{LABELS[m]}**" if m == ci.get("_reference") else LABELS[m]
            L.append(f"| {name} | {r['rmse']:.2f} | {r['mae']:.2f} | {r['bias']:+.2f} | {cov} | {crps} |")
        L.append("")
    ref = LABELS[ci["_reference"]]
    L.append(f"## RMSE difference versus {ref} (patient bootstrap, 95 % CI)\n")
    L.append("Positive = the method is worse than the reference.\n")
    L.append("| Method | ΔRMSE 95 % CI |")
    L.append("|---|---|")
    for m, v in ci.items():
        if m.startswith("_"):
            continue
        L.append(f"| {LABELS[m]} | {v[0]:+.2f} to {v[1]:+.2f} |")
    L.append("\n## Event: eGFR < 30 at a held-out visit (patients ≥ 30 at the end of the window)\n")
    L.append("| Method | Visits | Events | AUC | Brier |")
    L.append("|---|---|---|---|---|")
    for m, r in ev.items():
        L.append(f"| {LABELS[m]} | {r['n']} | {r['events']} | {r['auc']:.3f} | {r['brier']:.3f} |")
    pr = meta.get("prior")
    if pr is not None:
        L.append("\n## Population prior learned from the training patients (empirical Bayes)\n")
        L.append(f"K = {pr['K']} component(s), chosen by held-out marginal likelihood "
                 f"({pr['heldout']}); mean importance-sampling efficiency on the test "
                 f"patients {100 * meta['ess']:.0f} %.\n")
        L.append("| Component | Weight | Mean q | SD q | Median injury rate | SD log rate |")
        L.append("|---|---|---|---|---|---|")
        for k, c in enumerate(pr["components"]):
            L.append(f"| {k + 1} | {c['weight']:.2f} | {c['q_mean']:.2f} | {c['q_sd']:.2f} | "
                     f"{c['rate_median']:.2f}× | {c['log_rate_sd']:.2f} |")
    return "\n".join(L) + "\n"


def plot(df, tables, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 7, "axes.spines.top": False, "axes.spines.right": False,
                         "savefig.dpi": 300})
    methods = [m for m in METHODS if m in set(df.method)]
    col = {"population": "#999999", "ls_rate": "#56B4E9", "mlp": "#D55E00", "npe": "#0072B2",
           "npe_eb": "#000000",
           "ols": "#E69F00", "lmm_eb": "#CC79A7", "gbm": "#009E73"}
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.4))
    hs = list(HORIZONS)
    w = 0.8 / len(methods)
    for k, m in enumerate(methods):
        ax[0].bar(np.arange(len(hs)) + (k - len(methods) / 2 + 0.5) * w,
                  [tables[h][m]["rmse"] for h in hs], w, color=col[m], label=LABELS[m])
    ax[0].set_xticks(range(len(hs)), hs)
    ax[0].set_xlabel("horizon after the last observed visit")
    ax[0].set_ylabel("RMSE (mL/min/1.73 m²)")
    ax[0].set_title("a  forecast error on held-out visits", loc="left", fontsize=7)
    ax[0].legend(frameon=False, fontsize=5)
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


def run(cohort, patients_csv=None, noise_sd=8.7, net=None, mlp=None, seed=0,
        n_boot=1000, verbose=True, eb_prior=True):
    patients, excl = build_patients(cohort)
    if len(patients) < 20:
        raise ValueError(f"only {len(patients)} eligible patients; check the cohort")
    is_test, split = split_patients(patients, patients_csv, seed=seed)
    train = [p for p, z in zip(patients, is_test) if not z]
    test = [p for p, z in zip(patients, is_test) if z]
    if verbose:
        print(f"{len(patients)} eligible patients; split {split}: "
              f"{len(train)} train / {len(test)} test")
    if net is None:
        net = npe.load(noise_sd=noise_sd)
    if mlp is None:
        mlp = pz.get_estimator()
    lmm = fit_lmm_eb(train)
    gbm = fit_gbm(train, seed=seed)
    prior = fit_population_prior(train, net, seed=seed, verbose=verbose) if eb_prior else None
    df = score_rows(test, net, mlp, lmm, gbm, noise_sd, prior=prior, verbose=verbose)
    tables, ev, ci = summarize(df, n_boot=n_boot, seed=seed)
    meta = dict(n_cohort=int(cohort.patient_id.nunique()), n_eligible=len(patients),
                excl=excl, split=split, n_train=len(train), n_test=len(test),
                noise_sd=noise_sd)
    if prior is not None:
        meta["prior"] = dict(K=prior.extra["K"], components=prior.summary(),
                             heldout={k: round(v, 3) for k, v in
                                      prior.extra["heldout_loglik_per_patient"].items()})
        meta["ess"] = float(df.loc[df.method == "npe_eb", "ess"].mean())
    return df, tables, ev, ci, meta


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Patient-level validation on MIMIC-IV")
    ap.add_argument("--cohort", default=os.path.join(HERE, "..", "data", "_mimic_cohort.tsv"))
    ap.add_argument("--patients", default=None, help="hosp/patients.csv.gz (temporal split)")
    ap.add_argument("--noise-sd", type=float, default=8.7)
    ap.add_argument("--n-boot", type=int, default=1000)
    a = ap.parse_args()
    cohort = pd.read_csv(a.cohort, sep=None, engine="python", dtype={"patient_id": str})
    df, tables, ev, ci, meta = run(cohort, a.patients, a.noise_sd, n_boot=a.n_boot)
    os.makedirs(RESULTS, exist_ok=True)
    df.to_csv(os.path.join(RESULTS, "mimic_patient_validation_rows.csv"), index=False)  # git-ignored
    text = report(tables, ev, ci, meta)
    with open(os.path.join(RESULTS, "mimic_patient_validation.md"), "w", encoding="utf8") as f:
        f.write(text)
    plot(df, tables, os.path.join(RESULTS, "mimic_patient_validation.png"))
    print(text)
