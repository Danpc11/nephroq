"""
Benchmark of per-patient inference for the renal digital twin.

Question: given a short, irregular, noisy eGFR history, how well does each method
(a) recover the patient's latent parameters with honest uncertainty, and
(b) forecast the patient's eGFR five years after the last visit?

Methods, all scored on the SAME held-out virtual patients drawn from the NPE
prior (each patient has its own q, injury rate and true baseline eGFR):

  population            q = population value, injury rate = 1, start at the
                        first observed eGFR (no personalization)
  MLP ensemble          the app's current personalizer: 4 MLPs on 11 summary
                        features -> injury rate; q at population value
  LS, rate only         one-parameter least squares (the deployed problem)
  LS, rate + eGFR0      two-parameter least squares, q at population value
  NPE                   amortized neural posterior (GRU summary + spline flow),
                        full posterior over (q, log rate, eGFR0); forecast is the
                        posterior predictive
  NPE, q fixed          ablation: NPE draws with q replaced by the population
                        value (what "fix the unidentifiable parameter" costs)
  exact (grid)          brute-force Bayesian posterior on a 3-D grid under the
                        same prior and likelihood -- the gold standard

Outputs results/posterior_benchmark.md and results/posterior_benchmark.png.
Run:  python benchmark_posterior.py [--n-test 300]
"""
import argparse
import os
import time

import numpy as np
from scipy import stats
from scipy.optimize import least_squares

import model_core as core
import fast_sim
import personalize as pz
import npe_posterior as npe

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "..", "results")
HORIZON = 5.0
LEVELS = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95])
N_RANK = 99                       # posterior draws used for SBC ranks


def crps_samples(x, y):
    """Continuous ranked probability score from samples (lower is better)."""
    x = np.sort(np.asarray(x, float))
    n = len(x)
    e1 = np.mean(np.abs(x - y))
    # E|X - X'| for sorted samples in O(n)
    e2 = 2.0 * np.sum(x * (2 * np.arange(1, n + 1) - n - 1)) / (n * n)
    return e1 - 0.5 * e2


def crps_normal(mu, sd, y):
    sd = max(sd, 1e-6)
    z = (y - mu) / sd
    return sd * (z * (2 * stats.norm.cdf(z) - 1) + 2 * stats.norm.pdf(z) - 1 / np.sqrt(np.pi))


def central_hits(samples, y, levels=LEVELS):
    lo = np.quantile(samples, 0.5 - levels / 2)
    hi = np.quantile(samples, 0.5 + levels / 2)
    return (lo <= y) & (y <= hi), hi - lo


def ls_rate_and_egfr0(t, e, a1c, uacr0, sbp):
    """Least squares on (log rate, eGFR0) with q at the population value."""
    def resid(z):
        return fast_sim.simulate_egfr(npe.Q_POP, np.exp(z[0]), z[1], a1c, uacr0, sbp, t)[0] - e
    x0 = [0.0, float(np.clip(e[0], npe.PRIOR_LO[2] + 1e-6, npe.PRIOR_HI[2] - 1e-6))]
    sol = least_squares(resid, x0,
                        bounds=([npe.PRIOR_LO[1], npe.PRIOR_LO[2]],
                                [npe.PRIOR_HI[1], npe.PRIOR_HI[2]]))
    return float(np.exp(sol.x[0])), float(sol.x[1])


def train_mlp_baseline(n=20000, seed=7):
    """The app's MLP-ensemble personalizer, retrained under the NPE prior so the
    comparison is not decided by a prior mismatch."""
    d = npe.simulate_dataset(n, seed=seed)
    X = np.array([pz.features(t[m], e[m], a, u, s) for t, e, m, a, u, s in
                  zip(d["times"], d["egfr"], d["mask"], d["a1c"], d["uacr0"], d["sbp"])])
    Y = np.column_stack([d["theta"][:, 0], d["theta"][:, 1]])
    return pz.Personalizer(n_models=4).fit(X, Y)


def run(n_test=300, seed=2026, n_draws=500, verbose=True):
    rng = np.random.default_rng(seed)
    net = npe.load() if os.path.exists(npe.MODEL_PATH) else npe.train()[0]
    if verbose:
        print("Training the MLP-ensemble baseline under the NPE prior...")
    mlp = train_mlp_baseline()

    d = npe.simulate_dataset(n_test, seed=seed)
    rows, ranks_npe, ranks_grid = [], [], []
    hits = {k: [] for k in ("npe", "npe_qfix", "grid", "mlp")}
    for i in range(n_test):
        m = d["mask"][i]
        t, e = d["times"][i][m], d["egfr"][i][m]
        a1c, uacr0, sbp = d["a1c"][i], d["uacr0"][i], d["sbp"][i]
        th = d["theta"][i]
        t_f = t[-1] + HORIZON
        y = float(fast_sim.simulate_egfr(th[0], np.exp(th[1]), th[2], a1c, uacr0, sbp, [t_f])[0, 0])
        r = dict(i=i, n_visits=len(t), span=t[-1], q=th[0], log_scale=th[1], egfr0=th[2], y=y)

        def fc(q, scale, e0):
            return float(fast_sim.simulate_egfr(q, scale, e0, a1c, uacr0, sbp, [t_f])[0, 0])

        r["pop"] = fc(npe.Q_POP, 1.0, e[0])

        t0 = time.perf_counter()
        _, _, s_m, _ = mlp.predict(pz.features(t, e, a1c, uacr0, sbp))
        member = np.array([np.exp(mm.predict(mlp.scaler.transform(
            pz.features(t, e, a1c, uacr0, sbp)[None]))[0, 1]) for mm in mlp.models])
        r["t_mlp"] = time.perf_counter() - t0
        f_mem = fast_sim.simulate_egfr(npe.Q_POP, member, e[0], a1c, uacr0, sbp, [t_f])[:, 0]
        r["mlp"], r["mlp_sd"] = fc(npe.Q_POP, float(s_m[0]), e[0]), float(f_mem.std())

        t0 = time.perf_counter()
        s1 = pz.least_squares_scale_only(t, e, a1c, uacr0, sbp, q=npe.Q_POP)
        r["t_ls1"] = time.perf_counter() - t0
        r["ls1"] = fc(npe.Q_POP, s1, e[0])

        t0 = time.perf_counter()
        s2, e02 = ls_rate_and_egfr0(t, e, a1c, uacr0, sbp)
        r["t_ls2"] = time.perf_counter() - t0
        r["ls2"] = fc(npe.Q_POP, s2, e02)

        t0 = time.perf_counter()
        post = npe.posterior_samples(net, t, e, a1c, uacr0, sbp, n=n_draws, seed=i)
        f_npe = npe.predictive(post, a1c, uacr0, sbp, t_f)[:, 0]
        r["t_npe"] = time.perf_counter() - t0
        pq = post.copy(); pq[:, 0] = npe.Q_POP
        f_qfix = npe.predictive(pq, a1c, uacr0, sbp, t_f)[:, 0]
        r["npe"], r["npe_qfix"] = float(f_npe.mean()), float(f_qfix.mean())
        r["crps_npe"], r["crps_npe_qfix"] = crps_samples(f_npe, y), crps_samples(f_qfix, y)
        for k, name in enumerate(npe.PARAM_NAMES):
            r[f"npe_mean_{name}"] = float(post[:, k].mean())
            r[f"npe_var_{name}"] = float(post[:, k].var())
        ranks_npe.append([(post[:N_RANK, k] < th[k]).sum() for k in range(3)])

        t0 = time.perf_counter()
        pts, w, fut, cell = npe.grid_posterior(t, e, a1c, uacr0, sbp, t_future=t_f)
        r["t_grid"] = time.perf_counter() - t0
        gs, gi = npe.grid_samples(pts, w, cell, n_draws, rng)
        f_grid = fut[gi, 0]
        r["grid"] = float(np.sum(w * fut[:, 0]))
        r["crps_grid"] = crps_samples(f_grid, y)
        for k, name in enumerate(npe.PARAM_NAMES):
            mu = float(np.sum(w * pts[:, k]))
            r[f"grid_mean_{name}"] = mu
            r[f"grid_var_{name}"] = float(np.sum(w * (pts[:, k] - mu) ** 2) + cell[k] ** 2 / 12)
        ranks_grid.append([(gs[:N_RANK, k] < th[k]).sum() for k in range(3)])

        hits["npe"].append(central_hits(f_npe, y))
        hits["npe_qfix"].append(central_hits(f_qfix, y))
        hits["grid"].append(central_hits(f_grid, y))
        z = stats.norm.ppf(0.5 + LEVELS / 2)
        lo, hi = r["mlp"] - z * r["mlp_sd"], r["mlp"] + z * r["mlp_sd"]
        hits["mlp"].append(((lo <= y) & (y <= hi), hi - lo))
        r["crps_mlp"] = crps_normal(r["mlp"], r["mlp_sd"], y)
        rows.append(r)
        if verbose and (i + 1) % 25 == 0:
            print(f"  {i + 1}/{n_test} patients")

    import pandas as pd
    df = pd.DataFrame(rows)
    return df, np.array(ranks_npe), np.array(ranks_grid), hits


def summarize(df, ranks_npe, ranks_grid, hits):
    """Markdown report (tables) from the per-patient results."""
    def rmse(col):
        return float(np.sqrt(np.mean((df[col] - df.y) ** 2)))

    i90 = int(np.where(LEVELS == 0.9)[0][0])
    cov = {k: np.mean([h[0][i90] for h in v]) for k, v in hits.items()}
    wid = {k: np.mean([h[1][i90] for h in v]) for k, v in hits.items()}
    ms = lambda c: 1000 * float(df[c].mean())

    L = []
    L.append("# Per-patient inference benchmark\n")
    lo, hi = npe.PRIOR_LO, npe.PRIOR_HI
    L.append(f"{len(df)} held-out virtual patients drawn from the NPE prior "
             f"(q ~ U({lo[0]:g}, {hi[0]:g}); injury rate ~ log-U({np.exp(lo[1]):.3g}, "
             f"{np.exp(hi[1]):.3g}); true baseline eGFR ~ U({lo[2]:g}, {hi[2]:g})); "
             f"2-9 visits over 0.5-8 years; measurement noise SD {npe.NOISE_SD:g} mL/min/1.73 m2. "
             f"Target: latent eGFR {HORIZON:g} years after the last visit.\n")
    L.append("## Forecast\n")
    L.append("| Method | RMSE | 90% interval coverage | 90% interval width | CRPS | ms / patient |")
    L.append("|---|---|---|---|---|---|")
    fmt = lambda x: "—" if x is None else f"{x:.2f}"
    table = [
        ("Population parameters", "pop", None, None, None, None),
        ("MLP ensemble (current app)", "mlp", cov["mlp"], wid["mlp"], df.crps_mlp.mean(), ms("t_mlp")),
        ("Least squares, rate only", "ls1", None, None, None, ms("t_ls1")),
        ("Least squares, rate + eGFR0", "ls2", None, None, None, ms("t_ls2")),
        ("**NPE (posterior predictive)**", "npe", cov["npe"], wid["npe"], df.crps_npe.mean(), ms("t_npe")),
        ("NPE with q fixed (ablation)", "npe_qfix", cov["npe_qfix"], wid["npe_qfix"], df.crps_npe_qfix.mean(), None),
        ("Exact posterior (grid reference)", "grid", cov["grid"], wid["grid"], df.crps_grid.mean(), ms("t_grid")),
    ]
    for name, col, c, w_, crps, t in table:
        L.append(f"| {name} | {rmse(col):.2f} | {('—' if c is None else f'{100*c:.0f}%')} | "
                 f"{fmt(w_)} | {fmt(crps)} | {fmt(t)} |")

    L.append("\n## Parameters (posterior of the NPE vs the exact grid posterior)\n")
    L.append("| Parameter | Prior SD | NPE: RMSE of posterior mean | Exact: RMSE | NPE: contraction | "
             "Exact: contraction | NPE 90% coverage | SBC p (NPE) | SBC p (exact) |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    prior_sd = (npe.PRIOR_HI - npe.PRIOR_LO) / np.sqrt(12)
    for k, name in enumerate(npe.PARAM_NAMES):
        truth = df[name]
        r_n = float(np.sqrt(np.mean((df[f"npe_mean_{name}"] - truth) ** 2)))
        r_g = float(np.sqrt(np.mean((df[f"grid_mean_{name}"] - truth) ** 2)))
        c_n = 1 - float(df[f"npe_var_{name}"].mean()) / prior_sd[k] ** 2
        c_g = 1 - float(df[f"grid_var_{name}"].mean()) / prior_sd[k] ** 2
        z = np.abs(df[f"npe_mean_{name}"] - truth) / np.sqrt(df[f"npe_var_{name}"])
        # rank-based coverage: central 90% of the 99 draws
        rk = ranks_npe[:, k]
        cov90 = float(np.mean((rk >= 5) & (rk <= 94)))
        p_n = stats.chisquare(np.histogram(ranks_npe[:, k], bins=10, range=(0, N_RANK + 1))[0]).pvalue
        p_g = stats.chisquare(np.histogram(ranks_grid[:, k], bins=10, range=(0, N_RANK + 1))[0]).pvalue
        L.append(f"| {name} | {prior_sd[k]:.3f} | {r_n:.3f} | {r_g:.3f} | {c_n:.2f} | {c_g:.2f} | "
                 f"{100*cov90:.0f}% | {p_n:.2f} | {p_g:.2f} |")
    L.append("\nContraction = 1 − (mean posterior variance / prior variance): 0 means the data "
             "teach nothing about the parameter, 1 means it is pinned down. SBC p is a "
             "chi-square test of uniform simulation-based-calibration ranks (10 bins); "
             "p > 0.05 is consistent with a calibrated posterior.\n")
    return "\n".join(L), cov, wid


def plot(df, ranks_npe, hits, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 7, "axes.spines.top": False, "axes.spines.right": False,
                         "figure.dpi": 150, "savefig.dpi": 300})
    fig, ax = plt.subplots(1, 4, figsize=(7.2, 1.9))
    labels = {"q": "q (collapse exponent)", "log_scale": "log injury rate", "egfr0": "true baseline eGFR"}
    for k, name in enumerate(npe.PARAM_NAMES):
        ax[k].hist(ranks_npe[:, k], bins=10, range=(0, N_RANK + 1), color="#0072B2", alpha=0.85)
        ax[k].axhline(len(df) / 10, color="k", lw=0.6, ls="--")
        ax[k].set_title(chr(97 + k) + "  SBC ranks: " + labels[name], loc="left", fontsize=7)
        ax[k].set_xlabel("rank of truth among 99 draws")
    for key, col, lab in [("npe", "#0072B2", "NPE"), ("grid", "#009E73", "exact"),
                          ("npe_qfix", "#E69F00", "NPE, q fixed"), ("mlp", "#D55E00", "MLP ensemble")]:
        emp = np.mean([h[0] for h in hits[key]], axis=0)
        ax[3].plot(LEVELS, emp, "o-", ms=2.5, lw=1, color=col, label=lab)
    ax[3].plot([0, 1], [0, 1], "k--", lw=0.6)
    ax[3].set_xlabel("nominal coverage")
    ax[3].set_ylabel("empirical coverage")
    ax[3].set_title("d  forecast calibration", loc="left", fontsize=7)
    ax[3].legend(frameon=False, fontsize=5.5)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-test", type=int, default=300)
    ap.add_argument("--plot-only", action="store_true",
                    help="redraw the figure and report from the saved run")
    a = ap.parse_args()
    os.makedirs(RESULTS, exist_ok=True)
    raw = os.path.join(RESULTS, "posterior_benchmark_raw.npz")
    rows = os.path.join(RESULTS, "posterior_benchmark_rows.csv")
    if a.plot_only:
        import pandas as pd
        df = pd.read_csv(rows)
        z = np.load(raw)
        rn, rg = z["ranks_npe"], z["ranks_grid"]
        hits = {k: list(zip(z[f"hit_{k}"], z[f"width_{k}"])) for k in ("npe", "npe_qfix", "grid", "mlp")}
    else:
        df, rn, rg, hits = run(n_test=a.n_test)
        df.to_csv(rows, index=False)
        np.savez(raw, ranks_npe=rn, ranks_grid=rg,
                 **{f"hit_{k}": np.array([h[0] for h in v]) for k, v in hits.items()},
                 **{f"width_{k}": np.array([h[1] for h in v]) for k, v in hits.items()})
    text, _, _ = summarize(df, rn, rg, hits)
    with open(os.path.join(RESULTS, "posterior_benchmark.md"), "w", encoding="utf8") as f:
        f.write(text)
    plot(df, rn, hits, os.path.join(RESULTS, "posterior_benchmark.png"))
    print(text)
