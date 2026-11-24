"""
When does eGFR decline accelerate in this model?  (audit P0.3)

For fixed covariates, with G = G_MAX N^ALPHA and dN/dt = -N h:

    G'  = -ALPHA G h
    G'' =  ALPHA^2 G h^2 (1 + eps),      eps = d ln h / d ln G

so the decline ACCELERATES (|G'| grows) exactly where eps < -1 and decelerates
where eps > -1 (model_core.hazard_elasticity gives eps analytically). All hazard
terms are bounded, so there is no finite-time singularity and no bifurcation: the
exponent q only sets how strongly the hazard rises with load before saturation.

This script maps the accelerating region over (eGFR, q) for representative
covariates, finds the smallest q at which acceleration occurs anywhere, and
draws example trajectories. Outputs dynamics_regimes.md / .png in the output
directory (results/, --out-dir or $NEPHROQ_OUT_DIR).

Run:  python dynamics_regimes.py
"""
import argparse

import numpy as np

import model_core as core
import personalize as pz
import fast_sim
import outputs

G_GRID = np.linspace(12.0, 110.0, 99)
Q_GRID = np.linspace(0.8, 3.5, 55)
UACRS = (30.0, 300.0, 1500.0)
COV = dict(a1c=8.0, sbp=140.0)


def eps_map(uacr, scale=1.0, egfr0=110.0):
    """eps over (q, eGFR) for one albuminuria level. Shape (len(Q_GRID), len(G_GRID))."""
    out = np.empty((len(Q_GRID), len(G_GRID)))
    for i, q in enumerate(Q_GRID):
        _, out[i] = core.hazard_elasticity(G_GRID, egfr0, COV["a1c"], uacr, COV["sbp"],
                                           0.0, pz.patient_params(q, scale))
    return out


def q_threshold(uacr, scale=1.0):
    """Smallest q on the grid at which the decline accelerates at some eGFR."""
    m = eps_map(uacr, scale)
    rows = np.flatnonzero((m < -1).any(axis=1))
    return float(Q_GRID[rows[0]]) if rows.size else np.inf


def accelerating_range(q, uacr, scale=1.0):
    _, eps = core.hazard_elasticity(G_GRID, 110.0, COV["a1c"], uacr, COV["sbp"], 0.0,
                                    pz.patient_params(q, scale))
    acc = G_GRID[eps < -1]
    return (float(acc.min()), float(acc.max())) if acc.size else None


def report():
    L = ["# When does eGFR decline accelerate?\n",
         "For fixed covariates, G'' = ALPHA² G h² (1 + ε) with ε = d ln h / d ln G: the "
         "decline accelerates exactly where ε < −1. Covariates: HbA1c "
         f"{COV['a1c']:g} %, SBP {COV['sbp']:g} mmHg; injury rate 1× unless stated.\n",
         "| UACR (mg/g) | Smallest q with acceleration | Accelerating eGFR range at q = 1.52 "
         "(trial calibration) | at q = 2.9 |",
         "|---|---|---|---|"]
    for u in UACRS:
        qt = q_threshold(u)
        r1, r2 = accelerating_range(1.52, u), accelerating_range(2.9, u)
        f = lambda r: "none (decelerates everywhere)" if r is None else f"{r[0]:.0f}–{r[1]:.0f}"
        L.append(f"| {u:g} | {qt:.2f} | {f(r1)} | {f(r2)} |")
    L.append("\nThe injury rate barely moves these boundaries (it scales the injury terms "
             "together, leaving their shares of the hazard almost unchanged once k0 is "
             "small). Below the saturation point (eGFR ≈ 44 for S_SAT = 3.5) the "
             "hyperfiltration elasticity vanishes and the decline decelerates in every "
             "case; there is no terminal collapse in the model.")
    return "\n".join(L) + "\n"


def plot(path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 7, "axes.spines.top": False, "axes.spines.right": False,
                         "savefig.dpi": 300})
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.3))
    m = eps_map(30.0)
    im = ax[0].contourf(G_GRID, Q_GRID, 1 + m, levels=np.linspace(-1.2, 1.0, 12), cmap="RdBu")
    ax[0].contour(G_GRID, Q_GRID, 1 + m, levels=[0.0], colors="k", linewidths=0.8)
    ax[0].axhline(1.52, color="k", ls=":", lw=0.7)
    ax[0].set_xlabel("eGFR (mL/min/1.73 m²)")
    ax[0].set_ylabel("q")
    ax[0].set_title("a  sign of G'' (1 + ε), UACR 30", loc="left", fontsize=7)
    fig.colorbar(im, ax=ax[0], fraction=0.05, pad=0.02)
    t = np.linspace(0, 30, 601)
    for q, col in [(1.52, "#0072B2"), (2.9, "#D55E00")]:
        e = fast_sim.simulate_egfr(q, 1.0, 95.0, COV["a1c"], 30.0, COV["sbp"], t,
                                   params=pz.patient_params(q, 1.0))[0]
        ax[1].plot(t, e, color=col, label=f"q = {q:g}")
        ax[2].plot(e[1:], -np.diff(e) / np.diff(t), color=col, label=f"q = {q:g}")
    ax[1].set_xlabel("years"); ax[1].set_ylabel("eGFR")
    ax[1].set_title("b  trajectories from eGFR 95", loc="left", fontsize=7)
    ax[1].legend(frameon=False)
    ax[2].invert_xaxis()
    ax[2].set_xlabel("eGFR (falling →)"); ax[2].set_ylabel("rate of decline (/yr)")
    ax[2].set_title("c  speed of decline", loc="left", fontsize=7)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=None)
    a = ap.parse_args()
    text = report()
    with open(outputs.results_path("dynamics_regimes.md", a.out_dir), "w", encoding="utf8") as f:
        f.write(text)
    plot(outputs.results_path("dynamics_regimes.png", a.out_dir))
    print(text)
