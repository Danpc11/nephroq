"""
Figures for the two headline results, so the repo's strongest findings are
visualised. Uses seaborn for a cleaner aesthetic.

  1. Two regimes (Round 17, final MIMIC-pure fits): the collapse exponent q takes
     opposite regimes. Shown as the HAZARD-SHAPE curves themselves -- you see the
     regime change, not three bar heights -- with fit quality as context.
  2. In-silico trial replication as a forest plot: two calibration trials and the
     held-out DAPA-CKD prediction against published values.

Run:  python src/plot_key_results.py   -> results/*.png

Values are the FINAL reproduced numbers from docs/CHANGELOG.md Round 17
(q = 0.90 full cohort, 3.30 progressors; NOT the intermediate --anchor-weights
q=2.92 run) and the verified trial provenance.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sys.path.insert(0, os.path.dirname(__file__))
import model_core as core
import insilico_trial as it

sns.set_theme(style="ticks", context="paper", font_scale=0.9,
              rc={"axes.linewidth": 0.7, "font.family": "sans-serif"})
# colour-vision-safe
C_SUB, C_TRIAL, C_SUP = "#D55E00", "#009E73", "#0072B2"


def fig_two_regimes(path):
    """
    Left panel: the structural hazard term vs disease progression, for the three
    reproduced exponents. The SHAPE is the finding (q<1 decelerates, q~1.5 near-
    linear, q>3 collapses). Right panel (separate, not an inset): a single q fits
    progressors (chi2/n 3.03, pass) but not the full mixed cohort (8.06, warning).
    Legends and the second panel sit OUTSIDE the curve area so nothing is occluded.
    """
    fig, (ax, axq) = plt.subplots(
        1, 2, figsize=(6.4, 3.1), constrained_layout=True,
        gridspec_kw={"width_ratios": [2.4, 1]})

    egfr = np.linspace(90, 20, 200)
    N = np.array([core.N_of_egfr(e) for e in egfr])
    s_over = 1.0 / N

    regimes = [
        (0.90, C_SUB, "q = 0.90  MIMIC full cohort (sub-linear)"),
        (1.52, C_TRIAL, "q = 1.52  trials (near-linear)"),
        (3.30, C_SUP, "q = 3.30  MIMIC progressors (super-linear)"),
    ]
    for q, c, label in regimes:
        hf = s_over ** q / (1 + (s_over / core.S_SAT) ** q)
        hf = hf / hf.max()
        ax.plot(egfr, hf, color=c, lw=2.4, label=label)

    ax.axvline(44, color="0.6", lw=0.7, ls=(0, (3, 3)))
    ax.text(44, -0.07, "saturation\nbreakpoint", fontsize=6, color="0.5",
            ha="center", va="top")
    ax.set_xlabel("eGFR (mL/min/1.73m²)  →  disease progression")
    ax.set_ylabel("Structural hazard shape (normalised)")
    ax.invert_xaxis()
    ax.set_ylim(0, 1.05)
    # legend ABOVE the axes, clear of the curves
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=1,
              fontsize=6.5, frameon=False, borderaxespad=0)
    sns.despine(ax=ax)

    # right panel: fit quality, its own axes (no overlap possible)
    axq.scatter([0, 1], [8.06, 3.03], c=[C_SUB, C_SUP], s=70, zorder=3)
    axq.axhline(3.0, color="0.6", lw=0.6, ls=(0, (2, 2)))
    axq.set_xlim(-0.5, 1.5); axq.set_xticks([0, 1])
    axq.set_xticklabels(["full\ncohort", "progressors"], fontsize=7)
    axq.set_ylim(0, 9)
    axq.set_ylabel("χ²/n  (lower = better fit)", fontsize=7.5)
    axq.set_title("fit quality", fontsize=8)
    axq.annotate("warning", (0, 8.06), (0.06, 8.4), fontsize=6, color=C_SUB)
    axq.annotate("pass", (1, 3.03), (1.06, 3.4), fontsize=6, color=C_SUP)
    sns.despine(ax=axq)

    fig.suptitle("One hazard, three regimes: the collapse exponent depends on "
                 "the population it is fit on", fontsize=9)
    fig.savefig(path, dpi=300); plt.close(fig)


def fig_insilico(path):
    """
    Forest plot: two calibration trials (CREDENCE, EMPA-KIDNEY) and the held-out
    DAPA-CKD prediction vs its published CI. The held-out point landing on its
    published interval is the falsifiable result.
    """
    f = it.fit(n=220, seed=7)
    rows = []
    for name in ("CREDENCE", "EMPA-KIDNEY", "DAPA-CKD (T2D subgroup)"):
        D = it.TRIALS[name]
        arms = it.trial_arms(D, f["scale"], f["eff_met"], f["eff_hf"], f["eff_alb"],
                             n=220, seed=507)
        rows.append(dict(
            trial=name.replace(" (T2D subgroup)", "\n(T2D subgroup)"),
            predicted=arms["slope_diff"],
            published=D.get("chronic_slope_diff"),
            ci=D.get("chronic_slope_ci"),
            role="held-out" if "OUT-OF-SAMPLE" in D.get("role", "") else "calibration",
        ))

    fig, ax = plt.subplots(figsize=(4.6, 2.7), constrained_layout=True)
    y = np.arange(len(rows))[::-1]
    for yi, r in zip(y, rows):
        c = C_SUP if r["role"] == "held-out" else "0.55"
        # published CI as a horizontal line (where available)
        if r["ci"]:
            ax.plot(r["ci"], [yi, yi], color=C_TRIAL, lw=6, alpha=0.35,
                    solid_capstyle="round", zorder=1)
        if r["published"] is not None:
            ax.scatter(r["published"], yi, marker="|", s=180, color=C_TRIAL,
                       zorder=2, linewidths=1.6)
        ax.scatter(r["predicted"], yi, s=70, color=c, zorder=3,
                   edgecolor="white", linewidth=0.8)

    ax.set_yticks(y)
    ax.set_yticklabels([r["trial"] for r in rows], fontsize=7)
    ax.set_xlabel("Chronic eGFR slope difference (mL/min/1.73m² per year)")
    # title placed high, via suptitle, so it never collides with the legend below it
    fig.suptitle("In-silico replication: DAPA-CKD predicted out-of-sample",
                 fontsize=9, y=1.12)

    # legend by proxy
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=C_SUP,
               markeredgecolor="white", markersize=8, label="NephroQ predicted (held-out)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="0.55",
               markersize=8, label="NephroQ predicted (calibration)"),
        Line2D([0], [0], marker="|", color=C_TRIAL, markersize=12,
               label="published value"),
        Line2D([0], [0], color=C_TRIAL, lw=6, alpha=0.35, label="published 95% CI"),
    ]
    # legend sits between the title and the axes, in its own band (y just above 1)
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.005),
              ncol=2, fontsize=6, frameon=False, borderaxespad=0,
              columnspacing=1.4, handletextpad=0.5)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.margins(x=0.08)
    sns.despine(ax=ax)
    fig.savefig(path, dpi=300); plt.close(fig)


if __name__ == "__main__":
    out = os.path.join(os.path.dirname(__file__), "..", "results")
    os.makedirs(out, exist_ok=True)
    fig_two_regimes(os.path.join(out, "two_regimes_q.png"))
    fig_insilico(os.path.join(out, "insilico_dapackd.png"))
    print("saved two_regimes_q.png and insilico_dapackd.png to results/")
