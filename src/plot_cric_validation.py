"""
Figures for the CRIC external-validation check (population level).

Produces two panels:
  1. predicted vs observed chronic eGFR slope, by CRIC phase
  2. NephroQ-projected eGFR trajectories for each cohort's baseline profile

Run:  python src/plot_cric_validation.py
Saves PNGs to results/. These are population-level plausibility figures, not
patient-level validation (see external_validation.py).
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import model_core as core
import external_validation as ev

# clean, minimal, colour-vision-safe figure defaults
plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 300,
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7, "axes.labelsize": 7, "axes.titlesize": 8,
    "xtick.labelsize": 6, "ytick.labelsize": 6, "legend.fontsize": 6,
    "axes.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": False, "xtick.direction": "out", "ytick.direction": "out",
    "xtick.major.size": 3, "ytick.major.size": 3,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "legend.frameon": False, "lines.linewidth": 1.4, "lines.solid_capstyle": "round",
    "figure.facecolor": "white", "axes.facecolor": "white",
})
C_PRED, C_OBS = "#D55E00", "#009E73"
C_A, C_B = "#D55E00", "#0072B2"


def fig_slope_comparison(path):
    import math
    cohorts = ["phase_I", "phase_III"]
    labels = ["Phase I\n(eGFR 40.7)", "Phase III\n(eGFR 57.2)"]
    ns = [ev.CRIC_COHORTS[k]["n"] for k in cohorts]
    mid, lo, hi = [], [], []
    for k in cohorts:
        r = ev.validate_population(k)
        a, b = sorted(r.predicted_slope_range)
        lo.append(a); hi.append(b); mid.append((a + b) / 2)
    obs, sd = ev.CRIC_OBSERVED_SLOPE["value"], ev.CRIC_OBSERVED_SLOPE["sd"]
    n_overall = sum(ns)
    sem = sd / math.sqrt(n_overall)

    fig, ax = plt.subplots(figsize=(3.9, 2.8), constrained_layout=True)
    x = np.arange(len(cohorts))

    # CRIC's slope is a SINGLE overall diabetic value, not a per-phase measurement.
    # Draw it as one horizontal reference (line = mean, tight band = SEM, faint band
    # = between-patient SD), spanning both phases -- so it reads as "one reference
    # applied to both", never as two independent points.
    ax.axhspan(obs - sd, obs + sd, color=C_OBS, alpha=0.10, lw=0)
    ax.axhspan(obs - sem, obs + sem, color=C_OBS, alpha=0.35, lw=0)
    ax.axhline(obs, color=C_OBS, lw=1.0, ls=(0, (4, 3)),
               label=f"CRIC overall diabetic slope ({obs}±SEM)")

    # NephroQ predictions per phase (bar = proteinuria->UACR conversion range)
    yerr = [np.array(mid) - np.array(lo), np.array(hi) - np.array(mid)]
    ax.errorbar(x, mid, yerr=yerr, fmt="o", color=C_PRED, capsize=3,
                markersize=5, lw=1.2, label="NephroQ predicted (UACR range)", zorder=3)

    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("Chronic eGFR slope\n(mL/min/1.73m² per year)")
    ax.set_title("NephroQ vs CRIC overall diabetic slope", pad=22)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=1,
              handlelength=1.5, borderaxespad=0)
    ax.text(len(cohorts) - 0.52, obs + sd + 0.06,
            f"shaded = between-patient SD (±{sd:.1f}); no per-phase slope published",
            fontsize=4.5, color=C_OBS, ha="right", va="bottom")
    ax.set_ylim(-4.2, 0.4); ax.margins(x=0.22)
    fig.savefig(path); plt.close(fig)


def fig_trajectories(path):
    fig, ax = plt.subplots(figsize=(3.5, 2.6), constrained_layout=True)
    for key, color, name in (("phase_I", C_A, "Phase I (advanced)"),
                             ("phase_III", C_B, "Phase III (earlier)")):
        c = ev.CRIC_COHORTS[key]
        ulo, uhi = ev.proteinuria_to_uacr(c["proteinuria_g_day"])
        uacr = 0.5 * (ulo + uhi)
        t, e, _, _ = core.simulate_trajectory_v2(c["egfr"], c["hba1c"], uacr, c["sbp"],
                                                 u=0.0, p=core.TRIAL_CALIBRATION_V2,
                                                 years=10, n=100)
        ax.plot(t, e, color=color, label=name)
    ax.axhline(core.DIALYSIS_eGFR, color="0.5", lw=0.6, ls=(0, (4, 3)))
    ax.text(0.2, core.DIALYSIS_eGFR + 1.5, "Dialysis threshold", fontsize=5.5, color="0.5")
    ax.set_xlabel("Years from baseline")
    ax.set_ylabel("eGFR (mL/min/1.73m²)")
    ax.set_title("Projected trajectories, CRIC baseline profiles")
    ax.set_xlim(0, 10); ax.set_ylim(0, 65); ax.margins(x=0)
    ax.legend(loc="upper right", handlelength=1.5)
    fig.savefig(path); plt.close(fig)


if __name__ == "__main__":
    out = os.path.join(os.path.dirname(__file__), "..", "results")
    os.makedirs(out, exist_ok=True)
    fig_slope_comparison(os.path.join(out, "cric_slope_comparison.png"))
    fig_trajectories(os.path.join(out, "cric_trajectories.png"))
    print("saved cric_slope_comparison.png and cric_trajectories.png to results/")
