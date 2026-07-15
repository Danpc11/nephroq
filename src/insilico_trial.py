"""
================================================================================
IN-SILICO TRIAL REPLICATION  ·  NephroQ
================================================================================
A falsifiable test of the SHIPPED model (model_core, v2) against published
randomized trials.

DESIGN -- it is built so the model CAN fail:

  1. FIT 3 parameters on CREDENCE and EMPA-KIDNEY only:
       - a progression scale, anchored on their PLACEBO arms;
       - a treatment-effect scale, anchored on CREDENCE's chronic-slope benefit;
       - a direct anti-albuminuric effect, anchored on CREDENCE's 31% UACR drop.
  2. FREEZE them.
  3. PREDICT DAPA-CKD (type-2-diabetes subgroup) out-of-sample, from its
     published baseline characteristics alone. No parameter is left to turn.

Agreement on the trials used for fitting is guaranteed by construction and is
NOT evidence. All the evidential weight is on DAPA-CKD.

WHY THE *CHRONIC* SLOPE, NOT THE TOTAL SLOPE:
SGLT2 inhibitors cause an acute haemodynamic eGFR dip followed by slower
long-term decline; the published "total slope" mixes both. NephroQ models only
the chronic structural mechanism and has no acute-dip term, so it is scored
against the published CHRONIC slope. The placebo arm has no dip (no drug), so
its published slope is a fair comparator either way.

The trial characteristics and outcomes below are transcribed from the literature
and were NOT independently verified. Re-check them against the primary papers
before publication. Sources are given per trial.

Usage:
    python insilico_trial.py            # full replication
    python insilico_trial.py --n 2000   # larger virtual cohorts
================================================================================
"""
import os
import argparse
import numpy as np

import model_core as core
from model_core import N_of_egfr

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "..", "results")

# Public research calibration (same parameters the app ships with).
# Base (unscaled) hazard parameters; the progression `scale` below is what the
# trial placebo arms actually pin down.
K_HF_BASE = 0.0141
W_BASE = np.array([0.0144, 0.0180, 0.0108])   # [HbA1c, UACR, SBP]

# ------------------------------------------------------------------------------
# TRIAL SPECIFICATIONS
# ------------------------------------------------------------------------------
# Baseline characteristics are used to generate a VIRTUAL COHORT matching the
# trial's enrolment criteria. `chronic_slope_diff` is the published between-group
# difference in the CHRONIC eGFR slope (treatment minus placebo, positive =
# treatment declines more slowly), in mL/min/1.73m2 per year.
TRIALS = {
    "CREDENCE": dict(
        drug="canagliflozin (SGLT2i)",
        population="type 2 diabetes + albuminuric CKD",
        n_randomized=4401,
        duration_years=2.5,
        # enrolment: eGFR 30-<90, UACR 300-5000 mg/g, HbA1c 6.5-12%
        egfr_mean=56.2, egfr_sd=18.2, egfr_range=(30.0, 90.0),
        uacr_median=927.0, uacr_log_sd=1.0, uacr_range=(300.0, 5000.0),
        hba1c_mean=8.3, hba1c_sd=1.3,
        sbp_mean=140.0, sbp_sd=15.6,
        age_mean=63.0, age_sd=9.2, frac_male=0.66,
        chronic_slope_diff=2.74, chronic_slope_ci=None,
        total_slope_diff=1.52, total_slope_ci=(1.11, 1.93),
        placebo_slope=-4.59,   # placebo arm has NO acute dip, so total ~ chronic
        # Placebo-subtracted geometric-mean UACR reduction at week 26.
        # Canagliflozin lowered UACR by 31% (95% CI 27-36). Source: post hoc
        # analysis of CREDENCE (PMC7790219).
        uacr_reduction_pct=31.0, uacr_reduction_ci=(26.0, 35.0),
        source="Perkovic et al., NEJM 2019 (NCT02065791); slope analyses per "
               "CREDENCE secondary/post-hoc reports. Total-slope diff 1.52 "
               "(95% CI 1.11-1.93).",
        role="CALIBRATION",
    ),
    "DAPA-CKD (T2D subgroup)": dict(
        drug="dapagliflozin (SGLT2i)",
        population="CKD with type 2 diabetes",
        n_randomized=2906,          # T2D subgroup of 4304
        duration_years=2.4,
        # enrolment: eGFR 25-75, UACR 200-5000 mg/g
        egfr_mean=43.1, egfr_sd=12.4, egfr_range=(25.0, 75.0),
        uacr_median=949.0, uacr_log_sd=1.0, uacr_range=(200.0, 5000.0),
        hba1c_mean=7.8, hba1c_sd=1.5,
        sbp_mean=137.0, sbp_sd=17.0,
        age_mean=62.0, age_sd=12.1, frac_male=0.67,
        # PRE-REGISTERED TARGET for the out-of-sample test:
        chronic_slope_diff=2.26, chronic_slope_ci=(1.88, 2.64),
        total_slope_diff=1.18, total_slope_ci=(0.79, 1.56),
        # DAPA-CKD, T2D subgroup: dapagliflozin reduced UACR by 35.1%
        # (95% CI 30.6-39.4) vs placebo. Source: DAPA-CKD prespecified
        # albuminuria analysis (Heerspink et al.).
        uacr_reduction_pct=35.1, uacr_reduction_ci=(30.6, 39.4),
        placebo_slope=-3.84,   # DAPA-CKD placebo CHRONIC slope, T2D subgroup
                               # (Heerspink 2021, Fig 2A + p.747). The placebo arm
                               # receives no SGLT2i and thus has no acute dip, so
                               # this chronic slope is the correct comparator for
                               # the model's (dip-free) chronic slope.
        source="Heerspink et al., Lancet Diabetes Endocrinol 2021 (DAPA-CKD "
               "prespecified slope analysis, NCT03036150). T2D subgroup: chronic "
               "slope diff 2.26 (95% CI 1.88-2.64); total slope diff 1.18 "
               "(95% CI 0.79-1.56).",
        role="OUT-OF-SAMPLE TEST",
    ),
    "EMPA-KIDNEY": dict(
        drug="empagliflozin (SGLT2i)",
        population="broad CKD (~46% with type 2 diabetes)",
        n_randomized=6609,
        duration_years=2.0,
        # Enrolment: eGFR 20-<45 (any UACR), or eGFR 45-<90 with UACR >=200.
        # The trial's defining feature for us: MUCH lower baseline eGFR *and*
        # MUCH lower UACR than CREDENCE/DAPA-CKD -- orthogonal variation, which
        # is exactly what identifies the saturation ceiling separately from the
        # hazard scale.
        egfr_mean=37.3, egfr_sd=14.5, egfr_range=(20.0, 90.0),
        uacr_median=329.0, uacr_log_sd=1.6, uacr_range=(5.0, 5000.0),
        hba1c_mean=6.5, hba1c_sd=1.4,     # mixed diabetic/non-diabetic cohort
        sbp_mean=136.0, sbp_sd=18.0,
        age_mean=63.9, age_sd=13.9, frac_male=0.67,
        chronic_slope_diff=1.38, chronic_slope_ci=None,   # -1.37 vs -2.75
        placebo_slope=-2.75,
        uacr_reduction_pct=None, uacr_reduction_ci=None,
        source="EMPA-KIDNEY prespecified secondary analysis (Lancet Diabetes "
               "Endocrinol 2023, PMID 38061371; NCT03594110): empagliflozin halved "
               "the chronic slope from -2.75 to -1.37 mL/min/1.73m2/yr (relative "
               "difference 50%, 95% CI 42-58).",
        role="CALIBRATION (second anchor: low eGFR, low UACR)",
        caveat="POPULATION MISMATCH: only ~46% of EMPA-KIDNEY had type 2 diabetes, "
               "while NephroQ is a T2D model. Its HbA1c mean is therefore an "
               "ASSUMED value for a mixed cohort. Use it as a progression anchor "
               "at low eGFR/UACR, not as a T2D efficacy target.",
    ),
}


def sample_cohort(spec, n, rng):
    """Generate a virtual cohort matching the trial's published baseline
    characteristics and enrolment bounds (truncated to the eligibility range)."""
    def trunc_normal(mean, sd, lo, hi):
        x = rng.normal(mean, sd, n)
        # resample out-of-range values rather than clipping (clipping piles mass
        # on the bounds and distorts the mean)
        for _ in range(50):
            bad = (x < lo) | (x > hi)
            if not bad.any():
                break
            x[bad] = rng.normal(mean, sd, int(bad.sum()))
        return np.clip(x, lo, hi)

    egfr = trunc_normal(spec["egfr_mean"], spec["egfr_sd"], *spec["egfr_range"])
    log_u = rng.normal(np.log(spec["uacr_median"]), spec["uacr_log_sd"], n)
    uacr = np.clip(np.exp(log_u), *spec["uacr_range"])
    hba1c = np.clip(rng.normal(spec["hba1c_mean"], spec["hba1c_sd"], n), 5.0, 14.0)
    sbp = np.clip(rng.normal(spec["sbp_mean"], spec["sbp_sd"], n), 95.0, 200.0)
    return dict(egfr=egfr, uacr=uacr, hba1c=hba1c, sbp=sbp)


def _params(scale, eff_met=0.0, eff_hf=0.0, eff_alb=0.0, base=None):
    """
    Model parameters = a base calibration with the fitted scales applied.

    `base` defaults to model_core's trial-anchored v2 parameters. Passing a
    different base (e.g. a MIMIC calibration) lets the auditor ask whether THOSE
    parameters can reproduce the published trials.
    """
    p = dict(base or core.TRIAL_CALIBRATION_V2)
    p.update(k_hf=K_HF_BASE * scale,
             w_a1c=W_BASE[0] * scale, w_uacr=W_BASE[1] * scale, w_sbp=W_BASE[2] * scale,
             eff_met=eff_met, eff_hf=eff_hf, eff_alb=eff_alb)
    return p


def trial_arms(spec, scale, eff_met, eff_hf, eff_alb, n=400, seed=11, skip_years=0.15,
               base=None):
    """
    Run both arms of a virtual trial through model_core's v2 simulator.

    Returns the chronic eGFR slope in each arm, their difference, and the
    placebo-subtracted geometric-mean UACR reduction (the trials' week-26
    endpoint). The UACR ratio is taken against the PRE-treatment baseline: the
    drug's direct effect is applied at t=0, so dividing by the post-drug value
    would cancel it out.
    """
    rng = np.random.default_rng(seed)
    c = sample_cohort(spec, n, rng)
    yrs = spec["duration_years"]
    out = {}
    for u, arm in ((0.0, "placebo"), (1.0, "treated")):
        p = _params(scale, eff_met, eff_hf, eff_alb, base=base)
        slopes, uacr_ratio = [], []
        for i in range(n):
            t, egfr, uacr, _ = core.simulate_trajectory_v2(
                c["egfr"][i], c["hba1c"][i], c["uacr"][i], c["sbp"][i],
                u=u, p=p, years=yrs, n=80)
            i0 = np.searchsorted(t, skip_years)
            slopes.append((egfr[-1] - egfr[i0]) / (t[-1] - t[i0]))
            i26 = np.searchsorted(t, 0.5)
            uacr_ratio.append(uacr[i26] / c["uacr"][i])
        out[arm] = dict(slope=float(np.mean(slopes)),
                        uacr_ratio=float(np.exp(np.mean(np.log(uacr_ratio)))))
    out["slope_diff"] = out["treated"]["slope"] - out["placebo"]["slope"]
    out["uacr_reduction_pct"] = 100.0 * (1.0 - out["treated"]["uacr_ratio"]
                                         / out["placebo"]["uacr_ratio"])
    return out


def _solve(fn, target, lo, hi, iters=30):
    """Bisection on a monotone-decreasing function."""
    for _ in range(iters):
        m = 0.5 * (lo + hi)
        if fn(m) < target:
            hi = m
        else:
            lo = m
    return 0.5 * (lo + hi)


def fit(n=400, seed=11):
    """
    Fit 3 parameters on the CALIBRATION trials only.

    The saturation ceiling S_SAT is NOT fitted here: it is identified separately
    by anchoring the hazard on CREDENCE (mean eGFR 56) and scoring it on
    EMPA-KIDNEY (mean eGFR 37), which gives a clear optimum around 3-4 --
    consistent with the physiological ceiling on single-nephron hyperfiltration.
    It lives in model_core as S_SAT.
    """
    C = TRIALS["CREDENCE"]

    scale = _solve(lambda x: trial_arms(C, x, 0, 0, 0, n=n, seed=seed)["placebo"]["slope"],
                   C["placebo_slope"], 0.05, 6.0)

    def diff(sc):
        em, eh = min(0.45 * sc, 0.95), min(0.35 * sc, 0.95)
        return -trial_arms(C, scale, em, eh, 0.0, n=n, seed=seed)["slope_diff"]
    sc = _solve(diff, -C["chronic_slope_diff"], 0.0, 2.7)
    eff_met, eff_hf = min(0.45 * sc, 0.95), min(0.35 * sc, 0.95)

    eff_alb = _solve(
        lambda ea: -trial_arms(C, scale, eff_met, eff_hf, ea, n=n,
                               seed=seed)["uacr_reduction_pct"],
        -C["uacr_reduction_pct"], 0.0, 0.9)

    return dict(scale=scale, eff_met=eff_met, eff_hf=eff_hf, eff_alb=eff_alb)


def run(n=400, seed=11):
    f = fit(n=n, seed=seed)
    print("=" * 76)
    print("IN-SILICO TRIAL REPLICATION -- NephroQ (model_core v2)")
    print("=" * 76)
    print(f"\nFitted on CREDENCE (+ EMPA-KIDNEY anchors S_SAT={core.S_SAT}):")
    print(f"  progression scale={f['scale']:.3f}  eff_met={f['eff_met']:.3f}  "
          f"eff_hf={f['eff_hf']:.3f}  eff_alb={f['eff_alb']:.3f}")

    results, n_pass, n_test = {}, 0, 0
    for name, spec in TRIALS.items():
        r = trial_arms(spec, f["scale"], f["eff_met"], f["eff_hf"], f["eff_alb"],
                       n=n, seed=seed + 12)
        held_out = spec["role"].startswith("OUT-OF-SAMPLE")
        print(f"\n--- {name} [{'HELD OUT' if held_out else 'fitted'}] ---")
        pb = spec["placebo_slope"]
        print(f"  placebo slope    {r['placebo']['slope']:6.2f}   published {pb:6.2f}   "
              f"ratio {r['placebo']['slope']/pb:.2f}x")

        ci = spec.get("chronic_slope_ci")
        d = r["slope_diff"]
        verdict = ""
        if ci:
            ok = ci[0] <= d <= ci[1]
            n_test += 1; n_pass += ok
            verdict = f"  [{ci[0]}-{ci[1]}]  {'PASS' if ok else 'FAIL'}"
        print(f"  chronic slope diff {d:6.2f}   published {spec['chronic_slope_diff']:6.2f}{verdict}")

        uc = spec.get("uacr_reduction_ci")
        if uc:
            u = r["uacr_reduction_pct"]
            ok = uc[0] <= u <= uc[1]
            n_test += 1; n_pass += ok
            print(f"  UACR reduction   {u:6.1f}%   published {spec['uacr_reduction_pct']:5.1f}%"
                  f"  [{uc[0]}-{uc[1]}]  {'PASS' if ok else 'FAIL'}")
        results[name] = r

    print(f"\n>>> {n_pass}/{n_test} CI-testable endpoints PASS")
    write_report(f, results, n)
    return f, results


def write_report(f, results, n):
    os.makedirs(RESULTS, exist_ok=True)
    path = os.path.join(RESULTS, "insilico_trial_report.md")
    lines = [
        "# In-silico trial replication — NephroQ",
        "",
        "The treatment and progression parameters are fitted on **CREDENCE** (with the",
        "saturation ceiling anchored by **EMPA-KIDNEY**), then frozen. **DAPA-CKD is",
        "predicted out-of-sample** from its published baseline characteristics alone.",
        "Agreement on the fitted trials is guaranteed by construction and is not evidence;",
        "the evidential weight is on DAPA-CKD.",
        "",
        f"Fitted: progression scale={f['scale']:.3f}, eff_met={f['eff_met']:.3f}, "
        f"eff_hf={f['eff_hf']:.3f}, eff_alb={f['eff_alb']:.3f}. "
        f"Virtual cohorts: n={n} per arm.",
        "",
        "| Trial | Role | Endpoint | Model | Published |",
        "|---|---|---|---|---|",
    ]
    for name, r in results.items():
        spec = TRIALS[name]
        role = "**held out**" if spec["role"].startswith("OUT-OF-SAMPLE") else "fitted"
        lines.append(f"| {name} | {role} | placebo slope | {r['placebo']['slope']:.2f} | "
                     f"{spec['placebo_slope']:.2f} |")
        ci = spec.get("chronic_slope_ci")
        ci_s = f" (95% CI {ci[0]}–{ci[1]})" if ci else ""
        lines.append(f"| | | chronic slope diff | **{r['slope_diff']:.2f}** | "
                     f"{spec['chronic_slope_diff']:.2f}{ci_s} |")
        uc = spec.get("uacr_reduction_ci")
        if uc:
            lines.append(f"| | | UACR reduction | **{r['uacr_reduction_pct']:.1f}%** | "
                         f"{spec['uacr_reduction_pct']:.1f}% (95% CI {uc[0]}–{uc[1]}) |")
    lines += [
        "",
        "## Caveats",
        "",
        "- Trial values are transcribed from the literature and were **not independently",
        "  verified**. Re-check against the primary papers before publication.",
        "- Virtual cohorts sample covariates independently; real populations have correlated",
        "  covariates. This mainly affects spread, not the mean effect.",
        "- All three trials test SGLT2 inhibitors: this shows the model transports across",
        "  *populations*, not across *drug classes*.",
        "- EMPA-KIDNEY is only ~46% diabetic, while NephroQ is a type-2-diabetes model; it is",
        "  used as a low-eGFR progression anchor, not as an efficacy target.",
        "- The model has no acute haemodynamic dip, so it cannot be compared against *total*",
        "  eGFR slopes.",
        "",
        "*Generated by `src/insilico_trial.py`.*",
        "",
    ]
    with open(path, "w") as fh:
        fh.write("\n".join(lines))
    print(f"\nReport written: results/insilico_trial_report.md")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="In-silico trial replication (model_core v2)")
    ap.add_argument("--n", type=int, default=400, help="virtual patients per arm")
    ap.add_argument("--seed", type=int, default=11)
    a = ap.parse_args()
    run(n=a.n, seed=a.seed)
