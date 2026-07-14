"""
================================================================================
AUDIT A MIMIC CALIBRATION  ·  NephroQ
================================================================================
Run this the moment `calibrate_mimic.py` finishes. It answers the one question
that matters: does a calibration fitted on hospital records agree with what
randomized trials actually observed?

These are two INDEPENDENT sources of evidence:

  - the TRIAL-ANCHORED reference (model_core.TRIAL_CALIBRATION_V2), whose
    progression rate is pinned by the placebo arms of CREDENCE and EMPA-KIDNEY;
  - the MIMIC calibration, fitted on ICU/hospital creatinine trajectories.

Three checks, in order of severity:

  1. DID THE OPTIMIZER ACTUALLY MOVE? A frozen optimizer returns round-number
     parameters and a bootstrap with ~zero variance. If that happened, nothing
     below means anything.

  2. HOW FAR IS MIMIC FROM THE TRIALS? Reported as a hazard ratio. A ratio > 1
     means MIMIC thinks patients decline FASTER than the real trial placebo arms
     do. The most likely reason is not biology: it is that the MIMIC index date
     is the first available hospital creatinine, which is often drawn during an
     acute episode. This ratio therefore *quantifies* that bias.

  3. DOES THE MIMIC CALIBRATION SURVIVE THE IN-SILICO TRIALS? This is the real
     judge. We take the MIMIC parameters and ask them to reproduce the published
     placebo-arm slopes of CREDENCE, DAPA-CKD and EMPA-KIDNEY. A calibration that
     cannot reproduce a real trial's placebo arm should not be used to project
     patients, no matter how good its internal chi2/n looks.

Usage:
    python audit_calibration.py
    python audit_calibration.py --json ../calibration/mimic_calibration.json
================================================================================
"""
import os
import json
import argparse

import numpy as np

import model_core as core
import insilico_trial as it

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_JSON = os.path.join(HERE, "..", "calibration", "mimic_calibration.json")

# The unscaled base the trial anchoring scales from (see insilico_trial.py).
K_HF_BASE = 0.0141
W_BASE = np.array([0.0144, 0.0180, 0.0108])


def load(path):
    with open(path) as f:
        return json.load(f)


def check_optimizer(cal):
    print("\n[1] DID THE OPTIMIZER ACTUALLY MOVE?")
    q, khf = cal["q"], cal["k_hf"]
    w = [cal["w_a1c"], cal["w_uacr"], cal["w_sbp"]]

    # the old frozen optimum returned exactly these
    frozen = dict(q=1.5, k_hf=0.012, w_a1c=0.014, w_uacr=0.018, w_sbp=0.011)
    stuck = all(abs(cal[k] - v) < 1e-6 for k, v in frozen.items())

    boot = cal.get("bootstrap_params") or []
    if boot:
        qs = np.array([b["q"] for b in boot])
        ks = np.array([b["k_hf"] for b in boot])
        q_sd, k_sd = float(qs.std()), float(ks.std())
    else:
        q_sd = k_sd = None

    print(f"    q     = {q:.4f}")
    print(f"    k_hf  = {khf:.5f}")
    print(f"    w     = [{w[0]:.5f}, {w[1]:.5f}, {w[2]:.5f}]")
    if q_sd is not None:
        print(f"    bootstrap spread: sd(q)={q_sd:.4g}  sd(k_hf)={k_sd:.4g}  (n={len(boot)})")

    if stuck:
        print("    >>> FROZEN. These are exactly the initial guess. The fit is meaningless;")
        print("        nothing below should be interpreted. Check x_scale / gtol.")
        return False
    if q_sd is not None and q_sd < 1e-4:
        print("    >>> BOOTSTRAP DEGENERATE (spread ~ 0). The replicates did not move either.")
        return False
    print("    >>> OK: the optimizer moved off its initial guess.")
    return True


def compare_to_trials(cal):
    print("\n[2] HOW FAR IS MIMIC FROM THE TRIAL-ANCHORED REFERENCE?")
    ref = core.TRIAL_CALIBRATION_V2

    # the trial anchoring is a single scale on (k_hf, weights); recover MIMIC's
    # implied scale the same way, so the two are compared on the same axis
    scale_mimic = np.mean([cal["k_hf"] / K_HF_BASE,
                           cal["w_a1c"] / W_BASE[0],
                           cal["w_uacr"] / W_BASE[1],
                           cal["w_sbp"] / W_BASE[2]])
    scale_ref = ref["k_hf"] / K_HF_BASE

    print(f"    {'':22}{'MIMIC':>10}{'trials':>10}{'ratio':>9}")
    print(f"    {'q':22}{cal['q']:10.3f}{ref['q']:10.3f}{cal['q']/ref['q']:9.2f}x")
    print(f"    {'k_hf':22}{cal['k_hf']:10.5f}{ref['k_hf']:10.5f}"
          f"{cal['k_hf']/ref['k_hf']:9.2f}x")
    print(f"    {'w_uacr':22}{cal['w_uacr']:10.5f}{ref['w_uacr']:10.5f}"
          f"{cal['w_uacr']/ref['w_uacr']:9.2f}x")
    print(f"    {'implied hazard scale':22}{scale_mimic:10.3f}{scale_ref:10.3f}"
          f"{scale_mimic/scale_ref:9.2f}x")

    r = scale_mimic / scale_ref
    if r > 1.25:
        print(f"\n    >>> MIMIC says patients decline ~{r:.1f}x FASTER than the real trial")
        print("        placebo arms. The likeliest cause is NOT biology: the MIMIC index date")
        print("        is the first available hospital creatinine, often drawn during an acute")
        print("        episode, which inflates the apparent rate of decline. Read this ratio as")
        print("        a MEASUREMENT OF THAT BIAS, not as a better estimate of progression.")
    elif r < 0.8:
        print(f"\n    >>> MIMIC says patients decline ~{1/r:.1f}x SLOWER than the trials.")
        print("        Do NOT read this as a --chronic-only artifact: that filter selects")
        print("        DECLINING trajectories, so it biases TOWARD progressors -- it would")
        print("        push this ratio UP, not down. A ratio < 1 on this cohort points the")
        print("        other way: the insult covariates carry too little signal (MIMIC's")
        print("        median UACR is ~23 mg/g vs 927 in CREDENCE -- essentially")
        print("        normoalbuminuric), so the fit attributes almost all of the hazard to")
        print("        hyperfiltration and cannot express the albuminuric progression the")
        print("        trials are driven by. The fix is to ANCHOR the insult weights to the")
        print("        trials (--anchor-weights), not to change the cohort filter.")
    else:
        print(f"\n    >>> The two agree within {abs(1-r)*100:.0f}%. Two independent sources of")
        print("        evidence -- hospital records and randomized-trial placebo arms --")
        print("        converging is a strong result. Say so in the manuscript.")
    return scale_mimic


def survives_the_trials(cal, n=250, tol=0.30):
    """The real judge: can the MIMIC parameters reproduce a published placebo arm?"""
    print("\n[3] DOES THE MIMIC CALIBRATION SURVIVE THE IN-SILICO TRIALS?")
    print("    (its parameters are asked to reproduce the PUBLISHED placebo-arm slopes;")
    print("     the placebo arm gets no drug, so nothing can hide there)")

    p = dict(core.TRIAL_CALIBRATION_V2)
    p.update(q=cal["q"], k_hf=cal["k_hf"], w_a1c=cal["w_a1c"],
             w_uacr=cal["w_uacr"], w_sbp=cal["w_sbp"],
             eff_met=0.0, eff_hf=0.0, eff_alb=0.0)

    print(f"\n    {'trial':<26}{'model':>9}{'published':>11}{'ratio':>8}")
    ratios = []
    for name, spec in it.TRIALS.items():
        rng = np.random.default_rng(7)
        c = it.sample_cohort(spec, n, rng)
        yrs = spec["duration_years"]
        slopes = []
        for i in range(n):
            t, egfr, _, _ = core.simulate_trajectory_v2(
                c["egfr"][i], c["hba1c"][i], c["uacr"][i], c["sbp"][i],
                u=0.0, p=p, years=yrs, n=80)
            i0 = np.searchsorted(t, 0.15)
            slopes.append((egfr[-1] - egfr[i0]) / (t[-1] - t[i0]))
        model = float(np.mean(slopes))
        pub = spec["placebo_slope"]
        ratios.append(model / pub)
        print(f"    {name:<26}{model:9.2f}{pub:11.2f}{model/pub:8.2f}x")

    # SECOND ENDPOINT: the treated arm's UACR reduction. The placebo-slope check
    # above tests only the PROGRESSION parameters. A calibration can get those
    # right and still be wrong about albuminuria, which in v2 is an OUTPUT of the
    # same hazard -- so it is an independent constraint, and insilico_trial.py
    # already tests it. The auditor used to ignore it.
    print(f"\n    {'trial (UACR reduction)':<26}{'model':>9}{'published':>11}{'in CI?':>8}")
    uacr_ok = True
    for name, spec in it.TRIALS.items():
        if not spec.get("uacr_reduction_ci"):
            continue
        p_treat = dict(p)
        p_treat.update(eff_met=core.TRIAL_CALIBRATION_V2["eff_met"],
                       eff_hf=core.TRIAL_CALIBRATION_V2["eff_hf"],
                       eff_alb=core.TRIAL_CALIBRATION_V2["eff_alb"])
        arms = it.trial_arms(spec, 1.0, p_treat["eff_met"], p_treat["eff_hf"],
                             p_treat["eff_alb"], n=n, seed=7, base=p)
        u = arms["uacr_reduction_pct"]
        lo, hi = spec["uacr_reduction_ci"]
        ok = lo <= u <= hi
        uacr_ok &= ok
        print(f"    {name:<26}{u:9.1f}{spec['uacr_reduction_pct']:11.1f}"
              f"{'yes' if ok else 'NO':>8}")

    worst = max(abs(np.log(r)) for r in ratios)
    slope_ok = worst < np.log(1.0 + tol)
    print()
    if slope_ok and uacr_ok:
        print(f"    >>> PASSES: the MIMIC parameters reproduce every published placebo arm to")
        print(f"        within {100*tol:.0f}%, AND the treated-arm UACR reduction lands inside the")
        print("        published CIs. This calibration is usable.")
    else:
        if not slope_ok:
            print("    >>> FAILS on the PLACEBO SLOPES: the MIMIC parameters cannot reproduce")
            print("        the untreated progression seen in real trials.")
        if not uacr_ok:
            print("    >>> FAILS on the UACR reduction: the progression parameters distort")
            print("        albuminuria, which in v2 is an output of the same hazard.")
        print("        Do NOT ship these parameters in the app, regardless of how good the")
        print("        internal chi2/n looks -- an internally consistent fit to a biased")
        print("        cohort is still biased. The trial-anchored defaults remain the safer")
        print("        choice until the index date is made AKI-free.")


def quality(cal):
    print("\n[0] QUALITY GATE (from calibrate_mimic.py)")
    st = cal.get("quality_status", "unknown")
    reasons = cal.get("quality_reasons", [])
    print(f"    status: {st}")
    if reasons:
        for r in reasons:
            print(f"      - {r}")
    chi2 = cal.get("chi2_per_n")
    rmse = cal.get("rmse_mL_min")
    scale = cal.get("longitudinal_residual_scale")
    if chi2 is not None:
        print(f"    chi2/n = {chi2:.2f}   rmse = {rmse:.1f} mL/min"
              + (f"   (residual scale {scale:.1f})" if scale else ""))
        if scale:
            print(f"    sanity: (rmse/scale)^2 = {(rmse/scale)**2:.2f} should equal chi2/n")
    print(f"    n_patients = {cal.get('n_patients')}   n_obs = {cal.get('n_obs')}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Audit a MIMIC calibration against the trials")
    ap.add_argument("--json", default=DEFAULT_JSON)
    ap.add_argument("--n", type=int, default=250, help="virtual patients per trial arm")
    ap.add_argument("--tol", type=float, default=0.30,
                    help="How far a modelled placebo slope may sit from the published one "
                         "before the calibration is rejected, as a fraction. 0.30 (i.e. 30%%) "
                         "is a judgement call, not a standard: it is roughly the width of the "
                         "published slope CIs themselves, so a model inside it is not "
                         "distinguishable from the trial. Tighten it for a manuscript.")
    a = ap.parse_args()

    cal = load(a.json)
    print("=" * 74)
    print("CALIBRATION AUDIT  --  MIMIC vs published trials")
    print("=" * 74)
    quality(cal)
    moved = check_optimizer(cal)
    compare_to_trials(cal)
    if moved:
        survives_the_trials(cal, n=a.n, tol=a.tol)
    else:
        print("\n[3] SKIPPED -- the optimizer never moved, so its parameters mean nothing.")
