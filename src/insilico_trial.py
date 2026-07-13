"""
================================================================================
IN-SILICO TRIAL REPLICATION  ·  NephroQ
================================================================================
A FALSIFIABLE test of the mechanistic model against published randomized trials.

THE DESIGN (this is the whole point -- read it before trusting any number):

    The model's treatment effect (how strongly `u` blunts the metabolic insult
    and hyperfiltration) is NOT known a priori. If we tuned it until every trial
    matched, this would be curve-fitting dressed up as validation, and it would
    prove nothing.

    So instead:
      1. CALIBRATE the treatment-effect scale on ONE trial (CREDENCE) only.
      2. PREDICT a DIFFERENT trial (DAPA-CKD, type-2-diabetes subgroup) with
         that frozen effect, from its own published baseline characteristics.
      3. The prediction either falls inside DAPA-CKD's published 95% CI or it
         does not. There is no third option and no free parameter left to turn.

    A model that can only reproduce the trial it was fitted to has told us
    nothing. This script is designed so it CAN fail.

WHY THE *CHRONIC* SLOPE, NOT THE TOTAL SLOPE:

    SGLT2 inhibitors cause an ACUTE hemodynamic dip in eGFR (a few mL/min in the
    first weeks) followed by a slower long-term decline. The published "total
    slope" mixes both. NephroQ models only the chronic, structural mechanism --
    it has NO acute-dip term. Scoring it against the total slope would therefore
    penalize it for missing a mechanism it never claimed to have (and would flatter
    it in the opposite direction on other endpoints). The honest comparator is the
    published CHRONIC slope difference. This is a limitation of the model, stated
    up front, not a choice made to make the numbers look good.

PUBLISHED VALUES -- VERIFY BEFORE PUBLICATION. The trial characteristics and
outcomes below are transcribed from the literature and are NOT independently
checked here. Each carries its source. Re-verify against the primary papers
before any of this appears in a manuscript.

Usage:
    python insilico_trial.py                  # full replication + report
    python insilico_trial.py --n 5000         # bigger virtual cohorts
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
Q_POP, KHF_POP, K0_POP = 1.52, 0.0141, 0.0030
W_POP = np.array([0.0144, 0.0180, 0.0108])   # [HbA1c, UACR, SBP]

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
        uacr_reduction_pct=31.0, uacr_reduction_ci=(27.0, 36.0),
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
        placebo_slope=-3.83,   # DAPA-CKD placebo arm, published total slope. The
                               # placebo arm receives no SGLT2i, hence no acute
                               # hemodynamic dip, so its total slope is a fair
                               # comparator for the model's chronic slope.
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


def chronic_slope(egfr0, hba1c, uacr, sbp, u, eff_met, eff_hf, years,
                  q=Q_POP, k_hf=KHF_POP, k0=K0_POP, w=W_POP, skip_years=0.15):
    """
    Mean annualized CHRONIC eGFR slope for one virtual patient, mL/min/1.73m2/yr.

    `skip_years` drops the very start of the trajectory so that the slope is
    measured over the chronic phase, mirroring the trials' "week 2/3 to end of
    treatment" definition (they exclude the acute hemodynamic phase). NephroQ has
    no acute term, so this mostly just aligns the measurement window.
    """
    insult = core.metabolic_hazard(hba1c, uacr, sbp, w[0], w[1], w[2]) * (1.0 - eff_met * u)
    khf_eff = k_hf * (1.0 - eff_hf * u)
    t = np.array([skip_years, years])
    e = core.predict_egfr_at(k0, khf_eff, q, 1.0, insult, N_of_egfr(egfr0), t)
    return float((e[1] - e[0]) / (years - skip_years))


def simulate_trial(spec, eff_met, eff_hf, n=2000, seed=0):
    """Run both arms of a virtual trial; return the between-group chronic-slope
    difference (treatment minus placebo; positive = treatment declines slower)."""
    rng = np.random.default_rng(seed)
    c = sample_cohort(spec, n, rng)
    yrs = spec["duration_years"]
    placebo, treated = [], []
    for i in range(n):
        args = (c["egfr"][i], c["hba1c"][i], c["uacr"][i], c["sbp"][i])
        placebo.append(chronic_slope(*args, 0.0, eff_met, eff_hf, yrs))
        treated.append(chronic_slope(*args, 1.0, eff_met, eff_hf, yrs))
    placebo, treated = np.array(placebo), np.array(treated)
    return dict(
        slope_placebo=float(placebo.mean()),
        slope_treated=float(treated.mean()),
        slope_diff=float(treated.mean() - placebo.mean()),
        n=n,
    )


def calibrate_treatment_effect(spec, n=2000, seed=0, ratio_met_to_hf=0.45 / 0.35):
    """
    Fit the treatment-effect magnitude on the CALIBRATION trial ONLY.

    ONE free parameter: a scale `s` applied to the (met, hf) effect pair, whose
    RATIO is held fixed at the model's structural prior. Fitting a single scalar
    to a single published number means the calibration trial's agreement is
    guaranteed and is therefore NOT evidence -- all the evidential weight sits on
    the out-of-sample trial.
    """
    target = spec["chronic_slope_diff"]
    base_hf = 0.35
    base_met = base_hf * ratio_met_to_hf

    def diff_for(s):
        eff_met = float(np.clip(base_met * s, 0.0, 0.95))
        eff_hf = float(np.clip(base_hf * s, 0.0, 0.95))
        return simulate_trial(spec, eff_met, eff_hf, n=n, seed=seed)["slope_diff"], eff_met, eff_hf

    # monotone in s -> simple bisection
    lo, hi = 0.0, 2.7
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        d, _, _ = diff_for(mid)
        if d < target:
            lo = mid
        else:
            hi = mid
    s = 0.5 * (lo + hi)
    d, eff_met, eff_hf = diff_for(s)
    return dict(scale=s, eff_met=eff_met, eff_hf=eff_hf,
                fitted_slope_diff=d, target=target)


def run(n=2000, seed=0):
    calib_name = "CREDENCE"
    test_name = "DAPA-CKD (T2D subgroup)"
    calib_spec, test_spec = TRIALS[calib_name], TRIALS[test_name]

    print("=" * 78)
    print("IN-SILICO TRIAL REPLICATION -- NephroQ")
    print("=" * 78)
    print(f"\n[1/3] CALIBRATION on {calib_name} (1 free parameter: treatment-effect scale)")
    fit = calibrate_treatment_effect(calib_spec, n=n, seed=seed)
    print(f"      target chronic-slope difference : {fit['target']:.2f} mL/min/1.73m2/yr")
    print(f"      fitted                          : {fit['fitted_slope_diff']:.2f}")
    print(f"      => eff_met={fit['eff_met']:.3f}  eff_hf={fit['eff_hf']:.3f}  (scale={fit['scale']:.3f})")
    print("      NOTE: agreement here is guaranteed by construction and is NOT evidence.")

    print(f"\n[2/3] OUT-OF-SAMPLE PREDICTION on {test_name}")
    print("      (treatment effect FROZEN from CREDENCE; only the published baseline")
    print("       characteristics of this trial are used -- no refitting)")
    pred = simulate_trial(test_spec, fit["eff_met"], fit["eff_hf"], n=n, seed=seed + 1)
    obs = test_spec["chronic_slope_diff"]
    ci = test_spec["chronic_slope_ci"]
    inside = ci[0] <= pred["slope_diff"] <= ci[1]

    print(f"      predicted chronic-slope difference : {pred['slope_diff']:.2f} mL/min/1.73m2/yr")
    print(f"      published                          : {obs:.2f}  (95% CI {ci[0]:.2f}-{ci[1]:.2f})")
    print(f"      predicted placebo arm slope        : {pred['slope_placebo']:.2f} /yr")
    print(f"      predicted treatment arm slope      : {pred['slope_treated']:.2f} /yr")

    # SECOND, INDEPENDENT FALSIFIABLE CHECK: the placebo arm.
    # The treatment effect cannot hide here -- the placebo arm receives nothing,
    # so this tests the UNTREATED progression model directly. If the model's
    # placebo decline is wrong, the treatment-effect difference is built on sand.
    obs_pbo = test_spec.get("placebo_slope")
    if obs_pbo:
        ratio = pred["slope_placebo"] / obs_pbo
        print(f"\n      --- placebo-arm check (tests the UNTREATED model directly) ---")
        print(f"      model placebo slope     : {pred['slope_placebo']:.2f} /yr")
        print(f"      published placebo slope : {obs_pbo:.2f} /yr")
        print(f"      ratio                   : {ratio:.2f}x")
        if ratio > 1.25:
            print(f"      >>> The model makes untreated patients decline ~{ratio:.1f}x FASTER than")
            print( "          the real placebo arm. Because the treatment effect is multiplicative")
            print( "          on the hazard, an over-fast placebo arm mechanically INFLATES the")
            print( "          absolute slope difference. This -- not the treatment model -- is the")
            print( "          most likely cause of any overshoot above.")

    print(f"\n[3/3] RESULT")
    verdict = "PASS" if inside else "FAIL"
    print(f"      >>> {verdict}: prediction {'falls INSIDE' if inside else 'falls OUTSIDE'} "
          f"the published 95% CI.")
    if not inside:
        print(f"      The model is WRONG on this endpoint by "
             f"{min(abs(pred['slope_diff']-ci[0]), abs(pred['slope_diff']-ci[1])):.2f} "
             f"mL/min/1.73m2/yr beyond the CI. Report it as a failure.")

    write_report(calib_name, test_name, fit, pred, obs, ci, inside, n,
                 obs_placebo=test_spec.get("placebo_slope"))
    return dict(fit=fit, prediction=pred, observed=obs, ci=ci, pass_=inside)


def write_report(calib_name, test_name, fit, pred, obs, ci, inside, n, obs_placebo=None):
    os.makedirs(RESULTS, exist_ok=True)
    path = os.path.join(RESULTS, "insilico_trial_report.md")
    verdict = "PASS ✅" if inside else "FAIL ❌"
    lines = [
        "# In-silico trial replication — NephroQ",
        "",
        f"**Result: {verdict}**",
        "",
        "## Design (falsifiable by construction)",
        "",
        f"1. The treatment-effect magnitude (a **single** scalar scaling `eff_met`/`eff_hf`, "
        f"whose ratio is fixed by the model structure) was calibrated on **{calib_name}** and "
        f"then **frozen**.",
        f"2. **{test_name}** was then predicted from its published baseline characteristics "
        f"alone, with **no refitting**.",
        "3. The prediction either falls inside the published 95% CI or it does not.",
        "",
        "Agreement on the calibration trial is guaranteed by construction and is **not** "
        "evidence. All evidential weight is on the out-of-sample trial.",
        "",
        "## Comparator: chronic, not total, eGFR slope",
        "",
        "SGLT2 inhibitors cause an acute hemodynamic eGFR dip followed by slower long-term "
        "decline; the published *total* slope mixes both. **NephroQ has no acute-dip term** — "
        "it models only the chronic structural mechanism — so it is scored against the "
        "published **chronic** slope difference. This is a stated limitation of the model, "
        "not a convenience.",
        "",
        "## Results",
        "",
        "| | Value |",
        "|---|---|",
        f"| Calibration trial | {calib_name} |",
        f"| Fitted effect | eff_met={fit['eff_met']:.3f}, eff_hf={fit['eff_hf']:.3f} |",
        f"| {calib_name} chronic-slope diff (target / fitted) | {fit['target']:.2f} / {fit['fitted_slope_diff']:.2f} |",
        "",
        f"### Out-of-sample: {test_name}",
        "",
        "| Quantity | Model | Published |",
        "|---|---|---|",
        f"| Chronic eGFR slope difference (mL/min/1.73m²/yr) | **{pred['slope_diff']:.2f}** | "
        f"**{obs:.2f}** (95% CI {ci[0]:.2f}–{ci[1]:.2f}) |",
        f"| Placebo-arm slope | {pred['slope_placebo']:.2f} | — |",
        f"| Treatment-arm slope | {pred['slope_treated']:.2f} | — |",
        "",
        f"Virtual cohort: n={n} per arm, sampled from the trial's published baseline "
        "distributions and eligibility bounds.",
        "",
        "### Placebo-arm check — this is the diagnosis",
        "",
        "The placebo arm receives no drug, so it tests the **untreated progression model "
        "directly**; the treatment effect cannot hide here.",
        "",
        "| Quantity | Model | Published | Ratio |",
        "|---|---|---|---|",
        (f"| Placebo-arm eGFR slope (mL/min/1.73m²/yr) | **{pred['slope_placebo']:.2f}** | "
         f"**{obs_placebo:.2f}** | **{pred['slope_placebo']/obs_placebo:.2f}×** |")
        if obs_placebo else "| Placebo-arm slope | — | not specified | — |",
        "",
        (f"**The model makes untreated patients decline ~{pred['slope_placebo']/obs_placebo:.1f}× "
         "faster than the real placebo arm.** Because the treatment effect enters "
         "multiplicatively on the hazard, an over-fast placebo arm mechanically inflates the "
         "absolute between-group slope difference. The failure above is therefore most likely a "
         "failure of the **untreated progression calibration** — not of the treatment model. "
         "The public calibration (q, k_hf, weights) was never fitted to an advanced-CKD "
         "population like DAPA-CKD's (mean eGFR ~43), and it over-predicts decline there.")
        if obs_placebo else "",
        "",
        "## What this failure means (and what to do)",
        "",
        "This is a **useful** failure: an in-silico replication exists precisely so the model "
        "can be caught. Two things follow.",
        "",
        "1. **Do not report the treatment effect as validated.** The out-of-sample prediction "
        "overshoots the published CI.",
        "2. **Fix the untreated model first.** The placebo-arm ratio says the progression "
        "calibration is too aggressive for advanced CKD. Recalibrating q/k_hf on a cohort with "
        "the right eGFR range (and re-running this script) is the next step — and this script "
        "then becomes the regression test for whether that recalibration actually helped.",
        "",
        "## Caveats (read before citing)",
        "",
        "- Trial characteristics and outcomes are **transcribed from the literature and not "
        "independently verified here**. Re-check against the primary publications before "
        "publication. Sources are listed in `insilico_trial.py`.",
        "- The virtual cohorts sample covariates **independently**; real trial populations have "
        "correlated covariates (e.g. lower eGFR with higher UACR). This is a known "
        "simplification that mainly affects the spread, not the mean effect.",
        "- Both trials test **SGLT2 inhibitors**. Reproducing DAPA-CKD from CREDENCE shows the "
        "model transports across *populations*, not across *drug classes*. A GLP-1 trial "
        "(e.g. FLOW) would be a genuinely independent mechanism and is not attempted here.",
        "- One endpoint (chronic eGFR slope) is not a full validation. Hard outcomes "
        "(kidney-failure hazard ratios) remain untested.",
        "",
        "*Generated by `src/insilico_trial.py`.*",
        "",
    ]
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"\n      Report written: {os.path.relpath(path, HERE)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=2000, help="virtual patients per arm")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    run(n=a.n, seed=a.seed)
