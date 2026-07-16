"""
External validation against independent cohorts (roadmap block 12).

THREE POPULATIONS, THREE ROLES. NephroQ is not calibrated on one population and
"checked" on another as if they were the same phenomenon. Each cohort plays a
distinct role and occupies a distinct progression regime:

    Cohort          Role in NephroQ
    --------------- --------------------------------------------------------------
    MIMIC-IV        hospital EHR: AKI, irregular sampling, in-hospital dialysis;
                    near-normoalbuminuric. Source of the STRUCTURAL collapse
                    exponent (q ~ 2.9) at low UACR.
    CRIC Phase I    renal progression in MORE ADVANCED CKD (eGFR ~40), moderate
                    albuminuria. A distinct, ambulatory progression regime.
    CRIC Phase III  EARLIER progression (eGFR ~57), prevention / anticipation.
                    The regime where early-detection work would live.
    (SGLT2i trials) macroalbuminuric; source of the ALBUMINURIC term (q = 1.52).

The consequence for validation, and the thing to state plainly: applying the
DEFAULT (trial/MIMIC-anchored) parameters to CRIC is applying one regime's
parameters to another population. If the prediction differs from CRIC's observed
slope, that is a REGIME DIFFERENCE, not a failed validation. The scientifically
correct question is not "do MIMIC's parameters hit CRIC?" but "can the mechanistic
hazard, calibrated IN the CRIC domain, reproduce CRIC's progression?" -- the same
one-hazard-many-populations claim from CHANGELOG Rounds 16-17, extended to a third
regime.

TWO LEVELS OF VALIDATION, and this module is honest about which the data support:

  1. CROSS-REGIME REFERENCE (this file, now): given a cohort's baseline profile,
     what does the model predict under the DEFAULT parameters, and how does that
     compare to CRIC's observed slope? A difference here LOCATES CRIC relative to
     the calibrated regimes; it is not a pass/fail of the model.

  2. PATIENT-LEVEL, IN-DOMAIN (stub below, pending): given per-patient CRIC
     trajectories, calibrate/personalize within the CRIC domain and predict
     held-out future eGFR. This is the strong validation, and it needs the
     individual-level data (requested), not aggregate profiles.

CRIC baseline profiles are transcribed from the provided summary. The observed
slope is VERIFIED against the CRIC literature (see CRIC_OBSERVED_SLOPE).

PRELIMINARY CROSS-REGIME OBSERVATION (aggregate data, corrected):
    Against the VERIFIED CRIC overall diabetic slope (-2.7 mL/min/1.73m2/yr, AJKD
    2020), NephroQ under DEFAULT parameters predicts ~-2.5 (Phase I) and ~-3.1
    (Phase III). Phase I agrees closely; Phase III is modestly steeper than the
    overall reference. Two honest caveats temper this:
      - CRIC does not publish a per-phase slope, so the SAME overall value is the
        reference for both phases. It corresponds to CRIC's overall baseline eGFR
        (~43), closer to Phase I than to Phase III (eGFR ~57); less-advanced
        patients typically decline more slowly, so the overall -2.7 is a weaker
        comparator for Phase III, and the apparent Phase III "gap" may be an
        artifact of that mismatch rather than model error.
      - the estimands differ (mixed-model-with-censoring vs clean chronic slope).
    (An earlier version of this module used -1.83 from a metabolomics substudy;
    that was the wrong reference and has been corrected to the AJKD progression
    value.) This remains a regime-location signal, not a validation pass and not a
    model defect; the in-domain question needs patient-level data.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

import model_core as core


# The role each population plays in NephroQ. Kept as data (not just prose) so the
# distinction is explicit and queryable: three progression REGIMES, not one
# phenomenon measured three times.
POPULATION_ROLES = {
    "MIMIC-IV": "Hospital EHR: AKI, irregular sampling, in-hospital dialysis; "
                "near-normoalbuminuric. Calibrates the structural exponent q~2.9.",
    "CRIC Phase I": "Renal progression in more advanced CKD (eGFR ~40), moderate "
                    "albuminuria; ambulatory regime.",
    "CRIC Phase III": "Earlier progression (eGFR ~57), prevention/anticipation; the "
                      "early-detection regime.",
    "SGLT2i trials": "Macroalbuminuric; calibrates the albuminuric term (q=1.52).",
}


# ------------------------------------------------------------------------------
# CRIC diabetic-subgroup baseline profiles (from the provided aggregate summary)
# ------------------------------------------------------------------------------
CRIC_COHORTS = {
    "phase_I": dict(
        label="CRIC Phase I (advanced CKD, diabetic subgroup)",
        n=1908, egfr=40.7, egfr_sd=12.9, age=59.4, sbp=133.6, hba1c=7.7,
        proteinuria_g_day=0.38, proteinuria_iqr=(0.10, 1.74),
        egfr_under_30_pct=22,
    ),
    "phase_III": dict(
        label="CRIC Phase III (earlier CKD, diabetic subgroup)",
        n=908, egfr=57.2, egfr_sd=11.2, age=64.7, sbp=130.6, hba1c=7.6,
        proteinuria_g_day=0.44, proteinuria_iqr=(0.17, 1.21),
        egfr_under_30_pct=0,
    ),
}

# CRIC diabetic-subgroup eGFR slope, verified against the CRIC progression analysis.
#
# Source: Novel Risk Factors for Progression of Diabetic and Nondiabetic CKD,
# AJKD 2020 (S0272-6386(20)30925-2). Mean eGFR slope, diabetic subgroup:
# -2.7 (SD 4.7) mL/min/1.73m2/yr (non-diabetic -1.4 [SD 3.3]).
#
# IMPORTANT LIMITATIONS on how this number can be used (surfaced when the two
# "phases" were found to be plotted with an identical slope):
#   - This is the OVERALL diabetic slope. CRIC does NOT publish eGFR slopes broken
#     down by recruitment phase (Phase I vs Phase III), so there is no per-phase
#     observed slope to compare against. The overall value is the honest reference
#     for BOTH phases, but it is ONE number applied to two populations, not two
#     independent measurements.
#   - It corresponds to CRIC's overall baseline eGFR (~43). Phase III patients start
#     higher (eGFR ~57) and less advanced; such patients typically decline more
#     slowly, so the overall -2.7 is a WEAKER reference for Phase III than for
#     Phase I. Do not read agreement/disagreement per phase as if -2.7 were each
#     phase's own observed slope.
#   - CRIC's slope is a mixed-model estimate with death/KFRT censoring; NephroQ's is
#     a clean chronic slope. The estimands may not be identical.
#
# The albuminuria-stratified slopes below ARE published subgroup values and are the
# more defensible comparators, since NephroQ's cohorts differ in albuminuria and the
# model itself predicts an albuminuria-dependent slope.
CRIC_OBSERVED_SLOPE = dict(
    value=-2.7,                     # mL/min/1.73m2/yr, diabetic subgroup, OVERALL
    sd=4.7,
    plausible_range=(-3.5, -1.9),   # a conservative mean band (~ +/- 0.15 SD)
    per_phase_available=False,      # CRIC does not publish slope by recruitment phase
    baseline_egfr_of_slope=43.0,    # the eGFR this overall slope corresponds to
    source="CRIC diabetic subgroup, mean eGFR slope -2.7 (SD 4.7) mL/min/1.73m2/yr, "
           "AJKD 2020 S0272-6386(20)30925-2. NOT available per recruitment phase.",
    verified=True,
)

# Albuminuria-stratified diabetic slopes (published subgroup values, AJKD 2020).
CRIC_SLOPE_BY_ALBUMINURIA = {
    "severe":   dict(value=-5.2, sd=5.1),   # severely increased albuminuria
    "youngest": dict(value=-4.9, sd=7.5),   # age <44 (fastest by age)
}


def proteinuria_to_uacr(g_per_day, albumin_fraction_range=(0.5, 0.75),
                        urine_creatinine_g_day=1.0):
    """
    Convert 24-h total proteinuria (g/day) to an approximate UACR (mg/g), as a
    RANGE. This is a coarse equivalence: albumin is ~50-75% of total protein and
    urine creatinine excretion varies (~1 g/day assumed). The range is carried
    through every downstream number so the conversion's uncertainty is never
    hidden. Do not treat the midpoint as exact.
    """
    lo = g_per_day * 1000 * albumin_fraction_range[0] / urine_creatinine_g_day
    hi = g_per_day * 1000 * albumin_fraction_range[1] / urine_creatinine_g_day
    return lo, hi


def predicted_slope(cohort, params=None, years=5.0):
    """
    Model-predicted chronic eGFR slope for a cohort's BASELINE profile, untreated,
    as a range spanning the proteinuria->UACR conversion uncertainty.
    """
    p = dict(params or core.TRIAL_CALIBRATION_V2)
    ulo, uhi = proteinuria_to_uacr(cohort["proteinuria_g_day"])
    slopes = []
    for uacr in (ulo, uhi):
        t, e, _, _ = core.simulate_trajectory_v2(
            cohort["egfr"], cohort["hba1c"], uacr, cohort["sbp"],
            u=0.0, p=p, years=years, n=80)
        i0 = np.searchsorted(t, 0.15)     # drop the first, near-instant point
        slopes.append(float((e[-1] - e[i0]) / (t[-1] - t[i0])))
    return dict(uacr_range=(ulo, uhi),
                slope_range=(min(slopes), max(slopes)),
                slope_mid=float(np.mean(slopes)))


@dataclass
class PopulationValidationResult:
    cohort: str
    n: int
    predicted_slope_range: tuple
    observed_slope: Optional[float]
    observed_range: tuple
    within_observed_range: Optional[bool]
    caveats: list = field(default_factory=list)


def validate_population(cohort_key, params=None,
                        observed=CRIC_OBSERVED_SLOPE) -> PopulationValidationResult:
    """
    Population-level plausibility check: does the model's predicted slope for this
    cohort's baseline profile fall within the observed slope (range)?

    This is NOT the twin's validation. It is the most these aggregate data support,
    and its caveats are attached to the result so they travel with it.
    """
    cohort = CRIC_COHORTS[cohort_key]
    pred = predicted_slope(cohort, params)

    obs_val = observed.get("value")
    obs_range = observed.get("plausible_range")
    within = None
    if obs_val is not None:
        lo, hi = pred["slope_range"]
        within = (lo <= obs_val <= hi) or (min(lo, hi) <= obs_val <= max(lo, hi))
    elif obs_range is not None:
        # overlap of the two ranges
        plo, phi = sorted(pred["slope_range"])
        olo, ohi = sorted(obs_range)
        within = not (phi < olo or plo > ohi)

    caveats = [
        "CROSS-REGIME reference, NOT patient-level validation of the twin. CRIC is a "
        "distinct population/regime; this locates it relative to the default "
        "(trial/MIMIC-anchored) parameters, it does not pass/fail the model.",
        "A mismatch here is expected if CRIC's regime differs from the calibrated "
        "one -- the in-domain question needs per-patient data (see "
        "validate_patient_level).",
        "proteinuria->UACR conversion is approximate; the slope is reported as a range.",
        "CRIC's slope is a mixed-model estimate with death/KFRT censoring; the "
        "model's is a clean chronic slope -- the estimands may not be identical.",
    ]
    if not observed.get("verified", False):
        caveats.insert(0, "OBSERVED SLOPE IS A PLACEHOLDER -- verify against the CRIC "
                          "progression literature before reporting this result.")

    return PopulationValidationResult(
        cohort=cohort["label"], n=cohort["n"],
        predicted_slope_range=pred["slope_range"],
        observed_slope=obs_val, observed_range=obs_range,
        within_observed_range=within, caveats=caveats)


# ------------------------------------------------------------------------------
# Patient-level, in-domain validation -- declared with its full contract, not faked
# ------------------------------------------------------------------------------
# The specification below is fixed even though the data are pending, so the contract
# is unambiguous when the individual-level CRIC trajectories arrive.
#
# PROTOCOL (per patient):
#   1. Split each trajectory into an EARLY window (observed) and a HELD-OUT future.
#   2. Personalize on the early window with q FIXED at the population value; estimate
#      ONLY the individual susceptibility s_i (and N_0, sigma). Do NOT fit q per
#      patient -- it is not identifiable individually (CHANGELOG Round 17), and the
#      predictive gain lives in s_i.
#   3. Predict the held-out future and score:
#        - continuous:  eGFR(t+1), eGFR(t+3), eGFR(t+5)   (RMSE / calibration)
#        - events:      P(dEGFR <= -40%), P(G4, eGFR<30), P(G5, eGFR<15),
#                       and time-to-KRT if recorded (AUC / Brier / calibration)
#   4. Stratify by regime: Phase I (advanced, validate progression) vs Phase III
#      (earlier, validate early detection / prevention). Report separately.
#
# This is also the data on which the temporal AI component (NephroQ-AI) would be
# trained/validated: z_i = f_phi(X_i) -> s_i, N_0, sigma, with X_i the longitudinal
# {eGFR, proteinuria, HbA1c, SBP}. Same rule: learn s_i, keep q population.

HORIZONS_YEARS = (1.0, 3.0, 5.0)
EVENT_ENDPOINTS = ("decline_40pct", "reach_G4", "reach_G5", "krt")  # krt if recorded


def validate_patient_level(patient_states, params=None,
                           early_window_years=2.0, horizons=HORIZONS_YEARS):
    """
    STRONG, in-domain validation of the twin, per the protocol above: personalize
    s_i on an early window (q fixed population), predict held-out eGFR at multiple
    horizons and the event endpoints, and score against observation -- stratified by
    CRIC phase (Phase I: progression in advanced CKD; Phase III: early detection).

    Requires per-patient longitudinal trajectories (a list of PatientState). CRIC
    provides these on request; aggregate baseline profiles do not, so this raises
    until the individual-level data are available rather than faking a validation
    from means. When the data arrive: load via clinical_data.load_long_csv ->
    PatientState, then implement the early-window -> held-out-future scoring here
    following the PROTOCOL above.
    """
    raise NotImplementedError(
        "Patient-level CRIC validation needs per-patient trajectories (PatientState "
        "objects), not aggregate baseline profiles. The protocol is fixed (see the "
        "module docstring and HORIZONS_YEARS/EVENT_ENDPOINTS): personalize s_i on an "
        "early window with q fixed at the population value, predict eGFR(t+1/t+3/t+5) "
        "and the event endpoints on the held-out future, and score by CRIC phase. "
        "Request the individual-level data, then implement here.")
