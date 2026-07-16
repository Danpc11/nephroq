"""
External validation against independent cohorts (roadmap block 12).

The model has been calibrated on MIMIC-IV and anchored to the SGLT2i trials.
Validation requires a DIFFERENT population the model never saw. CRIC (Chronic Renal
Insufficiency Cohort) is that population, and it usefully sits BETWEEN the two we
have used: moderate albuminuria (~200-300 mg/g), where MIMIC is normoalbuminuric
(~23) and the trials are macroalbuminuric (~927).

TWO LEVELS OF VALIDATION, and this module is honest about which one the data
support:

  1. POPULATION-LEVEL (this file, now): given a cohort's BASELINE profile, does the
     model predict the cohort's OBSERVED mean eGFR slope? This is a plausibility
     check, not proof -- it compares one predicted number to one published number.
     It is what aggregate baseline data (means +/- SD) allow.

  2. PATIENT-LEVEL (stub below, pending): given each patient's trajectory, does the
     model predict their individual future eGFR? This is the strong validation of
     the twin, and it needs per-patient longitudinal data, which aggregate profiles
     do not provide. The function is declared and raises until that data arrives,
     rather than pretending aggregate data can do a job it cannot.

CRIC baseline profiles below are transcribed from the provided summary. The
observed-slope reference is a PLACEHOLDER and MUST be replaced with a verified
value from the CRIC progression literature before any result is reported.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

import model_core as core


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

# PLACEHOLDER -- replace with a verified CRIC diabetic eGFR-slope before reporting.
# The CRIC literature reports mean annual eGFR decline for diabetic participants;
# this is the number the population check compares against. Left as a range with an
# explicit 'unverified' flag so it cannot be mistaken for a confirmed value.
CRIC_OBSERVED_SLOPE = dict(
    value=None,                 # e.g. -2.4 mL/min/1.73m2/yr -- FILL FROM PUBLICATION
    plausible_range=(-2.7, -2.0),
    source="PLACEHOLDER -- verify against CRIC progression paper (e.g. Bundy et al.)",
    verified=False,
)


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
        "Population-level plausibility check, NOT patient-level validation of the twin.",
        "proteinuria->UACR conversion is approximate; the slope is reported as a range.",
        "Compares one predicted mean slope to a published cohort slope -- consistency "
        "here is necessary, not sufficient.",
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
# Patient-level validation -- declared, not faked (needs per-patient trajectories)
# ------------------------------------------------------------------------------
def validate_patient_level(patient_states, params=None):
    """
    STRONG validation of the twin: for each patient, fit/personalize on an early
    window and predict their held-out later eGFR, then score prediction vs
    observation across the cohort (external, and optionally temporal/geographic).

    Requires per-patient longitudinal trajectories (a list of PatientState). CRIC
    provides these on request; aggregate baseline profiles do not, so this raises
    until the individual-level data are available rather than producing a fake
    validation from means.
    """
    raise NotImplementedError(
        "Patient-level external validation needs per-patient CRIC trajectories "
        "(PatientState objects), not aggregate baseline profiles. Request the "
        "individual-level data, load via clinical_data, then implement the "
        "early-window -> held-out-future scoring here.")
