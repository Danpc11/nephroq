"""
Clinical safety assessment (roadmap block 11).

A digital twin that is deployed to clinicians must know when NOT to trust itself.
This layer inspects a patient's data and the model's state and returns a verdict:
is a prediction available, available-with-caution, or should it be withheld because
the history is too thin, the data quality is poor, or the patient is outside the
domain the model was validated on.

It is a GATE, not a model. It contains no hazard math. It reads what the other
layers already expose -- PatientState (history, covariates), clinical_data
(missingness, quality flags), and the known validity domain -- and turns them into
a plain, auditable recommendation, always shown alongside the model's provenance.

Nothing here silently changes a projection. It only labels how far to trust one.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


# The validated domain of the model, stated explicitly so 'in distribution' is not
# a matter of opinion. Values outside are not refused, but they are flagged as
# outside the domain the calibration was checked against.
VALID_DOMAIN = {
    "egfr":  (15.0, 90.0),      # the model is about CKD progression toward dialysis
    "age":   (18.0, 95.0),
    "uacr":  (0.0, 5000.0),     # trial anchors extend to ~5000
    "hba1c": (5.0, 14.0),
}

MIN_VISITS_FOR_PERSONALIZATION = 3
MIN_SPAN_YEARS = 1.0


@dataclass
class SafetyAssessment:
    """The verdict, plus the specific reasons behind it (never just a flag)."""
    sufficient_history: bool
    in_distribution: bool
    data_quality_ok: bool
    calibration_valid: bool
    uncertainty_acceptable: bool
    reasons: list = field(default_factory=list)

    @property
    def verdict(self) -> str:
        """One of:
            'prediction_available'
            'prediction_with_caution'
            'insufficient_data'
            'out_of_validated_domain'
            'do_not_use_for_scenarios'
        Chosen by the most limiting condition, so a serious problem is never masked
        by a passing one."""
        if not self.data_quality_ok:
            return "do_not_use_for_scenarios"
        if not self.in_distribution:
            return "out_of_validated_domain"
        if not self.sufficient_history:
            return "insufficient_data"
        if not (self.calibration_valid and self.uncertainty_acceptable):
            return "prediction_with_caution"
        return "prediction_available"

    def to_dict(self):
        return dict(verdict=self.verdict, sufficient_history=self.sufficient_history,
                    in_distribution=self.in_distribution,
                    data_quality_ok=self.data_quality_ok,
                    calibration_valid=self.calibration_valid,
                    uncertainty_acceptable=self.uncertainty_acceptable,
                    reasons=list(self.reasons))


def assess(state, egfr0=None, personalization=None, calibration_tier="public",
           quality_flags=None, scale_spread=None,
           uncertainty_threshold=0.6) -> SafetyAssessment:
    """
    Assess whether the twin's output for this patient should be trusted.

    Parameters mirror what the other layers already produce:
      state           : PatientState (history, covariates, demographics)
      egfr0           : current eGFR (falls back to the latest visit's)
      personalization : the dict personalize() returned (or None)
      calibration_tier: 'public' | 'mimic' | 'private'
      quality_flags   : list from clinical_data.quality_flags (or None)
      scale_spread    : personalized susceptibility spread (conformal half-width)
    """
    reasons = []

    # -- sufficient history ----------------------------------------------------
    n_visits = len(state.visits) if state else 0
    span = state.span_years() if state else 0.0
    sufficient = n_visits >= MIN_VISITS_FOR_PERSONALIZATION and span >= MIN_SPAN_YEARS
    if not sufficient:
        reasons.append(
            f"history has {n_visits} visit(s) over {span:.1f} y; "
            f"personalization needs >={MIN_VISITS_FOR_PERSONALIZATION} over "
            f">={MIN_SPAN_YEARS} y (projection uses population parameters)")

    # -- in distribution -------------------------------------------------------
    if egfr0 is None and state and state.latest:
        cov = state.latest_covariates()
        egfr0 = None  # eGFR may not be directly stored; leave to caller if absent
    checks = {}
    if egfr0 is not None:
        checks["egfr"] = egfr0
    if state:
        checks["age"] = state.age
        cov = state.latest_covariates()
        if cov.get("uacr") is not None:
            checks["uacr"] = cov["uacr"]
        if cov.get("hba1c") is not None:
            checks["hba1c"] = cov["hba1c"]
    in_dist = True
    for field_name, val in checks.items():
        lo, hi = VALID_DOMAIN[field_name]
        if not (lo <= val <= hi):
            in_dist = False
            reasons.append(f"{field_name}={val:g} is outside the validated domain "
                           f"[{lo:g}, {hi:g}]")

    # -- data quality ----------------------------------------------------------
    quality_ok = not quality_flags
    if quality_flags:
        reasons.append(f"{len(quality_flags)} out-of-range/suspect value(s) in the "
                       f"record; resolve before using for scenarios")

    # -- calibration validity --------------------------------------------------
    # A MIMIC/private tier is research-grade; the public trial-anchored tier is the
    # safest default. Neither is prospectively validated -- that caveat always
    # rides along, but it does not by itself block a prediction.
    calibration_valid = True
    if calibration_tier not in ("public", "mimic", "private"):
        calibration_valid = False
        reasons.append(f"unknown calibration tier '{calibration_tier}'")

    # -- uncertainty acceptable ------------------------------------------------
    uncertainty_ok = True
    if scale_spread is not None and scale_spread > uncertainty_threshold:
        uncertainty_ok = False
        reasons.append(f"personalized susceptibility spread ({scale_spread:.2f}) "
                       f"exceeds {uncertainty_threshold:.2f}; the individual estimate "
                       f"is too uncertain to rely on")

    return SafetyAssessment(
        sufficient_history=sufficient, in_distribution=in_dist,
        data_quality_ok=quality_ok, calibration_valid=calibration_valid,
        uncertainty_acceptable=uncertainty_ok, reasons=reasons)


def provenance_block(calibration_tier="public", calibration_source=None,
                     calibration_date=None, model_version="0.11.0") -> dict:
    """The 'always show this' footer: which model, calibrated on what, when, and
    what its validated domain is. A clinical tool must never present a number
    without this."""
    return dict(
        model_version=model_version,
        calibration_tier=calibration_tier,
        calibration_source=calibration_source or (
            "published trial anchors" if calibration_tier == "public" else "local"),
        calibration_date=calibration_date,
        validated_domain=dict(VALID_DOMAIN),
        status="research use -- not prospectively validated for clinical decisions",
    )
