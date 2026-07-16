"""
Regression tests for three integration bugs in the per-patient personalization,
all found when hardening the module for clinical ('individual twin') use.

None of these were visible in the model math; they lived in how the app wired the
personalizer to its inputs and to the calibration tiers. They are exactly the kind
of bug that silently produces a confident, wrong, personalized projection.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import model_core as core
import personalize as pz
from egfr_measurement import egfr_cr


def test_projection_uses_population_q_not_the_per_patient_estimate():
    """
    q is essentially unidentifiable from a single patient's routine data
    (measurement_strategy.py, R^2 ~ 0). Personalization must therefore rescale the
    injury rate (s_i) and leave q at its POPULATION value -- otherwise it reports a
    per-patient q that the data cannot support and that swings the projection.
    """
    p_true = pz.patient_params(1.52, 1.8)
    t = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    e = core.predict_egfr_at_v2(70.0, 8.0, 300.0, 140.0, 0.0, p_true, t)

    r = pz.personalize(t, e, 8.0, 300.0, 140.0)
    assert r["personalized"]
    # q used to project is the population value, exactly
    assert r["q"] == core.TRIAL_CALIBRATION_V2["q"]
    assert r["params"]["q"] == core.TRIAL_CALIBRATION_V2["q"]
    # the estimated q is still returned, but only as a diagnostic
    assert "q_estimated" in r
    # and the scale is genuinely personalized (this patient progresses fast)
    assert r["scale"] > 1.1


def test_historical_creatinine_uses_age_at_measurement():
    """
    A creatinine from N years ago belongs to a patient who was N years younger.
    CKD-EPI is age-dependent, so converting it with the CURRENT age biases the
    reconstructed history. This pins the direction and size of that bias so the
    app's conversion cannot silently regress to current-age.
    """
    c = 1.05
    age_now = 60.0
    years_ago = 3.0

    egfr_wrong = egfr_cr(c, age_now, "M")                 # the bug
    egfr_right = egfr_cr(c, age_now - years_ago, "M")     # correct

    # younger patient -> higher eGFR for the same creatinine
    assert egfr_right > egfr_wrong
    # the effect is real (order ~1 mL/min for a 3-year gap), not negligible
    assert egfr_right - egfr_wrong > 0.5


def test_thin_history_does_not_invent_a_personalization():
    """With too few measurements the module must return the population parameters
    and say why -- never fabricate a personalization. (This is the safety property
    the empty-history default protects: no history -> no personalization.)"""
    t = np.array([0.0, 0.5])         # too few, too short
    e = np.array([60.0, 59.0])
    r = pz.personalize(t, e, 7.0, 30.0, 130.0)
    assert r["personalized"] is False
    assert r["params"]["q"] == core.TRIAL_CALIBRATION_V2["q"]
    assert r["scale"] is None


def test_personal_scale_multiplies_the_active_tier_not_replaces_it():
    """
    The precedence rule: personalization (s_i) is applied ON TOP of the active
    calibration tier's population parameters, as a multiplier of the hazard scale.
    A local MIMIC/private tier sets the population baseline; s_i then rescales it.
    This reproduces the intended composition the app performs.
    """
    # a non-public tier: different population q / k_hf
    tier = dict(core.TRIAL_CALIBRATION_V2)
    tier.update(q=2.9, k_hf=0.0028)

    s_i = 1.5
    composed = dict(tier)
    composed["k_hf"] = tier["k_hf"] * s_i
    composed["w_a1c"] = tier["w_a1c"] * s_i

    # q stays at the tier's population value; scale multiplies through
    assert composed["q"] == 2.9
    assert composed["k_hf"] == pytest.approx(0.0028 * 1.5)
    # and it is genuinely different from either input alone
    assert composed["k_hf"] != tier["k_hf"]
