"""
Tests for external validation against CRIC (block 12). The module must be honest
about the difference between a population plausibility check (which aggregate data
allow) and patient-level validation of the twin (which they do not).
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import external_validation as ev


def test_proteinuria_conversion_returns_a_range_not_a_point():
    lo, hi = ev.proteinuria_to_uacr(0.38)
    assert lo < hi                      # it is a range, honestly
    assert 100 < lo < hi < 400          # sane magnitude for ~0.4 g/day


def test_cross_regime_reference_predicts_a_slope_range_per_cohort():
    for key in ("phase_I", "phase_III"):
        r = ev.validate_population(key)
        lo, hi = r.predicted_slope_range
        assert lo <= hi < 0             # eGFR declines
        assert r.n > 0
        # framed as a cross-regime reference, never as a pass/fail
        assert any("CROSS-REGIME" in c or "not pass/fail" in c or "pass/fail" in c
                   for c in r.caveats)


def test_population_roles_are_explicit_data():
    # the three-regime distinction must be queryable, not just prose
    assert "MIMIC-IV" in ev.POPULATION_ROLES
    assert "CRIC Phase I" in ev.POPULATION_ROLES
    assert "CRIC Phase III" in ev.POPULATION_ROLES


def test_patient_level_protocol_is_fixed_even_though_data_pending():
    # the validation contract is specified now: multi-horizon + events
    assert ev.HORIZONS_YEARS == (1.0, 3.0, 5.0)
    assert "reach_G5" in ev.EVENT_ENDPOINTS


def test_observed_slope_is_verified_and_result_stays_population_level():
    """The observed CRIC slope is now verified against the literature; the result
    must no longer flag a placeholder, but must still say it is NOT patient-level
    validation of the twin."""
    r = ev.validate_population("phase_I")
    assert ev.CRIC_OBSERVED_SLOPE["verified"] is True
    assert not any("PLACEHOLDER" in c for c in r.caveats)
    assert any("NOT patient-level" in c for c in r.caveats)


def test_patient_level_validation_refuses_without_individual_data():
    """It must not fake a validation from aggregate profiles."""
    with pytest.raises(NotImplementedError):
        ev.validate_patient_level([])


def test_cohorts_are_inside_the_models_validated_domain():
    """Both CRIC phases should sit inside eGFR 15-90, HbA1c 5-14."""
    for c in ev.CRIC_COHORTS.values():
        assert 15 <= c["egfr"] <= 90
        assert 5 <= c["hba1c"] <= 14
