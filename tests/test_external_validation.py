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


def test_population_check_predicts_a_slope_range_for_each_cohort():
    for key in ("phase_I", "phase_III"):
        r = ev.validate_population(key)
        lo, hi = r.predicted_slope_range
        assert lo <= hi < 0             # eGFR declines
        assert r.n > 0


def test_placeholder_observed_slope_is_flagged_loudly():
    """The observed reference is a placeholder; the result must SAY SO, so a
    consistency verdict cannot be mistaken for a validated one."""
    r = ev.validate_population("phase_I")
    assert any("PLACEHOLDER" in c for c in r.caveats)
    # and it is never presented as patient-level validation
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
