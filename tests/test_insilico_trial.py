"""Tests for the in-silico trial replication.

These guard the INTEGRITY OF THE TEST ITSELF, not the model's success: the whole
point of an in-silico replication is that it CAN fail, so a suite that quietly
forced it to pass would defeat the purpose.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import insilico_trial as it


def test_treatment_reduces_decline_in_a_virtual_trial():
    spec = it.TRIALS["CREDENCE"]
    r = it.trial_arms(spec, scale=0.73, eff_met=0.67, eff_hf=0.52, eff_alb=0.29,
                      n=25, seed=1)
    assert r["placebo"]["slope"] < 0                 # untreated CKD declines
    assert r["treated"]["slope"] > r["placebo"]["slope"]
    assert r["slope_diff"] > 0
    assert r["uacr_reduction_pct"] > 0               # and albuminuria falls


def test_zero_effect_gives_identical_arms():
    spec = it.TRIALS["CREDENCE"]
    r = it.trial_arms(spec, scale=0.73, eff_met=0.0, eff_hf=0.0, eff_alb=0.0,
                      n=20, seed=1)
    assert r["slope_diff"] == pytest.approx(0.0, abs=1e-9)
    assert r["uacr_reduction_pct"] == pytest.approx(0.0, abs=1e-9)


def test_held_out_trial_has_a_published_ci_to_fail_against():
    """The out-of-sample trial must carry real published CIs, or the
    'validation' would be unfalsifiable."""
    spec = it.TRIALS["DAPA-CKD (T2D subgroup)"]
    assert spec["role"].startswith("OUT-OF-SAMPLE")
    lo, hi = spec["chronic_slope_ci"]
    assert lo < spec["chronic_slope_diff"] < hi
    ulo, uhi = spec["uacr_reduction_ci"]
    assert ulo < spec["uacr_reduction_pct"] < uhi
    assert spec["placebo_slope"] < 0


def test_virtual_cohort_respects_trial_eligibility():
    spec = it.TRIALS["DAPA-CKD (T2D subgroup)"]
    c = it.sample_cohort(spec, 300, np.random.default_rng(0))
    assert c["egfr"].min() >= spec["egfr_range"][0] - 1e-9
    assert c["egfr"].max() <= spec["egfr_range"][1] + 1e-9
    assert c["uacr"].min() >= spec["uacr_range"][0] - 1e-9
    assert c["uacr"].max() <= spec["uacr_range"][1] + 1e-9


def test_three_trials_are_defined_and_span_the_egfr_range():
    """The saturation ceiling is only identifiable because the trials span very
    different baseline eGFR levels."""
    egfrs = sorted(s["egfr_mean"] for s in it.TRIALS.values())
    assert len(it.TRIALS) == 3
    assert egfrs[-1] - egfrs[0] > 15.0
