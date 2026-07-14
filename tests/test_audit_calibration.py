"""Tests for the calibration audit.

The audit exists to stop a bad calibration from reaching the app, so what has to
be guarded is that it CATCHES the failure modes -- not that it is polite."""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import audit_calibration as audit
import model_core as core


def _cal(**kw):
    base = dict(q=1.52, k_hf=0.0103, w_a1c=0.0105, w_uacr=0.0131, w_sbp=0.0079,
                bootstrap_params=[dict(q=1.52 + 0.05 * (i - 2), k_hf=0.0103,
                                       w_a1c=0.0105, w_uacr=0.0131, w_sbp=0.0079)
                                  for i in range(5)])
    base.update(kw)
    return base


def test_detects_a_frozen_optimizer():
    """The old failure mode: the fit returns exactly the initial guess."""
    frozen = _cal(q=1.5, k_hf=0.012, w_a1c=0.014, w_uacr=0.018, w_sbp=0.011,
                  bootstrap_params=[dict(q=1.5, k_hf=0.012, w_a1c=0.014,
                                         w_uacr=0.018, w_sbp=0.011)] * 5)
    assert audit.check_optimizer(frozen) is False


def test_detects_a_degenerate_bootstrap():
    """Replicates that all return the same numbers mean the optimizer never moved,
    NOT that the parameters are precisely known."""
    degenerate = _cal(bootstrap_params=[dict(q=1.52, k_hf=0.0103, w_a1c=0.0105,
                                             w_uacr=0.0131, w_sbp=0.0079)] * 5)
    assert audit.check_optimizer(degenerate) is False


def test_accepts_a_calibration_that_actually_moved():
    assert audit.check_optimizer(_cal()) is True


def test_hazard_ratio_flags_an_inflated_calibration(capsys):
    """A MIMIC fit that says patients decline twice as fast as real trial placebo
    arms must be reported as a bias measurement, not as a better estimate."""
    inflated = _cal(k_hf=0.0103 * 2, w_a1c=0.0105 * 2, w_uacr=0.0131 * 2,
                    w_sbp=0.0079 * 2)
    audit.compare_to_trials(inflated)
    out = capsys.readouterr().out
    assert "FASTER" in out


def test_agreement_is_reported_when_the_sources_converge(capsys):
    audit.compare_to_trials(_cal())
    out = capsys.readouterr().out
    assert "agree" in out.lower()


def test_reference_scale_matches_model_core():
    """The audit must compare against the SAME anchored parameters the app ships."""
    ref = core.TRIAL_CALIBRATION_V2
    assert ref["k_hf"] / audit.K_HF_BASE == pytest.approx(0.730, rel=1e-3)
