"""
Tests for block 7 (full predictive uncertainty) and block 8 (clinical outputs).
Both compose model_core; neither contains hazard math. Also pins a solver-guard
hardening found while building them.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import model_core as core
import uncertainty as unc
import clinical_outputs as co


# ---- Block 7 ----------------------------------------------------------------
def test_uncertainty_budget_components_are_separable_and_sum():
    b = unc.UncertaintyBudget(population=4.0, personalization=9.0, measurement=16.0,
                              future=1.0, model=4.0)
    # total SD is sqrt of the summed variances
    assert b.total_sd == pytest.approx(np.sqrt(34.0))
    # fractions partition to 1
    assert sum(b.fractions().values()) == pytest.approx(1.0)
    # measurement is the largest here
    assert b.dominant_source() == "measurement"


def test_predictive_band_brackets_the_point_and_widens_with_horizon():
    p = core.TRIAL_CALIBRATION_V2
    r = unc.predictive_band(50, 8.0, 300, 140, p, [2.0, 5.0, 10.0],
                            scale_spread=0.2, n_draws=150)
    # band contains the point projection at every horizon
    assert np.all(r["lo"] <= r["point"] + 1e-6)
    assert np.all(r["hi"] >= r["point"] - 1e-6)
    # a thin-history patient (large scale_spread) has a wider band than a well
    # characterized one
    narrow = unc.predictive_band(50, 8.0, 300, 140, p, [5.0], scale_spread=0.1,
                                 n_draws=150)
    wide = unc.predictive_band(50, 8.0, 300, 140, p, [5.0], scale_spread=0.6,
                               n_draws=150)
    assert (wide["hi"][0] - wide["lo"][0]) > (narrow["hi"][0] - narrow["lo"][0])


def test_confidence_label_tracks_total_spread():
    tight = unc.UncertaintyBudget(1, 1, 1, 0, 1)      # sd 2 -> high
    loose = unc.UncertaintyBudget(50, 60, 40, 10, 30) # sd ~14 -> low
    assert unc.confidence_label(tight) == "high"
    assert unc.confidence_label(loose) == "low"


# ---- Block 8 ----------------------------------------------------------------
def test_clinical_outputs_are_coherent_probabilities():
    p = core.TRIAL_CALIBRATION_V2
    out = co.clinical_outputs(42, 8.0, 500, 145, p, horizons=(2.0, 5.0, 10.0),
                              scale_spread=0.2, n_draws=200)
    # all probabilities are in [0,1]
    for d in (out.p_40pct_decline, out.p_g4, out.p_g5):
        assert all(0.0 <= v <= 1.0 for v in d.values())
    # risk is monotone non-decreasing with horizon (longer -> at least as likely)
    hs = sorted(out.p_g4)
    for a, b in zip(hs, hs[1:]):
        assert out.p_g4[b] >= out.p_g4[a] - 1e-9
    # a G3b patient reaching G4 gets a referral flag and a finite time-to-G4
    assert out.current_category == "G3b"
    assert out.referral_flag is True
    assert out.time_to_g4 is not None


def test_clinical_outputs_do_not_claim_a_dialysis_date():
    """The panel reports P(eGFR<15) and time-to-G5, never a KRT start date --
    eGFR<15 is not the same as initiating dialysis, and the project keeps that
    distinction."""
    p = core.TRIAL_CALIBRATION_V2
    out = co.clinical_outputs(55, 7.5, 100, 135, p, horizons=(5.0,), n_draws=150)
    d = out.to_dict()
    assert "p_g5" in d and "time_to_g5" in d
    assert not any("dialysis_date" in k or "krt_date" in k for k in d)


# ---- solver hardening found while building block 7 --------------------------
def test_predict_egfr_survives_a_degenerate_single_short_horizon():
    """A very short horizon in a Monte-Carlo draw once left solve_ivp's .y as a
    list with no .shape, raising deep in the loop. The guard must fall back
    cleanly and return a finite number."""
    p = core.TRIAL_CALIBRATION_V2
    for h in (0.25, 0.5, 1.0, 2.0):
        val = core.predict_egfr_at_v2(48, 8.0, 300, 140, 0.0, p,
                                      np.array([h]), years=float(h))
        assert np.isfinite(val[0])
