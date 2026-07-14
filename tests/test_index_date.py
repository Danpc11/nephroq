"""
The index date is the single largest source of bias in a hospital cohort.

People come to hospital BECAUSE they feel unwell. Their creatinine is high that
day for reasons that have nothing to do with lost nephrons -- dehydration, sepsis,
NSAIDs, contrast. Recording that value as the patient's chronic baseline projects
a healthy kidney to dialysis, and the error runs in the dangerous direction: it
OVER-diagnoses.

The fix is the rule nephrologists already use (KDIGO): an abnormality only counts
as chronic if it PERSISTS. These tests pin that behaviour.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import model_core as core
from mimic_loader import _confirmed_index, _first_index, INDEX_STRATEGIES

T0 = pd.Timestamp("2150-01-01")


def _times(days):
    return [T0 + pd.Timedelta(days=int(d)) for d in days]


def test_an_acute_episode_that_recovers_is_not_used_as_baseline():
    """THE WHOLE POINT. eGFR 40 on presentation, back to 70 four months later:
    that 40 was an acute dip, and must not become the patient's baseline."""
    t = _times([0, 7, 30, 120, 400, 700])
    e = np.array([40, 52, 63, 70, 69, 68], dtype=float)

    assert e[_first_index(t, e)] == 40          # the old rule takes the acute value
    i = _confirmed_index(t, e)
    assert i is not None
    assert e[i] > 60                            # the confirmed rule skips past it


def test_a_genuinely_low_egfr_that_persists_IS_the_baseline():
    """The rule must not throw away real CKD. eGFR 40 that stays ~40 is real."""
    t = _times([0, 30, 120, 400, 700])
    e = np.array([40, 41, 39, 37, 35], dtype=float)

    i = _confirmed_index(t, e)
    assert i == 0
    assert e[i] == pytest.approx(40)


def test_a_progressing_patient_is_not_mistaken_for_an_acute_episode():
    """
    THE FAILURE MODE THIS GUARDS AGAINST. A patient who is genuinely declining
    reads LOWER at the confirmation window, not higher. If the rule keyed on any
    change instead of RECOVERY, it would discard exactly the patients the model
    exists to identify -- the progressors.
    """
    t = _times([0, 120, 400, 700])
    e = np.array([55, 50, 40, 30], dtype=float)

    i = _confirmed_index(t, e)
    assert i == 0                               # the first value stands
    assert e[i] == pytest.approx(55)


def test_a_patient_who_only_ever_presents_acutely_is_dropped_not_guessed():
    """If no value can be confirmed, there is no baseline. Dropping the patient is
    honest; inventing one is what caused the bias in the first place."""
    t = _times([0, 5, 20])                      # nothing >= 90 days out
    e = np.array([35, 48, 60], dtype=float)

    assert _confirmed_index(t, e) is None


def test_the_confirmed_index_changes_the_projection_it_is_meant_to_change():
    """
    Ties the cohort rule to the clinical consequence: the same patient, indexed on
    an acute creatinine versus their true level, gets a completely different fate.
    This is the number that justifies the whole change.
    """
    p = dict(core.TRIAL_CALIBRATION_V2, q=2.79, k_hf=0.0034,
             w_a1c=0.0027, w_uacr=0.0058, w_sbp=0.0018)

    _, _, _, t_true = core.simulate_trajectory_v2(70.0, 7.0, 20.0, 132.0, u=0.0,
                                                  p=p, years=25, n=300)
    _, _, _, t_acute = core.simulate_trajectory_v2(40.0, 7.0, 20.0, 132.0, u=0.0,
                                                   p=p, years=25, n=300)

    assert t_acute < 15.0            # indexed on the acute value: dialysis within 15y
    assert t_true > t_acute + 5.0    # indexed on the truth: far later (or never)


def test_both_strategies_are_registered():
    assert set(INDEX_STRATEGIES) == {"first", "confirmed"}
