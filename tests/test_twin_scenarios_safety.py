"""
Tests for blocks 4 (continuous twin update), 5 (treatment scenarios), 6 (acute
events / AKI), and 11 (clinical safety). These layers turn the personalized
simulator into a deployable digital twin: they orchestrate, they never contain
hazard math of their own.
"""
import os
import sys
from datetime import date

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import model_core as core
from patient_state import PatientState, Visit, Measured
from digital_twin import RenalDigitalTwin
import treatment_engine as te
import acute_events as ae
import clinical_safety as cs


# ---- Block 4: continuous update ---------------------------------------------
def test_twin_scores_its_previous_forecast_against_the_new_observation():
    """The property that makes it a twin, not a one-shot simulator: it remembers
    what it predicted and measures the error when reality arrives."""
    ps = PatientState("P", 62, "M", visits=[
        {"date": "2020-01-01", "creatinine": 1.0, "hba1c": 8.0, "uacr": 200, "sbp": 140},
        {"date": "2021-01-01", "creatinine": 1.15, "uacr": 240},
        {"date": "2022-01-01", "creatinine": 1.32, "hba1c": 7.8, "sbp": 138},
    ])
    twin = RenalDigitalTwin(ps)
    twin.forecast()
    res = twin.update(Visit(date="2023-01-01", creatinine=Measured(1.50)))

    assert res.previous_forecast is not None
    assert res.observed_egfr is not None
    assert res.predicted_egfr is not None
    assert res.prediction_error == pytest.approx(res.observed_egfr - res.predicted_egfr)
    # a patient progressing faster than predicted should push susceptibility UP
    assert res.scale_after >= res.scale_before


def test_twin_keeps_a_chronological_forecast_history():
    ps = PatientState("P", 60, "M", visits=[
        {"date": "2020-01-01", "creatinine": 1.0},
        {"date": "2021-06-01", "creatinine": 1.2},
        {"date": "2022-06-01", "creatinine": 1.35},
    ])
    twin = RenalDigitalTwin(ps)
    twin.forecast()
    twin.update(Visit(date="2023-06-01", creatinine=Measured(1.5)))
    assert len(twin.forecasts) == 2
    assert twin.forecasts[1].as_of > twin.forecasts[0].as_of


# ---- Block 5: treatment scenarios -------------------------------------------
def test_more_drugs_slow_progression_and_combine_below_100pct():
    patient = dict(egfr0=48, a1c=8.0, uacr0=600, sbp=145)
    none = te.evaluate_scenario(**patient, regimen=te.Regimen(()), years=8)
    sglt = te.evaluate_scenario(**patient, regimen=te.Regimen(("sglt2i",)), years=8)
    triple = te.evaluate_scenario(**patient,
                                  regimen=te.Regimen(("sglt2i", "raasi", "finerenone")),
                                  years=8)
    e5 = lambda r: np.interp(5, r["t"], r["egfr"])
    assert e5(sglt) > e5(none)              # treatment helps
    assert e5(triple) >= e5(sglt)           # more helps at least as much

    # multiplicative combination never exceeds full blockade
    eff = te.Regimen(("sglt2i", "raasi", "finerenone")).effective_effects()
    assert all(0.0 <= v < 1.0 for v in eff.values())


def test_scenario_carries_its_evidence_and_is_not_a_recommendation():
    ev = te.regimen_evidence(te.Regimen(("sglt2i", "finerenone")))
    assert len(ev) == 2
    assert all(e["evidence"] and e["population"] for e in ev)


# ---- Block 6: acute events ---------------------------------------------------
def test_aki_drops_then_partially_recovers_leaving_a_step():
    t = np.linspace(0, 6, 200)
    chronic = np.full_like(t, 55.0)
    aki = ae.aki_event(t_onset=2.0, drop_mL=20, permanent_fraction=0.2)
    obs = ae.apply_acute(t, chronic, [aki])

    before = obs[np.searchsorted(t, 1.9)]
    nadir = obs[np.searchsorted(t, 2.0)]
    late = obs[np.searchsorted(t, 5.0)]
    assert nadir < before - 15               # sharp drop
    assert late > nadir + 10                 # recovers most of it
    assert late < before                     # but leaves a permanent step


def test_aki_raises_susceptibility_but_sglt2_dip_does_not():
    aki = ae.aki_event(2.0, 20, severity=1.0)
    dip = ae.sglt2_dip_event(0.0, 60.0)
    assert ae.susceptibility_after(1.0, [aki]) > 1.0
    assert ae.susceptibility_after(1.0, [dip]) == pytest.approx(1.0)


def test_acute_reading_detector_flags_a_recovering_dip():
    egfr = np.array([60, 40, 58, 55, 52])       # the 40 recovers to 58
    times = np.array([0.0, 0.5, 0.7, 1.5, 2.5])
    flags = ae.is_probably_acute(egfr, times)
    assert flags[1] and not flags[0] and not flags[3]


# ---- Block 11: clinical safety ----------------------------------------------
def _good_state():
    return PatientState("P", 62, "M", visits=[
        {"date": "2020-01-01", "creatinine": 1.1, "uacr": 200, "hba1c": 8.0},
        {"date": "2021-06-01", "creatinine": 1.2, "uacr": 240, "hba1c": 7.8},
        {"date": "2023-01-01", "creatinine": 1.35, "uacr": 300, "hba1c": 7.9},
    ])


def test_safety_verdicts_pick_the_most_limiting_condition():
    good = _good_state()
    assert cs.assess(good, egfr0=48).verdict == "prediction_available"
    # data quality is the hardest stop -- it wins even if everything else is fine
    assert cs.assess(good, egfr0=48,
                     quality_flags=[{"field": "sbp"}]).verdict == "do_not_use_for_scenarios"
    # out of domain
    assert cs.assess(good, egfr0=110).verdict == "out_of_validated_domain"
    # thin history
    thin = PatientState("P", 55, "F", visits=[{"date": "2023-01-01", "creatinine": 1.0}])
    assert cs.assess(thin, egfr0=60).verdict == "insufficient_data"
    # high uncertainty -> caution
    assert cs.assess(good, egfr0=48, scale_spread=0.9).verdict == "prediction_with_caution"


def test_safety_reasons_are_always_specific():
    good = _good_state()
    a = cs.assess(good, egfr0=110)
    assert a.reasons and any("domain" in r for r in a.reasons)


def test_provenance_block_always_states_research_use():
    p = cs.provenance_block()
    assert "not prospectively validated" in p["status"]
    assert p["validated_domain"]
