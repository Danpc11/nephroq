"""
Tests for the longitudinal PatientState / Visit model (block 2) and the clinical
data layer (block 10). These are the twin's foundation: a data-handling layer that
feeds the model without ever containing model logic.
"""
import os
import sys
from datetime import date

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from patient_state import PatientState, Visit, Measured
import clinical_data as cd


def test_visits_are_kept_in_date_order_however_they_are_added():
    ps = PatientState(patient_id="P", age=60, sex="M", visits=[
        {"date": "2023-01-01", "creatinine": 1.3},
        {"date": "2021-01-01", "creatinine": 1.0},
        {"date": "2022-01-01", "creatinine": 1.15},
    ])
    dates = [v.date for v in ps.visits]
    assert dates == sorted(dates)
    ps.add_visit(Visit(date=date(2020, 1, 1), creatinine=Measured(0.9)))
    assert ps.visits[0].date == date(2020, 1, 1)     # inserted in order
    assert ps.baseline.date == date(2020, 1, 1)
    assert ps.latest.date == date(2023, 1, 1)


def test_creatinine_history_gives_years_ago_for_age_correct_conversion():
    """
    The history must expose years_ago so each creatinine can be converted with the
    age AT THAT TIME. This is the same bug (historical age) fixed in the app, now
    prevented structurally: the state hands out years_ago, not just values.
    """
    ps = PatientState(patient_id="P", age=62, sex="M", visits=[
        {"date": "2021-01-01", "creatinine": 1.0},
        {"date": "2023-01-01", "creatinine": 1.3},
    ])
    yrs, creat = ps.creatinine_history()
    assert creat == [1.0, 1.3]
    assert yrs[0] == pytest.approx(2.0, abs=0.02)     # ~2 years before latest
    assert yrs[1] == pytest.approx(0.0, abs=0.02)
    # age at each measurement is reconstructable and correct
    ages_at = [ps.age - y for y in yrs]
    assert ages_at[0] == pytest.approx(60.0, abs=0.05)


def test_latest_covariates_returns_most_recent_OBSERVED_value():
    """Missing cells must be skipped, not treated as the current value."""
    ps = PatientState(patient_id="P", age=60, sex="F", visits=[
        {"date": "2021-01-01", "uacr": 180, "hba1c": 8.0},
        {"date": "2022-01-01", "uacr": 240},                 # hba1c missing here
    ])
    cov = ps.latest_covariates()
    assert cov["uacr"] == 240          # most recent
    assert cov["hba1c"] == 8.0         # falls back to last OBSERVED, not None
    assert cov["sbp"] is None          # never measured -> honestly None


def test_measured_rejects_a_non_finite_value():
    with pytest.raises(ValueError):
        Measured(value=float("nan"))


def test_load_long_csv_groups_by_patient_and_orders_visits(tmp_path):
    f = tmp_path / "clin.tsv"
    f.write_text(
        "patient_id\tdate\tage\tsex\tcreatinine\tuacr\n"
        "A\t2022-01-01\t60\tM\t1.2\t100\n"
        "A\t2021-01-01\t60\tM\t1.0\t80\n"
        "B\t2021-06-01\t55\tF\t0.9\t40\n"
    )
    states = cd.load_long_csv(str(f))
    assert len(states) == 2
    a = next(s for s in states if s.patient_id == "A")
    assert [v.date.isoformat() for v in a.visits] == ["2021-01-01", "2022-01-01"]


def test_out_of_range_value_is_flagged_not_dropped(tmp_path):
    f = tmp_path / "clin.tsv"
    f.write_text(
        "patient_id\tdate\tage\tsex\tsbp\n"
        "A\t2022-01-01\t60\tM\t999\n"       # implausible SBP
    )
    states = cd.load_long_csv(str(f))
    flags = cd.quality_flags(states)
    assert len(flags) == 1
    assert flags[0]["field"] == "sbp"
    assert flags[0]["quality"] == "out_of_range"
    # the value is still present (flagged, not silently discarded)
    assert states[0].visits[0].sbp.value == 999


def test_missingness_report_counts_absent_values(tmp_path):
    f = tmp_path / "clin.tsv"
    f.write_text(
        "patient_id\tdate\tage\tsex\tcreatinine\thba1c\n"
        "A\t2021-01-01\t60\tM\t1.0\t8.0\n"
        "A\t2022-01-01\t60\tM\t1.2\t\n"       # hba1c missing
    )
    states = cd.load_long_csv(str(f))
    rep = cd.missingness_report(states)
    assert rep["creatinine"] == 0.0
    assert rep["hba1c"] == pytest.approx(0.5)


def test_future_adapters_fail_honestly():
    """FHIR/OMOP are declared but not faked -- an honest NotImplementedError beats
    an adapter that silently drops fields."""
    with pytest.raises(NotImplementedError):
        cd.load_fhir({})
    with pytest.raises(NotImplementedError):
        cd.load_omop({})
