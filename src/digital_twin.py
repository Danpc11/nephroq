"""
Renal digital twin -- continuous longitudinal update (roadmap block 4).

This is the piece that changes what NephroQ *is*. A personalized simulator
projects once and freezes. A twin is re-estimated every time a new visit arrives,
and -- crucially -- it keeps a record of how its previous forecast compared to what
actually happened. That prediction-vs-observation trace is what lets a twin earn
(or lose) trust over time, and it is the raw material for calibration monitoring.

    new visit -> update state -> re-personalize -> new forecast
                                              \\-> score the PREVIOUS forecast

DESIGN: the twin holds a PatientState and a history of forecasts. It delegates the
biology to personalize()/model_core (it contains no hazard math of its own) and the
data to PatientState. It only orchestrates the loop and remembers what it predicted.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Optional

import numpy as np

import model_core as core
from patient_state import PatientState, Visit
from egfr_measurement import egfr_cr
import personalize as pz


@dataclass
class Forecast:
    """A single projection made at a point in time, kept so it can be scored later
    against what actually happened."""
    as_of: date                       # the date the forecast was made
    scale: Optional[float]            # personalized susceptibility used (None = population)
    personalized: bool
    t_years: np.ndarray               # horizon grid, years from as_of
    egfr: np.ndarray                  # projected eGFR on that grid
    egfr0: float                      # baseline eGFR the projection started from


@dataclass
class UpdateResult:
    """What changed when a new visit arrived. This is the twin's answer to 'was I
    right last time, and what do I think now?'."""
    forecast: Forecast
    previous_forecast: Optional[Forecast]
    observed_egfr: Optional[float]            # the eGFR actually seen at this visit
    predicted_egfr: Optional[float]           # what the PREVIOUS forecast said for now
    prediction_error: Optional[float]         # observed - predicted (mL/min)
    scale_before: Optional[float]
    scale_after: Optional[float]
    uncertainty_change: Optional[float]       # change in personalized scale spread


class RenalDigitalTwin:
    """
    A twin wrapped around one patient's longitudinal state.

    Lifecycle:
        twin = RenalDigitalTwin(patient_state)
        twin.forecast()                 # initial projection
        result = twin.update(new_visit) # each new consultation
        # result carries: prior forecast, the error against it, the new forecast,
        # and how the personalization moved.

    The twin never mutates the model's parameters globally; personalization is
    per-patient and applied on top of the active population parameters (q stays at
    the population value -- only susceptibility s_i is individual).
    """

    def __init__(self, state: PatientState, horizon_years: float = 15.0,
                 base_params: Optional[dict] = None):
        self.state = state
        self.horizon_years = float(horizon_years)
        self.base_params = dict(base_params or core.TRIAL_CALIBRATION_V2)
        self.forecasts: list = []           # chronological history of forecasts

    # -- reconstruct the model inputs from the longitudinal state --------------
    def _model_inputs(self):
        """eGFR trajectory (t, egfr) from the creatinine history, converted with
        the age AT EACH MEASUREMENT, plus the latest observed covariates."""
        yrs, creat = self.state.creatinine_history()
        if not creat:
            return None
        # years_ago -> t measured forward from the earliest visit
        yrs = np.asarray(yrs, float)
        t = (yrs.max() - yrs)                      # 0 at the oldest, max at latest
        order = np.argsort(t)
        t = t[order]
        ages_at = self.state.age - yrs[order]
        egfr = np.array([egfr_cr(c, a, self.state.sex)
                         for c, a in zip(np.asarray(creat)[order], ages_at)])
        cov = self.state.latest_covariates()
        return t, egfr, cov

    def _current_egfr(self):
        mi = self._model_inputs()
        return None if mi is None else float(mi[1][-1])

    # -- make a forecast from the current state --------------------------------
    def forecast(self, estimator=None) -> Forecast:
        mi = self._model_inputs()
        as_of = self.state.latest.date if self.state.latest else date.today()
        cov = mi[2] if mi else dict(hba1c=None, uacr=None, sbp=None)
        a1c = cov["hba1c"] if cov["hba1c"] is not None else 7.0
        uacr = cov["uacr"] if cov["uacr"] is not None else 30.0
        sbp = cov["sbp"] if cov["sbp"] is not None else 130.0

        # personalize (only if the history supports it)
        scale, personalized = None, False
        params = dict(self.base_params)
        if mi is not None:
            t, egfr, _ = mi
            r = pz.personalize(t, egfr, a1c, uacr, sbp, estimator=estimator)
            if r["personalized"]:
                scale, personalized = r["scale"], True
                # apply s_i on top of the base (population) params; q stays population
                params["k_hf"] = self.base_params["k_hf"] * scale
                params["w_a1c"] = self.base_params["w_a1c"] * scale
                params["w_uacr"] = self.base_params["w_uacr"] * scale
                params["w_sbp"] = self.base_params["w_sbp"] * scale
            egfr0 = float(egfr[-1])
            self._last_scale_sd = r.get("scale_sd")
        else:
            egfr0 = 60.0
            self._last_scale_sd = None

        tq = np.linspace(0.0, self.horizon_years, 60)
        proj = core.predict_egfr_at_v2(egfr0, a1c, uacr, sbp, 0.0, params, tq,
                                       years=self.horizon_years)
        fc = Forecast(as_of=as_of, scale=scale, personalized=personalized,
                      t_years=tq, egfr=proj, egfr0=egfr0)
        self.forecasts.append(fc)
        return fc

    # -- the continuous-update entry point -------------------------------------
    def update(self, visit, estimator=None) -> UpdateResult:
        """
        Register a new visit, score the PREVIOUS forecast against what actually
        happened, re-personalize, and produce a fresh forecast.
        """
        prev = self.forecasts[-1] if self.forecasts else None
        prev_scale = prev.scale if prev else None

        # what did the previous forecast predict for the date of THIS visit?
        v = visit if isinstance(visit, Visit) else Visit(**visit)
        predicted = observed = error = None
        if prev is not None:
            dt_years = (_as_date(v.date) - prev.as_of).days / 365.25
            if dt_years >= 0:
                predicted = float(np.interp(dt_years, prev.t_years, prev.egfr))
        # observed eGFR at this visit (from creatinine if needed)
        observed = _visit_egfr(v, self.state.age, self.state.sex)
        if predicted is not None and observed is not None:
            error = observed - predicted

        # extend the state and re-forecast
        self.state.add_visit(v)
        new_fc = self.forecast(estimator=estimator)

        unc_change = None
        if prev_scale is not None and new_fc.scale is not None:
            unc_change = new_fc.scale - prev_scale

        return UpdateResult(
            forecast=new_fc, previous_forecast=prev,
            observed_egfr=observed, predicted_egfr=predicted,
            prediction_error=error,
            scale_before=prev_scale, scale_after=new_fc.scale,
            uncertainty_change=unc_change,
        )


# ------------------------------------------------------------------------------
# helpers
# ------------------------------------------------------------------------------
def _as_date(x):
    if isinstance(x, date):
        return x
    return date.fromisoformat(str(x)[:10])


def _visit_egfr(v: Visit, age, sex):
    """eGFR at a visit: use a recorded egfr if present, else compute from
    creatinine at the patient's age."""
    if v.egfr is not None:
        return float(v.egfr.value)
    if v.creatinine is not None:
        return float(egfr_cr(v.creatinine.value, age, sex))
    return None
