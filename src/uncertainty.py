"""
Full predictive uncertainty (roadmap block 7).

The README is honest that the current bands propagate PARAMETER uncertainty only.
A predictive interval a clinician can act on needs more than that. This module
composes the five sources the roadmap names into one predictive band, and -- just
as importantly -- keeps them SEPARATE so the interface can answer *why* a
projection is uncertain:

    sigma_total^2 = sigma_population^2      how well are the population parameters known
                  + sigma_personalization^2 how well do we know THIS patient
                  + sigma_measurement^2      noise in the inputs (eGFR is a noisy draw)
                  + sigma_future^2           unknown future covariates / adherence / events
                  + sigma_model^2            structural: the model is a simplification

The point of separating them is actionable: a band dominated by
sigma_personalization shrinks if the patient brings more history; one dominated by
sigma_future does not, and the clinician should be told which case they are in.

This module composes and reports. It does not refit anything. It draws on
model_core for the forward map and on the personalizer's spreads; it holds no
hazard math.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

import model_core as core


# Default magnitudes for the sources that are not estimated per-patient. These are
# the same figures used elsewhere in the project, gathered here so the assumptions
# are in one auditable place.
SIGMA_MEASUREMENT_MLMIN = 8.7      # within-patient eGFR dispersion (cohort diagnostics)
SIGMA_MODEL_FRACTION = 0.08        # structural model error, as a fraction of eGFR
FUTURE_COVARIATE_LOG_SD = 0.10     # per-year drift in unknown future insult, log scale


@dataclass
class UncertaintyBudget:
    """The five variance components at a given horizon, plus the composed band.
    Every field is in (mL/min)^2 except where noted, so they add directly."""
    population: float
    personalization: float
    measurement: float
    future: float
    model: float

    @property
    def total_sd(self) -> float:
        return float(np.sqrt(max(self.population + self.personalization +
                                 self.measurement + self.future + self.model, 0.0)))

    def dominant_source(self) -> str:
        comps = dict(population=self.population, personalization=self.personalization,
                     measurement=self.measurement, future=self.future, model=self.model)
        return max(comps, key=comps.get)

    def fractions(self) -> dict:
        tot = (self.population + self.personalization + self.measurement +
               self.future + self.model)
        if tot <= 0:
            return {k: 0.0 for k in ("population", "personalization",
                                     "measurement", "future", "model")}
        return dict(population=self.population / tot,
                    personalization=self.personalization / tot,
                    measurement=self.measurement / tot,
                    future=self.future / tot, model=self.model / tot)


def predictive_band(egfr0, a1c, uacr0, sbp, params, horizons, u=0.0,
                    scale_spread=None, param_bootstrap=None,
                    meas_sd=SIGMA_MEASUREMENT_MLMIN,
                    model_frac=SIGMA_MODEL_FRACTION,
                    future_log_sd=FUTURE_COVARIATE_LOG_SD,
                    n_draws=300, seed=0):
    """
    Compose the full predictive band at each horizon, and return the point
    projection, the band (5th/95th), and the variance budget per horizon.

    Parameters:
      scale_spread    : personalized susceptibility half-width (None -> that
                        component is zero; the projection is population-level)
      param_bootstrap : list of parameter dicts (population uncertainty); if None,
                        that component is estimated from a small default jitter
    Sources are propagated by Monte Carlo over the mechanistic forward map, so the
    band respects the model's non-linearity rather than assuming a Gaussian output.
    """
    horizons = np.atleast_1d(np.asarray(horizons, float))
    rng = np.random.default_rng(seed)
    base = dict(params)

    # --- draw each source -----------------------------------------------------
    # measurement: baseline eGFR is one noisy observation
    egfr_draws = np.clip(egfr0 + rng.normal(0, meas_sd, n_draws), 5.0, 150.0) \
        if meas_sd > 0 else np.full(n_draws, float(egfr0))

    # personalization: individual susceptibility multiplier
    if scale_spread and scale_spread > 0:
        # scale_spread is a half-width; treat as ~1 SD on the log scale
        rate_mult = np.exp(rng.normal(0.0, scale_spread, n_draws))
    else:
        rate_mult = np.ones(n_draws)

    # population: bootstrap of q/k_hf if provided, else a small default jitter
    if param_bootstrap:
        pool = [param_bootstrap[i] for i in rng.integers(0, len(param_bootstrap), n_draws)]
    else:
        pool = []
        for _ in range(n_draws):
            p = dict(base)
            p["k_hf"] = base["k_hf"] * (1 + rng.normal(0, 0.10))
            pool.append(p)

    # future: unknown drift in the insult over the horizon (grows with time)
    # applied as a per-draw multiplier on the metabolic/albuminuric insult
    future_mult = np.exp(rng.normal(0.0, future_log_sd, n_draws))

    # --- forward map for each draw at each horizon ----------------------------
    H = len(horizons)
    sims = np.empty((n_draws, H))
    for d in range(n_draws):
        p = dict(pool[d])
        s_i = rate_mult[d] * future_mult[d]
        p["k_hf"] = p["k_hf"] * rate_mult[d]
        p["w_a1c"] = p["w_a1c"] * s_i
        p["w_uacr"] = p["w_uacr"] * s_i
        p["w_sbp"] = p["w_sbp"] * s_i
        sims[d] = core.predict_egfr_at_v2(float(egfr_draws[d]), a1c, uacr0, sbp, u,
                                          p, horizons, years=float(horizons[-1]))

    point = core.predict_egfr_at_v2(egfr0, a1c, uacr0, sbp, u, base, horizons,
                                    years=float(horizons[-1]))
    lo = np.percentile(sims, 5, axis=0)
    hi = np.percentile(sims, 95, axis=0)

    # --- variance budget per horizon, by leave-one-source-in -----------------
    budgets = []
    for j, h in enumerate(horizons):
        var_meas = _var_from(egfr0, a1c, uacr0, sbp, base, h, u, rng,
                             meas_sd=meas_sd)
        var_pers = _var_from(egfr0, a1c, uacr0, sbp, base, h, u, rng,
                             scale_spread=scale_spread) if scale_spread else 0.0
        var_pop = _var_from(egfr0, a1c, uacr0, sbp, base, h, u, rng,
                            param_bootstrap=param_bootstrap)
        var_fut = _var_from(egfr0, a1c, uacr0, sbp, base, h, u, rng,
                            future_log_sd=future_log_sd)
        var_mod = (model_frac * float(point[j])) ** 2
        budgets.append(UncertaintyBudget(population=var_pop, personalization=var_pers,
                                         measurement=var_meas, future=var_fut,
                                         model=var_mod))

    return dict(horizons=horizons, point=point, lo=lo, hi=hi, budgets=budgets)


def _var_from(egfr0, a1c, uacr0, sbp, base, h, u, rng, meas_sd=0.0, scale_spread=0.0,
              param_bootstrap=None, future_log_sd=0.0, n=120):
    """Variance of eGFR(h) attributable to a SINGLE source, holding others fixed.
    Used to build the budget so the components are separable and add up."""
    vals = np.empty(n)
    for i in range(n):
        e0 = egfr0 + rng.normal(0, meas_sd) if meas_sd > 0 else egfr0
        e0 = float(np.clip(e0, 5.0, 150.0))
        p = dict(base)
        if param_bootstrap:
            p = dict(param_bootstrap[rng.integers(0, len(param_bootstrap))])
        m = 1.0
        if scale_spread and scale_spread > 0:
            m *= float(np.exp(rng.normal(0, scale_spread)))
        if future_log_sd and future_log_sd > 0:
            m *= float(np.exp(rng.normal(0, future_log_sd)))
        p["k_hf"] = p["k_hf"] * m
        p["w_a1c"] = p["w_a1c"] * m
        p["w_uacr"] = p["w_uacr"] * m
        p["w_sbp"] = p["w_sbp"] * m
        vals[i] = core.predict_egfr_at_v2(e0, a1c, uacr0, sbp, u, p,
                                          np.array([h]), years=float(h))[0]
    return float(np.var(vals))


def confidence_label(budget: UncertaintyBudget) -> str:
    """Translate the total spread into a plain personalization-confidence label,
    the way the roadmap asks the app to show it."""
    sd = budget.total_sd
    if sd < 6:
        return "high"
    if sd < 12:
        return "moderate"
    return "low"
