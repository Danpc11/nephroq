"""
Clinical outputs (roadmap block 8).

The core returns an eGFR trajectory and a modelled time to eGFR<15. A clinician
needs that turned into the quantities they actually reason about: the probability
of crossing KDIGO categories, the expected time to G4/G5, the chance of rapid
progression, and a 40% eGFR decline (a validated surrogate endpoint). This module
computes those from the predictive band (block 7), so every probability carries the
full uncertainty, not just parameter spread.

DELIBERATE OMISSION, kept from the project's stance: this does NOT report a
'dialysis start date'. eGFR<15 is not the same as initiating kidney replacement
therapy -- the decision depends on symptoms, access, and patient choice -- and the
project has been careful about that distinction throughout. We report P(eGFR<15 by
horizon) and 'potential need for specialist referral', not a KRT date.

This module composes model outputs; it holds no hazard math.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

import model_core as core
import uncertainty as unc


# KDIGO category thresholds (upper bound of each, mL/min/1.73m^2)
CATEGORY_BOUNDS = {"G2": 90, "G3a": 60, "G3b": 45, "G4": 30, "G5": 15}
RAPID_PROGRESSION_ML_PER_YEAR = 5.0    # KDIGO 'rapid progression' definition


def _time_to_egfr(t, egfr, threshold):
    """First time the trajectory crosses below `threshold`, by interpolation, or
    None if it never does within the horizon."""
    egfr = np.asarray(egfr, float)
    below = np.where(egfr < threshold)[0]
    if below.size == 0:
        return None
    i = below[0]
    if i == 0:
        return float(t[0])
    # linear interpolation between the straddling points
    t0, t1 = t[i - 1], t[i]
    e0, e1 = egfr[i - 1], egfr[i]
    if e0 == e1:
        return float(t1)
    return float(t0 + (threshold - e0) * (t1 - t0) / (e1 - e0))


@dataclass
class ClinicalOutputs:
    egfr0: float
    current_category: str
    # probabilities (0..1) at the requested horizons
    p_40pct_decline: dict          # horizon -> P(eGFR drops >=40% from baseline)
    p_g4: dict                     # horizon -> P(reach G4, eGFR<30)
    p_g5: dict                     # horizon -> P(reach G5, eGFR<15)
    p_rapid_progression: float     # P(mean slope steeper than 5 mL/min/yr)
    time_to_g3b: Optional[float]
    time_to_g4: Optional[float]
    time_to_g5: Optional[float]
    expected_category_change: dict # horizon -> most likely KDIGO category
    referral_flag: bool            # potential need for specialist assessment

    def to_dict(self):
        return {k: (v if not isinstance(v, dict) else {str(kk): vv for kk, vv in v.items()})
                for k, v in self.__dict__.items()}


def clinical_outputs(egfr0, a1c, uacr0, sbp, params, horizons=(2.0, 5.0, 10.0),
                     u=0.0, scale_spread=None, param_bootstrap=None,
                     n_draws=400, seed=0):
    """
    Compute the clinical output panel with full predictive uncertainty.

    Probabilities are Monte-Carlo estimates over the same five-source draw as the
    predictive band, so P(G4 by 5y) reflects measurement noise and individual
    susceptibility, not just parameter spread.
    """
    horizons = tuple(float(h) for h in horizons)
    rng = np.random.default_rng(seed)
    base = dict(params)

    # draw a population of trajectories (reusing the band's source model)
    Hmax = max(horizons)
    tgrid = np.linspace(0.0, Hmax, 80)
    point = core.predict_egfr_at_v2(egfr0, a1c, uacr0, sbp, u, base, tgrid,
                                    years=float(Hmax))

    # rebuild the per-draw simulations for probability estimates
    sims = _simulate_population(egfr0, a1c, uacr0, sbp, base, tgrid, u,
                                scale_spread, param_bootstrap, n_draws, rng)
    p40, pg4, pg5, cat_change = {}, {}, {}, {}
    for h in horizons:
        j = int(np.searchsorted(tgrid, h))
        j = min(j, len(tgrid) - 1)
        eh = sims[:, j]
        p40[h] = float(np.mean(eh <= 0.6 * egfr0))
        pg4[h] = float(np.mean(eh < 30.0))
        pg5[h] = float(np.mean(eh < 15.0))
        cat_change[h] = core.gfr_category(float(np.median(eh)))

    # rapid progression: mean slope over the first 3 years (or the horizon)
    t_slope = min(3.0, Hmax)
    js = int(np.searchsorted(tgrid, t_slope))
    slopes = (sims[:, 0] - sims[:, js]) / max(tgrid[js], 1e-6)
    p_rapid = float(np.mean(slopes > RAPID_PROGRESSION_ML_PER_YEAR))

    return ClinicalOutputs(
        egfr0=float(egfr0),
        current_category=core.gfr_category(egfr0),
        p_40pct_decline=p40, p_g4=pg4, p_g5=pg5,
        p_rapid_progression=p_rapid,
        time_to_g3b=_time_to_egfr(tgrid, point, 45.0),
        time_to_g4=_time_to_egfr(tgrid, point, 30.0),
        time_to_g5=_time_to_egfr(tgrid, point, 15.0),
        expected_category_change=cat_change,
        # refer if there is a non-trivial chance of reaching G4 within 5 years
        referral_flag=(pg4.get(5.0, pg4[min(horizons)]) > 0.20),
    )


def _simulate_population(egfr0, a1c, uacr0, sbp, base, tgrid, u,
                         scale_spread, param_bootstrap, n_draws, rng):
    """Monte-Carlo population of full trajectories, sharing the band's source
    model, for probability estimates over KDIGO thresholds."""
    meas_sd = unc.SIGMA_MEASUREMENT_MLMIN
    egfr_draws = np.clip(egfr0 + rng.normal(0, meas_sd, n_draws), 5.0, 150.0)
    if scale_spread and scale_spread > 0:
        rate = np.exp(rng.normal(0.0, scale_spread, n_draws))
    else:
        rate = np.ones(n_draws)
    if param_bootstrap:
        pool = [param_bootstrap[i] for i in rng.integers(0, len(param_bootstrap), n_draws)]
    else:
        pool = []
        for _ in range(n_draws):
            p = dict(base); p["k_hf"] = base["k_hf"] * (1 + rng.normal(0, 0.10))
            pool.append(p)
    sims = np.empty((n_draws, len(tgrid)))
    for d in range(n_draws):
        p = dict(pool[d])
        p["k_hf"] = p["k_hf"] * rate[d]
        p["w_a1c"] = p["w_a1c"] * rate[d]
        p["w_uacr"] = p["w_uacr"] * rate[d]
        p["w_sbp"] = p["w_sbp"] * rate[d]
        sims[d] = core.predict_egfr_at_v2(float(egfr_draws[d]), a1c, uacr0, sbp, u,
                                          p, tgrid, years=float(tgrid[-1]))
    return sims
