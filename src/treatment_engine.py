"""
Treatment scenario engine (roadmap block 5).

The model's core takes a single scalar u in [0,1] that attenuates the
hyperfiltration, metabolic, and albuminuric terms with one set of efficacies. That
was right for validating the mechanism, but it cannot represent 'this patient on an
SGLT2i AND a RAAS inhibitor', or 'started finerenone at month 6'. Clinical support
needs named drugs, each with its own mechanism, evidence base, and uncertainty.

This engine sits ABOVE model_core. It does not rewrite the ODE. It composes a
per-mechanism effective u (u_hf, u_met, u_alb) from a drug regimen, and hands the
model a parameter set whose eff_* reflect that regimen. The tested core is
untouched; scenarios are a layer, which is exactly where a policy-like 'what if'
belongs.

IMPORTANT FRAMING (kept, deliberately): this is SCENARIO EVALUATION, not a
recommendation of the single best drug. Recommending an individual therapy needs
causal evidence this model does not carry. The engine answers 'what would this
regimen do, under the model', with an uncertainty band, and says which population
each drug's effect was estimated in.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

import model_core as core


# ------------------------------------------------------------------------------
# Per-drug mechanisms. Efficacies are fractional attenuations of a mechanism,
# with a plausible range, and a note on the evidence + population.
#
# These are ILLUSTRATIVE defaults anchored to the direction and rough magnitude of
# the published trials, NOT freshly fitted parameters. They are the knobs a
# scenario turns; the honest uncertainty band comes from the (lo, hi) ranges.
# ------------------------------------------------------------------------------
@dataclass
class DrugEffect:
    name: str
    # fractional attenuation of each mechanism (0 = no effect, 1 = fully blocked)
    hf: float = 0.0            # hyperfiltration
    met: float = 0.0           # metabolic insult
    alb: float = 0.0           # albuminuria
    hf_range: tuple = (0.0, 0.0)
    alb_range: tuple = (0.0, 0.0)
    evidence: str = ""
    population: str = ""


# Defaults. SGLT2i mirrors the model's existing trial-anchored eff_* (that is where
# they came from). The others are directionally reasonable placeholders, flagged as
# such, so the engine's shape is usable while the exact numbers are refined.
DRUGS = {
    "sglt2i": DrugEffect(
        "SGLT2 inhibitor", hf=0.521, met=0.669, alb=0.286,
        hf_range=(0.40, 0.62), alb_range=(0.20, 0.40),
        evidence="CREDENCE / DAPA-CKD / EMPA-KIDNEY (placebo-controlled)",
        population="T2D + CKD, mostly albuminuric"),
    "raasi": DrugEffect(
        "RAAS inhibitor (ACEi/ARB)", hf=0.30, met=0.0, alb=0.35,
        hf_range=(0.20, 0.40), alb_range=(0.25, 0.45),
        evidence="RENAAL / IDNT (background therapy in the SGLT2i trials)",
        population="T2D nephropathy, albuminuric"),
    "finerenone": DrugEffect(
        "Finerenone (ns-MRA)", hf=0.15, met=0.0, alb=0.30,
        hf_range=(0.08, 0.22), alb_range=(0.20, 0.40),
        evidence="FIDELIO-DKD / FIGARO-DKD",
        population="T2D + CKD, albuminuric, on RAASi"),
    "glp1ra": DrugEffect(
        "GLP-1 receptor agonist", hf=0.10, met=0.40, alb=0.20,
        hf_range=(0.05, 0.18), alb_range=(0.10, 0.30),
        evidence="FLOW (semaglutide, kidney outcomes)",
        population="T2D + CKD"),
}


@dataclass
class Regimen:
    """A set of drugs the patient is on. Effects on each mechanism combine
    multiplicatively (each drug attenuates what the previous left), which avoids
    the impossible >100% blockade that naive addition would give."""
    drugs: tuple = ()

    def _combined(self, which, lo_hi=None):
        """Combined attenuation of one mechanism across the regimen.
        multiplicative: 1 - prod(1 - e_i)."""
        remaining = 1.0
        for d in self.drugs:
            eff = DRUGS.get(d)
            if eff is None:
                continue
            if lo_hi is None:
                e = getattr(eff, which)
            else:
                rng = getattr(eff, f"{which}_range", None)
                e = rng[lo_hi] if rng else getattr(eff, which)
            remaining *= (1.0 - e)
        return 1.0 - remaining

    def effective_effects(self, lo_hi=None):
        return dict(hf=self._combined("hf", lo_hi),
                    met=self._combined("met", lo_hi),
                    alb=self._combined("alb", lo_hi))


def params_for_regimen(base_params: dict, regimen: Regimen, lo_hi=None) -> dict:
    """
    Build a parameter set whose eff_* encode this regimen, so the model can be run
    with u=1 to apply exactly this combination. lo_hi selects the low (0) or high
    (1) end of the efficacy ranges for the uncertainty band; None uses point
    estimates.
    """
    eff = regimen.effective_effects(lo_hi)
    p = dict(base_params)
    p["eff_hf"] = eff["hf"]
    p["eff_met"] = eff["met"]
    p["eff_alb"] = eff["alb"]
    return p


def evaluate_scenario(egfr0, a1c, uacr0, sbp, regimen: Regimen,
                      base_params: Optional[dict] = None, years=15, n=200):
    """
    Project the trajectory under a drug regimen, with an uncertainty band from the
    per-drug efficacy ranges. Returns t, the point projection, and lo/hi bands.

    This is scenario EVALUATION: 'what would this regimen do, under the model'.
    """
    base = dict(base_params or core.TRIAL_CALIBRATION_V2)
    u = 1.0 if regimen.drugs else 0.0

    def run(lo_hi):
        p = params_for_regimen(base, regimen, lo_hi)
        t, e, _, td = core.simulate_trajectory_v2(egfr0, a1c, uacr0, sbp, u=u, p=p,
                                                  years=years, n=n)
        return t, e, td

    t, e_mid, td_mid = run(None)
    # band: 'lo' efficacy -> faster decline (worse); 'hi' efficacy -> slower (better)
    _, e_worse, _ = run(0)
    _, e_better, _ = run(1)
    return dict(t=t, egfr=e_mid, egfr_worse=e_worse, egfr_better=e_better,
                time_to_threshold=td_mid, regimen=[d for d in regimen.drugs])


def compare_regimens(egfr0, a1c, uacr0, sbp, regimens: dict,
                     base_params: Optional[dict] = None, years=15):
    """Evaluate several named regimens on the same patient for side-by-side
    scenario comparison. Returns {name: scenario_result}."""
    return {name: evaluate_scenario(egfr0, a1c, uacr0, sbp, reg, base_params, years)
            for name, reg in regimens.items()}


def regimen_evidence(regimen: Regimen) -> list:
    """The evidence and population behind each drug in a regimen -- so a scenario is
    presented with its provenance, never as a bare recommendation."""
    out = []
    for d in regimen.drugs:
        eff = DRUGS.get(d)
        if eff:
            out.append(dict(drug=eff.name, evidence=eff.evidence,
                            population=eff.population))
    return out
