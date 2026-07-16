"""
Acute renal events (roadmap block 6): AKI episodes and the SGLT2i hemodynamic dip.

The core model's N(t) is monotonic -- it represents irreversible loss of functional
nephron mass, and that is correct for CHRONIC progression. But two clinically
important things are NOT chronic nephron loss:

  1. An AKI episode: eGFR drops sharply, partly recovers, and often leaves the
     patient more susceptible afterwards. Feeding the acute nadir in as a chronic
     baseline is exactly the bias the confirmed-index date was built to avoid;
     modelling AKI explicitly is the principled fix.
  2. The SGLT2i acute dip: starting an SGLT2 inhibitor causes a small, reversible
     eGFR drop over the first weeks (the README notes the core does not model it),
     which then reverses into a slower chronic slope. Reading that dip as
     progression would misjudge the drug.

Both are TRANSIENT deflections ON TOP of the chronic trajectory, so both are
modelled the same way and kept OUT of the core ODE:

    eGFR_observed(t) = eGFR_chronic(t) - D(t)

where D(t) is a sum of decaying acute deflections. N(t) stays monotonic; the acute
layer is additive and reversible. An AKI episode may additionally raise the
patient's susceptibility going forward:

    s_i  ->  s_i * (1 + rho * severity)

This module computes D(t) and the susceptibility bump. It does not touch the core.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


# Default recovery time constants (years). AKI resolves over weeks; the SGLT2i dip
# over a few weeks too, but it is smaller and essentially fully reversible.
TAU_AKI = 0.12          # ~6 weeks e-folding
TAU_SGLT2_DIP = 0.08    # ~4 weeks
RHO_SUSCEPTIBILITY = 0.15   # fractional s_i increase per unit AKI severity


@dataclass
class AcuteEvent:
    """A transient eGFR deflection. `magnitude` is the initial drop in mL/min;
    `tau` is its recovery time constant in years; `permanent_fraction` is the part
    that does NOT recover (a step, folded into chronic loss elsewhere)."""
    t_onset: float                 # years, on the same clock as the trajectory
    magnitude: float               # mL/min, initial acute drop (positive number)
    tau: float = TAU_AKI
    permanent_fraction: float = 0.0    # 0..1 of magnitude that never recovers
    kind: str = "aki"              # 'aki' | 'sglt2_dip'
    severity: float = 1.0          # for the susceptibility bump (AKI only)


def deflection(t, events) -> np.ndarray:
    """
    D(t): total transient eGFR loss at each time, summed over events.

    Each event contributes, for t >= onset:
        recoverable part:  mag*(1 - permanent) * exp(-(t - onset)/tau)
        permanent part:    mag*permanent            (a step from onset onward)
    Before onset it contributes nothing.
    """
    t = np.asarray(t, float)
    D = np.zeros_like(t)
    for ev in events:
        after = t >= ev.t_onset
        dt = np.where(after, t - ev.t_onset, 0.0)
        recoverable = ev.magnitude * (1.0 - ev.permanent_fraction) * np.exp(-dt / ev.tau)
        permanent = ev.magnitude * ev.permanent_fraction
        D = D + np.where(after, recoverable + permanent, 0.0)
    return D


def apply_acute(t, egfr_chronic, events) -> np.ndarray:
    """eGFR_observed = eGFR_chronic - D(t), floored at a physiological minimum."""
    obs = np.asarray(egfr_chronic, float) - deflection(t, events)
    return np.clip(obs, 1.0, None)


def sglt2_dip_event(t_onset, egfr_at_onset, dip_fraction=0.06) -> AcuteEvent:
    """
    The SGLT2i initiation dip: a reversible drop of ~5-6% of current eGFR over the
    first weeks (consistent with the published acute eGFR change on drug start).
    Fully recoverable (permanent_fraction 0): it reverses into the chronic benefit,
    which the core already models via the eff_* attenuation.
    """
    return AcuteEvent(t_onset=t_onset,
                      magnitude=float(dip_fraction) * float(egfr_at_onset),
                      tau=TAU_SGLT2_DIP, permanent_fraction=0.0, kind="sglt2_dip")


def aki_event(t_onset, drop_mL, permanent_fraction=0.2, severity=1.0) -> AcuteEvent:
    """
    An AKI episode: a sharp drop, mostly recovering, with a fraction that becomes
    permanent nephron loss and a susceptibility bump afterward (see
    susceptibility_after).
    """
    return AcuteEvent(t_onset=t_onset, magnitude=float(drop_mL), tau=TAU_AKI,
                      permanent_fraction=float(permanent_fraction), kind="aki",
                      severity=float(severity))


def susceptibility_after(scale, events, rho=RHO_SUSCEPTIBILITY) -> float:
    """
    Multiply the patient's susceptibility s_i by (1 + rho*severity) for each AKI
    episode: a patient who has had AKI progresses faster afterward. SGLT2i dips do
    NOT raise susceptibility (they are benign and reversible).
    """
    s = float(scale)
    for ev in events:
        if ev.kind == "aki":
            s *= (1.0 + rho * ev.severity)
    return s


def is_probably_acute(egfr_series, times, recovery_frac=0.30, window_days=90):
    """
    Flag whether an eGFR reading looks like an acute dip rather than a chronic
    level: it 'recovers' by more than recovery_frac within window_days afterward.
    This is the same logic as the confirmed-index rule, exposed for the twin so it
    can label a suspicious low reading instead of treating it as the new baseline.
    """
    e = np.asarray(egfr_series, float)
    t = np.asarray(times, float)  # years
    flags = np.zeros(len(e), dtype=bool)
    win = window_days / 365.25
    for i in range(len(e)):
        later = (t - t[i] > 0) & (t - t[i] <= win)
        if later.any() and e[i] > 0:
            if (e[later].max() - e[i]) / e[i] > recovery_frac:
                flags[i] = True
    return flags
