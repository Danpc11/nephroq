"""
================================================================================
MODEL CORE  ·  Single source of truth for the mechanistic renal model
================================================================================
Every script that simulates the renal trajectory for PRODUCTION use (the web
app, the MIMIC-IV calibration) MUST import from here instead of reimplementing
the equations.

This module exists because two independent implementations previously
diverged silently:
  - calibrate_mimic.py used an explicit fixed-step RK4 integrator.
  - mechanistic_twin.py (used by the app) used scipy's adaptive solve_ivp.
Both encoded "the same" hazard equation, but produced trajectories differing
by up to ~11 mL/min/1.73m² near the terminal collapse region -- exactly where
time-to-eGFR<15 decisions are made. See docs/KNOWN_ISSUES.md for the
before/after numeric comparison that caught this.

    dN/dt = -N * [ k0 + k_hf*(N_ref/N)^q + insult ]
    eGFR  = G_max * N^alpha
================================================================================
"""
import numpy as np
from scipy.integrate import solve_ivp

G_MAX = 120.0
ALPHA = 0.80
N_FLOOR = 1e-4
K0_DEFAULT = 0.0030
DIALYSIS_eGFR = 15.0


def egfr_of_N(N):
    return G_MAX * np.power(np.clip(N, 1e-9, None), ALPHA)


def N_of_egfr(egfr):
    """Maps eGFR -> N, clipped to (0,1] as documented (N(t) in (0,1])."""
    N = np.power(np.clip(egfr, 1e-6, None) / G_MAX, 1.0 / ALPHA)
    return np.clip(N, N_FLOOR, 1.0)


N_DIALYSIS = N_of_egfr(DIALYSIS_eGFR)


def metabolic_hazard(a1c, uacr, sbp, w_a1c, w_uacr, w_sbp):
    """Explicit-weights metabolic insult (already-scaled calibrated weights,
    e.g. from calibrate_mimic.py). I >= 0."""
    return (w_a1c * max(a1c - 6.5, 0.0)
            + w_uacr * np.log1p(uacr / 30.0)
            + w_sbp * max(sbp - 130.0, 0.0) / 10.0)


def literature_metabolic_hazard(a1c, uacr, sbp):
    """Original fixed-literature-weight insult (0.40, 0.50, 0.30), for the
    default (non-calibrated) physiological parameterization."""
    return (0.40 * max(a1c - 6.5, 0.0)
            + 0.50 * np.log1p(uacr / 30.0)
            + 0.30 * max(sbp - 130.0, 0.0) / 10.0)


def renal_hazard(N, k0, k_hf, q, N_ref, insult):
    """Hazard per nephron (1/year). Grows as N falls (hyperfiltration)."""
    N = max(N, N_FLOOR)
    return min(k0 + k_hf * (N_ref / N) ** q + insult, 50.0)


def simulate_trajectory(k0, k_hf, q, N_ref, insult, N0, years, n=600):
    """
    THE canonical trajectory simulator (adaptive-step solve_ivp). Both the
    app (via MechanisticRenalModel) and calibrate_mimic.py call this same
    function -- there is no second implementation of the integration anymore.

    Returns: (t_eval, N, egfr, t_dialysis)
    """
    def rhs(t, y):
        N = y[0]
        return [-N * renal_hazard(N, k0, k_hf, q, N_ref, insult)]

    t_eval = np.linspace(0, years, n)
    sol = solve_ivp(rhs, [0, years], [N0], t_eval=t_eval, rtol=1e-8, atol=1e-10)
    N = np.clip(sol.y[0], N_FLOOR, 1.0)
    egfr = egfr_of_N(N)
    below = np.where(N < N_DIALYSIS)[0]
    t_dial = t_eval[below[0]] if len(below) else np.inf
    return t_eval, N, egfr, t_dial


def predict_egfr_at(k0, k_hf, q, N_ref, insult, N0, t_query, dt_max=0.05):
    """
    Convenience wrapper for calibration: simulate once up to max(t_query) and
    interpolate at the requested query times. Used by calibrate_mimic.py's
    residual function (called many times per optimizer iteration).
    """
    t_max = float(np.max(t_query)) + dt_max
    n = max(int(t_max / dt_max), 50)
    t_eval, N, egfr, _ = simulate_trajectory(k0, k_hf, q, N_ref, insult, N0, t_max, n=n)
    return np.clip(np.interp(t_query, t_eval, egfr), 0, G_MAX)


def gfr_category(egfr):
    """
    KDIGO GFR category: G1>=90, G2 60-89, G3a 45-59, G3b 30-44, G4 15-29, G5<15.
    Single source of truth -- the app and tests both import this instead of
    each reimplementing the boundaries (a duplicate copy could silently drift
    from the real one and still pass its own tests).

    Note: a single eGFR value gives a GFR CATEGORY, not a CKD diagnosis --
    KDIGO defines CKD by abnormalities persisting >=3 months plus cause and
    albuminuria staging.
    """
    if egfr >= 90: return "G1"
    if egfr >= 60: return "G2"
    if egfr >= 45: return "G3a"
    if egfr >= 30: return "G3b"
    if egfr >= 15: return "G4"
    return "G5"
