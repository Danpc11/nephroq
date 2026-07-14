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
  - model_core.py (used by the app) used scipy's adaptive solve_ivp.
Both encoded "the same" hazard equation, but produced trajectories differing
by up to ~11 mL/min/1.73m² near the terminal collapse region -- exactly where
time-to-eGFR<15 decisions are made. See the README (Limitations) for the
before/after numeric comparison that caught this.

    dN/dt = -N * h(N)

    h(N) = k0
         + k_hf * s^q / (1 + (s/S_SAT)^q)        <- SATURATING hyperfiltration
         + insult(HbA1c, UACR(t), SBP)           <- UACR(t) is ENDOGENOUS

    with s = N_ref/N, and  UACR(t) = UACR_0 * (s(t)/s_0)^BETA * (1 - eff_alb*u).

    THIS is the production model (v2) -- the equation that produced every figure,
    every calibration and every validation in the repository. The unbounded power
    law k_hf*(N_ref/N)^q was v1: it diverged as nephrons were lost, over-predicted
    decline in advanced CKD, and FAILED the in-silico trial replication. It is
    retained below only because a few historical helpers still reference it; it is
    NOT used by the app, the calibration or the tests.
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


# ==============================================================================
# MODEL v2 -- saturating hyperfiltration + endogenous albuminuria
# ==============================================================================
# Two structural corrections, each forced by a concrete failure of v1 in the
# in-silico trial replication (see docs/TRIAL_DATA_AND_MODEL_IMPROVEMENT.md and
# src/insilico_trial.py).
#
# 1) HYPERFILTRATION SATURATES.
#    v1: h_hf = k_hf * s^q          (s = N_ref/N)  -> diverges as N -> 0.
#    A surviving nephron raises its single-nephron GFR by a bounded factor
#    (~3x), not without limit. The unbounded power law is why v1 over-predicted
#    decline hardest in advanced CKD.
#        v2: h_hf = k_hf * s^q / (1 + (s/S_SAT)^q)
#    S_SAT is IDENTIFIED from trial data (not fitted freely): anchoring the
#    hazard on CREDENCE's placebo arm (mean eGFR 56) and scoring on
#    EMPA-KIDNEY's (mean eGFR 37) gives a clear optimum at S_SAT ~ 3-4,
#    consistent with the physiological ceiling. The unbounded law (S_SAT -> inf)
#    is 15x worse on that held-out placebo arm.
#
# 2) ALBUMINURIA IS ENDOGENOUS.
#    v1 fed UACR in as a CONSTANT exogenous insult. That is mechanistically
#    backwards -- albuminuria is largely a CONSEQUENCE of glomerular
#    hypertension -- it double-counts the same process, and it made a published
#    fact structurally inexpressible (SGLT2i lower UACR by ~31-35%).
#        v2: UACR(t) = UACR_0 * (s(t)/s_0)^BETA * (1 - eff_alb * u)
#    Albuminuria becomes a dynamic readout of the hyperfiltration state plus a
#    direct drug effect, and the hazard uses the CURRENT UACR. It still enters
#    the hazard in its own right (proteinuria is tubulotoxic -- a pathway
#    distinct from hemodynamic injury), so this is a coupling, not a removal.
#
# WHAT IS ACTUALLY USED: v2 (below) is THE production model. The app, the
# calibration (calibrate_mimic.py), the own-data path (mvp_calibration.py), the
# personalizer and the in-silico validation all route through
# simulate_trajectory_v2 / predict_egfr_at_v2. The v1 helpers above are legacy and
# are NOT on any live path -- do not read the v1 equation as the one behind the
# results.
# ==============================================================================

S_SAT = 3.5      # ceiling on compensatory single-nephron hyperfiltration
BETA = 1.0       # albuminuria scales ~linearly with the hyperfiltration ratio
HAZARD_CAP = 50.0   # 1/yr; numerical guard only (see renal_hazard_v2), never biology

# Parameters anchored to PUBLISHED TRIAL DATA rather than to MIMIC.
# Progression (hazard scale) is fixed by the placebo arms of CREDENCE and
# EMPA-KIDNEY; the treatment effects by CREDENCE's chronic-slope difference and
# its 31% UACR reduction. DAPA-CKD is then predicted OUT-OF-SAMPLE and passes
# (see results/insilico_trial_report.md).
TRIAL_CALIBRATION_V2 = dict(
    q=1.52,
    k_hf=0.0141 * 0.730,                    # hazard scale 0.730
    k0=0.0030,
    w_a1c=0.0144 * 0.730,
    w_uacr=0.0180 * 0.730,
    w_sbp=0.0108 * 0.730,
    eff_met=0.669,
    eff_hf=0.521,
    eff_alb=0.286,
    s_sat=S_SAT,
    beta=BETA,
    source="in-silico replication: fitted on CREDENCE + EMPA-KIDNEY placebo arms; "
           "DAPA-CKD predicted out-of-sample (chronic slope and UACR reduction both "
           "inside the published 95% CI).",
)


def hyperfiltration_hazard_v2(N, k_hf, q, s_sat=S_SAT, n_ref=1.0):
    """Saturating hyperfiltration term. Bounded by k_hf * s_sat**q."""
    N = max(float(N), 1e-3)
    s = n_ref / N
    return k_hf * (s ** q) / (1.0 + (s / s_sat) ** q)


def uacr_of_state(N, N0, uacr0, u=0.0, eff_alb=0.0, beta=BETA):
    """Albuminuria as a readout of the hyperfiltration state + direct drug effect."""
    s = 1.0 / max(float(N), 1e-3)
    s0 = 1.0 / max(float(N0), 1e-3)
    return uacr0 * (s / s0) ** beta * (1.0 - eff_alb * u)


def renal_hazard_v2(N, N0, a1c, uacr0, sbp, u, p):
    """Total per-nephron hazard under v2. `p` is a TRIAL_CALIBRATION_V2-like dict."""
    uacr_t = uacr_of_state(N, N0, uacr0, u, p["eff_alb"], p.get("beta", BETA))
    hf = hyperfiltration_hazard_v2(N, p["k_hf"], p["q"], p.get("s_sat", S_SAT))
    hf *= (1.0 - p["eff_hf"] * u)
    insult = metabolic_hazard(a1c, uacr_t, sbp, p["w_a1c"], p["w_uacr"], p["w_sbp"])
    insult *= (1.0 - p["eff_met"] * u)
    # HAZARD CAP. 50/yr is a numerical guard, not biology: at that rate a nephron
    # population halves roughly every 5 days, which no patient survives. It exists
    # so the integrator cannot blow up while an optimizer explores an absurd corner
    # of parameter space. It should never bind for a plausible patient -- if it
    # does, the parameters are wrong, not the patient.
    return min(p["k0"] + hf + insult, HAZARD_CAP)


def simulate_trajectory_v2(egfr0, a1c, uacr0, sbp, u=0.0, p=None, years=15, n=400):
    """
    Canonical v2 simulation. Returns (t, egfr, uacr, t_threshold).

    Unlike v1 this also returns the PREDICTED ALBUMINURIA TRAJECTORY, which is a
    genuine model output now, not an input held constant.
    """
    p = TRIAL_CALIBRATION_V2 if p is None else p
    N0 = N_of_egfr(egfr0)

    def rhs(t, y):
        N = max(y[0], 1e-3)
        return [-N * renal_hazard_v2(N, N0, a1c, uacr0, sbp, u, p)]

    t_eval = np.linspace(0, years, n)
    sol = solve_ivp(rhs, (0, years), [N0], t_eval=t_eval, method="LSODA",
                    rtol=1e-6, atol=1e-9)
    N = np.maximum(sol.y[0], 1e-3)
    egfr = np.array([egfr_of_N(x) for x in N])
    uacr = np.array([uacr_of_state(x, N0, uacr0, u, p["eff_alb"], p.get("beta", BETA))
                     for x in N])

    t_thr = np.inf
    below = np.where(egfr < DIALYSIS_eGFR)[0]
    if len(below):
        i = below[0]
        if i == 0:
            t_thr = 0.0
        else:  # linear interpolation onto the threshold crossing
            e0, e1 = egfr[i - 1], egfr[i]
            f = (e0 - DIALYSIS_eGFR) / (e0 - e1) if e0 != e1 else 0.0
            t_thr = sol.t[i - 1] + f * (sol.t[i] - sol.t[i - 1])
    return sol.t, egfr, uacr, t_thr


def predict_egfr_at_v2(egfr0, a1c, uacr0, sbp, u, p, t_query, years=None):
    """
    v2 predictor for CALIBRATION: eGFR at arbitrary query times.

    Note it takes only BASELINE covariates. In v2 albuminuria is endogenous, so
    a patient's UACR trajectory is GENERATED by the model rather than fed in.
    That also removes the dependence on imputed per-visit UACR values, which
    were the least reliable input in any hospital dataset.

    PERFORMANCE: this integrates straight onto the requested times (`t_eval`)
    instead of building a dense grid and interpolating onto it. It is ~7x faster
    and numerically identical (agreement to <1e-6 mL/min), which matters because
    a calibration run evaluates this millions of times.
    """
    t_query = np.atleast_1d(np.asarray(t_query, dtype=float))
    t_end = float(max(t_query.max(), 1e-3)) if years is None else float(years)

    N0 = N_of_egfr(egfr0)
    t_clipped = np.clip(t_query, 0.0, t_end)

    # solve_ivp requires t_eval to be STRICTLY increasing. Real visit series are
    # not: hospital data routinely has several creatinines on the same day (ties),
    # and a caller may pass times in any order. Integrate on the unique sorted
    # times and scatter the results back -- otherwise this raises
    # "Values in `t_eval` are not properly sorted" on perfectly valid data.
    t_unique, inverse = np.unique(t_clipped, return_inverse=True)

    def rhs(t, y):
        N = max(y[0], 1e-3)
        return [-N * renal_hazard_v2(N, N0, a1c, uacr0, sbp, u, p)]

    sol = solve_ivp(rhs, (0.0, t_end), [N0], t_eval=t_unique, method="LSODA",
                    rtol=1e-6, atol=1e-9)
    if not sol.success or sol.y.shape[1] != len(t_unique):
        # fall back to the canonical simulator rather than returning garbage
        t, egfr, _, _ = simulate_trajectory_v2(egfr0, a1c, uacr0, sbp, u=u, p=p,
                                               years=t_end, n=200)
        return np.interp(t_query, t, egfr)
    egfr_unique = np.array([egfr_of_N(max(x, 1e-3)) for x in sol.y[0]])
    return egfr_unique[inverse]
