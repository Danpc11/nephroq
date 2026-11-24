"""
Vectorized simulator of the v2 renal model for MANY patients at once.

WHY THIS EXISTS
Simulation-based inference trains on hundreds of thousands of virtual patients,
and the exact reference posterior evaluates tens of thousands of parameter sets
per patient. `model_core.predict_egfr_at_v2` integrates one trajectory per
`solve_ivp` call; its per-call overhead, not the arithmetic, dominates. Here the
whole batch is integrated together with a fixed-step fourth-order Runge-Kutta
scheme on log N, which is smooth and non-stiff for every plausible patient
(the hazard is capped at HAZARD_CAP). Agreement with the LSODA reference is
checked in tests/test_fast_sim.py, across calibrations and treatment.

The hazard is `model_core.renal_hazard_v2`:
    h(N) = k0 + (1 - eff_hf u) k_hf s^q / (1 + (s/s_sat)^q)
              + (1 - eff_met u) [ w_a1c (A1c-6.5)+ + k_alb log(1 + UACR(N)/30)
                                  + w_sbp (SBP-130)+/10 ],
with s = 1/N the per-unit load index and UACR(N) = UACR0 (N0/N)^beta (1 - eff_alb u)
the endogenous albuminuria. A patient's personal injury rate `scale` multiplies
every injury coefficient (k_hf, w_a1c, k_alb, w_sbp) and leaves the baseline loss
k0 and the structure (q, s_sat, beta) untouched -- exactly as
`personalize.patient_params` does.

CONFIGURATION (audit P0.5). The parameters come from an explicit dict (default:
model_core.TRIAL_CALIBRATION_V2), never from hidden module state;
`model_core.config_hash(params)` identifies it, and trained artifacts store that
hash so an artifact built on another calibration is refused.
"""
import numpy as np

import model_core as core

DT_DEFAULT = 0.05          # years; RK4 step (max error vs LSODA ~5e-4 mL/min)


def _coeffs(params):
    c = core.sim_config(params)
    return c


def _hazard(lnN, q, scale, a1c_x, sbp_x, uacr0, lnN0, c, u):
    """Total per-nephron hazard for arrays of patients/parameter sets."""
    N = np.exp(lnN)
    s = 1.0 / np.clip(N, 1e-3, None)
    hf = c["k_hf"] * scale * s ** q / (1.0 + (s / c["s_sat"]) ** q) * (1.0 - c["eff_hf"] * u)
    uacr = uacr0 * np.exp(c["beta"] * (lnN0 - lnN)) * (1.0 - c["eff_alb"] * u)
    insult = scale * (c["w_a1c"] * a1c_x + c["k_alb"] * np.log1p(uacr / 30.0)
                      + c["w_sbp"] * sbp_x) * (1.0 - c["eff_met"] * u)
    return np.minimum(c["k0"] + hf + insult, core.HAZARD_CAP)


def simulate_egfr(q, scale, egfr0, a1c, uacr0, sbp, t_query, dt=DT_DEFAULT,
                  params=None, u=0.0):
    """
    eGFR at times `t_query` for a batch of B patients / parameter sets.

    All patient inputs broadcast to shape (B,). `t_query` is either a 1-D array
    shared by the whole batch, shape (T,), or a per-patient array, shape (B, T).
    `params` is the base calibration (default TRIAL_CALIBRATION_V2); its own `q`
    is ignored in favour of the `q` argument. `u` in [0, 1] is the treatment
    indicator. Returns an array of shape (B, T); B = 1 when every parameter is a
    scalar.
    """
    c = _coeffs(params)
    q, scale, egfr0, a1c, uacr0, sbp = np.broadcast_arrays(
        *(np.asarray(x, dtype=float) for x in (q, scale, egfr0, a1c, uacr0, sbp)))
    shape = q.shape
    q, scale, egfr0, a1c, uacr0, sbp = (x.ravel() for x in (q, scale, egfr0, a1c, uacr0, sbp))
    B = q.size

    tq = np.asarray(t_query, dtype=float)
    shared = tq.ndim == 1
    tq2 = np.broadcast_to(tq, (B, tq.size)) if shared else tq.reshape(B, -1)
    t_end = float(max(tq2.max(), dt))
    n_steps = int(np.ceil(t_end / dt))
    h = t_end / n_steps

    lnN0 = np.log(core.N_of_egfr(egfr0))
    a1c_x = np.maximum(a1c - 6.5, 0.0)
    sbp_x = np.maximum(sbp - 130.0, 0.0) / 10.0
    args = (q, scale, a1c_x, sbp_x, uacr0, lnN0, c, float(u))

    lnN = lnN0.copy()
    traj = np.empty((n_steps + 1, B))
    traj[0] = lnN
    for k in range(n_steps):
        k1 = -_hazard(lnN, *args)
        k2 = -_hazard(lnN + 0.5 * h * k1, *args)
        k3 = -_hazard(lnN + 0.5 * h * k2, *args)
        k4 = -_hazard(lnN + h * k3, *args)
        lnN = np.maximum(lnN + h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4), np.log(1e-3))
        traj[k + 1] = lnN

    # linear interpolation of log N onto each patient's query times
    pos = np.clip(tq2 / h, 0.0, n_steps)
    i0 = np.minimum(np.floor(pos).astype(int), n_steps - 1)
    w = pos - i0
    cols = np.arange(B)[:, None]
    lnN_q = (1 - w) * traj[i0, cols] + w * traj[i0 + 1, cols]
    egfr = core.egfr_of_N(np.exp(lnN_q))
    # scalar parameters still return a (1, T) batch, so callers index [0] safely
    return egfr.reshape((shape if shape else (1,)) + (tq2.shape[1],))
