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
checked in tests/test_fast_sim.py.

The hazard is the same as `model_core.renal_hazard_v2` with no treatment (u = 0):
    h(N) = k0 + k_hf s^q / (1 + (s/s_sat)^q)
              + w_a1c (A1c-6.5)+ + k_alb log(1 + UACR(N)/30) + w_sbp (SBP-130)+/10,
with s = 1/N the single-nephron load and UACR(N) = UACR0 (N0/N)^beta the
endogenous albuminuria. A patient's personal injury rate `scale` multiplies every
injury coefficient (k_hf, w_a1c, k_alb, w_sbp) and leaves the baseline loss k0
and the structure (q, s_sat, beta) untouched -- exactly as
`personalize.patient_params` does.
"""
import numpy as np

import model_core as core

# Population (trial-anchored) coefficients that a personal `scale` multiplies.
_P = core.TRIAL_CALIBRATION_V2
BASE_K_HF = float(_P["k_hf"])
BASE_W_A1C = float(_P["w_a1c"])
BASE_K_ALB = float(core._k_alb_of(_P))
BASE_W_SBP = float(_P["w_sbp"])
K0 = float(_P.get("k0", core.K0_DEFAULT))
S_SAT = float(_P.get("s_sat", core.S_SAT))
BETA = float(_P.get("beta", core.BETA))

DT_DEFAULT = 0.05          # years; RK4 step (max error vs LSODA ~5e-4 mL/min)


def _hazard(lnN, q, scale, a1c_x, sbp_x, uacr0, lnN0):
    """Total per-nephron hazard for arrays of patients/parameter sets."""
    N = np.exp(lnN)
    s = 1.0 / np.clip(N, 1e-3, None)
    hf = BASE_K_HF * scale * s ** q / (1.0 + (s / S_SAT) ** q)
    uacr = uacr0 * np.exp(BETA * (lnN0 - lnN))
    insult = scale * (BASE_W_A1C * a1c_x + BASE_K_ALB * np.log1p(uacr / 30.0)
                      + BASE_W_SBP * sbp_x)
    return np.minimum(K0 + hf + insult, core.HAZARD_CAP)


def simulate_egfr(q, scale, egfr0, a1c, uacr0, sbp, t_query, dt=DT_DEFAULT):
    """
    eGFR at times `t_query` for a batch of B patients / parameter sets.

    All inputs broadcast to shape (B,). `t_query` is either a 1-D array shared
    by the whole batch, shape (T,), or a per-patient array, shape (B, T) (pad
    unused entries with any time inside the horizon and ignore them).
    Returns an array of shape (B, T); B = 1 when every parameter is a scalar.
    """
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
    args = (q, scale, a1c_x, sbp_x, uacr0, lnN0)

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
