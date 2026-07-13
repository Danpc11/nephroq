"""
================================================================================
MODEL v2  ·  Structural improvements to the NephroQ hazard
================================================================================
Two changes, both motivated by concrete failures of v1 in the in-silico trial
replication (see docs/TRIAL_DATA_AND_MODEL_IMPROVEMENT.md).

--------------------------------------------------------------------------------
PROBLEM 1 -- the hyperfiltration term is an UNBOUNDED power law.

    v1:   h_hf = k_hf * s^q            with s = N_ref / N

    As N -> 0, this diverges. Biologically it should not: a surviving nephron
    hypertrophies and raises its single-nephron GFR by a factor of ~2-3, not
    without limit. This is exactly why v1 over-predicts decline hardest in the
    LOW-eGFR trial (DAPA-CKD, mean eGFR 43) and why a uniform rescale left a
    residual 1.12x error there -- the error is in the SHAPE, not the scale.

    v2:   h_hf = k_hf * s^q / (1 + (s/s_sat)^q)          [Hill saturation]

    - small s  ->  ~ k_hf * s^q          (recovers v1; early CKD unchanged)
    - large s  ->  ~ k_hf * s_sat^q      (bounded; no blow-up in advanced CKD)

    s_sat is NOT fitted. It is FIXED at 3.0 from physiology (the ceiling on
    compensatory single-nephron hyperfiltration). Fitting it would turn a
    prediction into a curve fit.

--------------------------------------------------------------------------------
PROBLEM 2 -- albuminuria is treated as an EXOGENOUS CONSTANT insult.

    v1:   I = ... + w_uacr * log1p(UACR_0 / 30) + ...     (UACR_0 fixed forever)

    Two things are wrong with this:
      (a) It is mechanistically backwards. Albuminuria is not an external driver
          poured onto the kidney; it is largely a CONSEQUENCE of glomerular
          hypertension/hyperfiltration. Holding it constant while nephrons are
          lost double-counts the same underlying process, and it is the reason
          the UACR term ends up carrying 33% of the hazard -- via the weight
          (w_uacr) that MIMIC could least identify (63% of UACR imputed).
      (b) It makes a published fact UNPREDICTABLE: SGLT2 inhibitors lower UACR
          by ~31-35%. v1 structurally cannot reproduce this, because UACR is an
          input that never moves.

    v2:   UACR(t) = UACR_0 * (s(t)/s_0)^beta * (1 - eff_alb * u)

    Albuminuria becomes a DYNAMIC READOUT of the hyperfiltration state, plus a
    direct pharmacological effect (SGLT2i constrict the afferent arteriole ->
    lower intraglomerular pressure -> less albumin leak). The hazard then uses
    the CURRENT UACR, not a frozen baseline.

    Albuminuria still enters the hazard on its own (proteinuria is genuinely
    tubulotoxic -- a pathway distinct from hemodynamic injury), so this is a
    coupling, not a removal.

    This buys a THIRD, INDEPENDENT FALSIFIABLE ENDPOINT: the model must now
    predict the observed UACR reduction, which it was previously incapable of
    even expressing.

--------------------------------------------------------------------------------
VALIDATION DESIGN (unchanged in spirit, stronger in content)

    Fit 3 free parameters on CREDENCE ONLY:
        - hazard scale         (progression)
        - treatment scale      (eff_met / eff_hf, ratio fixed)
        - eff_alb              (direct anti-albuminuric effect)
    ...against CREDENCE's 3 published numbers, then FREEZE and predict
    DAPA-CKD's 3 published numbers out-of-sample. 3 targets, 3 predictions,
    zero free parameters left.

Usage:  python model_v2.py
================================================================================
"""
import numpy as np
from scipy.integrate import solve_ivp

import insilico_trial as it

# ---- structural constants (NOT fitted) ---------------------------------------
S_SAT = 3.0     # ceiling on compensatory single-nephron hyperfiltration (~3x)
BETA = 1.0      # albuminuria scales ~linearly with the hyperfiltration ratio
N_FLOOR = 1e-3

Q, K_HF, K0 = it.Q_POP, it.KHF_POP, it.K0_POP
W = it.W_POP


def hazard_v2(N, uacr_t, a1c, sbp, u, k_hf, w, eff_met, eff_hf, s_sat=None, q=Q, k0=K0):
    # s_sat defaults to the module constant AT CALL TIME (a default argument would
    # bind once at import and silently ignore any later override -- e.g. in the
    # sensitivity analysis).
    s_sat = S_SAT if s_sat is None else s_sat
    """Per-nephron hazard with SATURATING hyperfiltration and the CURRENT UACR."""
    N = max(N, N_FLOOR)
    s = 1.0 / N                                   # N_ref = 1
    hf = k_hf * (s ** q) / (1.0 + (s / s_sat) ** q)    # <-- saturating (v2)
    hf *= (1.0 - eff_hf * u)
    insult = (w[0] * max(a1c - 6.5, 0.0)
              + w[1] * np.log1p(uacr_t / 30.0)          # <-- dynamic UACR (v2)
              + w[2] * max(sbp - 130.0, 0.0) / 10.0)
    insult *= (1.0 - eff_met * u)
    return min(k0 + hf + insult, 50.0)


def uacr_of_state(N, N0, uacr0, u, eff_alb, beta=BETA):
    """Albuminuria as a readout of hyperfiltration + direct drug effect."""
    s = 1.0 / max(N, N_FLOOR)
    s0 = 1.0 / max(N0, N_FLOOR)
    return uacr0 * (s / s0) ** beta * (1.0 - eff_alb * u)


def simulate_v2(egfr0, a1c, uacr0, sbp, u, k_hf, w, eff_met, eff_hf, eff_alb,
                years, n_out=60, s_sat=None):
    """Integrate the v2 ODE. Returns (t, eGFR, UACR)."""
    from model_core import N_of_egfr, egfr_of_N
    N0 = N_of_egfr(egfr0)

    def rhs(t, y):
        N = max(y[0], N_FLOOR)
        uacr_t = uacr_of_state(N, N0, uacr0, u, eff_alb)
        return [-N * hazard_v2(N, uacr_t, a1c, sbp, u, k_hf, w, eff_met, eff_hf, s_sat=s_sat)]

    t_eval = np.linspace(0, years, n_out)
    sol = solve_ivp(rhs, (0, years), [N0], t_eval=t_eval, method="LSODA",
                    rtol=1e-6, atol=1e-9)
    N = np.maximum(sol.y[0], N_FLOOR)
    egfr = np.array([egfr_of_N(x) for x in N])
    uacr = np.array([uacr_of_state(x, N0, uacr0, u, eff_alb) for x in N])
    return sol.t, egfr, uacr


def trial_arms_v2(spec, k_hf, w, eff_met, eff_hf, eff_alb, n=400, seed=11,
                  skip_years=0.15, s_sat=None):
    """Run both arms; return chronic slopes and the UACR change vs placebo."""
    rng = np.random.default_rng(seed)
    c = it.sample_cohort(spec, n, rng)
    yrs = spec["duration_years"]
    out = {}
    for u, arm in ((0.0, "placebo"), (1.0, "treated")):
        slopes, uacr_ratio = [], []
        for i in range(n):
            t, e, ua = simulate_v2(c["egfr"][i], c["hba1c"][i], c["uacr"][i], c["sbp"][i],
                                   u, k_hf, w, eff_met, eff_hf, eff_alb, yrs, s_sat=s_sat)
            i0 = np.searchsorted(t, skip_years)
            slopes.append((e[-1] - e[i0]) / (t[-1] - t[i0]))
            # UACR at ~6 months (trials report the early, week-26 effect).
            # IMPORTANT: the ratio must be taken against the PRE-TREATMENT
            # baseline (c["uacr"][i]), not against ua[0]. The drug's direct
            # anti-albuminuric effect is already applied at t=0, so dividing by
            # ua[0] would cancel it out and report ~0% reduction.
            i26 = np.searchsorted(t, 0.5)
            uacr_ratio.append(ua[i26] / c["uacr"][i])
        out[arm] = dict(slope=float(np.mean(slopes)),
                        uacr_ratio=float(np.exp(np.mean(np.log(uacr_ratio)))))  # geometric
    out["slope_diff"] = out["treated"]["slope"] - out["placebo"]["slope"]
    # placebo-subtracted geometric-mean % reduction in UACR
    out["uacr_reduction_pct"] = 100.0 * (1.0 - out["treated"]["uacr_ratio"] / out["placebo"]["uacr_ratio"])
    return out


def _solve(fn, target, lo, hi, iters=28):
    """Bisection on a monotone-decreasing fn."""
    for _ in range(iters):
        m = 0.5 * (lo + hi)
        if fn(m) < target:
            hi = m
        else:
            lo = m
    return 0.5 * (lo + hi)


def fit_on_credence(n=350, seed=11, s_sat=None):
    """Fit exactly 3 parameters to CREDENCE's 3 published numbers."""
    C = it.TRIALS["CREDENCE"]

    # (1) hazard scale -> CREDENCE placebo slope
    def pbo(s):
        return trial_arms_v2(C, K_HF * s, W * s, 0, 0, 0, n=n, seed=seed, s_sat=s_sat)["placebo"]["slope"]
    s_h = _solve(pbo, C["placebo_slope"], 0.05, 6.0)
    k_hf_s, w_s = K_HF * s_h, W * s_h

    # (2) treatment scale -> CREDENCE chronic slope difference
    def diff(sc):
        em, eh = min(0.45 * sc, 0.95), min(0.35 * sc, 0.95)
        return -trial_arms_v2(C, k_hf_s, w_s, em, eh, 0.0, n=n, seed=seed, s_sat=s_sat)["slope_diff"]
    sc = _solve(diff, -C["chronic_slope_diff"], 0.0, 2.7)
    eff_met, eff_hf = min(0.45 * sc, 0.95), min(0.35 * sc, 0.95)

    # (3) eff_alb -> CREDENCE UACR reduction (31%)
    def uacr_red(ea):
        return -trial_arms_v2(C, k_hf_s, w_s, eff_met, eff_hf, ea, n=n,
                              seed=seed, s_sat=s_sat)["uacr_reduction_pct"]
    eff_alb = _solve(uacr_red, -C["uacr_reduction_pct"], 0.0, 0.9)

    return dict(hazard_scale=s_h, k_hf=k_hf_s, w=w_s,
                eff_met=eff_met, eff_hf=eff_hf, eff_alb=eff_alb)


def run(n=350):
    C, D = it.TRIALS["CREDENCE"], it.TRIALS["DAPA-CKD (T2D subgroup)"]
    print("=" * 78)
    print("MODEL v2 -- saturating hyperfiltration + endogenous albuminuria")
    print("=" * 78)
    print(f"\nStructural constants (FIXED from physiology, not fitted): "
          f"s_sat={S_SAT}, beta={BETA}")

    print("\n[1/2] FIT on CREDENCE only (3 params <- 3 published numbers)")
    f = fit_on_credence(n=n)
    got = trial_arms_v2(C, f["k_hf"], f["w"], f["eff_met"], f["eff_hf"], f["eff_alb"], n=n)
    print(f"      hazard_scale={f['hazard_scale']:.3f}  eff_met={f['eff_met']:.3f}  "
          f"eff_hf={f['eff_hf']:.3f}  eff_alb={f['eff_alb']:.3f}")
    print(f"      placebo slope  {got['placebo']['slope']:6.2f}  (target {C['placebo_slope']})")
    print(f"      slope diff     {got['slope_diff']:6.2f}  (target {C['chronic_slope_diff']})")
    print(f"      UACR reduction {got['uacr_reduction_pct']:5.1f}%  (target {C['uacr_reduction_pct']}%)")

    print(f"\n[2/2] OUT-OF-SAMPLE on DAPA-CKD (T2D) -- 3 predictions, 0 free parameters")
    p = trial_arms_v2(D, f["k_hf"], f["w"], f["eff_met"], f["eff_hf"], f["eff_alb"], n=n, seed=23)
    rows = []
    lo, hi = D["chronic_slope_ci"]
    rows.append(("chronic slope diff", p["slope_diff"], D["chronic_slope_diff"], (lo, hi)))
    ul, uh = D["uacr_reduction_ci"]
    rows.append(("UACR reduction (%)", p["uacr_reduction_pct"], D["uacr_reduction_pct"], (ul, uh)))
    rows.append(("placebo slope", p["placebo"]["slope"], D["placebo_slope"], None))

    print(f"      {'endpoint':<22}{'model':>9}{'published':>11}{'95% CI':>16}  verdict")
    n_pass = 0
    for name, model, pub, ci in rows:
        if ci:
            ok = ci[0] <= model <= ci[1]
            n_pass += ok
            print(f"      {name:<22}{model:9.2f}{pub:11.2f}{f'{ci[0]}-{ci[1]}':>16}  "
                  f"{'PASS' if ok else 'FAIL'}")
        else:
            print(f"      {name:<22}{model:9.2f}{pub:11.2f}{'(no CI)':>16}  "
                  f"ratio {model/pub:.2f}x")
    print(f"\n      >>> {n_pass}/2 CI-testable endpoints PASS")
    return f, p


if __name__ == "__main__":
    run()
