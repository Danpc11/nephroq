"""
================================================================================
MECHANISTIC NONLINEAR MODEL  ·  Renal progression (Diabetes -> CKD)
================================================================================
Replaces a linear core with the REAL physics of the nephron:

  State variable:  N(t) = functional nephron mass fraction  (0..1)
                   irreversible and monotonically decreasing.

  (1) HYPERFILTRATION (positive feedback, power law):
        load per nephron ~ 1/N  ->  hazard per nephron  H ~ (N_ref/N)^q
        => accelerated collapse when N is small.

  (2) COMPENSATION (WEAK power law in the observable):
        eGFR = G_max * N^alpha,   alpha < 1
        => eGFR is buffered early and collapses late (sigmoid shape).

  Dynamics:
        dN/dt = -N * [ k0 + k_hf*(N_ref/N)^q + k_met * I ]
  with I = metabolic insult (A1c, UACR, blood pressure).

  Intervention u (SGLT2i / RAAS blockade):
        - reduces the metabolic insult:  I -> I*(1 - eff_met*u)
        - reduces hyperfiltration:        k_hf -> k_hf*(1 - eff_hf*u)
          (SGLT2 inhibitors lower intraglomerular pressure: real mechanism)

Requires: numpy, scipy, matplotlib.
================================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# ------------------------------------------------------------------------------
# Observable mapping  eGFR <-> N  (compensation = weak power law)
# ------------------------------------------------------------------------------
G_MAX = 120.0      # eGFR with intact kidney (N=1)
ALPHA = 0.80       # compensation exponent (<1 => buffering)

def egfr_of_N(N):          return G_MAX * np.power(np.clip(N, 1e-9, None), ALPHA)
def N_of_egfr(egfr):       return np.power(np.clip(egfr, 1e-6, None) / G_MAX, 1.0 / ALPHA)

DIALYSIS_eGFR = 15.0
N_DIALYSIS = N_of_egfr(DIALYSIS_eGFR)     # N threshold equivalent to dialysis


# ------------------------------------------------------------------------------
# Metabolic insult
# ------------------------------------------------------------------------------
def metabolic_insult(a1c, uacr, sbp):
    """I >= 0. Combines hyperglycemia, albuminuria, and blood pressure."""
    I = (0.40 * max(a1c - 6.5, 0.0)
         + 0.50 * np.log1p(uacr / 30.0)
         + 0.30 * max(sbp - 130.0, 0.0) / 10.0)
    return I


# ------------------------------------------------------------------------------
# Mechanistic model
# ------------------------------------------------------------------------------
class MechanisticRenalModel:
    def __init__(self, a1c, sbp, uacr, u=0.0,
                 k0=0.0030, k_hf=0.0120, q=1.6, N_ref=0.60, k_met=0.0360,
                 eff_met=0.45, eff_hf=0.35):
        self.a1c, self.sbp, self.uacr, self.u = a1c, sbp, uacr, u
        self.k0, self.q, self.N_ref = k0, q, N_ref
        self.k_hf = k_hf * (1 - eff_hf * u)      # SGLT2i reduces hyperfiltration
        I = metabolic_insult(a1c, uacr, sbp)
        self.k_met_I = k_met * I * (1 - eff_met * u)

    def hazard(self, N):
        """Hazard per nephron (1/year). Grows as N falls (hyperfiltration)."""
        N = max(N, 1e-4)
        return self.k0 + self.k_hf * (self.N_ref / N) ** self.q + self.k_met_I

    def rhs(self, t, y):
        N = y[0]
        return [-N * self.hazard(N)]

    def simulate(self, N0, years=25, n=600):
        t_eval = np.linspace(0, years, n)
        sol = solve_ivp(self.rhs, [0, years], [N0], t_eval=t_eval,
                        rtol=1e-8, atol=1e-10, dense_output=True)
        N = sol.y[0]
        egfr = egfr_of_N(N)
        # time to dialysis: first crossing of N < N_DIALYSIS
        below = np.where(N < N_DIALYSIS)[0]
        t_dial = t_eval[below[0]] if len(below) else np.inf
        return t_eval, N, egfr, t_dial


# ------------------------------------------------------------------------------
# Linear model (reference, for comparison)
# ------------------------------------------------------------------------------
def linear_egfr(a1c, sbp, uacr, u, egfr0, years=25, n=600, dt=1/12):
    """Linear decline at a constant rate (the earlier, simplified model)."""
    c0, cA, cB, cU, eff = -2.6, 0.6, 0.4, 2.0/300, 0.40
    bp = (sbp - 120) / 10.0
    rate = (c0 + cA*a1c + cB*bp + cU*uacr) * (1 - eff*u)   # mL/min/year
    t = np.linspace(0, years, n)
    egfr = np.maximum(egfr0 - rate * t, 0)
    below = np.where(egfr < DIALYSIS_eGFR)[0]
    t_dial = t[below[0]] if len(below) else np.inf
    return t, egfr, t_dial


# ==============================================================================
# DEMO + comparison
# ==============================================================================
def main():
    profiles = {
        "High risk, untreated": dict(a1c=9.0, sbp=150, uacr=300, u=0.0, color="#E24B4A"),
        "High risk, treated":   dict(a1c=9.0, sbp=150, uacr=300, u=1.0, color="#BA7517"),
        "Well controlled":      dict(a1c=6.8, sbp=125, uacr=30,  u=0.0, color="#1D9E75"),
    }
    egfr0 = 82.0
    N0 = N_of_egfr(egfr0)
    print(f"Initial N (eGFR={egfr0}) = {N0:.3f}   |   Dialysis N = {N_DIALYSIS:.3f}\n")
    print(f"{'Profile':<22}{'t_dialysis mechanistic':>24}{'t_dialysis linear':>20}")
    print("-" * 66)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axM, axR = axes

    for name, p in profiles.items():
        col = p.pop("color")
        m = MechanisticRenalModel(**p)
        t, N, egfr, t_dial = m.simulate(N0, years=25)
        tl, egfr_l, t_dial_l = linear_egfr(egfr0=egfr0, **p)

        s1 = f"{t_dial:.1f}" if np.isfinite(t_dial) else ">25"
        s2 = f"{t_dial_l:.1f}" if np.isfinite(t_dial_l) else ">25"
        print(f"{name:<22}{s1:>24}{s2:>20}")

        axM.plot(t, egfr, lw=2.4, color=col, label=name)
        axM.plot(tl, egfr_l, lw=1.2, color=col, ls=":", alpha=0.7)
        p["color"] = col

    # Left panel: mechanistic trajectories (solid) vs linear (dotted)
    axM.axhline(DIALYSIS_eGFR, color="k", lw=1.2)
    axM.text(0.3, DIALYSIS_eGFR + 1.5, "dialysis threshold", fontsize=9)
    axM.set_xlabel("years"); axM.set_ylabel("eGFR (mL/min/1.73m²)")
    axM.set_title("Mechanistic (solid) vs linear (dotted)")
    axM.legend(fontsize=9); axM.set_ylim(0, 90); axM.set_xlim(0, 25)

    # Right panel: the mechanism -- hazard per nephron and eGFR=G·N^alpha mapping
    Ngrid = np.linspace(0.05, 1.0, 200)
    m_demo = MechanisticRenalModel(a1c=9, sbp=150, uacr=300)
    hz = np.array([m_demo.hazard(n) for n in Ngrid])
    axR.plot(Ngrid, hz, color="#7F77DD", lw=2.4, label="hazard per nephron H(N)")
    axR.axvline(N_DIALYSIS, color="k", ls="--", lw=1, label="dialysis N")
    axR.set_xlabel("N (functional nephron fraction)")
    axR.set_ylabel("hazard (1/year)", color="#7F77DD")
    axR.tick_params(axis='y', labelcolor="#7F77DD")
    axR.set_title("Mechanism: hyperfiltration (↑ hazard as N falls)")
    ax2 = axR.twinx()
    ax2.plot(Ngrid, egfr_of_N(Ngrid), color="#1D9E75", lw=2.0, ls="-.",
             label="eGFR = G·Nᵅ (compensation)")
    ax2.set_ylabel("eGFR (mL/min/1.73m²)", color="#1D9E75")
    ax2.tick_params(axis='y', labelcolor="#1D9E75")
    lines = axR.get_legend_handles_labels()[0] + ax2.get_legend_handles_labels()[0]
    labs = axR.get_legend_handles_labels()[1] + ax2.get_legend_handles_labels()[1]
    axR.legend(lines, labs, fontsize=8, loc="upper center")

    plt.tight_layout()
    plt.savefig("../results/mechanistic_twin_demo.png", dpi=130)
    print("\nFigure saved: mechanistic_twin_demo.png")

if __name__ == "__main__":
    main()
