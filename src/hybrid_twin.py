"""
================================================================================
HYBRID DIGITAL TWIN  ·  Diabetes -> Chronic Kidney Disease
================================================================================
Couples TWO matrix formalisms:

  (A) LINEAR STATE-SPACE  (continuous variables: eGFR, UACR)
      x_{k+1} = A x_k + B u + c        (dynamics = matrix algebra)
      + exact KALMAN FILTER             (assimilation of patient data)

  (B) ABSORBING MULTI-STATE MARKOV CHAIN   (KDIGO stages G1..G5; G5 = dialysis)
      covariate-dependent generator Q
      EXPECTED TIME TO DIALYSIS = -T^{-1} · 1   (fundamental matrix, closed form)

Designed for physics students: everything is linear algebra.
Requires: numpy, matplotlib.
================================================================================
"""

import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(7)

# ------------------------------------------------------------------------------
# Minimal clinical utilities
# ------------------------------------------------------------------------------
# KDIGO eGFR thresholds (mL/min/1.73m^2) defining stages G1..G5.
KDIGO_EDGES = [90, 60, 30, 15]     # G1>=90, G2 60-89, G3 30-59, G4 15-29, G5<15
STAGE_NAMES = ["G1", "G2", "G3", "G4", "G5 (dialysis)"]
DIALYSIS_eGFR = 15                  # crossing this threshold = renal replacement therapy

def egfr_to_stage(egfr):
    """Converts a continuous eGFR into its KDIGO stage (0..4)."""
    if egfr >= 90:  return 0
    if egfr >= 60:  return 1
    if egfr >= 30:  return 2
    if egfr >= 15:  return 3
    return 4


# ==============================================================================
# (A) LINEAR STATE-SPACE MODEL  +  KALMAN FILTER
# ==============================================================================
class LinearRenalStateSpace:
    """
    Continuous state:  x = [eGFR, UACR].
    Patient covariates (A1c, blood pressure) and intervention u (0/1) fix the
    time-invariant linear system matrices:

        x_{k+1} = A x_k + c          (the intervention is already inside A, c)

    Strictly matrix algebra: simulating = multiplying and adding.
    """
    def __init__(self, a1c, sbp, u=0.0, dt=1/12,
                 eff_egfr=0.40, eff_uacr=0.55):
        self.dt = dt
        # ---- parameters of the filtration loss rate (calibratable) ----
        c0, cA, cU, cB = -2.6, 0.6, 2.0, 0.4     # base, A1c, UACR, blood pressure
        bp = (sbp - 120) / 10.0                  # scaled blood pressure
        # annual eGFR decline rate (part NOT depending on UACR):
        base_decl = (c0 + cA * a1c + cB * bp) * (1 - eff_egfr * u)
        coupl_uacr = (cU / 300.0) * (1 - eff_egfr * u)   # UACR -> eGFR coupling

        # ---- UACR dynamics: reversion to a setpoint (linear, bounded) ----
        #   UACR trends toward an equilibrium depending on risk;
        #   the drug lowers that setpoint. kappa = reversion speed.
        s0, sA, sB, kappa = -260.0, 42.0, 8.0, 0.5
        uacr_set = max((s0 + sA * a1c + sB * bp) * (1 - eff_uacr * u), 5.0)

        # ---- assemble A (2x2) and c (2,) ----
        self.A = np.array([
            [1.0, -dt * coupl_uacr],          # eGFR_{k+1} = eGFR - dt*(base + coupl*UACR)
            [0.0, 1.0 - dt * kappa]           # UACR_{k+1} = UACR + dt*kappa*(set - UACR)
        ])
        self.c = np.array([-dt * base_decl, dt * kappa * uacr_set])

        # ---- process and measurement noise (for Kalman) ----
        self.Q = np.diag([0.05, 4.0])          # model uncertainty
        self.R = np.diag([3.0, 25.0])          # lab noise
        self.H = np.eye(2)                      # we measure eGFR and UACR directly

    def step(self, x):
        return self.A @ x + self.c

    def simulate(self, x0, n_steps, noisy=False):
        """Returns the trajectory (n_steps+1, 2). Pure linear algebra."""
        X = np.zeros((n_steps + 1, 2)); X[0] = x0
        for k in range(n_steps):
            x = self.step(X[k])
            if noisy:
                x = x + rng.multivariate_normal(np.zeros(2), self.Q)
            X[k + 1] = np.maximum(x, 0.0)
        return X

    # ---------------- KALMAN FILTER (data assimilation) ----------------
    def kalman_filter(self, observations):
        """
        observations: list of (eGFR, UACR) measurements (noisy), or None if missing.
        Keeps the twin 'locked on' to the patient: predict -> correct.
        Returns filtered means and covariances.
        """
        n = len(observations)
        x = np.array([observations[0][0], observations[0][1]], dtype=float)
        P = np.diag([10.0, 100.0])
        means, covs = [x.copy()], [P.copy()]
        for k in range(1, n):
            # --- prediction (model) ---
            x = self.A @ x + self.c
            P = self.A @ P @ self.A.T + self.Q
            # --- correction (new data), if it exists ---
            z = observations[k]
            if z is not None:
                z = np.array(z, dtype=float)
                S = self.H @ P @ self.H.T + self.R          # innovation
                K = P @ self.H.T @ np.linalg.inv(S)         # Kalman gain
                x = x + K @ (z - self.H @ x)
                P = (np.eye(2) - K @ self.H) @ P
            means.append(x.copy()); covs.append(P.copy())
        return np.array(means), np.array(covs)


# ==============================================================================
# (B) ABSORBING MULTI-STATE MARKOV CHAIN  ->  TIME TO DIALYSIS (closed form)
# ==============================================================================
class AbsorbingCKDMarkov:
    """
    Continuous-time Markov chain over KDIGO stages G1..G5.
    G5 (dialysis) is ABSORBING. Progression intensities depend on patient
    covariates (A1c, UACR, blood pressure) and the intervention.

    Key closed-form result (fundamental matrix):
        expected time to absorption  m = -T^{-1} · 1
    where T is the transient block of the generator. A single matrix inversion!
    """
    def __init__(self, eff=0.40):
        self.eff = eff
        self.lmbda = np.array([0.08, 0.10, 0.12, 0.16])   # base rate per stage/year
        self.beta_a, self.beta_u, self.beta_b = 0.25, 0.40, 0.20

    def generator(self, a1c, uacr, sbp, u=0.0):
        """Builds the generator Q (5x5) for a given patient."""
        risk = np.exp(self.beta_a * (a1c - 6.5)
                      + self.beta_u * np.log1p(uacr / 30.0)
                      + self.beta_b * (sbp - 130) / 10.0)
        rates = self.lmbda * risk * (1 - self.eff * u)     # 4 intensities i->i+1
        Q = np.zeros((5, 5))
        for i in range(4):
            Q[i, i + 1] = rates[i]
            Q[i, i] = -rates[i]
        # row 4 (G5) stays zero -> absorbing state
        return Q

    def expected_time_to_dialysis(self, start_stage, a1c, uacr, sbp, u=0.0):
        """
        Expected time (years) to dialysis from 'start_stage', assuming
        covariates fixed at their current value.
        m = -T^{-1} · 1   (fundamental matrix of the absorbing chain).
        """
        Q = self.generator(a1c, uacr, sbp, u)
        T = Q[:4, :4]                          # transient block (4x4)
        m = -np.linalg.solve(T, np.ones(4))    # vector of expected times
        return float(m[start_stage])


# ==============================================================================
# (C) HYBRID TWIN  =  state-space  +  Markov
# ==============================================================================
class HybridDigitalTwin:
    """Combines both: the continuous trajectory and the closed-form time-to-dialysis estimate."""
    def __init__(self, a1c, sbp, eGFR0, UACR0, u=0.0, dt=1/12):
        self.a1c, self.sbp, self.u, self.dt = a1c, sbp, u, dt
        self.x0 = np.array([eGFR0, UACR0], dtype=float)
        self.ss = LinearRenalStateSpace(a1c, sbp, u=u, dt=dt)
        self.mk = AbsorbingCKDMarkov()

    def simulate(self, years=20):
        n = int(years / self.dt)
        X = self.ss.simulate(self.x0, n)
        t = np.arange(n + 1) * self.dt
        # realized time to dialysis: first crossing of eGFR < 15
        below = np.where(X[:, 0] < DIALYSIS_eGFR)[0]
        t_dialysis_sim = t[below[0]] if len(below) else np.inf
        return t, X, t_dialysis_sim

    def analytic_time_to_dialysis(self):
        stage = egfr_to_stage(self.x0[0])
        if stage >= 4:
            return 0.0
        return self.mk.expected_time_to_dialysis(
            stage, self.a1c, self.x0[1], self.sbp, self.u)


# ==============================================================================
# DEMO
# ==============================================================================
def main():
    # Three scenarios on the SAME at-risk patient:
    #  - untreated (u=0)
    #  - treated (u=1): SGLT2i / RAAS  -> slows decline and lowers UACR
    #  - well-controlled patient (reference)
    profiles = {
        "At risk, untreated":  dict(a1c=9.0, sbp=150, eGFR0=82, UACR0=300, u=0.0),
        "At risk, treated":    dict(a1c=9.0, sbp=150, eGFR0=82, UACR0=300, u=1.0),
        "Well controlled":     dict(a1c=6.8, sbp=125, eGFR0=82, UACR0=30,  u=0.0),
    }

    print("=" * 70)
    print(f"{'Profile':<22}{'t_dialysis sim (years)':>22}{'t_dialysis analytic (years)':>28}")
    print("-" * 72)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    colors = {"At risk, untreated": "#E24B4A",
              "At risk, treated":    "#BA7517",
              "Well controlled":    "#1D9E75"}

    for name, p in profiles.items():
        twin = HybridDigitalTwin(**p)
        t, X, t_sim = twin.simulate(years=20)
        t_an = twin.analytic_time_to_dialysis()
        s_sim = f"{t_sim:.1f}" if np.isfinite(t_sim) else ">20"
        print(f"{name:<22}{s_sim:>22}{t_an:>28.1f}")
        axes[0].plot(t, X[:, 0], lw=2.2, color=colors[name], label=name)

    # --- left panel: eGFR trajectories with KDIGO bands ---
    ax = axes[0]
    for edge in KDIGO_EDGES:
        ax.axhline(edge, color="0.8", lw=0.8, ls="--")
    ax.axhline(DIALYSIS_eGFR, color="k", lw=1.2)
    ax.text(0.2, DIALYSIS_eGFR + 1.5, "dialysis threshold (eGFR<15)", fontsize=9)
    ax.set_xlabel("years"); ax.set_ylabel("eGFR (mL/min/1.73m²)")
    ax.set_title("(A) Linear state-space: eGFR trajectory")
    ax.legend(fontsize=9); ax.set_ylim(0, 95)

    # --- right panel: Kalman assimilation for one patient ---
    p = profiles["At risk, untreated"]
    twin = HybridDigitalTwin(**p)
    t, X_true, _ = twin.simulate(years=12)
    # noisy quarterly observations (with some gaps)
    obs = []
    for k in range(len(t)):
        if k % 3 == 0:   # measured every 3 months
            z = X_true[k] + rng.multivariate_normal([0, 0], twin.ss.R)
            obs.append((max(z[0], 0), max(z[1], 0)))
        else:
            obs.append(None)
    means, covs = twin.ss.kalman_filter(obs)
    sd = np.sqrt(covs[:, 0, 0])

    ax = axes[1]
    ax.plot(t, X_true[:, 0], "k-", lw=1.5, label="true state (hidden)")
    obs_t = [t[k] for k in range(len(t)) if obs[k] is not None]
    obs_e = [obs[k][0] for k in range(len(t)) if obs[k] is not None]
    ax.scatter(obs_t, obs_e, s=22, color="#7F77DD", label="noisy measurements", zorder=3)
    ax.plot(t, means[:, 0], color="#BA7517", lw=2, label="twin (Kalman)")
    ax.fill_between(t, means[:, 0] - 2*sd, means[:, 0] + 2*sd,
                    color="#BA7517", alpha=0.2, label="±2σ")
    ax.set_xlabel("years"); ax.set_ylabel("eGFR (mL/min/1.73m²)")
    ax.set_title("(B) Kalman filter: the twin locks onto the data")
    ax.legend(fontsize=9)

    print("=" * 70)
    plt.tight_layout()
    plt.savefig("../results/hybrid_twin_demo.png", dpi=130)
    print("Figure saved: hybrid_twin_demo.png")

if __name__ == "__main__":
    main()
