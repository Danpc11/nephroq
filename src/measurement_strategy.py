"""
================================================================================
WHICH MEASUREMENTS SHOULD YOU ACTUALLY ORDER?  ·  NephroQ
================================================================================
The per-patient personalization (personalize.py) infers two things from a
patient's history: their INJURY RATE and the collapse exponent q. How well it can
do that depends on what you measured. This script answers, by simulation, what is
worth ordering -- and what is not.

Headline results (reproduce with: python measurement_strategy.py):

  1. q is essentially UNIDENTIFIABLE from routine data (R2 ~ 0.0-0.08), and NO
     assay fixes it -- cystatin C included. Stop optimizing for q.

  2. The patient's INJURY RATE, which is what actually drives the forecast, IS
     recoverable -- and what recovers it best is not an expensive assay:

         creatinine only, short history .................. R2 0.48
         + cystatin C ..................................... R2 0.67
         + duplicate creatinine per visit, LONG history ... R2 0.71   <-- free
         + cystatin C AND long history .................... R2 0.75

  3. TIME SPAN beats measurement COUNT, by a lot. The same 4-6 creatinines spread
     over 4-8 years recover the rate ~3x better than the same number crammed into
     1-2 years (0.59 vs 0.18) -- and better than 10-14 values inside a short
     window (0.34).

  => PRACTICAL CONSEQUENCE: pull the patient's OLD creatinine results out of the
     chart. They already exist, they are free, and they beat buying a new assay.

  4. Serial UACR does NOT help (0.47 vs 0.48 baseline). In this model albuminuria
     is a deterministic function of the same latent state, so it adds no
     independent information -- and it carries large biological noise (~40% CV).
     This refutes the intuition that a second, cheap biomarker must add signal.

These are simulation experiments under this model's assumptions, not a clinical
study.
================================================================================
"""
import numpy as np, pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import model_core as core, personalize as P

Q_R, S_R = P.Q_RANGE, P.SCALE_RANGE

def make(n, seed, noise, n_visits_rng, span_rng, use_uacr):
    rng = np.random.default_rng(seed); X, Y = [], []
    for _ in range(n):
        q = rng.uniform(*Q_R)
        s = float(np.exp(rng.uniform(np.log(S_R[0]), np.log(S_R[1]))))
        e0 = rng.uniform(20, 110); a1c = rng.uniform(5.5, 12)
        ua0 = float(np.exp(rng.uniform(np.log(5), np.log(3000)))); sbp = rng.uniform(105, 185)
        nv = rng.integers(*n_visits_rng); span = rng.uniform(*span_rng)
        t = np.sort(rng.uniform(0, span, nv)); t[0] = 0.0
        p = P.patient_params(q, s)
        tt, eg, ua, _ = core.simulate_trajectory_v2(e0, a1c, ua0, sbp, u=0.0, p=p,
                                                    years=max(span, .5), n=150)
        e_obs = np.clip(np.interp(t, tt, eg) + rng.normal(0, noise, nv), 3, None)
        f = P.features(t, e_obs, a1c, ua0, sbp)
        if use_uacr:
            # serial UACR, with realistic biological noise (log-sd 0.35 ~ 40% CV)
            ua_true = np.interp(t, tt, ua)
            ua_obs = ua_true * np.exp(rng.normal(0, 0.35, nv))
            lg = np.log(ua_obs)
            slope_u = np.polyfit(t, lg, 1)[0] if nv > 1 and span > 0 else 0.0
            f = np.concatenate([f, [slope_u, float(lg[-1] - lg[0]), float(np.std(lg))]])
        X.append(f); Y.append([q, np.log(s)])
    return np.array(X), np.array(Y)

def run(label, noise, nv=(3, 9), span=(0.8, 6.0), use_uacr=False, n=4000):
    Xtr, Ytr = make(n, 0, noise, nv, span, use_uacr)
    Xte, Yte = make(600, 777, noise, nv, span, use_uacr)
    sc = StandardScaler().fit(Xtr)
    preds = []
    for i in range(3):
        m = MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=500, early_stopping=True, random_state=i)
        m.fit(sc.transform(Xtr), Ytr); preds.append(m.predict(sc.transform(Xte)))
    Pm = np.mean(preds, 0)
    r2 = lambda a, b: 1 - np.sum((a-b)**2)/np.sum((a-np.mean(a))**2)
    q_r2 = r2(Yte[:,0], np.clip(Pm[:,0], *Q_R))
    s_r2 = r2(np.exp(Yte[:,1]), np.exp(Pm[:,1]))
    print(f"  {label:<44}{q_r2:8.3f}{s_r2:9.3f}")
    return q_r2, s_r2

print("Recovery of the patient's parameters, by what you measure")
print(f"  {'strategy':<44}{'q  R2':>8}{'rate R2':>9}")
print("  " + "-"*61)
run("BASELINE: creatinine only (noise 6.0)",            6.0)
run("+ cystatin C  (noise 3.0)",                        3.0)
run("+ SERIAL UACR (creatinine noise 6.0)",             6.0, use_uacr=True)
run("+ duplicate creatinine per visit (noise 4.2)",     4.2)
run("+ more visits (6-14 instead of 3-8)",              6.0, nv=(6, 15))
run("+ longer follow-up (3-10y instead of 0.8-6y)",     6.0, span=(3.0, 10.0))
run("BEST CHEAP COMBO: serial UACR + more visits",      6.0, nv=(6, 15), use_uacr=True)

print()
print("Disentangling: is it the NUMBER of visits, or the TIME SPAN?")
print(f"  {'strategy':<44}{'q  R2':>8}{'rate R2':>9}")
print("  " + "-"*61)
run("4-6 visits over 1-2 years  (short, dense)",    6.0, nv=(4,7),  span=(1.0, 2.0))
run("4-6 visits over 4-8 years  (long,  sparse)",   6.0, nv=(4,7),  span=(4.0, 8.0))
run("10-14 visits over 1-2 years (short, denser)",  6.0, nv=(10,15), span=(1.0, 2.0))
print()
print("Cheap combos vs cystatin:")
print("  " + "-"*61)
run("cystatin C alone (noise 3.0)",                            3.0)
run("CHEAP: duplicate creatinine + long span",                 4.2, span=(3.0,10.0))
run("cystatin C + long span (best possible)",                  3.0, span=(3.0,10.0))
