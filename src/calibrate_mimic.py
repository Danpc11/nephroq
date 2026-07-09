"""
================================================================================
CALIBRATION WITH LOCAL MIMIC-IV  ·  100% on your machine, uploads nothing
================================================================================
Takes your LOCAL copy of MIMIC-IV (hosp module), builds the cohort of
diabetic patients with eGFR trajectories (via mimic_loader.py), calibrates
the mechanistic model, and writes A SINGLE FILE:

    calibration/mimic_calibration.json

That JSON contains only AGGREGATE PARAMETERS (q, k_hf, weights) -- never
patient data. It is not pushed to git (see .gitignore). It is the file that:
  - the web app (app_web.py) uses automatically if present, as the
    research/demo calibration.
  - you can share "upon reasonable request" in the publication, consistent
    with the MIMIC-IV license (see docs/MIMIC_COMPLIANCE.md) -- the
    aggregate parameters are not PHI, but you control who receives them.

USAGE:
    python calibrate_mimic.py --mimic-dir /path/to/your/mimic-iv/hosp

Requires: numpy, pandas, scipy (already in requirements.txt). No network needed.
================================================================================
"""
import argparse, json, os, subprocess, sys, datetime
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from mimic_loader import main as build_mimic_csv  # reuses the already-tested loader

# ---- same mechanistic core and identifiable parameterization as amortized_ai.py / bayesian_model.py ----
G_MAX, ALPHA, N_FLOOR, K0_FIX = 120.0, 0.80, 0.05, 0.0030

def N_of_egfr(e): return np.power(np.clip(e, 1e-6, None) / G_MAX, 1 / ALPHA)
def egfr_of_N(N): return G_MAX * np.power(np.clip(N, 1e-9, None), ALPHA)

def insult(cov, w):
    a1c, uacr, sbp = cov
    return w[0]*max(a1c-6.5, 0) + w[1]*np.log1p(uacr/30) + w[2]*max(sbp-130, 0)/10

def simulate(q, khf, cov, w, t_max, egfr0, dt=0.05):
    I = insult(cov, w); N = N_of_egfr(egfr0)
    n = int(t_max/dt) + 1; ts = np.linspace(0, t_max, n); Ns = np.empty(n); Ns[0] = N
    for k in range(1, n):
        Nc = min(max(Ns[k-1], N_FLOOR), 1.0)
        h = min(K0_FIX + khf*(1.0/Nc)**q + I, 50.0)
        Ns[k] = min(max(Ns[k-1] - dt*Ns[k-1]*h, N_FLOOR), 1.0)
    return ts, Ns

def predict_egfr(q, khf, cov, w, t_query, egfr0):
    ts, Ns = simulate(q, khf, cov, w, float(np.max(t_query)) + 0.1, egfr0)
    return np.clip(egfr_of_N(np.interp(t_query, ts, Ns)), 0, G_MAX)

def load_cohort(csv_path):
    df = pd.read_csv(csv_path)
    patients = []
    for pid, g in df.groupby("patient_id"):
        g = g.sort_values("time_years")
        if len(g) < 3:
            continue
        cov = (float(g["hba1c"].median()), float(g["uacr"].median()), float(g["sbp"].median()))
        patients.append(dict(cov=cov, egfr0=float(g["egfr"].iloc[0]),
                             t=g["time_years"].values.astype(float),
                             e=g["egfr"].values.astype(float)))
    return patients

LO = np.array([0.5, 1e-4, 1e-4, 1e-4, 1e-4])
HI = np.array([3.0, 0.06, 0.06, 0.06, 0.06])
def unpack(p): return LO + (HI - LO) / (1 + np.exp(-p))
def pack(th):
    z = np.clip((th - LO) / (HI - LO), 1e-4, 1 - 1e-4)
    return np.log(z / (1 - z))

def calibrate(patients, noise_sd=3.5, seed=0):
    def residuals(p):
        q, khf, wa, wu, wb = unpack(p)
        w = np.array([wa, wu, wb])
        r = [(predict_egfr(q, khf, pac["cov"], w, pac["t"], pac["egfr0"]) - pac["e"]) / noise_sd
            for pac in patients]
        r = np.concatenate(r)
        return np.where(np.isfinite(r), r, 100.0)

    rng = np.random.default_rng(seed)
    base = np.array([1.5, 0.012, 0.014, 0.018, 0.011])
    best = None
    for s in range(5):
        p_init = pack(base) if s == 0 else pack(np.clip(base*rng.uniform(0.5, 1.8, 5), LO*1.01, HI*0.99))
        sol = least_squares(residuals, p_init, method="trf", max_nfev=3000)
        if best is None or sol.cost < best.cost:
            best = sol

    q, khf, wa, wu, wb = unpack(best.x)
    n_obs = sum(len(pac["t"]) for pac in patients)
    chi2_n = 2 * best.cost / n_obs
    rmse = float(np.sqrt(chi2_n) * noise_sd)
    return dict(q=float(q), k_hf=float(khf), w_a1c=float(wa), w_uacr=float(wu), w_sbp=float(wb),
               n_patients=len(patients), n_obs=int(n_obs),
               chi2_per_n=float(chi2_n), rmse_mL_min=rmse)

def diagnose_cohort(patients, noise_sd=3.5):
    """
    Cheap, per-patient diagnostics computed from trajectories ALREADY in
    memory (no need to re-read labevents.csv.gz). Flags whether the cohort
    looks like smooth chronic decline (what the mechanistic model assumes)
    or acute, fluctuating trajectories (common in a hospital/critical-care
    cohort like MIMIC-IV) -- the latter will make a global population fit
    converge to a boundary/degenerate q with a huge chi2/n, even though
    nothing is wrong with the code.
    """
    net_decline = 0
    volatilities = []
    for pat in patients:
        t, e = pat["t"], pat["e"]
        if len(t) < 3:
            continue
        # net trend: simple linear fit slope
        slope = np.polyfit(t, e, 1)[0]
        if slope < 0:
            net_decline += 1
        # volatility: residual std around a linear trend, vs assumed noise_sd
        resid = e - np.polyval(np.polyfit(t, e, 1), t)
        volatilities.append(np.std(resid))
    n = len(patients)
    frac_declining = net_decline / n if n else 0.0
    median_volatility = float(np.median(volatilities)) if volatilities else float("nan")
    print(f"\n[diagnostics] Patients with net-declining eGFR trend: "
          f"{net_decline}/{n} ({100*frac_declining:.0f}%)")
    print(f"[diagnostics] Median within-patient volatility (residual std around "
          f"a straight line): {median_volatility:.1f} mL/min/1.73m² "
          f"(assumed measurement noise: {noise_sd})")
    if median_volatility > 3 * noise_sd:
        print("[diagnostics] WARNING: within-patient volatility is much larger than the "
              "assumed measurement noise -- trajectories look ACUTE/fluctuating rather "
              "than smooth chronic decline. A single global fit will likely converge to a "
              "degenerate q (e.g. stuck at the lower bound) with a very high chi2/n. "
              "Consider --chronic-only, or use the hierarchical model instead of a flat fit.")
    if frac_declining < 0.6:
        print(f"[diagnostics] WARNING: only {100*frac_declining:.0f}% of patients show a net "
              "declining trend -- the mechanistic model (monotonic decline only) structurally "
              "cannot represent the rest. Consider --chronic-only to fit on the subset it can "
              "represent.")
    return dict(n_patients=n, frac_net_declining=round(frac_declining, 3),
               median_volatility_mL_min=round(median_volatility, 2) if volatilities else None)

def filter_chronic_like(patients, max_volatility_ratio=2.5, noise_sd=3.5):
    """
    Optional stricter cohort: keep only patients with a net-declining trend
    AND within-patient volatility not too far above measurement noise --
    i.e. trajectories the monotonic mechanistic model can plausibly represent.
    This is a subset, not a fix: it trades cohort size for a fit the model
    can actually explain, useful for a first honest proof-of-concept.
    """
    kept = []
    for pat in patients:
        t, e = pat["t"], pat["e"]
        if len(t) < 3:
            continue
        slope = np.polyfit(t, e, 1)[0]
        resid = e - np.polyval(np.polyfit(t, e, 1), t)
        vol = np.std(resid)
        if slope < 0 and vol <= max_volatility_ratio * noise_sd:
            kept.append(pat)
    return kept

def main():
    ap = argparse.ArgumentParser(description="Calibrates the twin with your LOCAL MIMIC-IV copy.")
    ap.add_argument("--mimic-dir", required=True, help="Path to the hosp/ folder of your local MIMIC-IV copy")
    ap.add_argument("--out", default=os.path.join(HERE, "..", "calibration", "mimic_calibration.json"))
    ap.add_argument("--mimic-version", default="3.1")
    ap.add_argument("--min-span-days", type=int, default=180)
    ap.add_argument("--min-points", type=int, default=4)
    ap.add_argument("--chronic-only", action="store_true",
                    help="Keep only patients with a net-declining, low-volatility "
                         "trajectory (a subset the monotonic mechanistic model can "
                         "plausibly represent). Use if diagnostics flag an acute/"
                         "fluctuating cohort.")
    a = ap.parse_args()

    tmp_csv = os.path.join(HERE, "..", "data", "_mimic_tmp.csv")
    os.makedirs(os.path.dirname(tmp_csv), exist_ok=True)

    print("[1/3] Building the cohort from local MIMIC-IV (never leaves your machine)...")
    build_mimic_csv(a.mimic_dir, tmp_csv, a.min_span_days, a.min_points)

    print("\n[2/3] Calibrating the mechanistic model...")
    patients = load_cohort(tmp_csv)
    if len(patients) < 5:
        print(f"Only {len(patients)} patients with a usable trajectory -- insufficient, aborting.")
        os.remove(tmp_csv)
        return

    diagnostics = diagnose_cohort(patients)

    if a.chronic_only:
        before = len(patients)
        patients = filter_chronic_like(patients)
        print(f"[diagnostics] --chronic-only: kept {len(patients)}/{before} patients "
             f"with a net-declining, lower-volatility trajectory.")
        if len(patients) < 5:
            print("Too few patients remain after --chronic-only filtering -- aborting.")
            os.remove(tmp_csv)
            return

    result = calibrate(patients)
    result["diagnostics"] = diagnostics
    result["chronic_only_filter"] = bool(a.chronic_only)

    result["source"] = "MIMIC-IV (calibrated locally, not redistributed -- see docs/MIMIC_COMPLIANCE.md)"
    result["mimic_version"] = a.mimic_version
    result["calibration_date"] = datetime.datetime.now().isoformat(timespec="seconds")
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=HERE,
                                capture_output=True, text=True).stdout.strip()
        result["code_commit"] = commit or "unknown"
    except Exception:
        result["code_commit"] = "unknown"

    out_path = os.path.abspath(a.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    os.remove(tmp_csv)   # do not leave the intermediate per-patient CSV behind

    print(f"\n[3/3] Saved: {out_path}")
    print(f"       q={result['q']:.2f}  k_hf={result['k_hf']:.4f}  "
         f"n_patients={result['n_patients']}  chi2/n={result['chi2_per_n']:.2f}")

    if result["q"] <= LO[0] + 1e-6 or result["q"] >= HI[0] - 1e-6:
        print(f"\nWARNING: q converged AT its bound ({LO[0]}-{HI[0]}). This usually means "
             "the model could not find an interior optimum -- treat this fit as unreliable, "
             "not a physical result. See the [diagnostics] messages above.")
    if result["chi2_per_n"] > 5:
        print(f"WARNING: chi2/n = {result['chi2_per_n']:.1f} is far above the ~1 expected for "
             "a good fit (rmse={:.1f} mL/min vs assumed noise={:.1f}). Do NOT treat this as a "
             "trustworthy calibration for the app/demo without further investigation "
             "(try --chronic-only, or use the hierarchical model).".format(
                 result["rmse_mL_min"], 3.5))

    print("\nThis JSON is NOT pushed to git (see .gitignore). It is the file that:")
    print("  - the web app uses automatically as the research/demo calibration.")
    print("  - you can share 'upon reasonable request' in the publication.")
    print("  - see calibration/README.md for the handling policy of this file.")

if __name__ == "__main__":
    main()
