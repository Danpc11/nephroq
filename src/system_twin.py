"""
================================================================================
NEPHROQ INTEGRATED SYSTEM  ·  Type 2 Diabetes -> CKD   (TRL4 entrypoint)
================================================================================
Orchestrates the FULL pipeline as a single system with a single invocation:

    physical component -> calibration -> validation -> report

instead of running 10 separate scripts by hand. This is the leap from
"components that work separately" (TRL3) to "integrated, jointly validated,
reproducible system" (TRL4).

Usage:
    python system_twin.py                          # full run, synthetic data
    python system_twin.py --csv ../data/mine.csv    # full run, real data
    python system_twin.py --skip-bayes              # faster (skips MCMC)

Output: ../results/system_run_<timestamp>/  with ALL artifacts of the run
       + manifest.json (what ran, how long it took, whether each stage
       passed or failed) -> this is the auditable evidence TRL4 requires.
================================================================================
"""
import argparse, json, os, shutil, subprocess, sys, time
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "..", "results")

STAGES = [
    ("mechanistic_core",     ["python", "mechanistic_twin.py"], None),
    ("egfr_measurement",     ["python", "egfr_measurement.py"], None),
    ("inverse_problem",      ["python", "inverse_fit.py"], None),
    ("identifiability",      ["python", "noise_identifiability.py"], "skip_slow"),
    ("amortized_ai",         ["python", "amortized_ai.py"], "skip_slow"),
    ("hierarchical_model",   ["python", "hierarchical_model.py"], None),
    ("bayesian_model",       ["python", "bayesian_model.py"], "skip_bayes"),
    ("forecast_comparison",  ["python", "forecast_comparison.py"], "skip_bayes"),
    ("real_data_validity",   ["python", "real_data_validity.py"], None),
    ("mvp_calibration",      ["python", "mvp_calibration.py"], None),
]

def run_stage(name, cmd, env, log_dir):
    t0 = time.time()
    log_path = os.path.join(log_dir, f"{name}.log")
    with open(log_path, "w") as logf:
        proc = subprocess.run(cmd, cwd=HERE, env=env, stdout=logf, stderr=subprocess.STDOUT)
    dt = time.time() - t0
    ok = proc.returncode == 0
    status = "OK" if ok else "FAILED"
    print(f"  [{status}] {name:<24} {dt:6.1f}s   (log: {os.path.relpath(log_path)})")
    return dict(stage=name, ok=ok, seconds=round(dt, 1), returncode=proc.returncode,
               log=os.path.relpath(log_path, RESULTS))

def main():
    ap = argparse.ArgumentParser(description="Integrated system run (TRL4 entrypoint)")
    ap.add_argument("--csv", default=None, help="Real-data CSV (schema: patient_id,time_years,egfr,hba1c,uacr,sbp)")
    ap.add_argument("--skip-slow", action="store_true", help="Skip slow stages (identifiability, amortized AI)")
    ap.add_argument("--skip-bayes", action="store_true", help="Skip MCMC sampling (faster, ~1 min)")
    args = ap.parse_args()

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(RESULTS, f"system_run_{run_id}")
    os.makedirs(log_dir, exist_ok=True)

    env = os.environ.copy()
    if args.csv:
        env["CKD_CSV"] = os.path.abspath(args.csv)

    print(f"=== DIGITAL TWIN SYSTEM — run {run_id} ===")
    print(f"Data: {'real (' + args.csv + ')' if args.csv else 'synthetic (demo)'}\n")

    manifest = dict(run_id=run_id, started=datetime.now().isoformat(),
                    csv=args.csv, stages=[])
    t_start = time.time()
    for name, cmd, flag in STAGES:
        if flag == "skip_slow" and args.skip_slow: continue
        if flag == "skip_bayes" and args.skip_bayes: continue
        manifest["stages"].append(run_stage(name, cmd, env, log_dir))

    manifest["total_seconds"] = round(time.time() - t_start, 1)
    manifest["all_passed"] = all(s["ok"] for s in manifest["stages"])
    manifest["finished"] = datetime.now().isoformat()

    with open(os.path.join(log_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"System {'PASSED' if manifest['all_passed'] else 'FAILED'} — "
          f"{sum(s['ok'] for s in manifest['stages'])}/{len(manifest['stages'])} stages OK "
          f"in {manifest['total_seconds']:.0f}s")
    print(f"Evidence (logs + manifest.json): {os.path.relpath(log_dir)}")
    print(f"{'='*60}")
    sys.exit(0 if manifest["all_passed"] else 1)

if __name__ == "__main__":
    main()
