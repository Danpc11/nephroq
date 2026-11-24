"""
Public synthetic cohort in the mimic_loader.py output format (audit P0.6).

Generated from the mechanistic model with a known two-regime population and
heavy-tailed measurement noise, so the complete patient-level validation
pipeline (validate_mimic_patients.py) can run in CI and on any machine without
restricted data. It contains no real patient information. Deterministic for a
given seed; the SHA-256 of the written files is printed for the record.

Run:
    python make_synthetic_cohort.py --out ../data/synthetic_cohort.tsv \\
        --patients-out ../data/synthetic_patients.csv.gz
"""
import argparse
import hashlib
import os

import numpy as np
import pandas as pd

import fast_sim

GROUPS = ["2008 - 2010", "2011 - 2013", "2014 - 2016", "2017 - 2019", "2020 - 2022"]


def make(n=1200, seed=2026):
    rng = np.random.default_rng(seed)
    rows, pts = [], []
    for i in range(n):
        fast = rng.uniform() < 0.15
        q = rng.uniform(1.2, 2.2)
        scale = float(np.exp(rng.normal(1.2 if fast else -0.8, 0.3)))
        e0 = rng.uniform(25.0, 105.0)
        a1c, sbp = rng.uniform(6.0, 11.0), rng.uniform(115.0, 170.0)
        uacr = float(np.exp(rng.uniform(np.log(10.0), np.log(2000.0))))
        age, sex = rng.uniform(40, 85), ("F" if rng.uniform() < 0.45 else "M")
        t = np.sort(np.concatenate([[0.0], rng.uniform(0.0, rng.uniform(2.5, 8.0),
                                                       rng.integers(6, 40))]))
        e = fast_sim.simulate_egfr(q, scale, e0, a1c, uacr, sbp, t)[0] \
            + 5.0 * rng.standard_t(3.5, len(t))
        missing_a1c = rng.uniform() < 0.2
        for tt, ee in zip(t, e):
            rows.append(dict(patient_id=f"S{i:05d}", time_years=round(float(tt), 4),
                             egfr=round(float(max(ee, 3.0)), 2),
                             age_at_index=round(age, 1), sex=sex,
                             hba1c=np.nan if missing_a1c else round(a1c, 2),
                             uacr=round(uacr, 1), sbp=round(sbp, 1),
                             hba1c_imputed=missing_a1c, uacr_imputed=False,
                             sbp_imputed=False))
        pts.append(dict(subject_id=f"S{i:05d}", anchor_year_group=GROUPS[i % len(GROUPS)]))
    return pd.DataFrame(rows), pd.DataFrame(pts)


def _sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()[:16]


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Synthetic cohort in the MIMIC loader format")
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--out", default=os.path.join(here, "..", "data", "synthetic_cohort.tsv"))
    ap.add_argument("--patients-out",
                    default=os.path.join(here, "..", "data", "synthetic_patients.csv.gz"))
    ap.add_argument("--n", type=int, default=1200)
    ap.add_argument("--seed", type=int, default=2026)
    a = ap.parse_args()
    cohort, patients = make(a.n, a.seed)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    cohort.to_csv(a.out, sep="\t", index=False)
    # mtime=0 keeps the gzip byte-identical across runs, so its hash is reproducible
    patients.to_csv(a.patients_out, index=False,
                    compression=dict(method="gzip", mtime=0) if a.patients_out.endswith(".gz") else None)
    print(f"{a.out}: {cohort.patient_id.nunique()} patients, {len(cohort)} rows, sha256 {_sha(a.out)}")
    print(f"{a.patients_out}: sha256 {_sha(a.patients_out)}")
