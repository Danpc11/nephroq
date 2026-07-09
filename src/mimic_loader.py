"""
================================================================================
MIMIC-IV LOADER  ·  builds longitudinal eGFR trajectories
================================================================================
Converts raw MIMIC-IV files (hosp module) into the schema expected by the
pipeline:  patient_id, time_years, egfr, hba1c, uacr, sbp

REQUIREMENTS (download from physionet.org/content/mimiciv, hosp module --
tested with v3.1, compatible with v2.2+):
    hosp/patients.csv.gz
    hosp/admissions.csv.gz
    hosp/labevents.csv.gz      (large, several GB -- filter by itemid on read)
    hosp/d_labitems.csv.gz
    hosp/diagnoses_icd.csv.gz
    hosp/omr.csv.gz            (NEW in v3.0+: outpatient vital signs,
                                 including real blood pressure -- optional,
                                 falls back to a population placeholder if missing)

STRATEGY (to approximate CHRONIC progression, not just one acute episode):
  1) Filter patients with a type 2 diabetes diagnosis (ICD-9 250.x0/250.x2,
     ICD-10 E11.x).
  2) Take ALL of their creatinine measurements in labevents (labs from the
     whole hospitalization AND outpatient labs, across ALL their encounters,
     potentially years) -> approximate longitudinal trajectory.
  3) Keep only patients with measurements spread over >180 days (filters out
     single-admission acute episodes; keeps those with real follow-up over
     time).
  4) Compute eGFR with CKD-EPI 2021 (egfr_measurement.py) using age
     (anchor_age) and sex.
  5) HbA1c: search labevents for it dynamically (labeled from d_labitems, not
     a fixed itemid, for robustness across versions).
  6) UACR and systolic blood pressure: MIMIC rarely records these outside the
     ICU. Left as optional columns -- if not found, they are filled with the
     population median (explicit imputation, flagged in the output CSV with
     a *_imputed column).

LICENSE COMPLIANCE (PhysioNet Credentialed Health Data License 1.5.0):
  - The output CSV (data/*.csv) must NEVER be pushed to git, to the deployed
    web app, or to any cloud service -- already excluded in .gitignore. Only
    the calibrated (aggregate) PARAMETERS may be published.
  - Do not send this data to third-party APIs (LLMs, cloud services).
  - See docs/MIMIC_COMPLIANCE.md for full detail and the citation required
    in publications.

HONEST LIMITATION: MIMIC-IV is a hospital/critical-care cohort, biased toward
sicker patients, with selection bias and acute comorbidity. Useful to test
the method with real noise; does not replace a chronic outpatient cohort
(CRIC, HCHS/SOL) for the final clinical validation.

Usage:
    python mimic_loader.py --mimic-dir /path/to/mimic-iv/hosp --out ../data/mimic_ckd.csv
================================================================================
"""
import argparse, gzip, sys
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from egfr_measurement import egfr_cr, egfr_cr_cys, egfr_cys  # noqa

# --------------------------------------------------------------------------
# Keywords to find itemids WITHOUT relying on the exact number matching
# between MIMIC versions (more robust than hardcoding the itemid).
# --------------------------------------------------------------------------
LABEL_KEYWORDS = {
    "creatinine": ["creatinine"],
    "hba1c":      ["hemoglobin a1c", "% hemoglobin a1c", "a1c"],
    "cystatin":   ["cystatin"],
    "uacr":       ["albumin/creatinine", "microalbumin", "albumin, urine"],
}
DM_ICD10_PREFIXES = ("E11",)                 # type 2 diabetes (ICD-10 E11.x is specifically type 2)

def is_type2_icd9(code):
    """
    ICD-9 diabetes codes follow the pattern 250XY (5 chars, no decimal point
    in MIMIC): X = complication category (0-9), Y = type/control digit.
    Y in {0, 2} = type 2 or unspecified type; Y in {1, 3} = type 1.
    A bare '250' prefix (as used previously) does NOT distinguish type 1 from
    type 2 -- this checks the actual type digit.
    """
    code = (code or "").strip()
    if not code.startswith("250") or len(code) < 5:
        return False
    return code[4] in ("0", "2")

def find_itemids(d_labitems, keywords, exclude=()):
    lab = d_labitems["label"].str.lower().fillna("")
    mask = np.zeros(len(d_labitems), dtype=bool)
    for kw in keywords:
        mask |= lab.str.contains(kw, na=False)
    for ex in exclude:
        mask &= ~lab.str.contains(ex, na=False)
    return d_labitems.loc[mask, "itemid"].tolist()

def load_diabetic_patient_ids(mimic_dir):
    dx = pd.read_csv(f"{mimic_dir}/diagnoses_icd.csv.gz",
                     usecols=["subject_id", "icd_code", "icd_version"],
                     dtype=str)
    is_dm9  = (dx.icd_version == "9")  & dx.icd_code.apply(is_type2_icd9)
    is_dm10 = (dx.icd_version == "10") & dx.icd_code.str.startswith(DM_ICD10_PREFIXES)
    ids = set(dx.loc[is_dm9 | is_dm10, "subject_id"].unique())
    print(f"[1/5] Patients with a type 2 diabetes diagnosis (approx.): {len(ids)}")
    return ids

def load_lab_series_multi(mimic_dir, analyte_itemids, subject_ids, chunksize=2_000_000):
    """
    Reads labevents.csv.gz ONCE, in chunks, splitting rows into each requested
    analyte on the same pass -- instead of re-reading and re-decompressing the
    (potentially multi-GB) file once per analyte. Important at real MIMIC-IV
    scale (labevents.csv.gz can be several GB compressed).

    analyte_itemids: dict like {"creatinine": [50912], "hba1c": [50852], ...}
    Returns: dict of DataFrames, same keys, each with columns
             [subject_id, charttime, valuenum, valueuom?].
    """
    all_ids = set()
    for ids in analyte_itemids.values():
        all_ids |= set(ids)
    if not all_ids:
        return {k: pd.DataFrame(columns=["subject_id", "charttime", "valuenum"]) for k in analyte_itemids}

    header = pd.read_csv(f"{mimic_dir}/labevents.csv.gz", nrows=0)
    has_uom = "valueuom" in header.columns
    cols = ["subject_id", "itemid", "charttime", "valuenum"] + (["valueuom"] if has_uom else [])
    buckets = {k: [] for k in analyte_itemids}
    id_to_key = {}
    for key, ids in analyte_itemids.items():
        for i in ids:
            id_to_key[i] = key

    reader = pd.read_csv(f"{mimic_dir}/labevents.csv.gz", usecols=cols,
                         dtype={"subject_id": str, "itemid": "Int64"},
                         parse_dates=["charttime"], chunksize=chunksize)
    n_chunks = 0
    for chunk in reader:
        n_chunks += 1
        m = chunk.itemid.isin(all_ids) & chunk.subject_id.isin(subject_ids) & chunk.valuenum.notna()
        sub = chunk.loc[m]
        if len(sub):
            for key in analyte_itemids:
                part = sub.loc[sub.itemid.isin(analyte_itemids[key])]
                if len(part):
                    buckets[key].append(part)
        if n_chunks % 20 == 0:
            print(f"        ...{n_chunks} chunks read from labevents.csv.gz")

    out = {}
    for key, parts in buckets.items():
        if not parts:
            out[key] = pd.DataFrame(columns=["subject_id", "charttime", "valuenum"])
            continue
        df = pd.concat(parts, ignore_index=True)
        if has_uom:
            expected = EXPECTED_UNITS.get(key)
            if expected:
                unit_ok = df.valueuom.isin(expected) | df.valueuom.isna()
                n_bad = int((~unit_ok).sum())
                if n_bad:
                    print(f"      WARNING: {n_bad} '{key}' measurements dropped for "
                          f"unexpected units (expected {expected}).")
                df = df.loc[unit_ok]
        out[key] = df[["subject_id", "charttime", "valuenum"]]
    return out

EXPECTED_UNITS = {"creatinine": {"mg/dL"}, "hba1c": {"%"}, "cystatin": {"mg/L"}, "uacr": {"mg/g"}}

def load_sbp_from_omr(mimic_dir, subject_ids):
    """
    NEW in MIMIC-IV v3.0+: hosp/omr.csv.gz (Online Medical Record) carries
    real OUTPATIENT vital signs, including blood pressure -- something that
    previously did not exist outside the ICU. Typical format:
    result_name='Blood Pressure', result_value='128/76'. If the file doesn't
    exist (MIMIC-IV <3.0), falls back to the population placeholder.
    """
    path = f"{mimic_dir}/omr.csv.gz"
    import os as _os
    if not _os.path.exists(path):
        print("      omr.csv.gz not found (MIMIC-IV <3.0?) -> SBP will be imputed.")
        return pd.DataFrame(columns=["subject_id", "chartdate", "sbp"])
    rows = []
    reader = pd.read_csv(path, dtype={"subject_id": str}, chunksize=1_000_000)
    for chunk in reader:
        m = (chunk.result_name == "Blood Pressure") & chunk.subject_id.isin(subject_ids)
        sub = chunk.loc[m, ["subject_id", "chartdate", "result_value"]].dropna()
        if len(sub):
            sbp = sub.result_value.str.split("/").str[0]
            sbp = pd.to_numeric(sbp, errors="coerce")
            rows.append(pd.DataFrame({"subject_id": sub.subject_id, "chartdate": sub.chartdate, "sbp": sbp}))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["subject_id", "chartdate", "sbp"])

def main(mimic_dir, out_path, min_span_days=180, min_points=4):
    patients = pd.read_csv(f"{mimic_dir}/patients.csv.gz",
                           usecols=["subject_id", "gender", "anchor_age", "anchor_year"],
                           dtype={"subject_id": str})
    d_labitems = pd.read_csv(f"{mimic_dir}/d_labitems.csv.gz", dtype={"itemid": "Int64"})

    dm_ids = load_diabetic_patient_ids(mimic_dir)

    id_creat = find_itemids(d_labitems, LABEL_KEYWORDS["creatinine"],
                            exclude=["urine", "albumin", "ratio", "clearance"])
    id_a1c   = find_itemids(d_labitems, LABEL_KEYWORDS["hba1c"])
    id_cys   = find_itemids(d_labitems, LABEL_KEYWORDS["cystatin"])
    id_uacr  = find_itemids(d_labitems, LABEL_KEYWORDS["uacr"])
    print(f"[2/5] itemids -> creatinine:{id_creat}  hba1c:{id_a1c}  cystatin:{id_cys}  uacr:{id_uacr}")

    print("[3/5] Reading labevents.csv.gz ONCE, splitting analytes in the same pass "
         "(may take several minutes for large files)...")
    series = load_lab_series_multi(mimic_dir,
        {"creatinine": id_creat, "hba1c": id_a1c, "cystatin": id_cys, "uacr": id_uacr},
        dm_ids)
    creat, a1c, cys, uacr = series["creatinine"], series["hba1c"], series["cystatin"], series["uacr"]
    print(f"      creatinine: {len(creat)} measurements  |  hba1c: {len(a1c)}  |  "
          f"cystatin: {len(cys)}  |  uacr: {len(uacr)}")

    if creat.empty:
        print("No creatinine measurements found. Check the MIMIC-IV path.")
        return

    creat = creat.merge(patients, on="subject_id", how="left")
    creat["sex"] = np.where(creat.gender == "F", "F", "M")

    rows = []
    n_ok = 0
    for pid, g in creat.groupby("subject_id"):
        g = g.sort_values("charttime")
        span_days = (g.charttime.max() - g.charttime.min()).days
        if span_days < min_span_days or len(g) < min_points:
            continue                                    # discard single acute episodes
        t0 = g.charttime.min()
        age = float(g.anchor_age.iloc[0]); sex = g.sex.iloc[0]
        # eGFR by creatinine (age approximated at the time of the lab; MIMIC
        # only gives anchor_age at anchor_year -> corrected by elapsed years)
        anchor_year = g.anchor_year.iloc[0]
        for _, r in g.iterrows():
            years_since_anchor = (r.charttime.year - anchor_year)
            age_at_lab = age + years_since_anchor
            eg = egfr_cr(r.valuenum, age_at_lab, sex)
            t_years = (r.charttime - t0).days / 365.25
            rows.append(dict(patient_id=pid, time_years=t_years, egfr=eg,
                             hba1c=np.nan, uacr=np.nan, sbp=np.nan))
        n_ok += 1
    df = pd.DataFrame(rows)
    print(f"[4/5] Patients with a usable trajectory (>= {min_points} measurements, "
          f">= {min_span_days} days of follow-up): {n_ok}")

    # Attach the closest HbA1c in time (if present) per patient
    if not a1c.empty:
        a1c_med = a1c.groupby("subject_id").valuenum.median()
        df["hba1c"] = df.patient_id.map(a1c_med)
    # UACR: almost certainly sparse/absent -> impute with global median if missing
    if not uacr.empty:
        uacr_med = uacr.groupby("subject_id").valuenum.median()
        df["uacr"] = df.patient_id.map(uacr_med)

    print("      Looking for real blood pressure in omr.csv.gz (v3.0+)...")
    omr_sbp = load_sbp_from_omr(mimic_dir, dm_ids)
    if not omr_sbp.empty:
        sbp_med = omr_sbp.groupby("subject_id").sbp.median()
        df["sbp"] = df.patient_id.map(sbp_med)
        print(f"      Real SBP found for {sbp_med.notna().sum()} patients.")

    # Explicit, flagged imputation (only what is truly missing)
    df["hba1c_imputed"] = df["hba1c"].isna()
    df["uacr_imputed"]  = df["uacr"].isna()
    df["sbp_imputed"]   = df["sbp"].isna()
    df["hba1c"] = df["hba1c"].fillna(df["hba1c"].median() if df["hba1c"].notna().any() else 7.5)
    df["uacr"]  = df["uacr"].fillna(df["uacr"].median() if df["uacr"].notna().any() else 30.0)
    df["sbp"]   = df["sbp"].fillna(df["sbp"].median() if df["sbp"].notna().any() else 135.0)

    df.to_csv(out_path, index=False)
    print(f"[5/5] Saved: {out_path}  ({df.patient_id.nunique()} patients, {len(df)} rows)")
    print("\nNOTE: *_imputed columns flag values that were not measured and were filled "
          "with the population median. Review before final calibration.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mimic-dir", required=True, help="Path to the hosp/ folder of your MIMIC-IV copy")
    ap.add_argument("--out", default="../data/mimic_ckd.csv")
    ap.add_argument("--min-span-days", type=int, default=180)
    ap.add_argument("--min-points", type=int, default=4)
    a = ap.parse_args()
    main(a.mimic_dir, a.out, a.min_span_days, a.min_points)
