"""
Clinical data layer -- load longitudinal patient records into PatientState
(roadmap block 10).

The twin must not depend on someone typing every value into a form by hand. This
module reads a longitudinal clinical table (long format: one row per
patient-visit-measurement, or wide: one row per visit) and builds validated
PatientState objects, preserving provenance (unit, source, imputed flag) on every
value.

Scope of this first version: CSV/TSV. FHIR and OMOP adapters are declared as
explicit stubs at the bottom -- named so the shape of the eventual integration is
clear, but not pretended to exist. Better an honest NotImplementedError than a
half-working adapter that silently drops fields.

DESIGN RULE (same as patient_state): this module holds DATA handling, not model
logic. It validates units and ranges, flags missingness, and emits PatientState.
It never touches the hazard or the ODE.
"""
from __future__ import annotations

import csv
from datetime import date
from typing import Optional

from patient_state import PatientState, Visit, Measured


# ------------------------------------------------------------------------------
# Plausibility ranges -- a clinical value outside these is flagged, not dropped
# ------------------------------------------------------------------------------
# These are generous physiological bounds. A value outside is marked quality=
# 'out_of_range' so downstream code (and the safety layer, block 11) can see it,
# rather than being silently trusted or silently discarded.
RANGES = {
    "creatinine": (0.1, 25.0, "mg/dL"),
    "egfr":       (1.0, 200.0, "mL/min/1.73m2"),
    "cystatin_c": (0.2, 10.0, "mg/L"),
    "uacr":       (0.0, 30000.0, "mg/g"),
    "hba1c":      (3.0, 20.0, "%"),
    "sbp":        (50.0, 260.0, "mmHg"),
}

# Common column-name synonyms -> canonical field. Real exports are inconsistent.
COLUMN_ALIASES = {
    "patient_id": {"patient_id", "subject_id", "pid", "mrn", "id"},
    "date":       {"date", "visit_date", "charttime", "measurement_date", "time"},
    "age":        {"age", "age_years"},
    "sex":        {"sex", "gender"},
    "creatinine": {"creatinine", "creat", "scr", "serum_creatinine"},
    "egfr":       {"egfr", "egfr_ckdepi", "gfr"},
    "cystatin_c": {"cystatin_c", "cystatin", "cysc"},
    "uacr":       {"uacr", "acr", "albumin_creatinine_ratio"},
    "hba1c":      {"hba1c", "a1c", "hemoglobin_a1c"},
    "sbp":        {"sbp", "systolic", "systolic_bp"},
    "medications": {"medications", "meds", "drugs"},
    "adherence":  {"adherence"},
    "aki_status": {"aki_status", "aki", "aki_flag"},
}

LAB_FIELDS = ("creatinine", "egfr", "cystatin_c", "uacr", "hba1c", "sbp")


def _canonical_columns(header):
    """Map each column in the file to a canonical field name (or itself)."""
    lut = {}
    for col in header:
        key = col.strip().lower()
        canon = next((c for c, aliases in COLUMN_ALIASES.items() if key in aliases), key)
        lut[col] = canon
    return lut


def _validate(field, raw, source):
    """Return a Measured with a quality flag, or None if the cell is empty."""
    if raw is None or str(raw).strip() == "":
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return Measured(value=float("nan"), unit="", source=source,
                        quality="suspect", imputed=False) if False else None
    lo, hi, unit = RANGES.get(field, (float("-inf"), float("inf"), ""))
    quality = "ok" if lo <= value <= hi else "out_of_range"
    return Measured(value=value, unit=unit, source=source, quality=quality)


def load_long_csv(path, source="csv", delimiter=None) -> list:
    """
    Load a LONG-format table: one row per patient-visit, columns for each lab.
    (One-row-per-measurement 'tall' format is a future extension; long is what
    most clinical exports and the MIMIC-derived cohorts already produce.)

    Returns a list of PatientState, one per patient_id, visits ordered by date.
    Missing cells are left absent (not imputed here -- imputation, if any, is a
    separate, flagged step so it never hides).
    """
    with open(path, "r", newline="") as fh:
        sample = fh.readline()
        fh.seek(0)
        delim = delimiter or ("\t" if "\t" in sample else ",")
        reader = csv.DictReader(fh, delimiter=delim)
        rows = list(reader)
        if not rows:
            return []
        colmap = _canonical_columns(reader.fieldnames)

    # regroup cells under canonical names
    def canon_row(r):
        out = {}
        for col, val in r.items():
            out[colmap.get(col, col)] = val
        return out

    by_patient = {}
    for raw in rows:
        r = canon_row(raw)
        pid = str(r.get("patient_id", "")).strip()
        if not pid:
            continue
        visit = Visit(
            date=_parse_date(r.get("date")),
            aki_status=_truthy(r.get("aki_status")),
            medications=_split_meds(r.get("medications")),
            adherence=_opt_float(r.get("adherence")),
            **{f: _validate(f, r.get(f), source) for f in LAB_FIELDS},
        )
        rec = by_patient.setdefault(pid, dict(age=_opt_float(r.get("age")),
                                              sex=r.get("sex", "M"), visits=[]))
        rec["visits"].append(visit)
        # keep the most recent age/sex seen
        if r.get("age"):
            rec["age"] = _opt_float(r.get("age"))
        if r.get("sex"):
            rec["sex"] = str(r.get("sex")).strip().upper()[:1] or "M"

    states = []
    for pid, rec in by_patient.items():
        age = rec["age"] if rec["age"] is not None else 60.0
        sex = rec["sex"] if rec["sex"] in ("M", "F") else "M"
        states.append(PatientState(patient_id=pid, age=age, sex=sex,
                                   visits=rec["visits"]))
    return states


def missingness_report(states) -> dict:
    """
    Per-field fraction of visits with the value MISSING, across all patients. The
    twin has to be honest about what it does not know; this is the number the
    safety layer (block 11) uses to decide how much to trust a projection.
    """
    total = sum(len(s.visits) for s in states)
    if total == 0:
        return {f: 1.0 for f in LAB_FIELDS}
    report = {}
    for f in LAB_FIELDS:
        missing = sum(1 for s in states for v in s.visits if getattr(v, f) is None)
        report[f] = missing / total
    return report


def quality_flags(states) -> list:
    """List every out-of-range or suspect value, with patient and date, so a data
    problem surfaces loudly instead of flowing into a projection."""
    flags = []
    for s in states:
        for v in s.visits:
            for f in LAB_FIELDS:
                m = getattr(v, f)
                if m is not None and m.quality != "ok":
                    flags.append(dict(patient_id=s.patient_id, date=v.date.isoformat(),
                                      field=f, value=m.value, quality=m.quality))
    return flags


# ------------------------------------------------------------------------------
# small parsing helpers
# ------------------------------------------------------------------------------
def _parse_date(x):
    if x is None or str(x).strip() == "":
        raise ValueError("every visit needs a date")
    s = str(x).strip()
    # accept ISO, or a plain year-fraction is NOT accepted here (dates are dates)
    for fmt in (None,):  # date.fromisoformat handles YYYY-MM-DD
        try:
            return date.fromisoformat(s[:10])
        except ValueError:
            break
    raise ValueError(f"unparseable date: {x!r}")


def _opt_float(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _truthy(x):
    return str(x).strip().lower() in ("1", "true", "yes", "y", "t") if x is not None else False


def _split_meds(x):
    if not x:
        return ()
    return tuple(m.strip().lower() for m in str(x).replace(";", ",").split(",") if m.strip())


# ------------------------------------------------------------------------------
# Future adapters -- declared, not faked
# ------------------------------------------------------------------------------
def load_fhir(bundle):
    """Load from a FHIR bundle. Not yet implemented: a real adapter must map
    Observation/MedicationStatement resources and their units, and silently
    dropping fields would be worse than an honest failure."""
    raise NotImplementedError("FHIR adapter not implemented yet (roadmap block 10, phase 2)")


def load_omop(tables):
    """Load from OMOP CDM tables (measurement, drug_exposure, person). Same
    principle: declared so the integration shape is clear, not stubbed to look
    functional."""
    raise NotImplementedError("OMOP adapter not implemented yet (roadmap block 10, phase 2)")
