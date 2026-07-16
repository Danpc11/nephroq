"""
Longitudinal patient state -- the object that turns a personalized simulator into
a digital twin (roadmap block 2).

The distinction matters. Until now the app consumed a flat snapshot: a history of
creatinines plus the CURRENT HbA1c / UACR / SBP and a treatment switch. That is
enough to project once. A twin instead carries the patient's clinical history as a
sequence of visits, so that (a) covariates are time-varying, not fixed at their
latest value, and (b) the state can be UPDATED when a new visit arrives rather than
recomputed from scratch (block 4 builds on this).

DESIGN RULE: this module holds DATA, not model logic. It knows nothing about the
hazard or the ODE. It validates, orders, and exposes a patient's clinical record in
the shape the model wants, and it records provenance (units, source, imputed-or-not)
because a clinical tool has to be auditable about where every number came from. The
model imports from here; this never imports the model.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from datetime import date, datetime
from typing import Optional


# ------------------------------------------------------------------------------
# Value with provenance -- every clinical number carries where it came from
# ------------------------------------------------------------------------------
@dataclass
class Measured:
    """
    A single clinical value plus the provenance a clinical tool must keep: the
    unit it was recorded in, the source it came from, and whether it was observed
    or imputed. A bare float loses all of this, and 'was this UACR measured or
    guessed?' is exactly the kind of question that changes how much a projection
    should be trusted.
    """
    value: float
    unit: str = ""
    source: str = "unknown"          # e.g. 'central_lab', 'chart', 'FHIR', 'user'
    imputed: bool = False
    quality: str = "ok"              # 'ok' | 'suspect' | 'out_of_range'

    def __post_init__(self):
        # a value that is present must at least be a finite number
        if self.value is not None and not (isinstance(self.value, (int, float))
                                           and math.isfinite(float(self.value))):
            raise ValueError(f"Measured.value must be a finite number, got {self.value!r}")


@dataclass
class Visit:
    """
    One clinical encounter. Only `date` is required; everything else is optional,
    because real records are sparse and a twin must cope with missingness rather
    than demand a full panel every time.

    Labs are stored as Measured (value + provenance). Convenience floats can be
    passed to the constructor and are wrapped automatically, so callers loading a
    plain CSV do not have to build Measured objects by hand.
    """
    date: date
    creatinine: Optional[Measured] = None       # mg/dL
    egfr: Optional[Measured] = None             # mL/min/1.73m^2 (if pre-computed)
    cystatin_c: Optional[Measured] = None       # mg/L
    uacr: Optional[Measured] = None             # mg/g
    hba1c: Optional[Measured] = None            # %
    sbp: Optional[Measured] = None              # mmHg
    medications: tuple = ()                     # e.g. ('sglt2i', 'raasi')
    adherence: Optional[float] = None           # 0..1, if known
    aki_status: bool = False                    # was this visit during an AKI episode?

    def __post_init__(self):
        self.date = _as_date(self.date)
        for fld in ("creatinine", "egfr", "cystatin_c", "uacr", "hba1c", "sbp"):
            v = getattr(self, fld)
            if v is not None and not isinstance(v, Measured):
                setattr(self, fld, Measured(float(v)))
        if self.adherence is not None and not (0.0 <= self.adherence <= 1.0):
            raise ValueError("adherence must be in [0, 1]")
        self.medications = tuple(self.medications)


@dataclass
class PatientState:
    """
    A patient's full longitudinal record, plus enough demographics to run the
    model. Visits are kept sorted by date. This is the object a twin is built
    around: `add_visit` extends it as care continues, and the `to_model_inputs`
    method projects it into the flat arrays the model currently expects, so the
    existing model code keeps working unchanged while the richer state accumulates
    underneath.
    """
    patient_id: str
    age: float                                  # years, at the most recent visit
    sex: str                                    # 'M' | 'F'
    visits: list = field(default_factory=list)
    comorbidities: tuple = ()
    events: list = field(default_factory=list)  # e.g. AKI episodes, hospitalizations

    def __post_init__(self):
        if self.sex not in ("M", "F"):
            raise ValueError("sex must be 'M' or 'F'")
        self.visits = sorted((_as_visit(v) for v in self.visits), key=lambda v: v.date)

    # -- building the record ---------------------------------------------------
    def add_visit(self, visit) -> "PatientState":
        """Add a visit and keep the list ordered. Returns self so calls can chain.
        This is the entry point a twin's `update(new_visit)` will call."""
        self.visits.append(_as_visit(visit))
        self.visits.sort(key=lambda v: v.date)
        return self

    @property
    def latest(self) -> Optional[Visit]:
        return self.visits[-1] if self.visits else None

    @property
    def baseline(self) -> Optional[Visit]:
        return self.visits[0] if self.visits else None

    def span_years(self) -> float:
        if len(self.visits) < 2:
            return 0.0
        return (self.visits[-1].date - self.visits[0].date).days / 365.25

    # -- projecting into what the model consumes -------------------------------
    def creatinine_history(self):
        """
        (years_ago, creatinine) for every visit that has a creatinine, measured
        from the LATEST visit. This is what the personalizer reconstructs a
        trajectory from. years_ago lets the caller convert each creatinine with the
        age AT THAT TIME (age - years_ago), which is the correct, bug-free path.
        """
        if not self.visits:
            return [], []
        t_end = self.latest.date
        yrs, creat = [], []
        for v in self.visits:
            if v.creatinine is not None:
                yrs.append((t_end - v.date).days / 365.25)
                creat.append(v.creatinine.value)
        return yrs, creat

    def latest_covariates(self):
        """Most recent observed HbA1c / UACR / SBP, each falling back to None if
        never measured. The model's caller decides how to handle a missing one
        (population default), and knows it was missing rather than guessing."""
        def most_recent(fieldname):
            for v in reversed(self.visits):
                m = getattr(v, fieldname)
                if m is not None:
                    return m.value
            return None
        return dict(hba1c=most_recent("hba1c"),
                    uacr=most_recent("uacr"),
                    sbp=most_recent("sbp"))

    def on_treatment(self) -> bool:
        """Whether the latest visit records any renoprotective medication."""
        return bool(self.latest and self.latest.medications)

    def to_dict(self) -> dict:
        """Plain-dict view (for serialization / logging). Dates become ISO strings."""
        d = asdict(self)
        for v in d["visits"]:
            v["date"] = v["date"].isoformat() if hasattr(v["date"], "isoformat") else v["date"]
        return d


# ------------------------------------------------------------------------------
# small coercion helpers
# ------------------------------------------------------------------------------
def _as_date(x) -> date:
    if isinstance(x, datetime):
        return x.date()
    if isinstance(x, date):
        return x
    if isinstance(x, str):
        return date.fromisoformat(x)
    raise ValueError(f"cannot interpret {x!r} as a date")


def _as_visit(v) -> Visit:
    if isinstance(v, Visit):
        return v
    if isinstance(v, dict):
        return Visit(**v)
    raise ValueError(f"cannot interpret {v!r} as a Visit")
