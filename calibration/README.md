# calibration/

This folder holds `mimic_calibration.json`: the mechanistic model parameters
calibrated locally with a copy of MIMIC-IV (see `src/calibrate_mimic.py`).
**The file is NOT pushed to git** — only this README stays in the
repository.

## How to generate it

```bash
cd src
python calibrate_mimic.py --mimic-dir /path/to/your/mimic-iv/hosp
```

Runs 100% locally. Builds the cohort (via `mimic_loader.py`), calibrates the
model, writes `mimic_calibration.json` here, and **deletes** the
intermediate per-patient CSV — only the aggregate parameters remain.

## What the JSON contains

Only aggregate population parameters: `q`, `k_hf`, the three metabolic
insult weights, and calibration metadata (number of patients, goodness of
fit, MIMIC-IV version, date, code commit used). **It contains no
identifiable patient data.**

## Handling policy

- Use in this repository: if present, `app_web.py` automatically uses it as
  the active calibration for the research demo.
- Publication: the manuscript states *"model parameters calibrated with
  MIMIC-IV, available upon reasonable request"* — the standard pattern for
  publications using PhysioNet's restricted-access data.
- Never redistributed together with patient data, and never uploaded to
  cloud services outside your control.

See [`../docs/MIMIC_COMPLIANCE.md`](../docs/MIMIC_COMPLIANCE.md) for the
full detail on MIMIC-IV license compliance.
