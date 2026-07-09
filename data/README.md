# data/

Place your real data (CSV) here to calibrate the model. **This folder is
excluded from git** (see `.gitignore`) — never push patient data to the
repository.

## Expected schema

`mvp_calibration.py` and `hierarchical_model.py` expect a CSV with these
columns (names are mapped automatically if you use common variants):

```
patient_id, time_years, egfr, hba1c, uacr, sbp
```

| Column | Description | Unit |
|---|---|---|
| `patient_id` | patient identifier | — |
| `time_years` | time since first visit | years |
| `egfr` | estimated glomerular filtration rate | mL/min/1.73m² |
| `hba1c` | glycated hemoglobin | % |
| `uacr` | urine albumin/creatinine ratio | mg/g |
| `sbp` | systolic blood pressure | mmHg |

## Sources of real data

See [`../docs/HOW_TO_DOWNLOAD_DATA.md`](../docs/HOW_TO_DOWNLOAD_DATA.md) for
the options evaluated (HCHS/SOL, CRIC, AASK, Synthea, MIMIC-IV) and how to
request them.

## Usage

```bash
cd ../src
CKD_CSV=../data/my_data.csv python mvp_calibration.py
```
