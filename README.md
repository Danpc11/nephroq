<h1>
  <img src="assets/nephroq_logo.png" alt="NephroQ logo" width="40">
  NephroQ
</h1>

## A mechanistic digital twin for Type 2 Diabetes → Chronic Kidney Disease

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![NumPy](https://img.shields.io/badge/NumPy-supported-blue)
![SciPy](https://img.shields.io/badge/SciPy-supported-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-app-FF4B4B?logo=streamlit&logoColor=white)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Danpc11/nephroq/blob/main/risk_notebook.ipynb)

A calibratable, nonlinear mechanistic model of renal function (eGFR) progression
in type 2 diabetes. It projects the trajectory of eGFR **and of albuminuria**, and
the time to a **modeled eGFR < 15 mL/min/1.73 m² threshold** — a kidney-function
threshold, **not** a prediction of when dialysis would actually start (real KRT
initiation depends on symptoms, labs and clinical judgment).

**Why "NephroQ":** the model's central, physically meaningful parameter is `q`,
the hyperfiltration feedback exponent that quantifies how abrupt the terminal
collapse of renal function is. It is estimated from clinical trajectories, not
fitted as a black box.

> ### ⚠️ Research prototype (TRL 4) — NOT a diagnostic tool
> Not validated on a prospective clinical cohort. Must not be used for clinical
> decisions without qualified medical supervision. Projections illustrate the
> model's *mechanism*; they are not individualized clinical predictions.

---

## Table of contents

- [Quick start (2 minutes)](#quick-start-2-minutes)
- [How the model works](#how-the-model-works)
- [Where the parameters come from](#where-the-parameters-come-from)
- [**Using your own data**](#using-your-own-data) ← start here to calibrate
- [Optional: calibrating with MIMIC-IV](#optional-calibrating-with-mimic-iv)
- [Validation: check the model yourself](#validation-check-the-model-yourself)
- [Repository structure](#repository-structure)
- [Limitations](#limitations)

---

## Quick start (2 minutes)

**You do not need any data to run NephroQ.** It ships with parameters anchored to
published clinical trials, so it works out of the box.

```bash
git clone https://github.com/Danpc11/nephroq.git
cd nephroq
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

streamlit run app_web.py          # interactive app (English / Spanish)
```

Click one of the **four example patients** in the sidebar to see the model's
behavior without typing anything. For a guided 7-minute walkthrough aimed at
clinicians, see [`docs/CLINICIAN_DEMO.md`](docs/CLINICIAN_DEMO.md).

**No installation at all:** open
[`risk_notebook.ipynb` in Colab](https://colab.research.google.com/github/Danpc11/nephroq/blob/main/risk_notebook.ipynb)
and run all cells. The code is hidden; you just get the controls.

### Using the model from Python

```python
import sys; sys.path.insert(0, "src")
import model_core as core

t, egfr, uacr, t_threshold = core.simulate_trajectory_v2(
    egfr0=47.7,      # baseline eGFR (mL/min/1.73m2)
    a1c=8.1,         # HbA1c (%)
    uacr0=145.0,     # baseline UACR (mg/g)
    sbp=142.0,       # systolic blood pressure (mmHg)
    u=0.0,           # 0 = untreated, 1 = renoprotective therapy (SGLT2i / ACEi-ARB)
    years=15,
)
print(f"eGFR<15 reached at {t_threshold:.1f} years")
print(f"UACR goes {uacr[0]:.0f} -> {uacr[-1]:.0f} mg/g")
```

Set `u=1.0` for the treated counterfactual of the same patient. Note that
**albuminuria is an output**, not an input held constant: it rises as nephrons are
lost, and falls under treatment.

---

## How the model works

A low-dimensional ODE for the surviving fraction of functional nephrons `N`:

```
dN/dt = -N * h(N)

h(N) = k0 + hyperfiltration(N) + insult(HbA1c, UACR(t), SBP)
```

Three ideas do the work:

1. **Hyperfiltration feedback.** As nephrons are lost, the survivors are
   overloaded and damaged faster — the source of the accelerating, non-linear
   collapse. It **saturates** at a physiological ceiling (a surviving nephron
   raises its single-nephron GFR ~3x, not without limit).
2. **Compensation.** eGFR stays roughly stable while reserve remains, then falls
   steeply near the end. This is why a single eGFR snapshot can look reassuring
   while the mechanism is already running.
3. **Endogenous albuminuria.** UACR is a *consequence* of glomerular
   hypertension, not an external driver. The model predicts its trajectory — and
   predicts that renoprotective therapy lowers it ~29% immediately (SGLT2i trials
   published **31–35%**).

Full mathematical specification: [`docs/MODEL_DOCUMENTATION.md`](docs/MODEL_DOCUMENTATION.md).

---

## Where the parameters come from

NephroQ resolves parameters across **three tiers**, highest priority first. The
app always displays which one is active.

| Tier | Source | When it is used |
|---|---|---|
| 1. Private | `.streamlit/secrets.toml` | If you have your own clinical-cohort calibration |
| 2. MIMIC-IV | `calibration/mimic_calibration.json` | Only if **you** generated it locally (see below) |
| 3. **Default (ships with the repo)** | **Published clinical trials** | **Automatically — no data required** |

**The default tier is trial-anchored.** Progression is fixed by the placebo arms
of **CREDENCE** and **EMPA-KIDNEY**; treatment effects by **CREDENCE**. Then
**DAPA-CKD is predicted out-of-sample**, and both its chronic eGFR slope and its
UACR reduction land inside the published 95% CI. No patient-level data is needed,
because the anchors are published aggregate results.

---

## Using your own data

**This is the recommended way to calibrate NephroQ for your population.** You do
**not** need MIMIC-IV. Any longitudinal cohort with repeated creatinine works.

### 1. Format your data as a CSV

One row per visit, per patient:

```csv
patient_id,time_years,egfr,hba1c,uacr,sbp
P001,0.0,58.2,8.4,180,145
P001,0.6,55.1,8.1,210,142
P001,1.4,51.7,8.6,260,148
P002,0.0,72.4,7.2,45,132
```

| Column | Meaning | Required |
|---|---|---|
| `patient_id` | any stable identifier | yes |
| `time_years` | years since that patient's index visit (first = 0) | yes |
| `egfr` | mL/min/1.73 m2 (CKD-EPI 2021; see `src/egfr_measurement.py`) | yes |
| `hba1c` | % | recommended |
| `uacr` | mg/g | recommended |
| `sbp` | mmHg | recommended |

Missing covariates are allowed (leave the cell empty) — they are imputed, and the
imputation is **reported**, not hidden. But `uacr` carries a large share of the
hazard, so a cohort with poor UACR coverage will not identify its weight well.

**Minimum per patient:** at least 4 eGFR measurements spanning at least 180 days.
Shorter trajectories cannot constrain the curvature.

### 2. Calibrate and validate

```bash
cd src
CKD_CSV=../data/my_cohort.csv python mvp_calibration.py
```

This fits the model, compares it against a linear-slope baseline, tests whether it
discriminates progressors, and writes a one-page report to
`results/validation_report.md` you can hand to clinical collaborators.

### 3. Read the diagnostics before trusting anything

The calibration prints an explicit quality verdict. If it reports
`quality_status = warning`, **the app refuses to use those parameters** and falls
back to the trial-anchored defaults unless you explicitly opt in. That is
deliberate: a warning printed above a plot does not undo the plot.

It will tell you, loudly, about: an optimizer that never moved off its initial
guess, a chi2/n far from 1, a cohort where most "patients" do not actually
decline, and how much of your covariate data was imputed.

---

## Optional: calibrating with MIMIC-IV

MIMIC-IV is **one option among several, not a requirement.** NephroQ ships and runs
without it. It is a useful large public testbed — but it is an ICU/hospital
database, which brings real problems (see [Limitations](#limitations)).

### You must request access yourself — we cannot give you the data

MIMIC-IV is a **credentialed** dataset. This repository contains **no patient
data**, and no MIMIC-derived file is ever committed (see `.gitignore`). To use it:

1. Create a [PhysioNet](https://physionet.org/) account.
2. Complete the required **CITI "Data or Specimens Only Research"** training.
3. Sign the MIMIC-IV **Data Use Agreement** and become credentialed.
4. Download MIMIC-IV v3.1 yourself.

See [`docs/MIMIC_COMPLIANCE.md`](docs/MIMIC_COMPLIANCE.md) for the handling rules
this project follows. **Please do not ask the authors for the data, and do not
redistribute it.**

### Running the calibration

The data never leaves your machine.

```bash
cd src
python calibrate_mimic.py --mimic-dir /path/to/mimiciv/3.1/hosp --chronic-only
```

It writes `calibration/mimic_calibration.json`, which the app then picks up
automatically (tier 2). That file is **git-ignored** — it is yours to keep, or to
share "upon reasonable request" in a publication.

Useful flags:

```bash
--chronic-only        # keep only net-declining, lower-volatility trajectories
--n-bootstrap 200     # 15 = smoke test ONLY; 100-200 preliminary; 500-1000 for a manuscript
--from-csv path.csv   # reuse a previously built cohort, skipping the slow rebuild
```

---

## Validation: check the model yourself

The model is validated against **published randomized trials**, and the test is
built so it **can fail**.

```bash
cd src
python insilico_trial.py       # in-silico replication of 3 trials
```

Parameters are fitted on **CREDENCE** (with **EMPA-KIDNEY** anchoring the
saturation ceiling), then **frozen**. **DAPA-CKD is predicted out-of-sample**,
from its published baseline characteristics alone — no parameter is left to turn.
Agreement on the fitted trials is guaranteed by construction and is *not*
evidence; all the weight is on DAPA-CKD:

| DAPA-CKD (held out) | Model | Published |
|---|---|---|
| Chronic eGFR slope difference | **2.10** | 2.26 (95% CI 1.88–2.64) — PASS |
| UACR reduction | **31.0 %** | 35.1 % (95% CI 30.6–39.4) — PASS |
| Placebo-arm slope | −3.51 | −3.83 |

The run writes `results/insilico_trial_report.md`.

```bash
python -m pytest tests -q      # 40 tests
```

---

## Repository structure

```
nephroq/
├── app_web.py                  # interactive app (English / Spanish)
├── risk_notebook.ipynb         # Colab notebook (code hidden, app-like)
├── requirements.txt
├── src/
│   ├── model_core.py           # THE model — single source of truth
│   ├── egfr_measurement.py     # CKD-EPI 2021
│   ├── i18n.py                 # UI strings (EN/ES) + example patients
│   ├── insilico_trial.py       # falsifiable validation against 3 published trials
│   ├── mvp_calibration.py      # calibrate + validate on YOUR data
│   ├── calibrate_mimic.py      # optional: calibrate on MIMIC-IV
│   └── mimic_loader.py         # optional: MIMIC-IV cohort builder
├── tests/                      # 40 tests
├── docs/
│   ├── MODEL_DOCUMENTATION.md  # mathematical specification
│   ├── CLINICIAN_DEMO.md       # 7-minute clinician demo script
│   ├── MIMIC_COMPLIANCE.md     # data-handling rules
│   ├── WEB_DEPLOYMENT.md       # free deployment
│   └── CHANGELOG.md
├── calibration/                # your calibration JSON lands here (git-ignored)
└── docker/Dockerfile
```

There is **one model**, in `model_core.py`. The app, the calibration scripts and
the validation all call the same simulator; nothing re-implements it.

---

## Limitations

Read before citing. These are not boilerplate.

- **Not validated on a prospective clinical cohort.** Trial anchoring is
  aggregate-level; that is not patient-level external validation.
- **No acute haemodynamic dip.** SGLT2 inhibitors cause an early, reversible eGFR
  drop that NephroQ does not model, so it can only be compared against *chronic*
  slopes, never total slopes.
- **MIMIC-IV index dates are not AKI-free baselines.** The index visit is the
  first available creatinine, which in a hospital database is often drawn during
  an acute episode. This is arguably the biggest threat to any MIMIC-based
  calibration.
- **The uncertainty band propagates only parameter uncertainty.** It is *not* a
  prediction interval: it excludes measurement noise, individual random effects,
  and unknown future covariates.
- **The KFRE comparison is exploratory**, using a proxy outcome (observed
  eGFR<15) rather than treated kidney failure.
- **Trial values are transcribed from the literature** and were not independently
  verified. Re-check them against the primary papers before publication.

---

## Citation

See [`CITATION.cff`](CITATION.cff).

## License

See [`LICENSE`](LICENSE).
