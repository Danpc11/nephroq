# Model documentation — NephroQ

### A mechanistic digital twin of type 2 diabetes → chronic kidney disease progression

This document specifies the model as it stands in the code today (`src/model_core.py`),
with the mathematics, the parameter values, how each parameter is calibrated, and the
empirical findings that shaped the current form. Where a number appears here it is the
number in the code; where a claim appears it is one that has been tested, and the test
is named.

---

## 1. What the model is

NephroQ is a low-dimensional dynamical system for how a type 2 diabetes patient's kidney
function declines over years. It is deliberately mechanistic rather than a statistical
fit, for one practical reason this project has repeatedly relied on: **when a mechanistic
parameter is wrong, you can see it.** A hazard term with the wrong shape fails a
placebo-arm reproduction test; an unidentifiable parameter pins to its bound under
cross-validation. Both have happened here, and both were caught because the parameters
mean something. Inside a black box they would have shown up only as a slightly larger
loss.

The state variable is `N`, the surviving fraction of functional nephron mass. It is
latent (never observed directly) and, in this model, monotonically non-increasing —
nephron loss is treated as irreversible. The observable is eGFR, a compressed function of
`N`. The dynamics encode two facts of renal physiology:

- **Hyperfiltration** — losing nephrons overloads the survivors, which damages them
  faster. This is positive feedback, and it is what makes late CKD accelerate.
- **Two competing insults** — one structural (hyperfiltration, dominant when albuminuria
  is low) and one albuminuric (dominant when albuminuria is high). The patient's own UACR
  decides which regime they are in. This split (Round 16) is what lets one parameter set
  describe both a near-normoalbuminuric hospital population and the macroalbuminuric
  populations of the SGLT2i trials.

---

## 2. Mathematical specification

### 2.1 State variable and observable

```
N(t) ∈ (0, 1]           surviving nephron mass fraction (latent)
eGFR(t) = G_max · N^α    observable
```

with `G_max = 120` mL/min/1.73m² (filtration of an intact kidney) and `α = 0.80`. Because
`α < 1`, eGFR is a *weak* power of `N`: filtration is buffered while reserve remains and
falls off only as `N` gets small. The dialysis threshold `eGFR < 15` corresponds to
`N < (15/120)^(1/0.8) ≈ 0.074`.

The inverse maps are `N_of_egfr` and `egfr_of_N`; `N_of_egfr` saturates at `N = 1` for
eGFR ≥ ~120.

### 2.2 The hazard (the core of the model)

The per-nephron hazard `h` drives the ODE

```
dN/dt = −N · h(N, covariates, u)
```

and is the sum of a baseline, a structural (hyperfiltration) term, an albuminuric term,
and a residual metabolic insult:

```
h = k0
  + k_hf · s^q / (1 + (s/S_SAT)^q)                    structural (saturating hyperfiltration)
  + k_alb · log(1 + UACR(t)/30)                       albuminuric
  + w_A1c·(HbA1c − 6.5)+ + w_SBP·(SBP − 130)+/10      metabolic residual
```

where `s = N_ref/N = 1/N` is the per-nephron overload (larger as `N` falls), and `(x)+`
denotes `max(x, 0)`.

**The structural term is *saturating*, not an unbounded power law.** An earlier version
used `k_hf · s^q`, which diverges as `N → 0` and drove untreated patients to dialysis far
too fast (it failed the in-silico trials). The current form
`k_hf · s^q / (1 + (s/S_SAT)^q)` rises like `s^q` at first, then levels off once
`s > S_SAT`. With `S_SAT = 3.5` the transition sits at **eGFR ≈ 44** (`egfr_of_N(1/3.5)`).
Physiologically this is a ceiling on how much a single surviving nephron can
hyperfiltrate; clinically, eGFR ~44 is where stage 3b decline is known to steepen.

**`q` is the collapse exponent** — it controls how *sharp* the transition is, not where it
sits (`S_SAT` sets the location). As `q` grows, the structural term goes from a gentle
gradient toward a near-step. This is the model's most scientifically interesting parameter,
and its value differs sharply between populations (see §4).

**The albuminuric term is separate** (Round 16). It was previously buried inside the
metabolic block as `w_UACR·log(1+UACR/30)`, which meant it was calibrated on cohorts that
had no albuminuria to learn from. It now has its own coefficient `k_alb` and is calibrated
where albuminuria is observable (the trials). The `log(1+UACR/30)` form is deliberate:
linear over-weights very high UACR, and a threshold form would wrongly zero out
microalbuminuria, which does predict progression. The log form is the one already validated
in-silico against all three trials.

In the endogenous-albuminuria simulator, `UACR(t)` is not a constant input — it is generated
by the model from the state:

```
UACR(t) = UACR_0 · (s(t)/s_0)^β · (1 − eff_alb·u)
```

with `β = 1.0`. Albuminuria rises as nephrons are lost and drops under treatment. This is
why UACR is an *output* the model can be checked against, not just an input.

### 2.3 Intervention (control)

A drug indicator `u ∈ {0,1}` (SGLT2i-like) enters by attenuating the mechanistically
appropriate terms:

```
structural  → × (1 − eff_hf·u)
metabolic   → × (1 − eff_met·u)
albuminuria → UACR reduced by (1 − eff_alb·u)
```

The efficacies are **anchored to trial effect sizes, not freely fitted** (see §3).

### 2.4 Numerical guard

The hazard is capped at `HAZARD_CAP = 50/yr`. This is a numerical guard, not biology: at
that rate a nephron population halves every ~5 days, which no patient survives. It exists so
the ODE solver cannot blow up while an optimizer explores an absurd corner of parameter
space, and it should never bind for a plausible patient. If it binds, the parameters are
wrong, not the patient.

### 2.5 The default (trial-anchored) parameter set

`TRIAL_CALIBRATION_V2` in `model_core.py`, the calibration the app ships with:

| Parameter | Value | Meaning |
|---|---|---|
| `q` | 1.52 | collapse exponent (trials) |
| `k_hf` | 0.01029 | structural hazard scale |
| `k_alb` | 0.01314 | albuminuric hazard scale |
| `w_a1c` | 0.01051 | HbA1c insult weight |
| `w_uacr` | 0.01314 | legacy UACR weight (migrated → `k_alb`) |
| `w_sbp` | 0.00788 | SBP insult weight |
| `k0` | 0.0030 | baseline (aging) loss |
| `s_sat` | 3.5 | saturation point (breakpoint eGFR ≈ 44) |
| `beta` | 1.0 | UACR–state coupling exponent |
| `eff_met` | 0.669 | drug effect on metabolic insult |
| `eff_hf` | 0.521 | drug effect on hyperfiltration |
| `eff_alb` | 0.286 | drug effect on albuminuria |

`w_uacr` is retained for backward compatibility: any parameter dict written before Round 16
carries `w_uacr` but no `k_alb`, and `_k_alb_of(p)` reads `w_uacr` as `k_alb` so old dicts
drive the hazard identically.

---

## 3. Calibration by domain

The central design decision: **each parameter is estimated from the data that actually
contain information about it.** Fitting everything on one source is what produced the
degeneracies this project spent several rounds diagnosing.

| Parameter(s) | Calibrated on | Why |
|---|---|---|
| `q`, `k_hf` | MIMIC-IV | structural signal; 2259 patients, ~109k observations, k-fold CV 3%/9% |
| `k_alb` | Trials (CREDENCE/DAPA/EMPA) | albuminuria observable there; MIMIC is near-normoalbuminuric (median UACR ~23) |
| `w_a1c`, `w_sbp` | Trials | k-fold flags them unidentifiable in MIMIC (CV 20%/54%) |
| `eff_met`, `eff_hf`, `eff_alb`, `S_SAT` | Trials | anchored to placebo-vs-treated effect sizes, not fitted to patient data |

The trial anchors and their sources are documented and verified in
`docs/TRIAL_DATA_PROVENANCE.md`. In-silico replication (`src/insilico_trial.py`) fits three
parameters on CREDENCE + EMPA-KIDNEY and predicts DAPA-CKD **out-of-sample**; the held-out
chronic slope difference lands inside the published CI (2.26, 95% CI 1.88–2.64) across seeds.

---

## 4. Two populations, one hazard — the central empirical finding

Calibrating independently on hospital records and on trial placebo arms revealed that they
do not share a collapse exponent:

| Population | Median UACR | `q` | Shape of decline |
|---|---|---|---|
| Trial placebo arms (macroalbuminuric) | ~927 | 1.52 | early, albuminuria-driven progression |
| MIMIC-IV (T2D, near-normoalbuminuric) | ~23 | **2.92** (k-fold CV 3%) | late, abrupt structural collapse below eGFR ~44 |

This is not a calibration artifact. `q ≈ 2.92` is identified (stable across CV folds, and it
does **not** run to the bound when `--q-max` is widened to 8). A single-regime hazard cannot
place its collapse "step" in two locations at once, which is why the Round 15 hybrid
(anchoring the weights to the trials) sent `q` to its ceiling and *raised* the cost. The
resolution is the two-term hazard of §2.2: at low UACR the structural term with `q ≈ 2.92`
dominates (MIMIC); at high UACR the albuminuric term switches on (trials). Because MIMIC is
near-normoalbuminuric, `k_alb` has almost no leverage there, so one parameter set can serve
both populations.

For a product aimed at an IMSS-like population (mostly normoalbuminuric, but including the
sickest patients who present already albuminuric), this two-regime structure is not optional:
a single-regime model calibrated on either population alone would dangerously misjudge the
other.

The index date matters here. In a hospital cohort the first available creatinine is often
drawn during an acute episode (dehydration, sepsis, contrast) and is not the patient's
chronic baseline. The loader supports a KDIGO-style **confirmed** index
(`--index-strategy confirmed`): an eGFR qualifies as baseline only if a measurement ≥90 days
later has not recovered by >30%. On the real cohort this moved the index off the first
creatinine for ~2155 patients.

---

## 5. Measurement model and identifiability

### 5.1 eGFR from the lab (`egfr_measurement.py`)

Three CKD-EPI equations, coefficients pinned by a regression test:

1. `egfr_cr` — creatinine only (CKD-EPI 2021, race-free).
2. `egfr_cys` — cystatin C only (CKD-EPI 2012).
3. `egfr_cr_cys` — combined (most precise).

All three are vectorised (accept whole cohort columns).

### 5.2 What is identifiable, and what is not — measured, not assumed

Observing only eGFR, some parameters are degenerate and must be fixed or anchored:

- **`α`, `G_max`, the scale of `N`** — fixed (`α = 0.80`, `G_max = 120`, `N_ref = 1`).
- **`k0` and `k_hf`** — partially confounded (both near-baseline); `k0` is fixed.
- **The insult weights** need covariate variation *across* patients; a single patient cannot
  identify them.

Two findings from simulation experiments (`measurement_strategy.py`) that corrected earlier
assumptions in this very document:

- **`q` is essentially unidentifiable from a single patient's routine data** (R² ≈ 0.0–0.08),
  and **no assay fixes this — cystatin C included.** An earlier version of this document
  claimed the combined assay reduced the error in `q` ~5×. That claim was **wrong and has
  been removed.** Cystatin C helps, but it helps the patient's *injury rate*, not `q`. (Note
  the contrast with §4: `q` is population-identifiable from thousands of patients even though
  it is not patient-identifiable. These are different questions.)
- **Follow-up time span matters far more than the number of measurements.** The same 4–6
  creatinines spread over 4–8 years recover the injury rate ~3× better than the same number
  crammed into 1–2 years. Practical consequence: pull the patient's old creatinine values out
  of the chart — they are free and beat buying a new assay. Serial UACR does **not** add
  independent information (it is a deterministic function of the same latent state).

### 5.3 Minimal panel per visit

- **Blood:** serum creatinine, HbA1c (cystatin C optional).
- **Urine:** UACR (single sample).
- **In clinic:** blood pressure.
- **Frequency:** every 3–6 months; a *long* history matters more than a dense one.

---

## 6. Per-patient personalization (`personalize.py`)

For a single patient, NephroQ infers two things from their eGFR history: an individual
injury-rate multiplier and (weakly) `q`. The method is **amortized inference**: an ensemble
of small neural networks trained *entirely on simulations from the mechanistic model* solves
the inverse problem, and the ODE does the forward projection. It is a hybrid, not a black box
— the network estimates parameters, the mechanism makes the prediction.

- On held-out virtual patients, +5y forecast RMSE ≈ 9.2 (network) vs ≈ 16 (population
  parameters, −43%) vs ≈ 10.6 (classical per-patient least squares), at ~30× the speed of
  least squares.
- The raw ensemble spread is **not** a calibrated uncertainty (a nominal 90% band covered the
  truth ~33% of the time for `q`). It is therefore **conformalized** on a held-out split; the
  reported 90% intervals then achieve ~90% coverage. The app shows conformal intervals, never
  "± spread".

---

## 7. Validation status

What is implemented and passing:

- **In-silico trial replication** — DAPA-CKD predicted out-of-sample, held-out chronic slope
  inside the published CI across seeds (`insilico_trial.py`, tests in
  `test_insilico_trial.py`).
- **MIMIC-IV calibration** — `q`, `k_hf` fitted with robust loss, patient-level train/test
  split, k-fold CV for identifiability, patient-level bootstrap for parameter uncertainty
  (`calibrate_mimic.py`).
- **Calibration audit** — a fitted parameter set is required to reproduce the published
  placebo-arm slopes of all three trials before it is trusted (`audit_calibration.py`). An
  internally consistent fit to a biased cohort is still biased; the audit is the external
  check.
- **KFRE head-to-head (Mode C)** — exploratory, AUC-based. Reported honestly as exploratory:
  the proxy outcome (observed eGFR<15) is not treated kidney failure, so absolute-risk metrics
  are not interpretable against KFRE, and the event counts are small.

Honest limitations to state in any write-up:

- The model is low-dimensional and treats `N` as monotonic; it does not represent
  inflammation, fibrosis, or acute events that can transiently move eGFR.
- `w_a1c` and `w_sbp` are not identifiable from a near-normoalbuminuric cohort and are
  anchored to trials rather than reported as MIMIC estimates.
- Trial anchors are transcribed from published articles and figures (verified against source
  PDFs, but with the reading uncertainty inherent in figure read-offs).
- No prospective clinical validation yet. Every calibration carries a "research-use, not
  externally validated" label regardless of internal quality metrics.

---

## 8. File map

| File | Content |
|---|---|
| `model_core.py` | The ODE, the hazard, the trial-anchored parameter set. One model, no dead code. |
| `egfr_measurement.py` | CKD-EPI 2021/2012 equations (creatinine / cystatin / combined), vectorised. |
| `calibrate_mimic.py` | MIMIC-IV calibration: robust fit, k-fold CV, bootstrap, three eval modes, `--anchor-weights`, `--index-strategy`, `--q-max`. |
| `mimic_loader.py` | Builds the cohort from local MIMIC-IV; confirmed-index logic. |
| `insilico_trial.py` | Falsifiable in-silico replication (fit CREDENCE+EMPA, predict DAPA-CKD). |
| `audit_calibration.py` | Post-calibration gate: reproduce the published placebo arms, or don't ship. |
| `personalize.py` | Amortized per-patient inference with conformalized uncertainty. |
| `measurement_strategy.py` | Simulation experiments on what is worth measuring. |
| `mvp_calibration.py` | Own-data calibration path (TSV input, same model_core). |
| `docs/TRIAL_DATA_PROVENANCE.md` | Every trial value, its source, and verification status. |
