# Model documentation — Mechanistic digital twin: Diabetes → CKD
### Specification, laboratory panel, step-by-step implementation, and publication analysis

This documentation accompanies the code built in this repository:

---

## 1. What the model is

It is a **low-dimensional dynamical system** describing how a diabetic
patient's kidney function deteriorates over the years, calibratable to
individual data and able to simulate interventions. It has three features
that distinguish it from an ordinary statistical fit:

**It is mechanistic.** The state variable is `N`, the functional nephron
mass fraction (irreversible, monotonically decreasing). Its dynamics encode
two nephrology facts: **hyperfiltration** (positive feedback: losing
nephrons overloads the remaining ones, damaging them faster → power law in
the hazard) and **compensation** (observed eGFR is buffered while there is
reserve, and collapses at the end → weak power law in the observable).

**It is identifiable and parsimonious.** After removing degeneracies, the
set of physical parameters is small (≈6), and the most interesting one is
`q`, the **feedback exponent** that quantifies how abrupt the terminal
collapse is — a physically meaningful number the model estimates from data.

**It is a digital twin, not a static model.** It synchronizes with the
patient via data assimilation and supports three inference regimes
depending on how much data is available: per-patient fitting, an amortized
AI estimator, and a hierarchical model with partial pooling.

---

## 2. Mathematical specification

### 2.1 State variable and observable
```
N(t) ∈ (0, 1]      functional nephron mass fraction (latent, not observed)
eGFR = G_max · N^α   observable (α < 1: compensation = weak power law)
```
`G_max ≈ 120` mL/min/1.73m² (filtration with an intact kidney). The dialysis
threshold is `eGFR < 15`, equivalent to `N < (15/G_max)^{1/α} ≈ 0.074`.

### 2.2 Dynamics (the core ODE)
```
dN/dt = −N · [ k0 + k_hf·(1/N)^q + I(covariates) ]
```
- `k0` — baseline nephron loss (aging).
- `k_hf·(1/N)^q` — **hyperfiltration**: the term that grows as `N` falls. The
  exponent `q` controls the regime:
  - `q < 1`: buffering (decline slows near the end).
  - `q ≈ 1`: nearly linear decline.
  - `q > 1`: **accelerated collapse** (the clinically realistic case; our
    reference value `q ≈ 1.6`).
- `I` — **metabolic insult**, a function of patient covariates:
```
I = w_A1c·(HbA1c − 6.5)₊ + w_UACR·log(1 + UACR/30) + w_BP·(SBP − 130)₊/10
```

### 2.3 Intervention (control)
A drug `u ∈ {0,1}` (SGLT2i / RAAS blockade) enters by modifying meaningful terms:
```
I → I·(1 − eff_met·u)        reduces the metabolic insult
k_hf → k_hf·(1 − eff_hf·u)   reduces hyperfiltration (real SGLT2i mechanism)
```
The `eff_*` efficacies are anchored to effect sizes from the DAPA-CKD /
CREDENCE / FLOW trials, not freely fitted.

### 2.4 Identifiability (critical — read before fitting)
Observing only eGFR, there are degeneracies that must be resolved or the fit
will not converge:
- **`α` is not identifiable** together with the rates (trade-off with the
  scale of `N`). → **Fix `α`** to a literature value (we use 0.80).
- **`k0` and `k_hf` are partially confounded** (both nearly baseline). →
  Fix `k0` or report only the identifiable combination.
- **The scale of `N` is free.** → Fix `N_ref = 1` and `G_max`.
- **The insult weights `w` need variation ACROSS patients** to separate;
  with a single patient they are not identifiable.

Minimal identifiable set: **`{ q, k_hf_eff, (w_A1c, w_UACR, w_BP) }`** with
`α`, `k0`, `G_max` fixed. The star parameter is `q`.

---

## 3. Measurement model and laboratory panel

### 3.1 Where each observed variable comes from
| Variable | Source | Sample type |
|---|---|---|
| eGFR | computed from creatinine and/or cystatin C + age + sex | **blood** |
| HbA1c | direct test | **blood** |
| UACR (albuminuria) | albumin/creatinine ratio | **urine** |
| SBP (systolic blood pressure) | cuff measurement | physical measurement |
| age, sex | demographics | — |

**Conclusion:** almost everything comes from a single blood draw (eGFR +
HbA1c), but the model **also requires a urine sample** (UACR, a strong
progression predictor) and a **blood pressure reading**. Everything comes
from a routine, inexpensive outpatient visit — exactly what a diabetic
patient's follow-up already generates.

### 3.2 The cystatin C option (implemented in `egfr_measurement.py`)
eGFR can be calculated three ways, from lower to higher precision:
1. `egfr_cr` — creatinine only (CKD-EPI 2021).
2. `egfr_cys` — cystatin C only.
3. `egfr_cr_cys` — combined (most precise). **Recommended.**

Why this matters physically: **lower measurement noise ⇒ `q` is much better

| Assay | noise σ | estimated q |
|---|---|---|
| creatinine only | 3.5 | 1.78 ± 0.15 |
| cystatin C | 2.6 | 1.61 ± 0.11 |
| creatinine + cystatin | 1.8 | **1.54 ± 0.03** |

Switching to the combined assay reduces the uncertainty of `q` by **5×** and
lowers the bias. For a study of the collapse exponent's identifiability, it
is worth requesting cystatin C.

### 3.3 Minimal laboratory panel per visit
What a physician would need to order to feed the twin:
- **Blood:** serum creatinine, HbA1c, and (recommended) cystatin C.
- **Urine:** UACR (albumin/creatinine ratio, single sample).
- **In clinic:** blood pressure.
- **Frequency:** every 3 months is ideal (as in the open UAE dataset); every
  6 months is viable.
- **Once (optional, CKDu module):** a blood sample for DNA/sequencing in
  atypical-etiology cases.

---

## 4. The three inference levels (when to use each)

| Level | File | When to use it | What it gives |
|---|---|---|---|

Practical rule: with real cohorts (many patients, uneven follow-up) the
correct model is **hierarchical**; the amortized one serves real-time
clinical scoring; the per-patient one only for the densest cases.

---

## 5. Step-by-step implementation

> Requirements: `pip install numpy scipy matplotlib scikit-learn`. Each
> script is self-contained and produces a figure.

**Step 1 — Understand the nonlinear core.**

the hyperfiltration mechanism. *Checkpoint:* mechanistic curves accelerate
near the end; the linear one does not.

**Step 2 — (Optional) Analytic matrix variant.**

to dialysis via the fundamental matrix (`m = −T⁻¹·1`), exact Kalman filter.
Use this for analytic tractability; the mechanistic one (Step 1) for
realism.

**Step 3 — eGFR measurement.**
`python egfr_measurement.py` → CKD-EPI 2021 equations. Decide the assay
(recommended: combined) and fix the corresponding noise `σ` for the
following steps.

**Step 4 — Inverse problem (verification on synthetic data).**

*Mandatory checkpoint:* chi²/n at θ_true must be ≈ 1 (if not, there is a bug
before touching real data). Confirms `q` and the weights are recovered;
`k0`/`k_hf` are degenerate (expected).

**Step 5 — Identifiability vs. noise.**

cystatin. Justifies the assay choice.

**Step 6 — Amortized AI estimator.**

parameters). *Checkpoint:* with few visits, the error in `q` is stable and
lower than per-patient fitting.

**Step 7 — Hierarchical model (the real-data one).**

*Checkpoint:* the error in `η_i` for patients with 3–5 visits drops ~2–3×
relative to no pooling.

**Step 8 — Connect real data.**
Prepare a CSV with columns `patient_id, time_years, egfr, hba1c, uacr, sbp`
(mapping the EAU / CRIC / AASK dataset) and run:

loader already accepts it.

**Logical order:** 1 → 3 → 4 → 5 → 7 are the main route; 2 and 6 are
extensions.

---

## 6. Publication analysis

### 6.1 What you already have (the contribution)
A **novel combination**: a mechanistic renal progression model with (a)
hyperfiltration feedback as a power law with an **identifiable collapse
exponent `q`**, (b) compensation as a weak power law in the observable,
embedded in (c) a digital-twin framework with amortized inference and (d) a
hierarchical model with *partial pooling*. The pieces exist separately in
the literature; the synthesis and the estimation of `q` from clinical
trajectories do not.

### 6.2 What is missing for publication (in priority order)
1. **Validation with real data.** This is gap #1: today everything is
   synthetic. You need to calibrate and validate on at least one real
   cohort (open UAE dataset for a prototype; CRIC/AASK with a DUA for the
   formal study).
2. **External validation by in-silico trial replication.** Reconstruct
   DAPA-CKD/CREDENCE/FLOW with virtual cohorts and verify you recover the
   reported hazard ratios. This is the credibility test reviewers and
   regulators expect.
3. **Comparison against established baselines.** The model must beat or
   match: the kidney failure risk equation (KFRE), linear mixed-effects
   models of eGFR slope, and joint longitudinal-survival models. Without
   this comparison, it will not be accepted.
4. **Full Bayesian inference.** Move from empirical-Bayes (EM) to NUTS
   (numpyro) to deconfound `q` from `k_hf` and give honest posteriors with
   propagated uncertainty.
5. **Calibrated uncertainty quantification.** Conformal prediction for
   per-patient intervals with guaranteed coverage, with equity auditing by
   sex and ancestry.
6. **Standards-compliant reporting.** Follow **TRIPOD+AI** (clinical
   prediction models with AI); pre-registration if applicable; data
   governance and ethics approval for real data.

### 6.3 Novelty analysis by angle (and where to publish)
- **Methodological angle (twin + identifiability of `q`):** *npj Digital
  Medicine*, *PLOS Computational Biology*, *Journal of the Royal Society
  Interface*.
- **Physics angle (the collapse exponent `q` as an order parameter; first-
  passage-time theory of the decline):** *PRX Life*, *Physical Review E*.
  Your profile carries more weight here.
- **Nephrology angle (predicting progression and the timing of
  intervention):** *Kidney International*, *JASN*, *CJASN*. Requires the
  full clinical validation (6.2, points 1–3).
- **Impact angle in Mexico (burden, cost, equity):** *Salud Pública de
  México*, *Value in Health*.

### 6.4 Realistic publication strategy
- **Paper 1 (achievable soon):** the mechanistic model + identifiability of
  `q` + the hierarchical inference framework, validated on the open UAE
  cohort. Methodological/physics venue.
- **Paper 2 (with CRIC/AASK + trial replication):** the twin as a clinical
  prediction tool, compared against KFRE. Nephrology venue.
- **Paper 3 (with the CKDu agent + genomics):** unknown etiology. Nephrology/
  genomics venue.

### 6.5 Limitations to state honestly
- The model is low-dimensional: it ignores mechanisms (inflammation,
  fibrosis, acute events) that can break the monotonicity of `N`.
- Population `q` is confounded with the distribution of `k_hf` under random
  effects → requires the Bayesian treatment to separate them.
- The weights `w` require covariate heterogeneity; in homogeneous cohorts
  they will be poorly identified.
- The causal validity of intervention simulations depends on anchoring
  `eff_*` to RCTs and on a causal layer (not included in this core).
- Synthetic data ≠ real data: external validation is essential before any
  clinical claim.

### 6.6 Minimal checklist before submission
- [ ] Calibrated and validated on ≥1 real cohort, with external validation.
- [ ] Compared against KFRE and a linear mixed model.
- [ ] In-silico replication of ≥1 trial (HR within the published CI).
- [ ] Calibrated uncertainty (conformal) + equity audit.
- [ ] Bayesian posteriors for `q` (not just point estimate + Hessian).
- [ ] Full TRIPOD+AI report; open code and synthetic data.
- [ ] Ethics approval / data use agreement documented.

---

## 7. File summary

| File | Content |
|---|---|

| `egfr_measurement.py` | CKD-EPI 2021 equations (creatinine/cystatin/combined) + noise |

