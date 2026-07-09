# Protocol — Mechanistic Digital Twin: Type 2 Diabetes → Chronic Kidney Disease
### v0.2

**Author:** Daniel Pérez-Calixto, Ph.D. — Research Scientist, INMEGEN / Adjunct Lecturer, UNAM Facultad de Ciencias

---

## 1. Rationale

Type 2 diabetes is the leading cause of chronic kidney disease (CKD) and
kidney failure worldwide and in Mexico. The economic asymmetry is stark:
managing diabetes costs a fraction of what dialysis costs per patient-year.
A tool that identifies, early and from routine labs, which diabetic
patients will progress to CKD — and when to intervene — targets the single
highest-leverage point in the health system's cost structure.

## 2. Objective

Build a mechanistic **digital twin** of an individual patient's diabetes →
CKD axis: a calibratable dynamical system that reproduces the patient's
observed trajectory, quantifies uncertainty, and simulates the effect of
interventions (SGLT2 inhibitors, RAAS blockade) on the time to dialysis.

## 3. General architecture

```
                    ┌─────────────────────────┐
   Patient data ──> │  DATA ASSIMILATION       │ <── model (mechanistic core)
   (labs, visits)   │  (Kalman / particle      │
                    │   filter)                │
                    └───────────┬─────────────┘
                                │
                    ┌───────────▼─────────────┐
                    │  CALIBRATED STATE         │
                    │  (N, eGFR, UACR, ...)     │
                    └───────────┬─────────────┘
                                │
              ┌─────────────────┼─────────────────┐
              ▼                 ▼                 ▼
     ┌────────────────┐┌────────────────┐┌────────────────┐
     │ RISK             ││ INTERVENTION    ││ UNCERTAINTY     │
     │ CLASSIFIERS      ││ OPTIMIZATION    ││ QUANTIFICATION  │
     └────────────────┘└────────────────┘└────────────────┘
```

## 4. Mechanistic core

The state variable `N(t)` (functional nephron mass fraction) evolves as:

```
dN/dt = -N * [ k0 + k_hf*(N_ref/N)^q + k_met * I(HbA1c, UACR, SBP) ]
eGFR = G_max * N^alpha
```

See `docs/MODEL_DOCUMENTATION.md` for the full mathematical specification,
identifiability analysis, and measurement model.

## 5. Agent and classifier layers (extended architecture)

Beyond the mechanistic core, the full architecture (for later phases)
includes:

- **C1 — Phenotype classifier:** typical diabetic pattern vs. atypical
  (candidate for unknown etiology, CKDu).
- **C2 — Progression risk classifier:** P(dialysis within 2/5 years). *The
  most clinically important.*
- **C3 — Stage and inflection classifier:** detects the change in eGFR
  slope (when decline accelerates).
- **C4 — Treatment response classifier:** will the patient respond to
  SGLT2/RAAS/GLP-1?
- **C5 — Data quality classifier:** is this lab value plausible?
- **Causal layer:** individualized treatment effect (ITE) estimation via
  `dowhy`/`econml`, feeding an offline-RL intervention recommender
  (Conservative Q-Learning) with a safety gate against contraindicated
  actions.
- **Etiology discovery agent:** for CKD of unknown origin (CKDu), a
  generate–critique–rank loop (Researcher / Reviewer / Moderator roles)
  with self-verification against databases, integrating a genomic
  component.

## 6. Section 17 — Frontier upgrades (roadmap)

- **PINN (Physics-Informed Neural Network):** solve the inverse problem by
  embedding the ODE residual in the loss function, robust to noisy, sparse
  data.
- **Schrödinger bridge / Wasserstein-Fisher-Rao transport:** frame
  intervention as minimum-action transport between pre/post-treatment
  distributions — has a closed-form solution in the linear-Gaussian case.
- **Amortized inference (NPE/SBI):** train once on simulated patients,
  infer a new patient's posterior instantly.
- **Conformal prediction:** guaranteed-coverage intervals for every
  prediction, with equity auditing across subgroups.
- **Multi-agent system (AI co-scientist style):** LLM-based orchestration
  for hypothesis generation, verification, and interpretable reporting.

## 7. Validation plan

1. **Synthetic verification:** parameter recovery, chi²/n ≈ 1 at ground
   truth (done, see `inverse_fit.py`, `bayesian_model.py`).
2. **Face validity:** real published baseline profiles must separate
   progressors from non-progressors in the correct direction (done, see
   `real_data_validity.py`).
3. **Real longitudinal cohort:** calibrate and validate on real trajectory
   data (CRIC, HCHS/SOL, MIMIC-IV, or a direct clinical collaboration).
4. **In-silico trial replication:** reconstruct DAPA-CKD/CREDENCE/FLOW with
   virtual cohorts; the simulated hazard ratio must fall within the
   published confidence interval.
5. **Comparison against clinical baselines:** KFRE, linear mixed models.

## 8. Data sources

See `docs/HOW_TO_DOWNLOAD_DATA.md` for the full list of datasets evaluated
(open, synthetic, and restricted-access) and how to request them.

## 9. Deliverables

- Verified, tested, documented source code (`src/`).
- Step-by-step methodology guide for students (`docs/STUDENT_METHODOLOGY.md`).
- Full technical documentation (`docs/MODEL_DOCUMENTATION.md`).
- MVP calibration and validation pipeline (`mvp_calibration.py`).
- Web interface for clinician demos (`app_web.py`).
- TRL4 evidence dossier (`docs/TRL4_dossier.md`).
