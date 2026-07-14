# Changelog

Notable fixes and changes to NephroQ, driven by several rounds of detailed code review. For currently open limitations, see the **Limitations** section of the [README](../README.md).

## Round 9 — calibration speed, a calibration auditor, and one model everywhere

### Fixed — three bugs, one of them capable of killing a running calibration

- 🔴 **`predict_egfr_at_v2` crashed on same-day lab draws.** The Round 9 speedup (below)
  integrates straight onto the visit times, and `solve_ivp` requires `t_eval` to be
  **strictly increasing**. Real data is not: hospital records routinely contain several
  creatinines drawn on the **same day**, and callers may pass times in any order. The result
  was `ValueError: Values in t_eval are not properly sorted` on perfectly valid patients.
  The predictor now deduplicates and sorts internally, then scatters the results back to the
  requested order.

- 🔴 **`mvp_calibration.py` was fitting a DIFFERENT model from the one the app projects
  with.** It carried its own fixed-step Euler integrator and the **old unbounded hazard**,
  bypassing `model_core` entirely. On the same patient with the same parameters it drifted by
  up to **13 mL/min at 10 years** (10.9 vs 24.0). Since this is the path the README recommends
  for calibrating on your *own* data, users were fitting one model and projecting with
  another. It now calls `model_core`; the two agree to **0.0000 mL/min**. This is the same
  class of bug as the two diverging integrators fixed in an earlier round, and it is now
  locked down by a test.

- 🟠 **Temporal leakage in the own-data loader.** Covariates were taken as the **median over
  each patient's whole trajectory**, which feeds a patient's own future into a supposedly
  baseline forecast. They are now taken at **baseline**, and a missing baseline is filled from
  the **cohort's baseline median** — never from that patient's later visits. The imputed
  fraction is reported.

### Changed

- **Input files for `mvp_calibration.py` are now TAB-separated (`.tsv`).** Clinical exports
  routinely contain commas inside fields (free-text sites, `"Apellido, Nombre"`), which
  silently corrupt a CSV. The delimiter is sniffed from the header, so an existing
  comma-separated file still works; `CKD_DATA` is the new environment variable and `CKD_CSV`
  is still honoured.

### Added — the calibration is now fast enough to iterate on

- **~7× algorithmic speedup, for free.** `predict_egfr_at_v2` used to build a dense grid (up
  to 580 points for a patient with 29 visits) and then interpolate onto the visit times. It
  now integrates **straight onto those times**: 3.62 ms → **0.49 ms** per patient, agreeing
  with the canonical simulator to <10⁻⁶ mL/min. This applies everywhere, with no flags — even
  the test suite dropped from 48 s to 23 s.
- **`--n-jobs`: parallel residual evaluation.** Patients are independent, so the residual
  vector splits cleanly across cores. The chunks are reassembled in the **original patient
  order**, so the optimizer sees bit-for-bit what it would have seen serially — verified, and
  locked by a regression test asserting serial and parallel agree to 10⁻⁹. Combined with the
  algorithmic speedup, a run that was heading for 5–15 hours becomes minutes.
  `joblib` is now declared explicitly rather than relied on as a transitive dependency of
  scikit-learn.

- **`src/audit_calibration.py` — do not trust a calibration until it has been audited.** Run
  it on the JSON that `calibrate_mimic.py` produces. Three checks, in order of severity:
  1. **Did the optimizer actually move?** A frozen fit returns round-number parameters and a
     bootstrap with ~zero variance. If that happened, nothing else means anything, and the
     script refuses to interpret the rest.
  2. **How far is MIMIC from the trial-anchored reference?** Reported as a hazard ratio. A
     ratio > 1 means MIMIC thinks patients decline faster than real trial placebo arms do —
     most likely because the MIMIC index date is the first available hospital creatinine,
     often drawn during an acute episode. The ratio *quantifies that bias* rather than
     estimating progression better.
  3. **Can the MIMIC parameters reproduce the published PLACEBO arms** of CREDENCE, DAPA-CKD
     and EMPA-KIDNEY? The placebo arm receives no drug, so nothing can hide there. This is an
     external judge, independent of the internal chi²/n: **an internally consistent fit to a
     biased cohort is still biased.**

## Round 8 — per-patient personalization (amortized inference)

### Added

- **`src/personalize.py` — the model is now personalized per patient.** Until now every
  patient was projected with the same population parameters: two patients with an identical
  eGFR today got an identical future, even if one had been collapsing for three years and
  the other was flat. Given a few past creatinine values, an **amortized (simulation-based)
  neural estimator** now infers that patient's own injury rate and collapse exponent `q`.
  - It is a **hybrid** model, not a black box: the network solves only the INVERSE problem,
    and the forward projection remains the mechanistic ODE. Its output is two physically
    meaningful numbers.
  - It is trained **entirely on simulations from the mechanistic model**, so no patient data
    is needed to train it.
  - Validated on held-out virtual patients (forecast 5 years past the last visit): RMSE
    **9.42** vs **15.55** for population parameters (**−39%**), also beating a classical
    per-patient least-squares fit (11.61) while being ~28× faster.
  - **Honest limit, reported in the app:** `q` is only weakly identifiable (R² ≈ 0.15) from
    sparse noisy measurements. Nearly all the benefit comes from inferring the injury rate.
  - With fewer than 3 measurements (or under 9 months of history) it **refuses to
    personalize** and falls back to population parameters, saying why.
- Measurement-history editor in the app sidebar, which drives the personalization.
- **`src/measurement_strategy.py` — what is actually worth measuring.** Simulation
  experiments on how well each strategy recovers a patient's parameters. Three findings, two
  of which contradicted our own prior assumptions:
  - **`q` is essentially unidentifiable from routine data** (R² ≈ 0.0–0.08) and **no assay
    fixes it, cystatin C included.** The app previously claimed cystatin C reduced the error
    in `q` roughly 5×; that claim was **wrong and has been removed**. Cystatin C helps, but it
    helps the patient's *injury rate*, not `q`.
  - **The time span of the history matters far more than the number of measurements.** The
    same 4–6 creatinines spread over 4–8 years recover the injury rate ~3× better than the
    same number crammed into 1–2 years (R² 0.59 vs 0.18), and better than 10–14 values inside
    a short window (0.34).
  - **Duplicate creatinine + a long history beats cystatin C alone** (R² 0.71 vs 0.67).
    Practical consequence: pull the patient's old creatinine results from the chart. They
    already exist and are free.
  - **Serial UACR does not help** (0.47 vs 0.48 baseline), refuting the intuition that a
    second cheap biomarker must add signal: in this model albuminuria is a deterministic
    function of the same latent state, so it adds no independent information while carrying
    large biological noise.

### Fixed

- **The shipped estimator can never become a liability.** `calibration/personalizer.pkl`
  (0.6 MB) is committed so the app starts instantly, but it is *never required*: if it is
  missing, or unloadable because scikit-learn changed its pickle format, `get_estimator()`
  retrains it from simulations (~13 s) instead of crashing. This removes the usual fragility
  of shipping a pickle. Training was retuned (2500 sims, ensemble of 4 nets) so that
  first-use cost is acceptable, at negligible accuracy cost.
- The estimator is persisted by saving its **components**, not the class instance: pickling
  the instance recorded it as `__main__.Personalizer` when the module was run as a script,
  which then failed to unpickle from the app. A missing or version-incompatible estimator now
  degrades gracefully to population parameters instead of crashing.
- `scikit-learn` was declared in `requirements.txt` but unused; it is now genuinely used.

## Round 7 — model v2, trial anchoring, and the release cleanup

This round replaced the model's two weakest structural assumptions, re-anchored its
parameters to published trials instead of to a hospital dataset, and reduced the
repository to what is actually needed to run and audit the application.

### Changed — the model itself

- **Hyperfiltration now saturates.** The hazard used an *unbounded* power law,
  `k_hf·(N_ref/N)^q`, which diverges as nephrons are lost. A surviving nephron raises its
  single-nephron GFR by a bounded factor (~3×), not without limit. It is now a Hill
  saturation, `k_hf·s^q / (1 + (s/S_SAT)^q)`. The ceiling `S_SAT` is **identified**, not
  guessed: anchoring the hazard on CREDENCE (mean eGFR 56) and scoring it on EMPA-KIDNEY
  (mean eGFR 37) gives a clear optimum around 3–4, matching the physiological ceiling.
- **Albuminuria is now endogenous.** UACR was fed in as a *constant exogenous insult*. That
  is mechanistically backwards — albuminuria is largely a *consequence* of glomerular
  hypertension — it double-counted the same process, and it made a published fact
  structurally inexpressible. UACR is now a model **output**:
  `UACR(t) = UACR₀·(s(t)/s₀)^β·(1 − eff_alb·u)`. The app plots it, and the model predicts a
  ~29% immediate reduction under renoprotective therapy (SGLT2i trials published 31–35%).
- **Parameters are anchored to published trials, not to MIMIC.** Progression is fixed by the
  placebo arms of CREDENCE and EMPA-KIDNEY; treatment effects by CREDENCE. The default
  calibration ships with the repo and needs no data.
- **Consequence — projections are slower, and correct.** The previous parameters made
  untreated patients decline ~1.9× faster than the real trial placebo arms. Example-patient
  timelines changed accordingly (the "fast progressor" moved from 5.0 to 13.4 years), which
  is the clinically plausible rate for CKD G3a. **Any demo script based on the old numbers
  must be updated.**

### Added

- `src/insilico_trial.py` — falsifiable in-silico replication of CREDENCE, DAPA-CKD and
  EMPA-KIDNEY. Parameters are fitted on CREDENCE (with EMPA-KIDNEY anchoring the saturation
  ceiling), frozen, and **DAPA-CKD is predicted out-of-sample**: its chronic eGFR slope
  (2.10 vs published 2.26, 95% CI 1.88–2.64) and its UACR reduction (31.0% vs 35.1%, 95% CI
  30.6–39.4) both land inside the published intervals. Writes
  `results/insilico_trial_report.md`.
- `src/i18n.py` — single source of truth for UI strings (English/Spanish) and the example
  patients, shared by the app and the notebook so the two cannot drift apart.

### Fixed

- **One model, one integrator.** The app, the calibration and the validation now all route
  through `model_core.simulate_trajectory_v2`. Previously the app and the MIMIC path used
  different model structures, and a second copy of the v2 equations lived in its own module
  — exactly the "silently diverging integrators" failure mode this project had already been
  bitten by once.
- **MIMIC calibration no longer depends on imputed albuminuria.** Because UACR is now
  generated by the model, only *baseline* covariates are fed in. This removes the
  dependence on per-visit UACR, which was imputed for the majority of patients and was the
  least identifiable input in the whole pipeline. It also makes the forecast baseline-anchored
  by construction, closing off temporal leakage.

### Removed (release cleanup)

Exploratory modules that nothing imported and that only obscured the shipped application:
`amortized_ai.py`, `bayesian_model.py`, `hierarchical_model.py`, `hybrid_twin.py`,
`inverse_fit.py`, `noise_identifiability.py`, `real_data_validity.py`,
`forecast_comparison.py`, `system_twin.py`, and the v1 model class `mechanistic_twin.py`
(superseded by `model_core`). `docs/KNOWN_ISSUES.md` was folded into the README's
**Limitations** section so there is one place to look.

## Round 6

### Added
- **Three explicit evaluation modes, no longer conflated.** The holdout evaluation previously used each test patient's FULL observed covariate
  history (via the dynamic per-visit matching) even to "predict" early visits -- this measures dynamic RECONSTRUCTION given known exposure
  history, not a genuine forecast, and is not comparable to a baseline model like KFRE (which only ever sees age/sex/eGFR/UACR at one index
  date). Fixed by explicitly splitting into:
  - **Mode A (dynamic reconstruction)** -- the existing holdout, now labeled `holdout_dynamic_reconstruction` and printed as "NOT comparable to KFRE".
  - **Mode B (baseline forecast)** -- NEW: `evaluate_baseline_forecast` uses ONLY each patient's baseline (index-date) covariates, held CONSTANT, to predict eGFR at fixed horizons (2, 5 years) via the constant-insult engine. This is the metric to compare against KFRE. Only evaluated on a `filter_kfre_comparable` cohort -- patients whose baseline HbA1c AND UACR were REALLY observed (see below), not imputed.
  - **Mode C (landmark updating)** -- explicitly deferred; see `KNOWN_ISSUES.md`.
- **Strict, backward-only baseline observation.** `mimic_loader.py` now computes `{hba1c,uacr,sbp}_baseline_observed` (boolean) and `{hba1c,uacr,sbp}_baseline_strict` (the actual value), both via backward-only matching (a real measurement at or before the index date, never after -- no forward tolerance at all), vectorized with `merge_asof`'s `by=` grouping. This is stricter than the existing
  baseline-window tier (Round 3), which allows a +14 day forward grace period defensible for retrospective reconstruction but not for a prospective forecast. `evaluate_baseline_forecast` was found to still use `hba1c_series[0]` (from the more permissive tiers) even after the strict flag was added for cohort filtering -- fixed to use `hba1c_baseline_strict` etc. directly, closing a residual leak where the eligibility check was strict but the value used wasn't. 
- **`quality_status` now considers holdout performance**, not just training-fit metrics: flags `holdout_much_worse_than_training` (possible overfitting), `high_holdout_chi2`, `no_baseline_forecast_evaluation` (no KFRE-comparable patients in the held-out set), and per-horizon `poor_baseline_forecast_accuracy_year_{h}`.

### Changed
- **Bootstrap interval renamed and its statistics corrected.** The app
  called its bootstrap-based band a "90% prediction interval" -- renamed to
  "90% bootstrap **parameter**-uncertainty interval/band" throughout, since
  it captures only calibrated-parameter uncertainty (from resampling
  patients), not residual variability, measurement error, individual
  random effects, unknown future covariate evolution, or structural model
  error (see `KNOWN_ISSUES.md`). Also fixed a statistical issue: the
  time-to-eGFR<15 interval previously dropped non-crossing (infinite-time)
  bootstrap trajectories before computing percentiles, which could show a
  falsely precise-looking interval even when most resamples never crossed
  the threshold. Now reports the fraction of resamples that DO reach the
  threshold within the horizon, and only shows a numeric interval when a
  majority (>=50%) actually cross it; otherwise states the result is
  better read as ">N years" for most resamples.

## Round 5

### Added
- **Dynamic (time-varying) covariates.** HbA1c/UACR/SBP were a single
  static value per patient (from the Round 3 baseline-window fix), applied
  across their whole trajectory. `mimic_loader.py` now attaches covariates
  in three tiers per row: (1) the measurement nearest to THAT visit's own
  date, within +/-60 days (genuinely time-varying); (2) falling back to the
  patient's baseline value (Round 3's index-date window) where no nearby
  measurement exists; (3) population-median imputation, unchanged. A new
  `model_core.simulate_trajectory_dynamic` integrates the ODE with a
  linearly-interpolated, time-varying insult instead of a constant one; the
  app's forward "what-if these labs stay the same" projection is
  unaffected (still uses the constant-insult engine, which is the correct
  framing for a forward scenario). New regression tests verify (a) the
  dynamic engine collapses to the same trajectory as the app's constant
  engine when covariates are literally constant, and (b) it correctly
  responds to a patient's HbA1c rising partway through follow-up.
  Performance note: `attach_dynamic_per_visit` was first written with an
  O(n_patients × n_measurements) per-patient table scan; fixed to
  pre-group once (O(n log n)) before the per-row loop, since the naive
  version would not have scaled to a 25k-patient cohort.
- **Bootstrap prediction intervals in the app.** `calibrate_mimic.py` now
  runs a patient-level bootstrap (`--n-bootstrap`, default 15) after the
  primary fit: resample patients with replacement, refit (1 cheap
  multistart, seeded at the point estimate) on each resample, and save the
  resulting parameter sets as `bootstrap_params` in the calibration JSON.
  This runs once, offline. `app_web.py` re-simulates (fast -- no fitting)
  a patient's projection under each bootstrap parameter set and displays a
  90% interval (shaded band on the trajectory plot, and a numeric range
  for "time to eGFR<15") instead of a bare point estimate, whenever
  `bootstrap_params` is present; falls back cleanly to a point estimate
  otherwise. Verified with a constructed cohort with genuine per-patient
  heterogeneity (bootstrap correctly shows a wide, non-degenerate interval)
  versus a homogeneous test cohort (correctly shows a tight interval,
  since there is no real between-patient variation to capture in that
  case) -- confirming the mechanism responds to actual data properties
  rather than always reporting a fixed-width interval regardless of
  content.

## Round 4

### Fixed
- **Imputed values used in the primary analysis with the same weight as
  observed ones.** `calibrate_mimic.py` now splits the cohort into a
  **primary analysis** (only patients with OBSERVED, non-imputed HbA1c AND
  UACR) and a **sensitivity analysis** (the full cohort, imputation
  included), fit separately. The primary result is what the app uses and
  what `quality_status` is computed from; the sensitivity result is
  reported alongside for comparison, never silently blended in. If fewer
  than 30 patients have both covariates observed, the code falls back to
  the full cohort as primary and flags this explicitly
  (`quality_reasons: ["primary_cohort_too_small_used_full_imputed_cohort"]`)
  rather than fitting 5 parameters on a handful of patients. Opt out with
  `--include-imputed` for quick iteration. New regression tests:
  `test_primary_sensitivity_split`, `test_primary_sensitivity_fallback_when_too_small`.

## Round 3

### Fixed
- **Temporal leakage in covariates.** HbA1c/UACR/SBP were attached to a
  patient's entire trajectory as a single whole-trajectory median --
  including measurements from years after an earlier eGFR observation
  being "explained." Example: a patient with HbA1c=7.0 at their 2014 index
  date, 8.2 in 2017, and 10.0 in 2020 had their 2014 eGFR explained using
  HbA1c=8.2 (the median), leaking three years of future information into
  the earliest prediction. Fixed with a **baseline covariate model**: each
  patient's index date is their first eGFR observation; HbA1c/UACR/SBP are
  attached from the nearest measurement within a defined window relative
  to that index date only (-90/+14 days for HbA1c and SBP, -180/+14 days
  for the sparser UACR). Patients without a measurement inside the window
  fall back to the existing population-median imputation (unchanged,
  still flagged via `*_imputed`). Verified with a constructed reproduction
  of the leakage example, now a permanent regression test
  (`test_no_temporal_leakage_in_covariates`), plus a full end-to-end
  calibration re-run (`quality_status="pass"`).

## Round 2

### Fixed
- **[Critical] Duplicated simulator.** `calibrate_mimic.py` had its own
  explicit fixed-step RK4 integrator; the app's `MechanisticRenalModel`
  used `solve_ivp`. Measured divergence: up to ~11 mL/min/1.73m² near the
  terminal collapse region -- exactly where time-to-eGFR<15 decisions are
  made. Fixed by creating `src/model_core.py` as the single source of
  truth; both the app and the calibration script now delegate to it. New
  regression test: `test_calibrator_and_app_produce_same_trajectory`
  (compares full trajectories, not just an instantaneous hazard value).
- **`gfr_category()` duplicated** between the app and its own test.
  Centralized into `model_core.py`; the test now imports the real function.
- **Observations were unweighted by patient**, letting heavily-monitored
  patients (e.g. ICU stays with labs drawn daily) dominate the fit. Fixed:
  residuals scaled by `1/sqrt(n_i)` per patient during fitting; reported
  RMSE/chi2 remain unweighted (interpretable, original units).
- **No missingness reporting.** `calibrate_mimic.py` now reports the
  fraction of patients with a fully-imputed covariate, and warns if UACR
  imputation exceeds 50%.
- **No train/test split.** Patients are now split 70/30 (fixed seed); the
  held-out set is never used to choose parameters or filters, only to
  report `holdout` metrics.
- **The app trusted any MIMIC calibration unconditionally.**
  `calibrate_mimic.py` now writes `quality_status`/`quality_reasons` to
  the JSON; `app_web.py` shows a visible error banner instead of silently
  presenting an unreliable calibration as trustworthy.
- **`--chronic-only` under-labeled** as a general-purpose filter. Clarified
  everywhere that it is a SECONDARY, outcome-selected sanity check, not a
  valid primary cohort for predictive comparisons (e.g. vs. KFRE).
- **Editorial:** `<your-username>` placeholders replaced with the real
  GitHub username; unit-test count kept in sync in the README.

## Round 1

### Fixed
- **[Critical] App/calibration parameterization mismatch.** The app
  monkeypatched `metabolic_insult()` with already-scaled calibrated
  weights, but `MechanisticRenalModel` still defaulted to `N_ref=0.60` and
  `k_met=0.036`, silently double-scaling the metabolic insult. On a sample
  patient this changed the modeled time to eGFR<15 from 14.5 years (buggy)
  to 5.0/6.3 years (correct). Fixed with explicit `w_a1c`/`w_uacr`/`w_sbp`
  parameters that force `N_ref=1`, `k_met=1`; the fragile monkeypatch was
  removed.
- **N could exceed its documented (0,1] range.** Clipped.
- **KDIGO category error**: missing G3a/G3b split. Fixed; relabeled
  "Approximate KDIGO stage" -> "KDIGO GFR category."
- **"Time to dialysis" mislabeling.** Relabeled to "modeled time to eGFR<15
  threshold" with an explicit disclaimer.
- **Treatment framing** relabeled as an explicit "illustrative" scenario.
- **Demonstration-mode banner** added for the public/synthetic calibration.
- **MIMIC-IV ICD-9 type-2 filter** fixed to check the actual type digit
  instead of matching any `250*` code (which includes type 1).
- **No lab unit validation.** `mimic_loader.py` now reads `valueuom` and
  drops measurements in unexpected units.
- **ZeroDivisionError on creatinine=0.0** (a real value found in MIMIC-IV).
  Fixed with a numerical floor in the CKD-EPI equations and a
  physiological plausibility filter on lab values.
- **`labevents.csv.gz` read 4x** (once per analyte, re-decompressing a
  multi-GB file each time). Fixed to a single pass.
