# Changelog

Notable fixes and changes to NephroQ, driven by several rounds of detailed
code review. For currently open limitations, see
[`KNOWN_ISSUES.md`](KNOWN_ISSUES.md).

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
