# Changelog

Notable fixes and changes to NephroQ, driven by several rounds of detailed code review. For currently open limitations, see the **Limitations** section of the [README](../README.md).

## Round 17 — the collapse exponent is a property of progressing CKD, not of the whole cohort

### Context

Round 16 left one question open: is the steep collapse exponent (`q ≈ 2.9`) seen under
`--chronic-only` a real feature of the population, or an artifact of selecting on the
outcome? `--chronic-only` keeps patients whose eGFR is net-declining, which is selection
on the dependent variable, and a reviewer will rightly ask whether the whole finding is
built on that filter. This round runs the controlled experiment: the same MIMIC-IV cohort,
calibrated with and without the filter, weights free (not anchored), everything else held
fixed.

### What was run

Two MIMIC-pure calibrations from the same kept cohort (`_mimic_cohort.tsv`), differing only
in `--chronic-only`. Weights were fitted, not anchored, so this isolates the effect of the
filter itself rather than confounding it with the trial anchor.

```
python calibrate_mimic.py --from-cohort ../data/_mimic_cohort.tsv \
    --n-jobs 20 --q-max 8 --cv-folds 5 --n-bootstrap 10           # full cohort
python calibrate_mimic.py --from-cohort ../data/_mimic_cohort.tsv \
    --chronic-only --n-jobs 20 --q-max 8 --cv-folds 5 --n-bootstrap 10   # progressors
```

### Result

| | Full cohort | `--chronic-only` (progressors) |
|---|---|---|
| n patients | 6395 | 2259 |
| **q (fit)** | **0.90** | **3.30** |
| q, k-fold CV | 11% | 1% |
| q pinned at bound? | no | no |
| k_hf | 0.0103 | 0.0015 |
| chi²/n | 8.06 | 3.03 |
| RMSE (mL/min) | 24.3 | 14.9 |
| quality gate | **warning** | pass |

Both fits identify `q` (neither pins to a bound; CV is 11% and 1% respectively), but they
converge to opposite regimes:

- `q ≈ 0.90` is **sub-linear** decline — progression that *decelerates* as nephrons are
  lost. This is what dominates a diabetic cohort that is mostly stable or slowly declining.
- `q ≈ 3.30` is **super-linear** decline — the accelerating terminal collapse of CKD that
  is actually progressing.

The quality gate tells the rest of the story. On the full cohort the single-regime model
fits **badly** (chi²/n = 8.06, three quality warnings including poor 5-year forecast
accuracy). On the progressor subset it fits cleanly (chi²/n = 3.03, pass). A one-`q` hazard
cannot describe a mixture of two regimes; forced to, it splits the difference and fits
everyone poorly.

### Interpretation

`--chronic-only` does **not** manufacture the steep exponent through selection bias. It
selects the population in which the collapse exponent is *defined*. `q` characterises
terminal acceleration; for a patient whose kidney function is flat, there is no
acceleration to measure, and averaging such patients in does not yield a less-biased `q` —
it yields a different, lower-quality fit of a single regime to a two-regime population.

This is the same logic by which KFRE is calibrated on established CKD rather than on the
general population. The model's domain of validity is progressing CKD. The full-cohort
chi²/n = 8.06 is not a failure to be hidden; it **quantifies the cost** of applying the
model outside that domain, and is itself a reportable number.

The honest caveat, which must be stated and not buried: `--chronic-only` is selection on
the outcome. The defence is one of framing, not denial — the tool is for patients who are
progressing, and it is calibrated on that population deliberately.

### Corrects a claim from earlier in the session

An intermediate run with `--anchor-weights` (weights fixed at the trial values) had shown
`q` un-identifiable on the full cohort (CV 26%, one fold at the bound), and it was tempting
to conclude that `q` is simply not estimable in non-progressors. **That was an artifact of
the anchor, not of the population.** With the weights freed (this round), `q` on the full
cohort is identified (CV 11%, no fold at a bound) — it just lands in the sub-linear regime.
The degeneracy was `q`↔weights, not `q`↔filter. The finding (two regimes) stands; the
mechanism was misattributed and is corrected here.

### Not over-claimed

The sensitivity fit on the *imputed* full cohort returned `q = 1.52`, numerically identical
to the trial-anchored value. This is intriguing — it would fit a picture in which the trial
populations (broad-inclusion, established CKD) sit at the population-average exponent while
the progressor and stable subgroups are its extremes — but it is **not reported as a
finding.** That cohort has ~63% of UACR imputed, which injects an artificial albuminuria
signal that could pull `q` toward the trial value by construction. Distinguishing a real
convergence from an imputation artifact would need a cohort with well-measured UACR, which
is a separate study. It is recorded here as a hypothesis, not a result.

### Also confirmed (unchanged, by a different route)

In both MIMIC-pure fits a metabolic weight pins to its bound — `w_uacr` on the full cohort
(CV 94%, 2/5 folds) and `w_sbp` on the progressor subset (CV 78%, 3/5 folds). This is the
same non-identifiability of the metabolic weights seen before, and it reinforces the
by-domain calibration policy: those weights belong to the trials, not to MIMIC.

## Round 16 — separating the albuminuric term so one hazard fits two populations

### Context

Rounds 10-15 established a robust, uncomfortable finding: MIMIC (the target IMSS-like
population -- type 2 diabetes, near-normoalbuminuric, median UACR ~23) and the trial placebo
arms (macroalbuminuric, UACR ~927) will not calibrate to a common `q`. MIMIC wants a late,
abrupt structural collapse (`q ~= 2.9`, identified at k-fold CV 3%); the trials want early
progression (`q = 1.52`). A single-regime hazard cannot put its collapse "step" in two places
at once, and the Round 15 hybrid confirmed it: anchoring the weights to the trials sent `q` to
its ceiling and raised the cost. These are two diseases of different shape.

The product is for the IMSS-like population, which is mostly normoalbuminuric but also includes
the sickest patients, who arrive already albuminuric. A single-regime model underestimates
those patients dangerously. So the model must carry BOTH regimes in one hazard and let the
patient's own UACR decide which dominates.

### Changed -- `model_core.py`

The UACR term was pulled out of `metabolic_hazard` (which is now A1c + SBP only) into its own
coexisting term, `albuminuria_hazard(uacr, k_alb) = k_alb * log1p(UACR/30)`:

```
h = k0
  + k_hf  * s^q / (1 + (s/S_SAT)^q)      structural  (hyperfiltration)
  + k_alb * log1p(UACR(t)/30)            albuminuric  (NEW: separate coefficient)
  + w_a1c*(A1c-6.5) + w_sbp*(SBP-130)/10 metabolic residual (no UACR)
```

The log form is unchanged -- it is the one already validated in-silico against the three
trials (linear over-weights very high UACR; a threshold contradicts microalbuminuria
predicting progression). The two insult terms keep the same `(1 - eff_met*u)` drug multiplier
the old combined block had, and albuminuria's endogenous `eff_alb` reduction is untouched, so
the drug response is identical.

The point of the separation is calibration by domain. `k_hf, q` are pinned on MIMIC (where the
structural signal is); `k_alb` is pinned on the trials (where albuminuria is observable),
NEVER on MIMIC. `k_alb` was added to `TRIAL_CALIBRATION_V2` (same value the trial-fitted
`w_uacr` had, `0.0180*0.730`). Because MIMIC is near-normoalbuminuric, `k_alb` has large
leverage at high UACR and almost none at low UACR (a measured ~5x endpoint-shift ratio between
UACR 927 and UACR 23 over 5 years), which is exactly what lets one parameter set serve both
populations.

### Backward compatibility -- migration chosen over a zero default

`w_uacr` is threaded through the entire calibration/audit/app/personalize pipeline and dozens
of tests. Rather than rename it everywhere (large, risky) or default `k_alb` to 0 (which would
silently disconnect the coefficient `calibrate_mimic` still fits from the term it is supposed
to control), the albuminuric coefficient is read via `_k_alb_of(p)`: use `k_alb` if present,
else fall back to `w_uacr`. So every pre-Round-16 parameter dict -- and every `calibrate_mimic`
fit, and every anchored run -- drives the hazard exactly as before. Under `--anchor-weights`,
fixing `w_uacr` to the trial value now IS anchoring `k_alb` to the trials: the hybrid run from
Round 15 already produces the dual-domain parameter set (MIMIC `q, k_hf` + trial `k_alb`), the
separation just makes the semantics honest. `--index-strategy`, `--anchor-weights` and the
conformal personalizer were not touched.

### Verified by running (not by reading)

- Default behaviour is BIT-IDENTICAL: `renal_hazard_v2` recomputed the old way (UACR inside
  the metabolic block) matches the new split to |Δ| = 0 across eGFR, UACR, and treated/placebo.
- Migration is exact: a pre-R16 dict (only `w_uacr`) simulates identically to one with
  `k_alb` set to that value (max |Δ eGFR| = 0).
- `batched == loop` still holds to < 1e-4 (the invariant the review demanded), including
  replicates that set `k_alb` explicitly and ones that rely on migration.
- In-silico trial replication still passes 3/3 (CREDENCE 0.99x, DAPA-CKD held-out 0.94x,
  EMPA-KIDNEY 0.99x).
- Suite: 93 tests green (86 + 7 new Round 16 regression tests covering the split, the
  migration, the independence of `k_alb` from `w_uacr` once set, the leverage asymmetry, and
  the batched/loop agreement).

### Falsifiable DUAL test -- offline result, and what remains for `fx-gpu`

The dual criterion: one parameter set must reproduce (a) MIMIC's late-abrupt collapse
(chi2/n ~ 3) AND (b) all three placebo arms (ratios in [0.8, 1.25]).

An offline feasibility scan (structural `q, k_hf` fixed at the MIMIC values, `w_a1c, w_sbp` at
the trials, sweeping `k_alb`) shows (b) is achievable with the SAME `k_alb` that leaves the
near-normoalbuminuric regime almost unmoved:

- with the free-MIMIC structural fit (`q=2.923, k_hf=0.00277`): a `k_alb ~ 0.007-0.013`
  window passes all three arms; the synthetic UACR~23 slope moves only ~6% across it.
- with the anchored structural fit (`q~3.97, k_hf~0.0005`): the trial `k_alb = 0.01314`
  itself passes all three arms (worst 1.07x).

So the two-term model can, in principle, reconcile the populations -- the leverage asymmetry is
real. This scan uses a simplified slope measure with drug effects off, so it is indicative, not
the verdict. Side (a) -- the MIMIC chi2/n -- and the exact audit ratios must come from the
`fx-gpu` run:

```
python calibrate_mimic.py --from-cohort ../data/_mimic_cohort.tsv --chronic-only \
    --n-jobs 20 --anchor-weights --cv-folds 5 --n-bootstrap 20 --q-max 8
python audit_calibration.py
```

If (a) and (b) both hold, the dual-regime model reconciles the two populations. If it improves
the placebos but breaks MIMIC (or vice versa), that is reported as-is: which term is not
enough, and why. No dependency on SMOTE, deep learning, SHAP or Optuna was introduced.

## Round 15 — hybrid calibration: estimate each parameter from the data that contain it

### Context

The MIMIC-IV fit (2259 patients, `--chronic-only`) produced `q = 2.923`, `k_hf = 0.00277`,
`quality = pass`, but `audit_calibration.py` rejected it: the parameters underestimate the
published placebo-arm slopes of CREDENCE (0.68×) and DAPA-CKD (0.79×), while nearly matching
EMPA-KIDNEY (0.95×). The ratio tracks baseline eGFR — worst where eGFR is highest.

Root cause is not the collapse exponent. k-fold CV shows `q` (CV 3%) and `k_hf` (CV 13%) are
identified and stable, and `q = 2.92` survives `--q-max 8`, so it is a real optimum, not a
censored bound. The weights are the problem: MIMIC's median UACR is ~23 mg/g versus 927 in
CREDENCE, so the cohort is essentially normoalbuminuric. There is no albuminuria signal to
learn, the insult weights are fitted on noise (CV: `w_a1c` 20%, `w_sbp` 54%), and the model
attributes ~87% of the hazard to hyperfiltration — which is exactly why it cannot reproduce
the albuminuria-driven progression of a high-eGFR CREDENCE patient.

### Added

- **`--anchor-weights`: hybrid calibration.** Each parameter is now estimable from the data
  that actually contain it: `q, k_hf` from MIMIC (identified here), and `w_a1c, w_uacr, w_sbp`
  from the trials (`model_core.TRIAL_CALIBRATION_V2`, whose weights are pinned by the CREDENCE
  and EMPA-KIDNEY placebo arms). The weights are fixed **before** fitting — the free vector
  shrinks from 5 to 2 and `(q, k_hf)` are re-optimised under that constraint. This is *not*
  the same as fitting five parameters and overwriting three afterwards: in the overwrite,
  `q, k_hf` stay optimised for the weights they were fitted alongside, leaving a fit that is
  optimal for no parameter set at all. A regression test asserts the two produce a different
  `(q, k_hf)`, so the constraint is demonstrably not inert. A side benefit is that the
  `q ↔ k_hf ↔ weights` degeneracy cannot arise with the weights held fixed.

  The flag propagates through `calibrate`, `cross_validate`, `bootstrap_calibrate` and the
  sensitivity fit. The k-fold identifiability summary scores only the free parameters — a
  trial-anchored weight is a constant, and reporting `CV = 0.00` for it would manufacture the
  precise "looks perfectly stable, is actually no information" artifact Round 10 added the
  check to catch. The bootstrap band correspondingly has zero width in the weight directions
  (they were not estimated), and the JSON records provenance (`anchor_weights`,
  `free_parameters`, `anchored_parameters`, `anchored_weight_source`) so a reader can tell a
  fitted `w_uacr` from an assumed one.

  Implemented via the existing `base_params=` hook (added for the S_SAT profile) plus a
  free-index restriction of `unpack`/`pack`. The ordinary 5-parameter fit is bit-for-bit
  unchanged (`free_idx=None` path), locked by a test.

### Fixed

- **`audit_calibration.py` check [2] gave backwards advice on an underestimate.** When MIMIC
  declined *slower* than the trials (ratio < 0.8), the auditor blamed `--chronic-only` for
  selecting "stable patients". That is reversed: `--chronic-only` selects **declining**
  trajectories, so it biases toward progressors and would push the ratio *up*, not down. It
  sent the user to look in the wrong place. The message now states the real cause — too little
  signal in the insult covariates (near-normoalbuminuric UACR) — and the real fix
  (`--anchor-weights`), not a change of cohort filter. Regression test added.

### Falsifiable success criterion (run on `fx-gpu`, not yet executed here)

```
python calibrate_mimic.py --from-cohort ../data/_mimic_cohort.tsv \
    --chronic-only --n-jobs 20 --anchor-weights --cv-folds 5 --n-bootstrap 200
python audit_calibration.py
```

- If the hybrid passes all three placebo arms → defensible population calibration; ship it.
- If it still fails CREDENCE (the high-eGFR arm) → the weights were not the whole story and
  `q = 2.92` is incompatible with early trial progression. The honest conclusion is then that
  these are **two populations needing two separate calibrations**, not one hybrid — report it
  that way rather than forcing the fit. Both outcomes are publishable.

  An a-priori feasibility scan (weights fixed at trial values, `S_SAT = 3.5`) shows the hybrid
  family *can* pass in principle: at `q = 2.92` every placebo arm lands within 30% for
  `k_hf ≈ 0.0015–0.0033`. Whether MIMIC's own `k_hf` falls in that band is the empirical
  question the run above settles; the scan only establishes the target is not empty.

### Not adopted this round (recorded for a later one)

An external statistical review (this round's) additionally proposed: a one-parameter
susceptibility-scale reduced model (`s_MIMIC · H_structural`); a joint/regularised objective
`L_MIMIC + λ·L_trials`; splitting `quality_status` into internal / stability / external-audit
levels with a `deployment.approved` gate the app checks instead of `quality == pass`; renaming
"primary analysis within the chronic-only cohort" to "complete-case analysis within the
chronic-trajectory sensitivity cohort"; a baseline-covariate comparison table (observed vs
missing UACR) to test biomarker-availability selection; and reporting the KFRE benchmark with
confidence intervals given its 2–5 events. Each is sound and each is its own piece of work with
its own falsifiable check; none is folded in silently here. No new ML dependency was
introduced (SMOTE, deep learning, SHAP, Optuna remain rejected — see earlier rounds).

## Round 14 — vectorisation where it actually costs, and two coefficients that were wrong

### Fixed

- **A code review proposed CKD-EPI equations with two WRONG coefficients.** Both errors are
  the kind that survive a glance, and adopting them would have silently changed every eGFR the
  app shows a clinician:
  - `eGFRcys` with a leading **135** and age base **0.9946** — those belong to the
    creatinine-cystatin equation. eGFRcys is **133** and **0.996**.
  - `eGFRcr-cys` with a male α of **−0.291** — the published value is **−0.144**. (−0.302 is
    the male α of the *creatinine-only* equation, which is presumably where −0.291 drifted in
    from.)

  Verified against NKF, NIDDK and Inker et al. NEJM 2021/2012. **The existing implementation
  was correct and was NOT changed.** A test now pins the exact published coefficients, so a
  well-meaning "refactor" cannot move them.

### Added

- **Vectorised the CKD-EPI equations** (the review's other, valid point): they now accept whole
  cohort columns instead of raising the classic "truth value of an array is ambiguous". Honest
  accounting: ~20× faster per row, but that is **2.1 s → 0.11 s** across 1.5M MIMIC
  creatinines. It is a correctness and ergonomics win, not a performance one.

- **Batched the bootstrap ODEs — the vectorisation that *does* pay.** The review proposed
  vectorising `P_NQ = (egfr < 15).mean(axis=1)`. Measured: that operation is **0.0042 %** of
  the cost. The expensive part is *generating* the projections — B separate `solve_ivp` calls
  per patient, whose per-call overhead dominates a 1-D problem. Those B replicates are
  independent trajectories, so they stack into a single B-dimensional system:
  `predict_egfr_at_v2_batched` does one solve instead of B. **66.5 ms → 3.47 ms (≈19×)**,
  agreeing with the per-replicate loop to 1e-4 mL/min, locked by a test.

  The reviewer's instinct — *vectorise* — was right. It was aimed at the wrong line.

### Not adopted (and why)

- **An ML classifier (RF/XGBoost) to detect AKI from ICD codes.** The proposal is to train a
  model to predict labels (`N17.x`, `584.x`) that *we already have*. If the ICD codes are
  trustworthy enough to be training labels, use them directly as a filter; if they are not,
  a classifier trained on them inherits the same unreliability. The genuinely useful ideas in
  that section are the **cleaner index date** and the **stability weighting**, neither of
  which needs machine learning — and those remain the top open item.
- **MICE / VAE imputation of covariates.** In v2 the per-visit UACR is no longer used at all
  (albuminuria is an *output*), which is what the 63 %-imputed figure was really about. Only
  the baseline value now matters, and the primary analysis already restricts to patients where
  it is observed. Also: imputing UACR from eGFR and then using UACR to predict the eGFR
  trajectory is close to circular.
- **Optuna / contextual bandits to tune `f_scale` and multistarts.** `f_scale` is set from the
  robust MAD of the residuals — a principled choice. Tuning it to minimise CV RMSE would push
  it toward down-weighting real signal as if it were outliers, which is precisely the failure
  mode robust loss exists to avoid.
- **JAX autodiff, global optimisers (basinhopping / differential_evolution).** With 5
  parameters, a finite-difference Jacobian costs 6 residual evaluations; after Round 9's ~7×
  speedup and `--n-jobs`, that is not the constraint. Worth revisiting only if the parameter
  count grows (e.g. if an explicit AKI state is added).
- **Full Bayesian (PyMC / NUTS).** Attractive, and it would express `q`'s unidentifiability
  honestly as a prior-dominated posterior. But we already *know* that from the k-fold check
  (Round 10), obtained without a heavy new dependency. This is a real future direction, not a
  fix.

## Round 13 — closing the file-by-file review

### Fixed

- **The auditor never checked albuminuria.** `audit_calibration.py` is the gate that
  decides whether a MIMIC calibration may be shipped, but it only tested the placebo-arm
  slopes — while `insilico_trial.py` had been testing the UACR endpoint all along. The
  auditor now checks both.

  Honest caveat, found by testing it: on a deliberately AKI-inflated calibration the UACR
  check **passes** while the placebo slopes fail (1.7–1.8×). The UACR reduction is dominated
  by `eff_alb` and is largely insensitive to the progression parameters, so it is a **weak
  second gate**, not an independent confirmation. Recorded in the code so it cannot be
  mistaken for one.

- **Failed bootstrap replicates were printed and forgotten.** If 12 of 15 failed, the JSON
  silently carried a three-replicate "uncertainty band" and no reader could tell. The counts
  (`n_requested`, `n_successful`, `n_failed`, and the first failures) are now written to
  `bootstrap_diagnostics`, and a warning is printed.

- **`mvp_calibration.py` fitted with a non-robust objective.** `calibrate_mimic.py` uses
  `soft_l1` with a data-driven `f_scale`; the own-data path used plain least squares, so a
  handful of AKI spikes could steer the whole fit. It now uses the same robust loss, with
  `f_scale` set from the cohort's own residual spread (robust MAD), not a hard-coded constant.

- **The "bootstrap degenerate" warning was written for modellers, not clinicians** (and
  ended in a dangling `('optimizer scaling')` fragment). Rewritten in both languages to say
  plainly what it means: the fit never moved, so its numbers carry no information — re-run it.

### Added

- **Seed-sensitivity test for the in-silico replication.** The virtual cohorts are random
  draws; if DAPA-CKD only landed inside its published CI for a lucky seed, the PASS would be
  noise. Across seeds the held-out chronic slope difference is **2.22 ± 0.07** (published CI
  1.88–2.64) and the UACR reduction **31.1 ± 0.1** (CI 30.6–39.4) — **6/6 seeds pass**, and a
  test now fails if that stops being true. Note the UACR prediction sits consistently near the
  *lower edge* of its interval; it passes, but it is not centred.

- **`pyproject.toml`** (pytest `pythonpath = ["src"]`, so tests no longer need `sys.path`
  surgery) and **`requirements-lock.txt`** with the exact versions the results were produced
  with. `requirements.txt` keeps lower bounds for easy installation; the lock file is what a
  manuscript should cite.

- **The hazard cap is documented and named** (`HAZARD_CAP = 50.0`). It is a numerical guard so
  the integrator cannot blow up while an optimizer explores an absurd corner of parameter
  space — at that rate a nephron population halves every ~5 days. It must never bind for a
  plausible patient; if it does, the parameters are wrong, not the patient.

### Not adopted

- **`logging` instead of `print`** — the calibration's `print` output *is* the diagnostic
  record a reviewer reads, and it is a research tool, not a service. Worth revisiting if this
  is ever deployed.
- **Type hints throughout, a central `config.py`** — readability, not correctness. Low return
  next to what is still missing (an AKI-free index date).
- **Narrowing the two `except Exception: pass` blocks** — the silent failure is deliberate
  there: personalization must never take down the app. The reviewer's point stands in general,
  though, and these should become specific exceptions with a logged warning once logging exists.

## Round 12 — honest uncertainty, and one model in the file

### Fixed

- **The ensemble spread was being sold as uncertainty, and it was ~7× too narrow.**
  `personalize.py` claimed "their disagreement IS the uncertainty". Measured on held-out
  virtual patients, a nominal 90% band built from that spread covered the truth **32.8%** of
  the time for `q` and **41.5%** for the injury rate. Quoting it as "±" was false precision.

  The spread is now **conformalized** (split-conformal, normalized nonconformity): on a
  held-out calibration split the ratio |θ_true − θ̂| / spread is computed and its 90th
  percentile is stored. The required inflation turned out to be **×7.29** (`q`) and **×6.47**
  (injury rate) — a measure of just how over-confident the raw spread was. Measured coverage
  of the calibrated interval: **89.5%** and **91.2%** against a nominal 90%. The raw spread is
  still exposed, but as `q_spread` — never as an interval. A test now fails if coverage drifts.

- **The app claimed the parameters came from "hierarchical Bayesian inference on synthetic
  data".** That module has not been part of the tree for several rounds; the active parameters
  are anchored to published trial data. The claim was false, and it appeared in the app's own
  "About the model" panel (EN and ES) and in `CITATION.cff`. Corrected.

- **`model_core.py` still carried the entire v1 model as dead code** — the unbounded
  hazard, its integrator, and its predictor — none of it called from anywhere. A reviewer
  opening the central file would find two families of equations and no way to tell which one
  produced the figures. Removed, and locked by a test.

### Notes

This round continues the same external review as Round 11. The three findings fixed there —
the pre-filled fictitious history, historical creatinines converted with the patient's
*current* age, and personalization being silently overwritten under a MIMIC/private
calibration — are covered by tests and are not repeated here.

## Round 11 — three critical bugs found by external review

### Fixed — critical

- **The app personalized every patient from a FICTITIOUS history.** The measurement editor
  shipped pre-filled with example creatinines (`1.05, 1.15, 1.22`). A user who entered only
  today's markers and never typed a single historical value still saw **"Personalized to this
  patient"** — computed from invented data. The editor now starts **empty**, and the example
  history is behind an explicit button labelled *"Example measurement history — NOT patient
  data."*

- **Personalization was silently discarded whenever a MIMIC or private calibration was
  loaded.** `project()` overwrote the personalized `q`/`k_hf`/weights with population values
  for any tier that was not `public`, while the interface went on announcing "Personalized to
  this patient". The inferred injury rate is a **multiplier relative to the population model**,
  so the fix is structural: the personalizer is now parameterized by the *active* population
  calibration, trained against it, and its estimator is cached per tier. An estimator built
  around the trial-anchored model cannot be transplanted onto a MIMIC calibration whose hazard
  is twice as fast.

- **The calibration fitted v2 but EVALUATED with v1.** Not in the review — found while
  checking it. `calibrate_mimic.py` still called the old unbounded `predict_egfr_at` in three
  places: the bootstrap-derived risk, Mode B (baseline forecast) and Mode C (KFRE comparison).
  Parameters were being estimated under one set of dynamics and scored under another. All
  three now use `predict_egfr_at_v2`. **No live path uses v1 any more.**

### Fixed — high

- **Historical creatinines were converted with the patient's CURRENT age.** A sample drawn
  ten years ago was run through CKD-EPI with today's age, which systematically *understates*
  the historical eGFR and makes the decline look flatter than it was. The bias grows with the
  length of the history — precisely the histories that carry the most information. Each value
  is now converted with the age the patient had at the time.

### Changed — honesty of the claims

- **`q` is described as what it is.** The README opened by presenting `q` as the central
  parameter "estimated from clinical trajectories", while the repository's own experiments
  show it is close to unidentifiable from routine data. It is now stated as a
  **population-level structural parameter**, with individual heterogeneity carried by the
  **injury-rate multiplier** — which is the recoverable quantity, and the more interesting
  claim.
- **Ensemble spread is no longer dressed up as a confidence interval.** The app reported
  `q = 1.72 ± 0.13`. That ± is the disagreement between the networks in the ensemble, not an
  interval with known coverage. It is now labelled "ensemble spread", and `q` is marked
  *experimental*.
- **The therapy toggle no longer claims a combined SGLT2i/ACEi-ARB effect.** Both the
  calibration (CREDENCE) and the out-of-sample validation (DAPA-CKD) are SGLT2 inhibitor
  trials. The scenario is now an "illustrative SGLT2i-like intervention".
- **The demo banner no longer says "synthetic calibration".** The public tier has been
  trial-anchored since Round 7; the banner had not caught up.
- **`model_core.py`'s header showed the v1 equation** and stated that "v2 is opt-in" — neither
  was true any more. A reviewer opening the central file would not have known which equation
  produced the figures. The header now carries the v2 equation, and the legacy v1 helpers are
  explicitly marked as being on no live path.

## Round 10 — cross-validation: is the parameter even identifiable?

### Added

- **`--cv-folds`: K-fold cross-validation, split by patient.** The point is *not* a slightly
  better RMSE estimate — it is a **stability / identifiability check on the parameters
  themselves**. If `q` swings from 1.1 to 2.4 depending on which patients land in the
  training set, then `q` is not identifiable from that cohort, and any point estimate of it
  is an artifact of the split.

  **A bootstrap cannot see this.** It resamples the *same* patients, so it measures sampling
  noise around one fit — not whether a genuinely different set of patients would have given a
  different answer. A tight bootstrap interval on an unidentifiable parameter is false
  precision, and this is the check that catches it.

- **Detection of parameters pinned at a bound — the dangerous false green.** Writing the
  check above immediately exposed a flaw in it. On a deliberately uninformative cohort
  (short follow-up, high noise), `q` came back as `[0.50, 0.50, 0.50, 0.50]`: a coefficient
  of variation of **0.00**, which a spread-based check reports as *perfectly stable*. It is
  the opposite. 0.50 is the optimizer's lower bound: the data carry no information, so the
  fit slams into the boundary every time. A parameter sitting on its bound is **degenerate,
  not identified**. Both failure modes — swinging across folds, and pinned at a bound — are
  now flagged, and both are covered by tests.

  Note how the out-of-fold RMSE fails to raise the alarm on its own: 8.80 mL/min on the
  uninformative cohort versus 2.94 on the informative one. Only ~3× worse, entirely
  publishable-looking — while the parameters underneath it are meaningless.

### Not adopted (and why)

A review was received that targets a **different codebase** (a LightGBM binary classifier on
kidney-transplant data: `grado_histologico`, `time_tx`, `inmunosupresion`, `cmv`). None of
those variables exist here, and NephroQ contains no classifier at all. Recorded for the
avoidance of doubt:

- **SMOTE / class balancing** — there are no classes. NephroQ predicts *trajectories* from an
  ODE; synthesising patients by interpolating in feature space would break the mechanistic
  coherence that makes the model falsifiable.
- **Deep learning "to capture complex non-linear interactions"** — this would destroy the
  property that lets the model be *caught being wrong*. The unbounded-hazard error of Round 7
  was found precisely because the parameters are physical. Inside a neural network, it would
  have shown up only as a slightly larger loss.
- **SHAP** — the parameters already have physical meaning; there is nothing opaque to explain.
- **MDRD for eGFR** — a regression. This project uses CKD-EPI 2021 (no race coefficient).
- **Optuna, drift monitoring** — no relevant hyperparameters, and nothing is in production.

What *did* apply from that review — cross-validation, and the warning about data leakage —
is implemented above and in Round 9 respectively.

## Round 9 — calibration speed, a calibration auditor, and one model everywhere

### Fixed — three bugs, one of them capable of killing a running calibration

- **`predict_egfr_at_v2` crashed on same-day lab draws.** The Round 9 speedup (below)
  integrates straight onto the visit times, and `solve_ivp` requires `t_eval` to be
  **strictly increasing**. Real data is not: hospital records routinely contain several
  creatinines drawn on the **same day**, and callers may pass times in any order. The result
  was `ValueError: Values in t_eval are not properly sorted` on perfectly valid patients.
  The predictor now deduplicates and sorts internally, then scatters the results back to the
  requested order.

- **`mvp_calibration.py` was fitting a DIFFERENT model from the one the app projects
  with.** It carried its own fixed-step Euler integrator and the **old unbounded hazard**,
  bypassing `model_core` entirely. On the same patient with the same parameters it drifted by
  up to **13 mL/min at 10 years** (10.9 vs 24.0). Since this is the path the README recommends
  for calibrating on your *own* data, users were fitting one model and projecting with
  another. It now calls `model_core`; the two agree to **0.0000 mL/min**. This is the same
  class of bug as the two diverging integrators fixed in an earlier round, and it is now
  locked down by a test.

- **Temporal leakage in the own-data loader.** Covariates were taken as the **median over
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
