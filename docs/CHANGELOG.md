# Changelog

Notable fixes and changes to NephroQ, driven by several rounds of detailed code review. For currently open limitations, see the **Limitations** section of the [README](../README.md).

## Round 30 — core cap for shared compute nodes (v0.18.1)

Network training set torch to every core of the machine (`os.cpu_count()`), ignoring
`OMP_NUM_THREADS`, and scikit-learn's OpenMP pools likewise took all cores.
`outputs.n_threads()` resolves the allowed cores (`--threads` > `NEPHROQ_THREADS` > the
process's CPU affinity), and `outputs.limit_threads()` caps torch, BLAS and OpenMP at that
number and exports it to child processes. `validate_mimic_patients.py` and
`benchmark_posterior.py` take `--threads`; NPE training and `calibrate_mimic.py --n-jobs -1`
use the same cap. Tests added.

## Round 29 — code and method audit; Phase B diagnostics (v0.18.0)

A full review of the publication pipeline after the leakage-free MIMIC-IV run.

**Errors fixed**

- **Long histories shifted the time origin** (`npe_posterior.posterior_samples`). Histories
  longer than the network input kept the last 10 visits and re-zeroed time at the first
  kept visit, while `predictive` reads times from the original first visit, so forecasts
  for long histories were shifted by the dropped span and eGFR0 referred to the wrong
  visit. Now the first visit and the most recent nine are kept (`reduce_history`); test
  added. (MIMIC windows have at most nine thinned visits, so the reported results were not
  affected; the app path was.)
- **Training never saw the longest histories**: schedules had 2–9 visits while the network
  accepts 10. Now 2–10.
- **Body-fluid creatinine could enter the cohort** (`mimic_loader`). MIMIC labels ascites,
  pleural and drain creatinine simply "Creatinine" and distinguishes them only in the
  `fluid` column; items were chosen by label. Creatinine, HbA1c and cystatin C items are now
  restricted to `fluid == Blood`, with body-fluid words excluded from labels as a fallback.
  **The cohort must be rebuilt.**
- **Unequal search spaces**: the least-squares comparator searched injury rates 0.25–3×
  while the NPE searched 0.25–6×. `least_squares_scale_only` takes `bounds`; the validation
  passes the NPE box.
- **Random-slope pooling** weighted patients by `1/(se² + var(slopes))` instead of
  `1/(se² + τ²)`; corrected.
- **Noise fit**: the degrees of freedom were clipped to 2.5 after fitting, so the scale was
  fitted for a different df than the one reported. The bound now lives inside the fit, and
  the report says when df sits at the limit.
- **Withdrawn terminology still shown to users**: the app (English and Spanish), the
  personalizer, the NPE and prior modules and two figure scripts still called `q` a
  "collapse exponent" and described a "terminal-collapse regime", contradicting §2.6. All
  replaced ("feedback exponent"; late-stage example rewritten).

**Method changes**

- **Injury-rate box widened to 0.02–6×** (was 0.25–6×). The stable MIMIC component sat
  against the 0.25× floor (Round 28); 0.02× is ageing-only decline (< 0.3 mL/min/1.73 m² per
  year at eGFR 75). Networks retrain on first use; the posterior benchmark must be re-run.
- **Between-visit deviation** (audit P1.6): held-out visits are modelled as latent eGFR +
  measurement noise + N(0, v0 + v1·Δt), Δt = years since the last observation; (v0, v1) by
  maximum likelihood on the training patients' held-out visits (the networks never read
  them), with the Student-t noise as a scale mixture of normals. New method
  "NPE, learned prior + between-visit deviation", now the mechanistic reference. Recovery
  test included.
- **Cross-fitted recalibration** of gradient boosting and hybrid intervals (5-fold
  out-of-fold standardized residuals): their quantile models were scored on their own
  training rows. Coverage test included.
- **New comparators** (audit P1.7): last observed value (persistence, with the training
  change by horizon as predictive distribution) and per-patient Theil–Sen slope.
- **Bootstrap CIs for every metric** (audit P1.8): differences in RMSE, MAE, CRPS, 90 %
  coverage and AUC against both references.
- **Identifiability report** (audit P1.1): posterior contraction of `q` and log rate and the
  `q`–rate correlation on the test patients, for both stages.
- **Mixture stability** (audit P1.2): K chosen among 1, 2, 3; components ordered by injury
  rate (identified) instead of `q`; patient-bootstrap refits with label alignment give
  intervals for weights, mean `q` and median rate; the report warns when a component lies
  within two SDs of a box edge.
- **Selection flow** (audit P1.5): selection table by exclusion reason with the fraction of
  recorded deaths (`dod`), to show informative censoring.
- **Sensitivity runs**: `--noise-scale/--noise-df` fix the noise model; `--tag` names the
  outputs; `--prior-boot` sets the stability refits.
- `grid_posterior` accepts a Student-t likelihood (`noise_df`).

Suite: 194 tests (183 + 11 new).

## Round 28 — MIMIC-IV re-run on the leakage-free protocol (v0.17.1)

The patient-level validation was re-run with v0.17.0 (cut-then-thin, window-only index and
covariates, training-median imputation, simulated-quantile noise fit). Cohort: 9 517 eligible
(3 806 newly excluded because the index could not be confirmed inside the observed window),
8 663 training / 854 test patients, 1 252 held-out visits. Results replace those of v0.16.1 in
`results/mimic_patient_validation.md` and the README.

- **Data-driven methods barely move**, so the leak itself was small: gradient boosting
  14.39 → 14.42, hybrid 14.24 → 14.34, least squares 16.25 → 16.31, MLP 16.42 → 16.60.
- **The mechanistic posterior moves substantially**: retrained NPE 15.34 → 16.64 (bias
  −0.15 → −2.09; coverage 79 % → 67 %). It is now tied with least squares (−1.28 to +0.68) and
  the MLP (−1.02 to +0.96) instead of ahead of them, and behind gradient boosting by +1.42 to
  +3.03. The main driver is the noise estimate: the corrected fit gives scale 2.77 / df 2.5
  (SD 6.19) instead of 5.26 / 3.3 (SD 8.38). The previous, inflated noise widened the
  intervals (hiding under-coverage) and changed the learned prior. Every method whose
  intervals depend on the noise model lost coverage (MLP 70 → 59 %, uniform NPE 80 → 64 %);
  gradient boosting and the hybrid, whose intervals come from residuals, kept 86 %.
- **Learned prior**: K = 2 (held-out log-likelihood 0.197 vs 0.102); 69 % at 0.30× (SD of log
  rate 0.07, against the 0.25× edge of the box) and 31 % at 2.89× (q 1.89); importance
  efficiency 15 % (was 7 %).
- Consequences recorded as open items for Phase B: (i) short-interval noise does not cover
  between-visit variation over years → explicit deviation term or interval calibration by
  horizon; (ii) the lower edge of the injury-rate box is binding → widen it and re-test the
  mixture; (iii) the attribution between leak fix, covariate source and noise estimator was
  not separated → an ablation run with the noise fixed at the v0.16 values.
- Version 0.17.1 (results and documentation only; no code change).

## Round 27 — Phase A of the external audit: blocking items (v0.17.0)

An external audit plan (24 Nov 2026) was adopted as the roadmap. This round closes its blocking
items (P0) plus P1.4 and the first part of P1.6.

- **P0.1 Load convention.** Writing the load as the physical per-unit filtration `N^(α−1)`
  instead of `1/N` is an exact reparametrization (`q' = q/(1−α)`, `S_SAT' = S_SAT^(1−α)`,
  `β' = β/(1−α)`; agreement 1e-15). No recalibration; documented, with the consequence that
  the numerical value of `q` is convention-dependent. Test added.
- **P0.2 Domain.** `N` is documented as an effective functional-capacity index, not a nephron
  count. eGFR above `G_MAX` is truncated to `N = 1`; `model_core.egfr_domain_flags` reports it
  and `personalize()` warns and returns `domain_warning`.
- **P0.3 Dynamics.** Derived `G'' = α² G h² (1 + ε)`, `ε = ∂ln h/∂ln G`: decline accelerates
  iff `ε < −1` (`model_core.hazard_elasticity`, `dynamics_regimes.py`). With the trial
  calibration (`q = 1.52`) the decline **decelerates at every eGFR**; acceleration needs `q`
  above 2.05–2.45 (UACR 30–1500) and occurs only above eGFR ≈ 36–40; below saturation it
  always decelerates; every term is bounded, so there is no collapse, bifurcation or
  finite-time singularity. "Collapse exponent", "near-step", "terminal collapse" and "falls
  steeply near the end" are withdrawn from the README and model documentation. Tests check
  `G''` against finite differences and the deceleration claim.
- **P0.4 Temporal leakage (confirmed and fixed).** `validate_mimic_patients.py` thinned
  before cutting at the forecast origin, so a 90-day bin straddling the origin could put
  post-origin draws into the observed window (reproduced: changing only a post-origin draw
  changed an observed value, 35 → 50). Trajectories are now cut first and thinned per side;
  the index is re-confirmed with observed-window data only (the loader could confirm it with a
  later value); covariates come from the window only, loader-imputed values (cohort-wide
  medians, test patients included) are discarded, and missing values are imputed with
  training medians. Tests rewrite all post-origin data and check invariance. MIMIC-IV results
  must be re-run.
- **P0.5 Simulator configuration.** `fast_sim.simulate_egfr` takes an explicit `params` dict
  and a treatment `u`; `model_core.sim_config`/`config_hash` identify the model; NPE
  checkpoints store the hash and `load()` refuses a mismatch. Equivalence with LSODA is tested
  across three calibrations, with and without treatment.
- **P1.4 Truncated densities.** `PopulationPrior` returns exactly −inf outside the prior box
  (mask before clipping); a test integrates the density to 1.
- **P1.6 (part) Noise fit.** The difference of two Student-t errors is not Student-t; scale
  and degrees of freedom are now fitted by matching quantiles of simulated differences
  (recovers 5.0/3.5 → 5.03/3.64 and 4.0/10 → 4.01/10.06). The explicit acute-deviation state
  remains for Phase B.
- **P1.5 (part) and P0.6/P1.11 (part).** The validation report now has a cohort-selection
  table (included vs excluded) and a provenance block (commit, version, configuration hash,
  cohort hash, weight hashes, package versions). `make_synthetic_cohort.py` writes a public
  synthetic cohort in the loader format (byte-reproducible); a new CI job runs the full
  pipeline on it (`--quick`), plus the dynamics map and in-silico trials, and uploads the
  outputs with `SHA256SUMS`.
- README claims tempered (audit P1.3, P1.7): the hybrid is "statistically indistinguishable"
  from gradient boosting, and the two-component prior is "heterogeneity, not phenotypes",
  pending stability tests.

Suite: 183 tests (166 + 17 new).

## Round 26 — analysis outputs can live outside the clone (v0.16.2)

On the cluster, running the benchmark and the MIMIC validation rewrote `results/*.md` and
`results/*.png`, files the repository also tracks, so the next `git pull` aborted. New module
`outputs.py` resolves the output directory as `--out-dir` > `$NEPHROQ_OUT_DIR` > `results/`; it is
used by `benchmark_posterior.py`, `validate_mimic_patients.py`, `insilico_trial.py`,
`mvp_calibration.py`, `plot_cric_validation.py` and `plot_key_results.py` (the first three also
take `--out-dir`). With `NEPHROQ_OUT_DIR` set on a compute node, its clone stays identical to the
remote. Committed results are updated deliberately from the aggregate reports.

Suite: 166 tests (162 + 4 new).

## Round 25 — results of the two-stage, Student-t and hybrid run on MIMIC-IV (v0.16.1)

Aggregate results committed in `results/mimic_patient_validation.md` and summarized in the README.
Noise learned from 27 782 short-interval pairs: Student-t, scale 5.26, 3.3 df (SD 8.38) — heavy
tails, as the acute spikes suggested. Pooled over 1 261 held-out visits (858 test patients):
retrained NPE RMSE 15.34 (bias −0.15, 90 % coverage 79 %); hybrid 14.24 (coverage 86 %, CRPS
7.45, AUC 0.891); gradient boosting 14.39 (85 %, 7.56, 0.889). Bootstrap ΔRMSE: hybrid vs NPE
−1.66 to −0.57; hybrid vs GBM −0.40 to +0.11 (tie); NPE vs least squares +0.07 to +1.64 and vs
the MLP ensemble +0.24 to +1.86 in favour of NPE; GBM vs NPE −1.60 to −0.39 in favour of GBM.
Retraining on the prior and reweighting to it are equivalent here (−0.47 to +0.05).

The learned prior again chose K = 2 (held-out log-lik/patient 0.225 vs 0.162), now as 87 % with
median injury rate 0.40× (mean q 1.38) and 13 % at 4.23× (mean q 1.73, SD 0.59). With the
injury rate free to exceed 3×, the fast regime moved from a high `q` to a high injury rate: the
existence of two regimes is robust, their attribution to `q` is not, consistent with `q` being
weakly identified. The README states it that way.

Posterior benchmark regenerated with the 0.25–6× box: NPE 9.47 vs exact 9.35 RMSE, 91 % vs 92 %
coverage, all SBC p ≥ 0.05; MLP ensemble 33 % coverage; `q` fixed 11.87 and 83 %. The benchmark
header now reads the prior box from the code instead of a hard-coded string.

## Round 24 — two-stage empirical Bayes, heavy-tailed noise and a hybrid residual model (v0.16.0)

### What the v0.15.0 run on MIMIC-IV showed
The empirical-Bayes prior chose K = 2 (held-out log-lik/patient 0.129 vs 0.053 for K = 1): a
stable regime (weight 0.93, mean q 1.02, median injury rate 0.49×) and a progressor regime (0.07,
q 2.93, 2.66×) — the two regimes Round 17 found by population calibration (q ≈ 0.90 and 3.30),
now recovered from per-patient posteriors without selecting on outcome. Reweighting removed the
decline bias (pooled −3.67 → +0.44; 5 y −8.33 → +1.20) and lowered MAE (11.30 → 11.04), but
pooled RMSE was unchanged (15.52 → 15.59, CI −0.49 to +0.30), 90 % coverage fell to 82 % and AUC
for eGFR < 30 to 0.853. The cause was sampling: importance efficiency was 7 % (≈ 70 effective of
1 000 draws), so tails were thin. Two further signals: the progressor component's injury rate
sat at 2.66× with SD 0.03 against the 3× ceiling of the prior box, and every method, gradient
boosting included, under-covered (82–85 %), pointing to heavy-tailed noise.

### Changes
- **Stage-2 network.** `npe_posterior.train(prior=...)` trains on simulations drawn from a
  learned `PopulationPrior` (`PopulationPrior.sample`); the validation trains it after the
  empirical-Bayes fit, so the population-prior posterior needs no importance weights.
  Weights are cached under the prior's hash (`npe_posterior_sd…_t…_pop<key>.pt`).
- **Wider injury-rate box**: 0.25–6× (was 0.25–3×). Checkpoints now record their box and
  `load()` refuses weights trained on a different one.
- **Student-t measurement noise.** `observation_noise(..., noise_df)`; the validation fits
  scale and degrees of freedom to short-interval (≤ 0.5 y) eGFR differences of the training
  patients and trains both networks with that noise model (`--noise-model gaussian` keeps the
  old behaviour).
- **Hybrid.** Gradient boosting on the residuals of the stage-2 forecast (with the forecast and
  its spread as inputs), trained on the training patients.
- The report gives bootstrap ΔRMSE against both the retrained NPE and gradient boosting, the
  fitted noise model, and the learned prior.
- The posterior benchmark table in the README was produced with the 3× box and is regenerated
  by `benchmark_posterior.py`.

Dry run on a synthetic cohort (1 500 patients, 85 % low-q regime, Student-t noise scale 6 with
3 df): fitted noise scale 6.41 with 4.1 df; retrained NPE 90 % coverage, pooled RMSE 13.19 vs
16.03 for the uniform-prior NPE. On data generated by the model itself the hybrid does not help
(13.94), as expected; whether it helps on MIMIC is the open question this round asks.

Suite: 162 tests (160 + 2 new).

## Round 23 — a population prior learned from the data removes the decline bias seen on MIMIC-IV (v0.15.0)

### What the first real-data run showed
`validate_mimic_patients.py` on MIMIC-IV 3.1 (25 062 patients from the loader; 9 530 eligible;
temporal split, test = anchor_year_group 2017-2019 and 2020-2022: 8 672 train / 858 test; NPE
trained for noise SD 8.7). Pooled over 1 261 held-out patient-visits, RMSE: population 18.21,
least squares (rate) 16.25, MLP ensemble 16.42, **NPE 15.54**, per-patient OLS 28.09,
empirical-Bayes random slope 20.57, gradient boosting trained on MIMIC 14.39. Patient-bootstrap
ΔRMSE vs NPE: population +1.58 to +3.84, MLP +0.18 to +1.64, OLS +10.09 to +15.07, random slope
+3.95 to +6.11, least squares −0.04 to +1.48 (tie), gradient boosting −1.86 to −0.49 (better).
NPE 90 % coverage 84 % (GBM 85 %, MLP 71 %). eGFR < 30 at a held-out visit (57 events / 1 157):
AUC NPE 0.877, MLP 0.887, GBM 0.889; Brier 0.050, 0.040, 0.038.

The NPE error was dominated by a **bias toward too-fast decline that grew with the horizon**
(−2.5, −5.5, −8.3 mL/min/1.73 m² at ~1, ~3, ~5 y). That is the signature of the uniform prior on
`q` (mean 2.15): `q` is barely identified per patient, so its posterior inherits the prior,
while Round 17 measured q ≈ 0.9 for the MIMIC cohort as a whole.

### New — `population_prior.py` (empirical Bayes over the amortized posterior)
Learns the population distribution of (q, log injury rate) from the TRAINING patients' NPE
draws, as a mixture of K components (Beta for q on its box × truncated normal for the log rate),
by EM with exact weighted-MLE M-steps (components constrained unimodal and not sharper than
q SD ≈ 0.12). New patients get the posterior under that prior by importance reweighting the same
draws — the network is not retrained. K ∈ {1, 2} is chosen by held-out marginal likelihood on
20 % of the training patients. Recovers a known two-regime population (weights 0.67/0.33 vs
0.70/0.30; mean q 1.06/3.07 vs 1.19/2.96). On a synthetic low-q cohort it moves the pooled bias
from −1.11 to +0.80 and brings 90 % coverage from 95 % to 90 % (CRPS 5.14 → 5.05).

### Validation now reports "NPE + population prior" and uses it as the reference
`validate_mimic_patients.py` fits the prior on the training split (300 draws per patient),
scores the reweighted posterior predictive as an eighth method, reports the fitted components
and the mean importance-sampling efficiency, and computes the bootstrap ΔRMSE of every method
against it.

Suite: 160 tests (159 + 1 new).

## Round 22 — patient-level validation on real trajectories, ready to run on MIMIC-IV (v0.14.0)

### New — `validate_mimic_patients.py`
Implements the patient-level protocol of `external_validation.validate_patient_level` on the
cohort written by `mimic_loader.py`. eGFR series are thinned to one median per 90 days; every
method personalizes on the first 2 years after the confirmed index; held-out visits are scored at
~1, ~3 and ~5 years after the last observed visit, one visit per patient per horizon. Split by
`anchor_year_group` (temporal; most recent group = test) when `patients.csv.gz` is supplied,
otherwise random 30 %. Seven methods on identical visits: population model, one-parameter least
squares, MLP ensemble, NPE posterior predictive (with measurement noise added for scoring against
observations), per-patient OLS slope, empirical-Bayes random-slope model and quantile gradient
boosting (the last two fitted on the training split only; the mechanistic methods are not fitted
on MIMIC). Reports RMSE/MAE/bias/90 % coverage/CRPS by horizon, AUC and Brier for eGFR < 30 at a
held-out visit, and patient-bootstrap 95 % CIs of every RMSE difference against NPE. Per-patient
rows go to git-ignored `results/*.csv`; only the aggregate `.md`/`.png` are meant to be shared.
Tested end to end on a synthetic cohort in the loader format (no real data in the repository),
including a check that no held-out visit leaks into the observed window.

### NPE trained for the data's noise level
`npe_posterior.train(noise_sd=...)` and `load(noise_sd=...)`; weights are stored per noise level
(`npe_posterior_sd8.7.pt` for hospital creatinine) and the checkpoint records the level.

### NPE training runs its full learning-rate schedule
On the cluster, early stopping (patience 6) fired at epoch 15 of 40, during the high-rate phase of
the one-cycle schedule, before the annealing third where the validation loss improves most (best
val NLL 3.854 vs 3.750 for the full run). The posterior stayed calibrated (90 % coverage 93 %, all
SBC p > 0.05) but forecast less sharply (RMSE 9.51 vs 8.91). `train()` now runs the full schedule
and keeps the best checkpoint; `patience`, if given, is honoured only after 80 % of the epochs.

### The cached personalizer can no longer outlive the model it learned
`personalizer.pkl` now stores a fingerprint of `patient_params`, priors, noise and training
sizes. A mismatched or cross-version cache (warnings during unpickling are now errors) is
retrained instead of reused. This is what let a pre-Round-21 cache, trained before the
albuminuric scaling fix, load silently on a cluster with a newer scikit-learn.

Suite: 159 tests (155 + 4 new).

## Round 21 — full Bayesian personalization, and the injury rate now reaches the albuminuric term (v0.13.0)

### P1 — the personal injury rate did not scale the albuminuric hazard
Round 16 moved the albuminuric coefficient from `w_uacr` to `k_alb`, and the hazard now reads
`k_alb`. `personalize.patient_params` kept scaling only `w_uacr`, so since Round 16 a patient's
injury rate multiplied the hyperfiltration and metabolic terms but **not** the albuminuric one,
which dominates in macroalbuminuric patients. `patient_params` now scales `k_alb` (and keeps
`w_uacr` in step). A regression test checks that doubling the rate doubles `k_alb` and speeds
decline in an albuminuria-dominated patient. Deployed-path validation after the fix (400 virtual
patients, eGFR 5 years after the last visit): population 16.01; network 10.33 (−35 %);
one-parameter least squares 11.27 (−30 %). The measurement-strategy experiment was re-run and its
README table is unchanged to two decimals.

### New — amortized neural posterior (`npe_posterior.py`, optional dependency)
Full posterior over (`q`, log injury rate, true baseline eGFR) from an irregular history: GRU
summary network over the visit sequence plus a conditional neural spline flow (zuko), trained by
maximum likelihood on 200 000 simulated patients (val NLL 3.750; ~9 min on 2 CPU cores). The
prior on `q` is widened to 0.8–3.5 so it covers the progressor regime found in Round 17
(q ≈ 3.3). Forecasts are posterior-predictive: every draw, `q` included, is projected with the
ODE.

### New — exact reference and benchmark (`benchmark_posterior.py`)
`grid_posterior` computes the exact posterior under the same prior and likelihood on a 3-D grid
(57 024 points per patient). On 300 held-out virtual patients: NPE RMSE 8.91, 90 % coverage
93 %, CRPS 4.09, 47 ms/patient; exact posterior 8.73, 95 %, 4.03, 910 ms; MLP ensemble 13.33
with 18 % coverage (overconfident); least squares 10.60 (rate only) and 10.09 (rate + baseline
eGFR); NPE with `q` fixed at the population value 12.07 with 68 % coverage. Outputs:
`results/posterior_benchmark.md` and `.png`.

### New — vectorized simulator (`fast_sim.py`)
RK4 on log N for whole batches of patients, the only re-expression of the hazard outside
`model_core`. Pinned by test to the LSODA reference (max disagreement 5e-4 mL/min/1.73 m²
across the prior; < 0.01 enforced). 20 000 patients over 8 years in < 1 s.

### Packaging
`requirements-npe.txt` and the `npe` optional extra (torch, zuko); CI installs the CPU wheels so
the NPE tests run; NPE tests skip cleanly when torch is absent. New modules listed in
`pyproject.toml`. Trained weights (`calibration/npe_posterior.pt`) are git-ignored and
reproducible with `python -c "import npe_posterior as n; n.train()"`.

The k-fold "false green" test still asserted the pre-Round-18 bound (`< 0.05`) although Round 18
records it as relaxed to `< 0.12`; it failed on macOS/Accelerate (0.065). The test now uses the
documented bound.

Suite: 155 tests (148 + 7 new).

## Round 20 — the personalizer is validated as it runs, against a fair rival (v0.12.0)

### P1 — the reported personalization gain measured a path the app does not use
`validate()` forecast with the network's OWN `q` estimate, but `personalize()` projects with the
POPULATION `q` and the inferred injury rate (Round 17: `q` is not identifiable per patient). The
README's −43 % therefore described a configuration the app never runs. `validate()` now scores
the deployed path; the own-`q` forecast is kept as a labelled diagnostic.

### P2 — the classical rival was not the fair one
The only classical baseline fitted BOTH `q` and the injury rate. Because `q` is not identifiable,
that rival spends its noise budget on the wrong parameter and is easy to beat. The new
`least_squares_scale_only()` solves exactly the deployed problem (injury rate, `q` at population
value) and is now the headline comparator; the two-parameter fit is kept as a diagnostic.

Result on 400 held-out virtual patients (each with its own true `q`), eGFR 5 years past the last
visit: population 13.20; network as deployed 9.82 (−26 %); one-parameter least squares 10.08
(−24 %); network with own `q` 9.19 (diagnostic); two-parameter least squares 11.15 (diagnostic).
Personalization clearly earns its place; against the fair rival the network matches accuracy and
is ~4× faster (0.9 vs 4.1 ms). README updated accordingly.

### P3 — the CRIC compatibility flag answered a different question
`within_observed_range` tested whether CRIC's observed POINT slope fell inside the model's
predicted range, which spans only the proteinuria→UACR conversion and is very narrow, so it came
back False for both phases even though both predicted ranges (Phase I −2.57 to −2.45; Phase III
−3.17 to −2.99) lie inside the published plausible band (−3.5 to −1.9). The flag now means
"predicted range inside the observed band"; the strict check is reported separately as
`observed_in_predicted_range`. The README no longer calls the CRIC slope an unverified
placeholder (it is the published AJKD 2020 value, `verified=True`).

### Housekeeping
Test-count references updated (148); stale comments about a committed `personalizer.pkl` fixed;
files written by the test suite (`results/validation_report.md`, `results/mvp_validation.png`)
are now git-ignored; version 0.12.0 in `pyproject.toml`, `CITATION.cff` and the provenance block.

Suite: 148 tests green (145 + 3 new).

## Round 19 — two loader edge cases the Round 18 provenance fix missed

A follow-up review found the CSV loader still violated the "age at the most recent visit"
contract and could mislabel an imputed sex. Both lived in the per-row demographic accumulation,
which decided by FILE ORDER rather than by visit date. Both reproduced directly; both now have
regression tests.

### P1 — age could bind to the wrong visit in an unordered CSV
The loader kept the last age seen in file order and only afterwards sorted visits by date. A
descending CSV (newest row first) therefore stored the OLDEST age: a patient whose latest visit
(2024) recorded age 70 came back with 66. That silently re-broke the age contract and shifted
every CKD-EPI conversion. The loader now tracks the age from the latest visit DATE that supplies
one, and carries it forward to the latest visit by calendar time when it was recorded earlier
(the same arithmetic add_visit uses -- not a population guess, so not flagged imputed).

### P2 — a fully-absent sex column was not flagged imputed
`sex=r.get("sex", "M")` seeded the default 'M' at accumulation time, so when the column was
missing entirely, `sex_imputed` came back False -- a guessed sex presented as observed. (The
column-present-but-blank case worked, because a blank cell is "" rather than a missing key.) Sex
is now recorded only from a VALID 'M'/'F' cell; a missing column, blank cell, or garbage value
leaves it unset and it is defaulted to 'M' with `sex_imputed=True`. The latest valid sex by date
wins, consistent with the age rule.

Suite: 145 tests green (139 + 6 new). Fix is confined to `src/clinical_data.py`.

## Round 18 — deployment-review fixes (Docker, twin state, provenance, packaging, CI)

An external review of the deployment layer found six issues plus a flaky test and no CI. Each
was verified against the code before fixing, and each fix carries a regression test.

### P1 — the Docker image could not start
`docker/Dockerfile` set `CMD ["python", "system_twin.py", ...]`, but `system_twin.py` was
removed when the pipeline was restructured (commit "Delete src/system_twin.py"), and its
orchestrated stage scripts no longer exist either -- so no current script accepts the old
`--skip-slow/--skip-bayes` flags. The build could finish; the container failed immediately on
`docker run`. CMD now runs `insilico_trial.py`, a fast, deterministic, self-contained
validation that reproduces the three published placebo arms and writes a report. Verified it
exits 0.

### P2 — patient age was frozen across twin updates
`PatientState.age` is defined as the age at the latest visit, but `add_visit()` appended
without advancing it, and `RenalDigitalTwin.update()` / `_model_inputs()` then reconstructed the
whole eGFR history through CKD-EPI with that stale age. Because CKD-EPI eGFR falls with age, a
too-young age inflates eGFR -- a bias that grew with every longitudinal update. `add_visit`
now advances `age` by the gap whenever a strictly later visit becomes the latest (and leaves it
unchanged for a back-dated insert); a new `age_at(date)` helper gives the correct age for any
visit, and `update()` scores the incoming visit at its own age. Regression tests assert the age
advances, that a back-dated visit does not move it, and that the same creatinine observed five
years later reconstructs to a LOWER eGFR (it stayed flat under the bug).

### P3 — `uncertainty_change` reported the wrong quantity
The field is documented as the change in the spread of the susceptibility estimate, but it
stored `new.scale - previous.scale` -- the shift in the POINT estimate, which could mislead a
clinical or audit consumer. `Forecast` now carries `scale_sd` (the personalizer already
computed it; it was just never stored), and `uncertainty_change` is `new.scale_sd -
prev.scale_sd`. Regression test pins the spread semantics and guards against the point-estimate
difference.

### P4 — README claimed a shipped artifact that is not in git
The README described `calibration/personalizer.pkl` as a pre-trained model that "ships with the
repo", but only `calibration/README.md` is committed. The auto-retrain path prevents a crash
but pays a one-time ~13 s training cost on first use and contradicts the "starts instantly"
promise. The README now states the estimator is trained on demand on first use and cached, with
the one-time cost stated -- matching what the code actually does.

### P5 — missing demographics were imputed silently
The CSV loader turned a missing age into 60 and a missing/invalid sex into 'M' with no record
that either was a guess -- and both feed CKD-EPI directly, so this corrupted the audit trail the
design promises. `PatientState` now carries `age_imputed` / `sex_imputed` flags, set by the
loader whenever it substitutes a default. Regression test checks both the value and the flag.

### P6 — the declared package installed no code
`pyproject.toml` had `py-modules = []`, so `pip install .` installed metadata and dependencies
but none of the modules in `src/`. Fixed with `package-dir = {"" = "src"}` and an explicit
`py-modules` list; a test (`test_packaging.py`) asserts that list stays exactly in sync with
`src/*.py` so it cannot silently drift. Verified in a clean venv that the modules import after
`pip install .`.

### Flaky k-fold assertion + no CI
The "false green" k-fold test asserted the pinned-`q` CV was `< 0.05`; that exact value depends
on BLAS/optimizer tie-breaking and was reported at ~0.065 on Python 3.10/3.12. The bound is now
`< 0.12` (still small enough to demonstrate the deceptively-tidy spread), while the load-bearing
assertions -- `q` in `at_bound` and in `unstable` -- are unchanged. A GitHub Actions workflow
(`.github/workflows/ci.yml`) now runs the suite on Python 3.10 and 3.12, so these regressions are
caught automatically.

Suite: 139 tests green (130 + 9 new). No new ML dependency introduced.

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
