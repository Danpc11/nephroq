# Trial data provenance and verification

This document records every published trial value used by NephroQ's in-silico
validation (`src/insilico_trial.py`), the primary source it was taken from, and
whether it has been checked against that source's PDF.

**Why this exists.** The in-silico test is only as trustworthy as the numbers it
is anchored to. NephroQ fits its treatment-effect and anti-albuminuric parameters
on CREDENCE and EMPA-KIDNEY, then predicts DAPA-CKD out-of-sample. If a target
were mis-transcribed, a "pass" would be meaningless. Every value below was
therefore read back against the source article.

**Verification status (2026-07-15).** All values were checked against the primary
PDFs. Two transcription errors were found and corrected; both were numerically
negligible and did not change any pass/fail verdict. They are listed at the end.

A note on slope phases, because it is the easiest thing to get wrong. SGLT2
inhibitors cause an acute, reversible eGFR dip in the first weeks, so a trial
reports three slopes:

- **acute** (baseline to ~week 2–3): dominated by the reversible haemodynamic dip;
- **chronic** (week 2–3 to end): the underlying rate of nephron loss;
- **total** (baseline to end): a weighted combination of the two.

The **placebo arm has no dip** (it receives no drug), so its total and chronic
slopes are nearly identical. NephroQ has no acute-dip term, so its slope is
intrinsically a *chronic* slope. It must therefore be compared against published
**chronic** figures, and against the placebo arm's slope directly. Mixing chronic
and total between trials would bias the calibration.

---

## CREDENCE — calibration anchor

Canagliflozin in type 2 diabetes with albuminuric CKD.

**Source:** Perkovic V, Jardine MJ, Neal B, et al. Canagliflozin and Renal
Outcomes in Type 2 Diabetes and Nephropathy. *N Engl J Med* 2019;380:2295–2306.
NCT02065791. (`NEJMoa1811744`)

| Field | Value | Source location | Verified |
|---|---|---|---|
| n randomized | 4401 | p.2298 | yes |
| Follow-up (median) | 2.62 yr | p.2298 | yes |
| Baseline eGFR (mean) | 56.2 | p.2298 | yes |
| Baseline UACR (median) | 927 | p.2298 | yes |
| Baseline HbA1c (mean) | 8.3% | p.2298; Table 1 | yes |
| Baseline SBP (mean) | 140.0 | Table 1 | yes |
| Age (mean) / % male | 63.0 / 66% | Table 1 | yes |
| **Placebo chronic slope** | **−4.59** mL/min/1.73m²/yr | p.2301 | yes |
| Canagliflozin chronic slope | −1.85 | p.2301 | yes |
| **Chronic slope difference** | **2.74** (= −1.85 − −4.59) | p.2301 | yes |
| Total slope difference | 1.52 (95% CI 1.11–1.93) | p.2301 | yes |
| **UACR reduction** | **31%** (95% CI 26–35) | p.2301 | yes |

Role in NephroQ: the placebo slope anchors the progression scale; the chronic
slope difference anchors the treatment-effect scale; the UACR reduction anchors
the direct anti-albuminuric effect.

---

## DAPA-CKD (type 2 diabetes subgroup) — out-of-sample test

Dapagliflozin in CKD. NephroQ predicts this trial's chronic slope difference
**without fitting to it**; it is the held-out target that carries the evidential
weight of the in-silico validation.

**Slope source:** Heerspink HJL, Jongs N, Chertow GM, et al. Effect of
dapagliflozin on the rate of decline in kidney function in patients with CKD with
and without type 2 diabetes: a prespecified analysis from the DAPA-CKD trial.
*Lancet Diabetes Endocrinol* 2021;9:743–754. NCT03036150. (`heerspink2021`)

**Albuminuria source:** Jongs N, Greene T, Chertow GM, et al. Effect of
dapagliflozin on urinary albumin excretion in patients with CKD with and without
type 2 diabetes: a prespecified analysis from the DAPA-CKD trial. *Lancet Diabetes
Endocrinol* 2021;9:755–766. (`jongs2021`)

| Field | Value | Source location | Verified |
|---|---|---|---|
| n (T2D subgroup) | 2906 of 4304 | Heerspink 2021 p.746 | yes |
| Follow-up (median) | 2.3 yr | Heerspink 2021 p.746 | yes |
| Baseline eGFR (mean) | 43 | Heerspink 2021 p.746 | yes |
| Baseline UACR (median) | 949 | Heerspink 2021 p.746 | yes |
| Baseline SBP (mean) | 137 | Heerspink 2021 Table 1 | yes |
| **Placebo chronic slope (T2D)** | **−3.84** mL/min/1.73m²/yr | Heerspink 2021 Fig 2A; p.747 | yes |
| Dapagliflozin chronic slope (T2D) | −1.58 | Heerspink 2021 p.747 | yes |
| **Chronic slope difference (T2D)** | **2.26** (95% CI 1.88–2.64) | Heerspink 2021 Fig 2A; p.747 | yes |
| Total slope difference (T2D) | 1.18 (95% CI 0.79–1.56) | Heerspink 2021 Fig 2B | yes |
| **UACR reduction (T2D)** | **35.1%** (95% CI 30.6–39.4) | Jongs 2021 p.760 | yes |

The HbA1c mean (7.8) used in the cohort simulator is an approximation for the T2D
subgroup; the slope papers do not report it for that subgroup directly. It only
affects the metabolic insult term, which is small, and does not enter the held-out
target.

---

## EMPA-KIDNEY — second calibration anchor (low eGFR, low UACR)

Empagliflozin in a broad CKD population. Used to identify the saturation ceiling
`S_SAT` separately from the hazard scale, because its baseline eGFR and UACR are
both much lower than CREDENCE/DAPA — orthogonal variation.

**Source:** The EMPA-KIDNEY Collaborative Group. Effects of empagliflozin on
progression of CKD: a prespecified secondary analysis from the EMPA-KIDNEY trial.
*Lancet Diabetes Endocrinol* 2024;12:39–50. PMID 38061371. NCT03594110.
(`PIIS2213858723003212`)

| Field | Value | Source location | Verified |
|---|---|---|---|
| n randomized | 6609 | Summary; p.43 | yes |
| Follow-up (median) | 2.0 yr | Summary | yes |
| Baseline eGFR (mean) | 37.3 | Fig 1 (all participants) | yes |
| Baseline UACR (median) | 329 | Table 1 (approx) | yes |
| **Placebo chronic slope** | **−2.75** mL/min/1.73m²/yr | p.44; Fig 2 | yes |
| Empagliflozin chronic slope | −1.37 | Summary; p.44 | yes |
| **Chronic slope difference** | **1.38** (= −1.37 − −2.75) | Summary; Fig 2 | yes |
| Relative reduction | 50% (95% CI 42–58) | Summary; p.44 | yes |

**Population caveat.** Only ~46% of EMPA-KIDNEY had type 2 diabetes, whereas
NephroQ is a T2D model. Its HbA1c mean (6.5) is an assumed value for this mixed
cohort. Use EMPA-KIDNEY as a progression anchor at low eGFR/UACR, **not** as a T2D
efficacy target. This is also why EMPA-KIDNEY sits near the upper edge of the
audit tolerance: it is a partly non-diabetic population.

---

## Corrections applied (2026-07-15)

Both found by reading the code values back against the source PDFs. Both are
numerically negligible and changed no pass/fail verdict; the held-out DAPA-CKD
prediction stayed inside its published CI before and after.

| Field | Was | Corrected to | Source |
|---|---|---|---|
| CREDENCE `uacr_reduction_ci` | (27.0, 36.0) | **(26.0, 35.0)** | NEJM 2019 p.2301 |
| DAPA-CKD `placebo_slope` | −3.83 | **−3.84** | Heerspink 2021 Fig 2A + p.747 |

A third item was investigated and found to be **correct as written**: CREDENCE's
`chronic_slope_diff = 2.74`. It is genuinely the *chronic* slope difference
(−1.85 − −4.59 = 2.74), not the total (1.52). An earlier reading mistook it for
the total; direct arithmetic from the placebo and treated chronic slopes confirms
the label is right, and it is consistent with the way DAPA-CKD's chronic (2.26)
and total (1.18) are stored separately.

---

## Standing caveat

These values were transcribed from published articles and figures by hand and
then verified against the source PDFs. Figure read-offs (e.g. EMPA-KIDNEY's UACR
median, DAPA-CKD's subgroup slopes read from Figure 2) carry a small reading
uncertainty. Where a value drives a result, the source location is given above so
a reviewer can re-check it directly. NephroQ's calibration is only as good as
these anchors, and this file exists so that dependency is auditable rather than
hidden.
