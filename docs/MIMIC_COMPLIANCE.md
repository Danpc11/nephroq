# MIMIC-IV license compliance (summary)

This project uses [MIMIC-IV](https://physionet.org/content/mimiciv/) for
research calibration and demonstration purposes, under the **PhysioNet
Credentialed Health Data License 1.5.0**.

## What this means in practice

- **Research and non-commercial demo use only.** MIMIC-IV data is used
  exclusively to calibrate and validate the mechanistic model for
  scientific publication and a non-commercial research demo.
- **No raw data is redistributed.** Patient-level data never enters this
  repository, the deployed web app, or any cloud service — `data/*.csv` and
  `calibration/*.json` are excluded from version control.
  `src/calibrate_mimic.py` runs entirely on the user's local machine and
  deletes the intermediate per-patient file after calibration.
- **No third-party sharing.** MIMIC data is never sent to external APIs or
  services during processing.
- **Only aggregate, derived parameters may be shared**, e.g. the
  population-level `q`, `k_hf`, and insult weights resulting from a local
  calibration — never row-level patient data. See
  [`calibration/README.md`](../calibration/README.md).
- **Open-source code**, as required by the license: this repository.

## Required citation

Any publication or communication derived from MIMIC-IV must cite:

> Johnson, A., Bulgarelli, L., Pollard, T., Horng, S., Celi, L. A., & Mark,
> R. (2023). MIMIC-IV (version 3.1). PhysioNet.
> https://doi.org/10.13026/hxp0-hg59

and the original MIMIC-IV paper (Johnson et al., *Scientific Data*, 2023).

## Scope

This summary covers research use, academic publication, and a
non-commercial demo tied to that publication. Any commercial or clinical
deployment beyond that scope requires a separate license review with
PhysioNet/MIT-LCP before proceeding.
