"""End-to-end test of the patient-level validation on a SYNTHETIC cohort in the
mimic_loader output format (no real data in the repository). Skipped without the
optional NPE dependencies."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")
pytest.importorskip("zuko")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import fast_sim
import npe_posterior as npe
import personalize as pz
import validate_mimic_patients as vm


def _synthetic_cohort(n=80, seed=0):
    rng = np.random.default_rng(seed)
    rows, groups = [], []
    for i in range(n):
        q, s = rng.uniform(0.9, 3.0), float(np.exp(rng.uniform(-1, 1)))
        e0, a1c = rng.uniform(30, 100), rng.uniform(6, 11)
        uacr, sbp = float(np.exp(rng.uniform(np.log(10), np.log(2000)))), rng.uniform(115, 170)
        t = np.sort(np.concatenate([[0.0], rng.uniform(0, 7, rng.integers(8, 25))]))
        e = fast_sim.simulate_egfr(q, s, e0, a1c, uacr, sbp, t)[0] + rng.normal(0, 8, len(t))
        for tt, ee in zip(t, e):
            rows.append(dict(patient_id=f"p{i}", time_years=tt, egfr=ee,
                             hba1c=a1c, uacr=uacr, sbp=sbp))
        groups.append(dict(subject_id=f"p{i}",
                           anchor_year_group=["2008 - 2010", "2011 - 2013", "2014 - 2016"][i % 3]))
    return pd.DataFrame(rows), pd.DataFrame(groups)


def test_thinning_keeps_one_median_per_window():
    t, e = vm.thin(np.array([0.0, 0.01, 0.02, 0.5]), np.array([50.0, 70.0, 60.0, 40.0]))
    assert len(t) == 2 and e[0] == pytest.approx(60.0)


def test_pipeline_runs_and_reports_only_aggregates(tmp_path):
    cohort, groups = _synthetic_cohort()
    pfile = tmp_path / "patients.csv.gz"
    groups.to_csv(pfile, index=False)
    net, _ = npe.train(n_train=3000, epochs=1, save=False, verbose=False)
    mlp = pz.train(n_train=600, save=False)
    df, tables, ev, ci, meta = vm.run(cohort, str(pfile), noise_sd=8.0, net=net, mlp=mlp,
                                      n_boot=50, verbose=False)
    assert meta["split"].startswith("temporal")
    assert set(df.method) == set(vm.METHODS)
    for m in vm.METHODS:
        assert np.isfinite(tables["all"][m]["rmse"])
    assert 0.0 <= tables["all"]["npe"]["cov90"] <= 1.0
    text = vm.report(tables, ev, ci, meta)
    assert "NPE" in text
    assert not any(f"p{i}" in text for i in range(80))   # no patient ids


def test_no_held_out_visit_leaks_into_the_observed_window():
    cohort, _ = _synthetic_cohort(n=40, seed=3)
    patients, _ = vm.build_patients(cohort)
    assert patients
    for p in patients:
        assert p["t"][0] == 0.0 and p["t"][-1] <= vm.EARLY_YEARS
        for t_f, _ in p["future"].values():
            assert t_f - p["t"][-1] > vm.HORIZONS["1y"][0]
