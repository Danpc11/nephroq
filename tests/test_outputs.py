"""Analysis outputs can be sent outside the repository, so a compute node's clone
stays identical to the remote and `git pull` never collides with generated files."""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import outputs


def test_default_is_the_repository_results_folder(monkeypatch):
    monkeypatch.delenv(outputs.ENV_VAR, raising=False)
    assert outputs.results_dir() == outputs.REPO_RESULTS


def test_environment_variable_redirects_every_script(monkeypatch, tmp_path):
    target = tmp_path / "runs"
    monkeypatch.setenv(outputs.ENV_VAR, str(target))
    assert outputs.results_path("x.md") == str(target / "x.md")
    assert target.is_dir()


def test_explicit_directory_wins_over_the_environment(monkeypatch, tmp_path):
    monkeypatch.setenv(outputs.ENV_VAR, str(tmp_path / "env"))
    assert outputs.results_dir(str(tmp_path / "flag")) == str(tmp_path / "flag")


def test_insilico_report_honours_the_output_directory(monkeypatch, tmp_path):
    import insilico_trial
    monkeypatch.setenv(outputs.ENV_VAR, str(tmp_path))
    insilico_trial.run(n=40, seed=1)
    assert (tmp_path / "insilico_trial_report.md").exists()


def test_thread_cap_follows_flag_then_environment(monkeypatch):
    import outputs
    monkeypatch.setenv(outputs.THREADS_VAR, "40")
    assert outputs.n_threads() == 40
    assert outputs.n_threads(8) == 8
    monkeypatch.delenv(outputs.THREADS_VAR)
    assert outputs.n_threads() >= 1


def test_limit_threads_caps_torch_and_exports_the_cap(monkeypatch):
    import outputs
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", outputs.THREADS_VAR):
        monkeypatch.delenv(v, raising=False)
    torch = pytest.importorskip("torch")
    before = torch.get_num_threads()
    try:
        n = outputs.limit_threads(2)
        assert n == 2 and os.environ["OMP_NUM_THREADS"] == "2"
        assert os.environ[outputs.THREADS_VAR] == "2"
        assert torch.get_num_threads() == 2
    finally:
        torch.set_num_threads(before)
        from threadpoolctl import threadpool_limits
        threadpool_limits(limits=None)
