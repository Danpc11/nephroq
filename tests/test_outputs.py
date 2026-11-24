"""Analysis outputs can be sent outside the repository, so a compute node's clone
stays identical to the remote and `git pull` never collides with generated files."""
import os
import sys

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
