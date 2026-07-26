"""
Packaging regression (review P6). `pip install .` must install the source
modules, not just metadata. With py-modules = [] it installed nothing importable.
This guards the fix from silently drifting: the declared py-modules must match
exactly the set of modules under src/, so a new src file cannot be left out (and a
deleted one cannot linger) without this test failing.
"""
import os
import sys

try:
    import tomllib          # py3.11+
except ModuleNotFoundError:  # pragma: no cover - py3.10
    import tomli as tomllib  # type: ignore

ROOT = os.path.join(os.path.dirname(__file__), "..")


def _load_pyproject():
    with open(os.path.join(ROOT, "pyproject.toml"), "rb") as f:
        return tomllib.load(f)


def test_pyproject_lists_all_src_modules():
    cfg = _load_pyproject()
    declared = set(cfg["tool"]["setuptools"]["py-modules"])
    on_disk = {f[:-3] for f in os.listdir(os.path.join(ROOT, "src"))
               if f.endswith(".py") and f != "__init__.py"}
    assert declared == on_disk, (
        "py-modules is out of sync with src/*.py.\n"
        f"  missing from pyproject: {sorted(on_disk - declared)}\n"
        f"  stale in pyproject:     {sorted(declared - on_disk)}")


def test_package_dir_points_at_src():
    cfg = _load_pyproject()
    assert cfg["tool"]["setuptools"]["package-dir"] == {"": "src"}
