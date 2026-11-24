"""
Where analysis scripts write their outputs.

Order of precedence:
  1. an explicit directory passed by the caller (the scripts' --out-dir flag);
  2. the environment variable NEPHROQ_OUT_DIR;
  3. results/ inside the repository (the default, as before).

Pointing NEPHROQ_OUT_DIR outside the clone on a compute node (e.g. in ~/.bashrc)
keeps that clone byte-identical to the remote, so `git pull` never collides with
reports or figures the node generated. The files committed under results/ are
then updated deliberately, from the aggregate reports, never as a side effect of
running an analysis.
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_RESULTS = os.path.normpath(os.path.join(HERE, "..", "results"))
ENV_VAR = "NEPHROQ_OUT_DIR"


def results_dir(explicit=None):
    """Resolve, create and return the output directory."""
    path = explicit or os.environ.get(ENV_VAR) or REPO_RESULTS
    path = os.path.abspath(os.path.expanduser(path))
    os.makedirs(path, exist_ok=True)
    return path


def results_path(name, explicit=None):
    """Full path of an output file inside the resolved directory."""
    return os.path.join(results_dir(explicit), name)
