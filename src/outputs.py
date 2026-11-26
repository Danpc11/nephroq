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

How many cores the analyses may use is resolved the same way: an explicit
`--threads`, else NEPHROQ_THREADS, else the CPUs this process may run on.
`limit_threads()` caps torch, the BLAS libraries and OpenMP (scikit-learn's
gradient boosting) at that number, so a run never takes more cores than allowed.
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


THREADS_VAR = "NEPHROQ_THREADS"


def n_threads(explicit=None):
    """Cores an analysis may use: explicit > $NEPHROQ_THREADS > CPUs available to
    this process (its affinity mask, not the machine's total)."""
    if explicit:
        return max(1, int(explicit))
    env = os.environ.get(THREADS_VAR)
    if env:
        return max(1, int(env))
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except AttributeError:                       # macOS / Windows
        return max(1, os.cpu_count() or 1)


def limit_threads(explicit=None):
    """Caps every thread pool of this process at n_threads(explicit) and returns
    the cap. Call it at the start of a script, before heavy work. The variables
    are also exported so that child processes inherit the cap."""
    n = n_threads(explicit)
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", THREADS_VAR):
        os.environ[var] = str(n)
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(limits=n)               # BLAS and OpenMP pools already loaded
    except Exception:
        pass
    try:
        import torch
        torch.set_num_threads(n)
    except Exception:
        pass
    return n
