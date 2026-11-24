"""
================================================================================
AMORTIZED NEURAL POSTERIOR ESTIMATION (NPE) for the renal digital twin
================================================================================
WHAT IT INFERS
For one patient, the full Bayesian posterior over three latent quantities, given
a handful of noisy eGFR values at irregular times plus baseline HbA1c, UACR and
systolic pressure:

    q          collapse exponent of the hyperfiltration feedback (structural;
               Round 17 found it differs between stable patients and
               progressors, so it is treated as uncertain, not fixed)
    log scale  the patient's personal injury rate, as a log multiple of the
               population rate
    eGFR0      the patient's true (noise-free) eGFR at the first visit

The forecast is the posterior predictive: every posterior draw is projected
forward with the mechanistic ODE, so the predictive band carries the parameter
uncertainty, including the unidentifiable q, instead of fixing q and hiding it.

HOW
1. Simulate virtual patients from the prior and the mechanistic model
   (`fast_sim`), with irregular visit schedules and measurement noise.
2. A summary network (a GRU over the visit sequence plus the baseline covariates)
   compresses each history into a fixed-size context vector. It reads visit
   times and values directly, so it handles any number of visits at any spacing.
3. A conditional neural spline flow (zuko NSF) models p(theta | context) on an
   unbounded transform of the prior box, so every posterior draw respects the
   prior support.
4. Both are trained jointly by maximum likelihood on (theta, history) pairs,
   which is exactly neural posterior estimation.

REFERENCE
`grid_posterior` computes the exact posterior under the same prior and
likelihood by brute-force evaluation on a 3-D grid. It is slow (~1 s per patient)
but has no approximation beyond the grid spacing, so it is the gold standard the
network must match.

Requires the optional dependencies `torch` and `zuko` (requirements-npe.txt).
================================================================================
"""
import os
import time
from dataclasses import dataclass

import numpy as np

import model_core as core
import fast_sim

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "..", "calibration", "npe_posterior.pt")

# ---- prior -------------------------------------------------------------------
PARAM_NAMES = ("q", "log_scale", "egfr0")
PRIOR_LO = np.array([0.8, np.log(0.25), 20.0])
PRIOR_HI = np.array([3.5, np.log(3.0), 110.0])
Q_POP = float(core.TRIAL_CALIBRATION_V2["q"])

# ---- observation model ---------------------------------------------------------
NOISE_SD = 6.0            # mL/min/1.73m2 per measurement
MAX_VISITS = 10
VISITS_RANGE = (2, 9)     # visits per history (inclusive)
SPAN_RANGE = (0.5, 8.0)   # years between first and last visit
SEQ_FEATURES = 3
STATIC_FEATURES = 5


# ==============================================================================
# Simulation
# ==============================================================================
def sample_prior(n, rng):
    """Draws theta = (q, log_scale, eGFR0) uniformly over the prior box."""
    return rng.uniform(PRIOR_LO, PRIOR_HI, size=(n, 3))


def sample_covariates(n, rng):
    """Baseline HbA1c (%), UACR (mg/g, log-uniform) and systolic BP (mmHg)."""
    a1c = rng.uniform(5.5, 12.0, n)
    uacr0 = np.exp(rng.uniform(np.log(5.0), np.log(3000.0), n))
    sbp = rng.uniform(105.0, 185.0, n)
    return a1c, uacr0, sbp


def sample_schedules(n, rng, visits_range=VISITS_RANGE, span_range=SPAN_RANGE):
    """Irregular visit times, first visit at t = 0, padded to MAX_VISITS.
    Returns (times (n, MAX_VISITS), mask (n, MAX_VISITS) bool)."""
    n_vis = rng.integers(visits_range[0], visits_range[1] + 1, n)
    span = rng.uniform(*span_range, n)
    u = np.sort(rng.uniform(0.0, 1.0, (n, MAX_VISITS)), axis=1)
    mask = np.arange(MAX_VISITS)[None, :] < n_vis[:, None]
    # place visits 1..n-1 uniformly in (0, span], with the last one AT span
    times = np.zeros((n, MAX_VISITS))
    for i in range(n):
        k = n_vis[i]
        inner = np.sort(rng.uniform(0.0, span[i], k - 2)) if k > 2 else np.array([])
        times[i, :k] = np.concatenate([[0.0], inner, [span[i]]])
        times[i, k:] = span[i]                    # padding: repeat last time
    return times, mask


def simulate_histories(theta, a1c, uacr0, sbp, times, mask, rng, noise_sd=NOISE_SD):
    """Noisy eGFR observations for each patient at its own visit times."""
    q, log_s, egfr0 = theta[:, 0], theta[:, 1], theta[:, 2]
    latent = fast_sim.simulate_egfr(q, np.exp(log_s), egfr0, a1c, uacr0, sbp, times)
    obs = np.clip(latent + rng.normal(0.0, noise_sd, latent.shape), 3.0, None)
    return np.where(mask, obs, 0.0)


def simulate_dataset(n, seed=0, chunk=20000):
    """Full training set: theta, covariates, schedules and noisy histories."""
    rng = np.random.default_rng(seed)
    out = {k: [] for k in ("theta", "a1c", "uacr0", "sbp", "times", "mask", "egfr")}
    for start in range(0, n, chunk):
        m = min(chunk, n - start)
        th = sample_prior(m, rng)
        a1c, uacr0, sbp = sample_covariates(m, rng)
        times, mask = sample_schedules(m, rng)
        egfr = simulate_histories(th, a1c, uacr0, sbp, times, mask, rng)
        for k, v in zip(out, (th, a1c, uacr0, sbp, times, mask, egfr)):
            out[k].append(v)
    return {k: np.concatenate(v) for k, v in out.items()}


# ==============================================================================
# Encoding
# ==============================================================================
def encode(times, egfr, mask, a1c, uacr0, sbp):
    """
    Sequence features per visit: time since first visit, eGFR, gap since the
    previous visit (all scaled to order one). Static features: HbA1c,
    log-albuminuria, systolic pressure, number of visits and span.
    Returns (seq (n, MAX, 3), lengths (n,), static (n, 5)) as float32 arrays.
    """
    times = np.atleast_2d(times)
    egfr = np.atleast_2d(egfr)
    mask = np.atleast_2d(mask)
    gap = np.diff(times, axis=1, prepend=0.0)
    seq = np.stack([times / 8.0, (egfr - 60.0) / 30.0, gap / 2.0], axis=-1)
    seq = np.where(mask[..., None], seq, 0.0)
    lengths = mask.sum(axis=1)
    span = np.array([t[m][-1] for t, m in zip(times, mask)])
    static = np.stack([
        (np.atleast_1d(a1c) - 8.5) / 2.0,
        (np.log1p(np.atleast_1d(uacr0) / 30.0) - 2.0) / 1.5,
        (np.atleast_1d(sbp) - 145.0) / 25.0,
        lengths / 10.0,
        span / 8.0,
    ], axis=-1)
    return seq.astype(np.float32), lengths.astype(np.int64), static.astype(np.float32)


def to_unbounded(theta):
    """Prior box -> R^3 (logit of the position inside the box)."""
    u = np.clip((theta - PRIOR_LO) / (PRIOR_HI - PRIOR_LO), 1e-6, 1 - 1e-6)
    return np.log(u) - np.log1p(-u)


def from_unbounded(z):
    u = 1.0 / (1.0 + np.exp(-z))
    return PRIOR_LO + u * (PRIOR_HI - PRIOR_LO)


# ==============================================================================
# Network
# ==============================================================================
def _torch():
    import torch
    import zuko
    return torch, zuko


def build_network(context=48, hidden=64, transforms=5, flow_hidden=(128, 128)):
    torch, zuko = _torch()
    nn = torch.nn

    class Summary(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Sequential(nn.Linear(SEQ_FEATURES, hidden), nn.GELU())
            self.gru = nn.GRU(hidden, hidden, batch_first=True)
            self.head = nn.Sequential(
                nn.Linear(hidden + STATIC_FEATURES, 128), nn.GELU(),
                nn.Linear(128, context))

        def forward(self, seq, lengths, static):
            x = self.embed(seq)
            packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False)
            _, h = self.gru(packed)
            return self.head(torch.cat([h[-1], static], dim=-1))

    class NPE(nn.Module):
        def __init__(self):
            super().__init__()
            self.summary = Summary()
            self.flow = zuko.flows.NSF(features=3, context=context,
                                       transforms=transforms,
                                       hidden_features=list(flow_hidden))

        def log_prob(self, z, seq, lengths, static):
            return self.flow(self.summary(seq, lengths, static)).log_prob(z)

        def sample(self, n, seq, lengths, static):
            return self.flow(self.summary(seq, lengths, static)).sample((n,))

    return NPE()


@dataclass
class TrainReport:
    n_train: int
    epochs: int
    best_val_nll: float
    seconds: float


def train(n_train=200_000, epochs=40, batch=512, lr=1e-3, seed=0, save=True,
          verbose=True, patience=6):
    """Simulate, then fit the summary network and flow jointly (maximum
    likelihood on the unbounded parameters). Early-stops on a held-out 5 %."""
    torch, _ = _torch()
    torch.manual_seed(seed)
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    t0 = time.perf_counter()
    if verbose:
        print(f"Simulating {n_train} virtual patients...")
    d = simulate_dataset(n_train, seed=seed)
    seq, lengths, static = encode(d["times"], d["egfr"], d["mask"],
                                  d["a1c"], d["uacr0"], d["sbp"])
    z = to_unbounded(d["theta"]).astype(np.float32)

    n_val = max(1000, n_train // 20)
    T = [torch.from_numpy(a) for a in (z, seq, lengths, static)]
    tr = [a[n_val:] for a in T]
    va = [a[:n_val] for a in T]

    net = build_network()
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-5)
    steps = epochs * int(np.ceil(len(tr[0]) / batch))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps)

    best, best_state, bad = np.inf, None, 0
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(len(tr[0]))
        for i in range(0, len(perm), batch):
            idx = perm[i:i + batch]
            loss = -net.log_prob(*(a[idx] for a in tr)).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step()
            sched.step()
        net.eval()
        with torch.no_grad():
            val = float(-net.log_prob(*va).mean())
        if verbose:
            print(f"  epoch {ep + 1:2d}  val NLL {val:7.4f}")
        if val < best - 1e-4:
            best, bad = val, 0
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    net.load_state_dict(best_state)
    net.eval()
    rep = TrainReport(n_train, ep + 1, best, time.perf_counter() - t0)
    if save:
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        torch.save({"state": net.state_dict(), "report": rep.__dict__}, MODEL_PATH)
    if verbose:
        print(f"Trained in {rep.seconds:.0f} s (best val NLL {best:.4f})")
    return net, rep


def load(path=MODEL_PATH):
    torch, _ = _torch()
    ck = torch.load(path, map_location="cpu", weights_only=False)
    net = build_network()
    net.load_state_dict(ck["state"])
    net.eval()
    return net


# ==============================================================================
# Inference
# ==============================================================================
def posterior_samples(net, t, egfr, a1c, uacr0, sbp, n=500, seed=0):
    """Draws n posterior samples of (q, log_scale, eGFR0) for ONE patient."""
    torch, _ = _torch()
    t = np.asarray(t, float)
    e = np.asarray(egfr, float)
    k = len(t)
    if k > MAX_VISITS:              # keep the most recent visits
        t, e, k = t[-MAX_VISITS:] - t[-MAX_VISITS], e[-MAX_VISITS:], MAX_VISITS
    times = np.full((1, MAX_VISITS), t[-1])
    times[0, :k] = t - t[0]
    vals = np.zeros((1, MAX_VISITS))
    vals[0, :k] = e
    mask = np.arange(MAX_VISITS)[None, :] < k
    seq, lengths, static = encode(times, vals, mask, a1c, uacr0, sbp)
    torch.manual_seed(seed)
    with torch.no_grad():
        z = net.sample(n, torch.from_numpy(seq), torch.from_numpy(lengths),
                       torch.from_numpy(static))[:, 0, :].numpy()
    return from_unbounded(z.astype(float))


def predictive(theta, a1c, uacr0, sbp, t_future):
    """Latent eGFR at `t_future` (years since the FIRST visit) for each draw."""
    return fast_sim.simulate_egfr(theta[:, 0], np.exp(theta[:, 1]), theta[:, 2],
                                  a1c, uacr0, sbp, np.atleast_1d(t_future))


# ==============================================================================
# Exact reference: brute-force posterior on a grid
# ==============================================================================
def grid_posterior(t, egfr, a1c, uacr0, sbp, nq=36, ns=48, ne=33,
                   noise_sd=NOISE_SD, t_future=None):
    """
    Exact posterior (up to grid spacing) under the NPE prior and likelihood.

    eGFR0 is gridded on +/- 4 measurement SDs around the first observation (the
    likelihood is negligible outside), clipped to the prior. Returns the grid
    points (G, 3), their normalized weights (G,), and, if `t_future` is given,
    the latent eGFR at those times for every grid point (G, len(t_future)).
    """
    t = np.asarray(t, float) - float(t[0])
    e = np.asarray(egfr, float)
    qg = np.linspace(PRIOR_LO[0], PRIOR_HI[0], nq)
    sg = np.linspace(PRIOR_LO[1], PRIOR_HI[1], ns)
    lo = max(PRIOR_LO[2], e[0] - 4 * noise_sd)
    hi = min(PRIOR_HI[2], e[0] + 4 * noise_sd)
    eg = np.linspace(lo, hi, ne)
    Q, S, E = np.meshgrid(qg, sg, eg, indexing="ij")
    pts = np.stack([Q.ravel(), S.ravel(), E.ravel()], axis=1)

    tq = t if t_future is None else np.concatenate([t, np.atleast_1d(t_future)])
    sim = fast_sim.simulate_egfr(pts[:, 0], np.exp(pts[:, 1]), pts[:, 2],
                                 a1c, uacr0, sbp, tq)
    pred = sim[:, :len(t)]
    loglik = -0.5 * np.sum(((pred - e) / noise_sd) ** 2, axis=1)
    w = np.exp(loglik - loglik.max())
    w /= w.sum()
    fut = None if t_future is None else sim[:, len(t):]
    cell = np.array([qg[1] - qg[0], sg[1] - sg[0], eg[1] - eg[0] if ne > 1 else 0.0])
    return pts, w, fut, cell


def grid_samples(pts, w, cell, n, rng):
    """Draws from the grid posterior, jittered uniformly within each cell."""
    idx = rng.choice(len(w), size=n, p=w)
    jit = (rng.uniform(-0.5, 0.5, (n, 3)) * cell)
    return np.clip(pts[idx] + jit, PRIOR_LO, PRIOR_HI), idx
