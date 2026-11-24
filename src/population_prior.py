"""
================================================================================
EMPIRICAL-BAYES POPULATION PRIOR FOR THE AMORTIZED POSTERIOR
================================================================================
The neural posterior (npe_posterior.py) is trained under a deliberately broad,
uniform prior. For a parameter the data barely constrain -- the collapse exponent
q -- the posterior then inherits that broad prior, and the forecast averages over
collapse regimes the target population may rarely show. On MIMIC-IV this shows up
as a forecast bias toward too-fast decline that grows with the horizon, exactly
what Round 17 predicts (whole-cohort q ~ 0.9, far below the uniform prior's 2.15).

The remedy is to LEARN the population distribution of (q, log injury rate) from
a set of training patients and use it as the prior for new patients -- a
hierarchical model -- without retraining the network:

  * The NPE gives, for each patient i, draws theta_ij from p_u(theta | x_i), the
    posterior under the uniform prior p_u.
  * Under a new prior pi_phi, the marginal likelihood of x_i is, up to a constant,
        p_phi(x_i) / p_u(x_i) = E_{theta ~ p_u(.|x_i)} [ pi_phi(theta) / p_u(theta) ].
  * phi is fitted by maximizing sum_i log of that Monte-Carlo average over the
    training patients (empirical Bayes / prior deconvolution).
  * For a new patient, the posterior under pi_phi is obtained by importance
    reweighting the same NPE draws with w_j proportional to pi_phi(theta_j)/p_u(theta_j).

The prior family is a mixture of K components, each the product of a Beta
distribution for q on the prior box and a truncated normal for the log injury
rate. K = 2 lets it represent the two regimes Round 17 found. The baseline eGFR
keeps its uniform prior (the data pin it down, so it does not matter).
================================================================================
"""
from dataclasses import dataclass, field

import numpy as np
from scipy import optimize, special, stats

import npe_posterior as npe

Q_LO, Q_HI = npe.PRIOR_LO[0], npe.PRIOR_HI[0]
S_LO, S_HI = npe.PRIOR_LO[1], npe.PRIOR_HI[1]


@dataclass
class PopulationPrior:
    weights: np.ndarray            # (K,)
    q_ab: np.ndarray               # (K, 2) Beta parameters for (q - Q_LO)/(Q_HI - Q_LO)
    s_mu_sd: np.ndarray            # (K, 2) truncated-normal mean and sd of log rate
    train_loglik: float = np.nan
    n_train: int = 0
    extra: dict = field(default_factory=dict)

    def component_logpdf(self, q, log_s):
        """Per-component log densities (K, ...) over (q, log_s), weights included."""
        u = np.clip((np.asarray(q, float) - Q_LO) / (Q_HI - Q_LO), 1e-9, 1 - 1e-9)
        ls = np.asarray(log_s, float)
        out = []
        for k in range(len(self.weights)):
            a, b = self.q_ab[k]
            mu, sd = self.s_mu_sd[k]
            lq = ((a - 1) * np.log(u) + (b - 1) * np.log1p(-u)
                  - special.betaln(a, b) - np.log(Q_HI - Q_LO))
            z = (ls - mu) / sd
            mass = special.ndtr((S_HI - mu) / sd) - special.ndtr((S_LO - mu) / sd)
            lsn = -0.5 * z * z - np.log(sd * np.sqrt(2 * np.pi)) - np.log(max(mass, 1e-300))
            out.append(np.log(self.weights[k]) + lq + lsn)
        return np.stack(out)

    def logpdf(self, q, log_s):
        """Log density over (q, log_s) on the prior box (eGFR0 stays uniform)."""
        return special.logsumexp(self.component_logpdf(q, log_s), axis=0)

    def summary(self):
        """Component means in natural units (shareable aggregate)."""
        rows = []
        for k in range(len(self.weights)):
            a, b = self.q_ab[k]
            mu, sd = self.s_mu_sd[k]
            ta, tb = (S_LO - mu) / sd, (S_HI - mu) / sd
            rows.append(dict(
                weight=float(self.weights[k]),
                q_mean=float(Q_LO + (Q_HI - Q_LO) * a / (a + b)),
                q_sd=float((Q_HI - Q_LO) * np.sqrt(a * b / ((a + b) ** 2 * (a + b + 1)))),
                rate_median=float(np.exp(stats.truncnorm.median(ta, tb, loc=mu, scale=sd))),
                log_rate_sd=float(sd)))
        return rows


def _uniform_logpdf():
    return -np.log(Q_HI - Q_LO) - np.log(S_HI - S_LO)


def _beta_mle(elog_u, elog_1mu, start=(2.0, 2.0)):
    """Weighted Beta MLE from its sufficient statistics E[log u], E[log(1-u)].
    a, b >= 1 keeps every component unimodal (a U-shaped component would absorb
    both regimes); a, b <= 60 bounds the sharpness (q SD >= ~0.12), a mild
    regularization against collapsing a component onto a spike."""
    def nll(x):
        a, b = x
        return -((a - 1) * elog_u + (b - 1) * elog_1mu - special.betaln(a, b))
    res = optimize.minimize(nll, np.maximum(start, 1.0), method="L-BFGS-B",
                            bounds=[(1.0, 60.0), (1.0, 60.0)])
    return tuple(res.x)


def _truncnorm_mle(ex, ex2, start):
    """Weighted MLE of a normal truncated to [S_LO, S_HI] from E[x], E[x^2]."""
    def nll(x):
        mu, lsd = x
        sd = np.exp(lsd)
        mass = special.ndtr((S_HI - mu) / sd) - special.ndtr((S_LO - mu) / sd)
        return 0.5 * (ex2 - 2 * mu * ex + mu * mu) / sd ** 2 + lsd + np.log(max(mass, 1e-300))
    res = optimize.minimize(nll, [start[0], np.log(start[1])], method="L-BFGS-B",
                            bounds=[(S_LO - 2.0, S_HI + 2.0), (np.log(0.03), np.log(5.0))])
    return float(res.x[0]), float(np.exp(res.x[1]))


def fit(draws, K=2, seed=0, n_iter=300, tol=1e-6):
    """
    Empirical-Bayes fit by EM over the NPE draws.

    draws: list of arrays (n_draws, >=2) of NPE posterior draws (q, log_s, ...),
    one per training patient. E-step: each draw of each patient gets the weight
    pi(theta)/p_u(theta) normalized within the patient (its posterior under the
    current prior), split across components by responsibility. M-step: component
    weights from the summed responsibilities; Beta (q) and normal (log rate)
    parameters by weighted moments. Monotone in the marginal likelihood up to the
    moment approximation; converges in tens of iterations.
    """
    Q = np.stack([d[:, 0] for d in draws]).astype(float)       # (N, J)
    S = np.stack([d[:, 1] for d in draws]).astype(float)
    N, J = Q.shape
    U = np.clip((Q - Q_LO) / (Q_HI - Q_LO), 1e-6, 1 - 1e-6)
    LU, L1U, S2 = np.log(U), np.log1p(-U), S * S
    rng = np.random.default_rng(seed)
    # start: components centred at spread quantiles of the pooled q draws
    centres = np.quantile(U, np.linspace(0.15, 0.85, K))
    pr = PopulationPrior(np.full(K, 1.0 / K),
                         np.array([(1 + 8 * c, 1 + 8 * (1 - c)) for c in centres]),
                         np.column_stack([np.full(K, np.mean(S)) + rng.normal(0, 0.05, K),
                                          np.full(K, max(np.std(S), 0.1))]))
    lu = _uniform_logpdf()
    prev = -np.inf
    for it in range(n_iter):
        comp = pr.component_logpdf(Q, S) - lu                  # (K, N, J)
        tot = special.logsumexp(comp, axis=0)                  # (N, J)
        ll_i = special.logsumexp(tot, axis=1) - np.log(J)      # (N,)
        ll = float(ll_i.sum())
        if it > 0 and ll - prev < tol * N:
            break
        prev = ll
        # E-step: each patient's posterior under the current prior, as weights
        # over its own draws, split across components (sums to 1 per patient)
        R = np.exp(comp - (ll_i + np.log(J))[None, :, None])  # (K, N, J)
        wk = R.sum(axis=(1, 2))
        weights = np.maximum(wk / N, 1e-4)
        weights /= weights.sum()
        # M-step: weighted maximum likelihood per component (monotone EM)
        q_ab, s_ms = [], []
        for k in range(K):
            r = R[k] / max(wk[k], 1e-12)
            q_ab.append(_beta_mle(np.sum(r * LU), np.sum(r * L1U), pr.q_ab[k]))
            s_ms.append(_truncnorm_mle(np.sum(r * S), np.sum(r * S2), pr.s_mu_sd[k]))
        pr = PopulationPrior(weights, np.array(q_ab), np.array(s_ms))
    order = np.argsort(Q_LO + (Q_HI - Q_LO) * pr.q_ab[:, 0] / pr.q_ab.sum(1))
    return PopulationPrior(pr.weights[order], pr.q_ab[order], pr.s_mu_sd[order],
                           train_loglik=ll, n_train=N, extra=dict(iterations=it + 1))


def reweight(draws, prior):
    """Self-normalized importance weights turning uniform-prior draws into draws
    under `prior`. Returns (weights, effective sample size fraction)."""
    lw = prior.logpdf(draws[:, 0], draws[:, 1])
    lw -= lw.max()
    w = np.exp(lw)
    w /= w.sum()
    ess = 1.0 / np.sum(w ** 2)
    return w, float(ess / len(w))


def resample(draws, prior, n=None, rng=None):
    """Draws under `prior` by multinomial resampling of the reweighted draws."""
    rng = np.random.default_rng(0) if rng is None else rng
    w, ess = reweight(draws, prior)
    idx = rng.choice(len(draws), size=n or len(draws), p=w)
    return draws[idx], ess
