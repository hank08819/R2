"""certify.py -- the diagnostic of the paper, on your own benchmark, in one pass.

Give it the per-instance losses of the models you are choosing between and the weight
your deployment implies, and it reports, for the model the benchmark picks:

  * the mismatch sd_mu(r) the stated deployment exhibits;
  * the ranking-stability radius kappa-star, the width of deployment the ranking
    survives;
  * which rivals are certified, meaning no deployment inside that width can overtake
    them, and which are open;
  * for each open rival, the two groups of instances to show a practitioner, the tenth
    where the choice does best and the tenth where the rival does, and the sign that
    answer implies (one instance per end recovers the sign no better than a coin);
  * the deployment regret test: a 95% upper bound on the regret ratio of the
    benchmark's choice (the excess deployment loss per unit of its benchmark risk),
    valid although the choice was made on the same benchmark, and a verdict against a
    tolerance fixed in advance: certified, exceeds, or undetermined.

Nothing is refitted and no deployment data are needed.  The losses are the ones the
benchmark already produced.

    from certify import certify
    out = certify(L, w, names)          # L: (K, n) losses, w: (n,) deployment weight
    print(out.report())

Running the file demonstrates it on a constructed example.
"""
from dataclasses import dataclass, field
import numpy as np


@dataclass
class Report:
    names: list
    choice: str
    sd_r: float
    kappa_star: float
    radius: dict = field(default_factory=dict)      # rival -> stability radius
    certified: list = field(default_factory=list)
    open_pairs: list = field(default_factory=list)
    questions: list = field(default_factory=list)   # (rival, tenth where the choice does best, tenth where the rival does)
    test: dict = field(default_factory=dict)        # the deployment regret test

    def report(self):
        out = [f'benchmark choice        {self.choice}',
               f'stated mismatch sd(r)   {self.sd_r:.3f}',
               f'ranking-stability radius {self.kappa_star:.3f}',
               '']
        if self.test:
            t = self.test
            out += ['Deployment regret test',
                    f'  verdict at a tolerance of {100 * t["tol"]:.0f}% regret: {t["verdict"]}',
                    f'  regret ratio: estimate {100 * t["rr_hat"]:.1f}%, '
                    f'95% upper bound {100 * t["rr_ucb"]:.1f}% of its benchmark risk',
                    f'  effective sample size of the weight                 {t["neff"]:.0f}'
                    + ('   (below 30: treat the test with caution)' if t['neff'] < 30 else ''),
                    '']
        if self.kappa_star >= self.sd_r:
            out.append('The ranking survives every deployment of this width: no rival '
                       'can overtake the\nbenchmark choice. No question needs asking.')
            return '\n'.join(out)
        out.append(f'{len(self.certified)} of {len(self.names) - 1} rivals are certified '
                   f'at this width; {len(self.open_pairs)} are open.')
        out.append('')
        out.append('Open comparisons, and the question to ask about each:')
        for rival, hi, lo in self.questions:
            out.append(f'  against {rival}: of the {len(hi)} instances where the choice does best '
                       f'(e.g. {", ".join(map(str, hi[:5]))}, ...) and the {len(lo)} where the rival '
                       f'does (e.g. {", ".join(map(str, lo[:5]))}, ...), which group does the '
                       f'deployment weight more heavily?')
        out.append('')
        out.append('Answering "the first" leaves the benchmark ordering; answering '
                   '"the second"\nputs that rival ahead once the correction is applied.')
        return '\n'.join(out)


def certify(L, w=None, names=None, rho_bar=0.05, tol=0.10):
    """L is (K, n) per-instance losses, w an (n,) deployment weight, names the models."""
    L = np.asarray(L, float)
    K, n = L.shape
    names = list(names) if names is not None else [f'model {i}' for i in range(K)]
    mu = np.full(n, 1.0 / n)
    if w is None:
        r = np.ones(n)
    else:
        w = np.asarray(w, float)
        if w.min() < 0:
            raise ValueError('a deployment weight cannot be negative')
        r = w / (mu @ w)
    sd_r = float(np.sqrt(mu @ (r - 1) ** 2))

    risk = L @ mu
    f = int(np.argmin(risk))
    rep = Report(names=names, choice=names[f], sd_r=sd_r, kappa_star=np.inf)
    for g in range(K):
        if g == f:
            continue
        d = L[f] - L[g]
        e = float(mu @ d)                       # negative: the choice is ahead
        sd = float(np.sqrt(mu @ (d - e) ** 2))
        kappa = abs(e) / sd if sd > 0 else np.inf
        rep.radius[names[g]] = kappa
        rep.kappa_star = min(rep.kappa_star, kappa)
        if kappa >= sd_r:
            rep.certified.append(names[g])
        else:
            rep.open_pairs.append(names[g])
            order = np.argsort(d)               # d low: the choice does better
            m = max(1, n // 10)                 # show each end as a group, its tenth
            rep.questions.append((names[g], [int(i) for i in order[:m]], [int(i) for i in order[-m:]]))
    if w is not None:
        rep.test = regret_test(L, w, tol=tol)
    return rep


def regret_test(L, w, tol=0.10, alpha=0.05, nboot=2000, seed=0):
    """Deployment regret test, from benchmark losses and a stated weight only.

    For every ordered pair (f, g), theta_fg = sum(w d) / sum(w) with d = l_f - l_g is
    the deployment gap, with influence function r (d - theta_fg), r = w / mean(w); for
    every f, m_f = mean(l_f) is the benchmark risk.  One multiplier-bootstrap critical
    value bounds all gaps from above and all risks from below at once, so the bound
    read off the benchmark's choice holds although that choice was made on the same
    sample (Supplementary Note 8).  Verdict against the tolerance tol on the regret
    ratio: 'certified' (upper bound <= tol), 'exceeds' (the two-sided band shows a
    rival ahead by more than tol), or 'undetermined'.  The weight is taken as stated;
    only benchmark sampling noise is quantified.
    """
    L = np.asarray(L, float); w = np.asarray(w, float)
    K, n = L.shape
    r = w / w.mean(); neff = float(w.sum() ** 2 / (w ** 2).sum())
    m = L.mean(1); f = int(np.argmin(m))
    I, J = np.where(~np.eye(K, dtype=bool))
    D = L[I] - L[J]
    theta = (D * r).mean(1)
    Psi = np.vstack([r * (D - theta[:, None]), -(L - m[:, None])])
    se = Psi.std(1, ddof=1) / np.sqrt(n)
    flat = se <= 1e-12 * max(1.0, float(np.abs(L).max()))     # known exactly up to rounding
    se = np.where(flat, 0.0, se)
    Zb = (np.random.default_rng(seed).standard_normal((nboot, n)) @ Psi.T) / n / np.where(flat, 1.0, se)
    Zb[:, flat] = 0.0
    c = float(np.quantile(Zb.max(1), 1 - alpha)); c2 = float(np.quantile(np.abs(Zb).max(1), 1 - alpha))
    P = len(I); row = I == f
    up = max(0.0, float((theta + c * se[:P])[row].max())); lo = float((theta - c2 * se[:P])[row].max())
    m_lo, m_up = m[f] - c * se[P + f], m[f] + c2 * se[P + f]
    rr_ucb = up / m_lo if m_lo > 0 else np.inf
    verdict = 'certified' if rr_ucb <= tol else ('exceeds' if lo > tol * m_up else 'undetermined')
    return dict(verdict=verdict, tol=tol, rr_hat=max(0.0, float(theta[row].max())) / m[f],
                rr_ucb=rr_ucb, neff=neff)


def corrected_order(L, w, names, signs, rho_bar=0.05):
    """Apply elicited signs to the benchmark ordering, as Research question 4 does."""
    L = np.asarray(L, float)
    K, n = L.shape
    mu = np.full(n, 1.0 / n)
    r = np.asarray(w, float); r = r / (mu @ r)
    sd_r = float(np.sqrt(mu @ (r - 1) ** 2))
    score = np.zeros(K)
    for i in range(K):
        for j in range(K):
            if i == j:
                continue
            d = L[i] - L[j]
            e = float(mu @ d)
            sd = float(np.sqrt(mu @ (d - e) ** 2))
            s = signs.get((names[i], names[j]), -signs.get((names[j], names[i]), 0))
            score[i] += e + s * rho_bar * sd_r * sd
    return [names[k] for k in np.argsort(score)]


if __name__ == '__main__':
    rng = np.random.default_rng(0)
    n = 400
    y = rng.normal(size=n)
    tail = y >= np.quantile(y, .90)             # the deployment acts on the top decile
    base = rng.gamma(2, .5, size=n)
    L = np.vstack([base,                                   # even everywhere
                   base * (1 - .05 * tail) + .04,          # better in the tail
                   base * (1 + .20 * tail) - .02])         # better in the bulk
    out = certify(L, w=tail.astype(float), names=['even', 'tail-strong', 'bulk-strong'])
    print(out.report())
