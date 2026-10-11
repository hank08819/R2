"""u5_regret.py -- the controlled grid of Theorem 1 and the bound of Theorem 2.

Part one varies the mismatch sd(r) and the correlation rho independently and
records how often the criterion orders a pair opposite to the deployment loss.
Part two draws a set of candidate models, lets the criterion pick one, and
compares the loss that choice incurs against the bound D(P) * sd(r).
"""
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, itertools
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / 'results' / 'universal'
OUT.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(11)


def make_r(n, target_sd, mu):
    """Lognormal density ratio with mean one under mu and the stated dispersion."""
    s2 = np.log(1 + target_sd ** 2)
    r = np.exp(np.sqrt(s2) * rng.standard_normal(n) - s2 / 2)
    return r / (mu @ r)


# ------------------------------------------------------- controlled grid ---
SD = [0.3, 0.8, 1.5, 3.0]
RHO = [0.0, 0.1, 0.3, 0.6]
rows = []
for sd_t in SD:
    for rho_t in RHO:
        inv = 0
        for _ in range(3000):
            n = 400; mu = np.full(n, 1 / n)
            r = make_r(n, sd_t, mu)
            rc = (r - 1) / max(1e-9, float(np.sqrt(mu @ (r - 1) ** 2)))
            e = rng.standard_normal(n); e = e - mu @ e
            d = rho_t * rc + np.sqrt(max(1e-9, 1 - rho_t ** 2)) * e + rng.normal(0, .3)
            e_mu, e_nu = float(mu @ d), float(mu @ (d * r))
            inv += int(abs(e_mu) > 1e-12 and np.sign(e_mu) != np.sign(e_nu))
        rows.append(dict(sd_r=sd_t, rho=rho_t, inv=100 * inv / 3000))
grid = pd.DataFrame(rows); grid.to_csv(OUT / 'controlled_grid.csv', index=False)

# ----------------------------------------------------------- regret bound ---
recs = []
for _ in range(3000):
    n, K = 400, 6
    mu = np.full(n, 1 / n)
    sd_t = float(rng.uniform(.2, 3.0))
    r = make_r(n, sd_t, mu)
    sd_r = float(np.sqrt(mu @ (r - 1) ** 2))
    base = rng.standard_normal(n)
    L = {k: base + rng.normal(0, rng.uniform(.2, 1.5), n) + rng.normal(0, .3)
         for k in range(K)}                          # per-instance losses
    S = {k: float(mu @ v) for k, v in L.items()}     # criterion: benchmark risk
    D = {k: float(mu @ (v * r)) for k, v in L.items()}   # deployment risk
    fstar = min(S, key=S.get)                        # chosen by the criterion
    gstar = min(D, key=D.get)                        # chosen by deployment
    regret = D[fstar] - D[gstar]
    diam = max(float(np.sqrt(mu @ (L[a] - L[b] - mu @ (L[a] - L[b])) ** 2))
               for a, b in itertools.combinations(range(K), 2))
    recs.append(dict(sd_r=sd_r, regret=regret, bound=diam * sd_r, diam=diam))
reg = pd.DataFrame(recs); reg.to_csv(OUT / 'regret.csv', index=False)

viol = int((reg.regret > reg.bound + 1e-12).sum())
pos = reg[reg.regret > 1e-12]
print(grid.pivot(index='sd_r', columns='rho', values='inv').round(1).to_string())
print(f'\nTheorem 2: violations {viol}/{len(reg)}   '
      f'positive-regret draws {len(pos)}   '
      f'median bound/regret {np.median(pos.bound / pos.regret):.1f}')
