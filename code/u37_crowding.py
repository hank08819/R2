"""u37_crowding.py -- does a benchmark settle less as the field gets larger?

The budget up to which the benchmark's choice is guaranteed is a minimum over
rivals of margin divided by dispersion.  Adding candidates can only lower a
minimum, and if margins have positive density at zero the minimum of K-1 draws
falls like 1/K.  The prediction is that a benchmark's certified budget shrinks
inversely in the size of the field.  This tests it by drawing model sets of each
size from the two settings the paper measures.
"""
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, itertools
from pathlib import Path

R = Path(__file__).resolve().parents[1] / 'results' / 'universal'
rng = np.random.default_rng(3)
DRAWS = 400


def kstar(E, S, idx):
    """Certified budget for the benchmark's choice within the given model set."""
    sub = np.ix_(idx, idx)
    e, s = E[sub], S[sub]
    r = e.sum(1)
    f = int(np.argmin(r))
    best = np.inf
    for j in range(len(idx)):
        if j == f or s[f, j] <= 0:
            continue
        best = min(best, (e[j, f]) / s[f, j])
    return best


def tables(g, left, right, K):
    ix = {m: i for i, m in enumerate(sorted(set(g[left]) | set(g[right])))}
    E = np.zeros((K, K)); S = np.zeros((K, K))
    for _, x in g.iterrows():
        i, j = ix[x[left]], ix[x[right]]
        E[i, j], E[j, i] = x.e_mu, -x.e_mu
        S[i, j] = S[j, i] = x.sd_d
    return E, S


rows = []
# ---- the regression suite: 10 candidates per dataset and split
d = pd.read_csv(R / 'scale_r2.csv')
d['left'] = d.pair.str.split('-').str[0]; d['right'] = d.pair.str.split('-').str[1]
for (ds, se), g in d.groupby(['dataset', 'seed']):
    K = len(set(g.left) | set(g.right))
    E, S = tables(g, 'left', 'right', K)
    for k in [2, 3, 4, 6, 8, 10]:
        if k > K:
            continue
        v = [kstar(E, S, rng.choice(K, k, replace=False)) for _ in range(40)]
        v = [x for x in v if np.isfinite(x)]
        if v:
            rows.append(dict(setting='regression suite', K=k, kstar=float(np.median(v))))

# ---- the leaderboard: 77 candidates, one deployment is enough for kappa*
l = pd.read_csv(R / 'llm_pairs.csv')
l = l[l.deployment == l.deployment.iloc[0]].copy()
l['left'] = l.pair.str.split('|').str[0]; l['right'] = l.pair.str.split('|').str[1]
K = len(set(l.left) | set(l.right))
E, S = tables(l, 'left', 'right', K)
for k in [2, 3, 5, 8, 12, 20, 35, 55, 77]:
    v = [kstar(E, S, rng.choice(K, k, replace=False)) for _ in range(DRAWS)]
    v = [x for x in v if np.isfinite(x)]
    rows.append(dict(setting='language models', K=k, kstar=float(np.median(v))))

t = pd.DataFrame(rows).groupby(['setting', 'K']).kstar.median().reset_index()
t.to_csv(R / 'crowding.csv', index=False)
for s, g in t.groupby('setting'):
    lg = np.polyfit(np.log(g.K), np.log(g.kstar), 1)
    print(f'\n{s}:  slope of log(kappa*) on log(K) = {lg[0]:+.2f}   '
          f'(the prediction is -1)')
    print('   K      median kappa*   K * kappa*')
    for _, x in g.iterrows():
        print(f'  {int(x.K):3d} {x.kstar:15.4f} {x.K*x.kstar:12.3f}')


# ---------------- is the exponent a law or a fit? measure alpha directly ----
print('\n' + '=' * 66)
print('the exponent predicted from the margin distribution, not fitted to K')


def ratios(g, left, right):
    """margin over dispersion, for every ordered pair, on the benchmark."""
    ix = {m: i for i, m in enumerate(sorted(set(g[left]) | set(g[right])))}
    K = len(ix); E = np.zeros((K, K)); S = np.zeros((K, K))
    for _, x in g.iterrows():
        i, j = ix[x[left]], ix[x[right]]
        E[i, j], E[j, i] = x.e_mu, -x.e_mu
        S[i, j] = S[j, i] = x.sd_d
    f = int(np.argmin(E.sum(1)))
    return np.array([E[j, f] / S[f, j] for j in range(K)
                     if j != f and S[f, j] > 0])


def alpha_of(x, q=.30):
    """If P(X<=x) ~ c x^(1+a) near zero, the log-log slope of the lower tail is 1+a."""
    x = np.sort(x[x > 0])
    m = max(30, int(q * len(x)))
    x = x[:m]
    F = (np.arange(1, m + 1) - .5) / len(np.sort(x))
    F = (np.arange(1, m + 1) - .5) / m * q
    b = np.polyfit(np.log(x), np.log(F), 1)[0]
    return b - 1.0, b


out = []
for name, g, L, Rt in [('regression suite', d, 'left', 'right'),
                       ('language models', l, 'left', 'right')]:
    if name == 'regression suite':
        pool = np.concatenate([ratios(h, L, Rt) for _, h in g.groupby(['dataset', 'seed'])])
    else:
        pool = ratios(g, L, Rt)
    a, b = alpha_of(pool)
    pred = -1.0 / (1.0 + a)
    obs = np.polyfit(np.log(t[t.setting == name].K),
                     np.log(t[t.setting == name].kstar), 1)[0]
    print(f'\n  {name}: {len(pool)} margin ratios')
    print(f'    lower-tail exponent 1+alpha = {b:.2f}  ->  alpha = {a:.2f}')
    print(f'    predicted slope -1/(1+alpha) = {pred:+.2f}')
    print(f'    observed slope               = {obs:+.2f}')
    print(f'    agreement                    = {abs(pred-obs):.2f}')
    out.append(dict(setting=name, n_ratios=len(pool), alpha=a, predicted=pred,
                    observed=obs, gap=abs(pred - obs)))

pd.DataFrame(out).to_csv(R / 'crowding_exponent.csv', index=False)
