"""u55_minimax_check.py -- the minimax theorem beyond its condition, computed directly.

Theorem 1 needs kappa <= 1/||z||_inf, which keeps both extreme reweightings 1 +/- kappa z
nonnegative.  At the budgets the deployments of this paper exhibit that condition fails,
so this script computes the worst case over the budget set exactly, with r >= 0
imposed, and asks whether the benchmark's choice is still the minimax choice.

For each comparison, the budget is the deployment's own mismatch sd_mu(r).  The upper
end of Cov_mu(d, r) over {r >= 0, E_mu r = 1, sd_mu(r) <= kappa} is attained at
r = (alpha + beta a)_+ with a the centred loss difference; a bisection on the threshold
finds it.  The suite refits the ten learners of u20_visible.py; the leaderboard reuses
the 77 models and eight deployments of u50_llm_entropy.py.  Writes
results/universal/minimax_check.csv, one row per comparison.
"""
import warnings, json, itertools; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.ensemble import (RandomForestRegressor, ExtraTreesRegressor,
    HistGradientBoostingRegressor, AdaBoostRegressor, BaggingRegressor)
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / 'results' / 'universal'
IDS = json.load(open(R / 'ctr23_ids.json'))
NMAX, SEEDS = 4000, [0, 1, 2]
NAME = dict(zip(pd.read_csv(R / 'scale_r2.csv').data_id.astype(str),
                pd.read_csv(R / 'scale_r2.csv').dataset))


def learners(s):
    """The ten of u7_scale.py, at library defaults."""
    return [RandomForestRegressor(100, random_state=s, n_jobs=-1),
            ExtraTreesRegressor(100, random_state=s, n_jobs=-1),
            HistGradientBoostingRegressor(random_state=s),
            AdaBoostRegressor(random_state=s),
            BaggingRegressor(n_estimators=30, random_state=s, n_jobs=-1),
            DecisionTreeRegressor(max_depth=5, random_state=s),
            DecisionTreeRegressor(max_depth=12, random_state=s),
            make_pipeline(StandardScaler(), LinearRegression()),
            make_pipeline(StandardScaler(), Ridge(alpha=10.0)),
            make_pipeline(StandardScaler(), KNeighborsRegressor(10))]


def deployments(y):
    """The five of u7_scale.py; the relative weight needs y away from zero."""
    lo, hi = y.min(), y.max()
    out = {'top decile of outcome':    (y >= np.quantile(y, .90)).astype(float),
           'top quartile of outcome':  (y >= np.quantile(y, .75)).astype(float),
           'bottom decile of outcome': (y <= np.quantile(y, .10)).astype(float),
           'cost rising in outcome':   (y - lo) / (hi - lo) if hi > lo else None}
    out['relative-error weighted'] = (1.0 / np.maximum(np.abs(y), 1e-6) ** 2
                                      if np.all(np.abs(y) > 1e-6) else None)
    # A weight that is the same on every instance is the benchmark itself, not a
    # deployment: on an outcome that is mostly zero, the top quartile selects all of it.
    return {k: v for k, v in out.items()
            if v is not None and v.sum() > 0 and np.ptp(v) > 0}




def sup_cov(a, mu, k):
    """Largest E_mu[a r] over r >= 0, E_mu r = 1, sd_mu(r) <= k, for a of mu-mean zero."""
    def er2(t):
        p = np.clip(a - t, 0, None); m = mu @ p
        return (mu @ p ** 2) / m ** 2, p / m
    sa = np.sqrt(mu @ a ** 2)
    lo, hi = a.min() - 10 * sa / max(k, 1e-9) - 1.0, a.max() - 1e-12 * max(sa, 1)
    while er2(lo)[0] > 1 + k * k:          # move left until the budget is slack
        lo = a.min() - 4 * (a.min() - lo)
    for _ in range(200):
        mid = (lo + hi) / 2
        if er2(mid)[0] > 1 + k * k: hi = mid
        else: lo = mid
    return float(mu @ (a * er2(lo)[1]))


def check(setting, L, mu, deps, rows):
    """One row per comparison and deployment, oriented so f is the benchmark's choice."""
    for dep, nu in deps.items():
        k = float(np.sqrt(mu @ (nu / mu - 1) ** 2))
        for i, j in itertools.combinations(range(len(L)), 2):
            dd = L[i] - L[j]; e = mu @ dd; s = np.sqrt(mu @ (dd - e) ** 2)
            if s == 0 or e == 0:
                continue
            if e > 0:
                dd, e = -dd, -e
            m, a = -e, dd - e
            up, low = sup_cov(a, mu, k), -sup_cov(-a, mu, k)
            rows.append(dict(setting=setting, deployment=dep, kappa=k, kfg=m / s,
                             inv_zinf=s / np.abs(a).max(),
                             reg_bench=max(0.0, up - m), reg_other=max(0.0, m - low)))


rows = []
for did in IDS:
    for seed in SEEDS:
        try:
            X, y = fetch_openml(data_id=int(did), return_X_y=True, as_frame=True)
            X = SimpleImputer(strategy='median').fit_transform(X.select_dtypes('number'))
            y = np.asarray(y, float)
            if len(y) > NMAX:
                X, _, y, _ = train_test_split(X, y, train_size=NMAX, random_state=seed)
            Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.4, random_state=seed)
            L = np.array([(m.fit(Xtr, ytr).predict(Xte) - yte) ** 2 for m in learners(seed)])
        except Exception as e:
            print(f'  {did} seed {seed}: {type(e).__name__}', flush=True); continue
        mu = np.full(len(yte), 1 / len(yte))
        check('suite', L, mu, {k: w / w.sum() for k, w in deployments(yte).items()}, rows)

# the leaderboard, filtered exactly as in u50_llm_entropy.py
d = pd.read_csv(R / 'llm_raw.csv')

AGG = {'leaderboard_bbh', 'leaderboard_gpqa', 'leaderboard_math_hard',
       'leaderboard_musr', 'leaderboard'}
d = d[~d.task.isin(AGG)]


def group(t):
    for g in ['bbh', 'math', 'gpqa', 'musr', 'mmlu_pro', 'ifeval']:
        if t.startswith('leaderboard_' + g):
            return g
    return 'other'


d['group'] = d.task.map(group)
d = d[d.group != 'other']
W = d.pivot(index='model', columns='task', values='score')
tasks = sorted(t for t in W.columns if W[t].notna().sum() >= .8 * len(W))
models = sorted(W[tasks].dropna(axis=0).index)
S = d.pivot(index='model', columns='task', values='score').loc[models, tasks]
L = 1.0 - S.values                       # per-task error, the per-instance score
G = d.drop_duplicates('task').set_index('task')['group']
N = d.drop_duplicates('task').set_index('task')['n'].astype(float).loc[tasks].values

# Definition 1: centre by model, standardise by instance, read the spectrum.
D = L - L.mean(0, keepdims=True)
D = D / (D.std(0, keepdims=True) + 1e-12)
_, s, Vt = np.linalg.svd(D, full_matrices=False)
s = s[s > 1e-10]
u = s / s.sum()
h = float(-(u * np.log2(u)).sum())
energy = s ** 2 / (s ** 2).sum()

# the leaderboard's own measure, and the eight deployments of u34
gsz = G.loc[tasks].value_counts()
mu = np.array([1.0 / (len(gsz) * gsz[G[t]]) for t in tasks]); mu /= mu.sum()
DEPLOY = {'by question count': N / N.sum()}
for g in sorted(gsz.index):
    w = np.array([1.0 if G[t] == g else 0.0 for t in tasks])
    DEPLOY[f'the {g} group alone'] = w / w.sum()
hard = (L.mean(0) >= np.quantile(L.mean(0), .75)).astype(float)
DEPLOY['the hardest quartile of tasks'] = hard / hard.sum()

check('leaderboard', L, mu, DEPLOY, rows)

t = pd.DataFrame(rows)
t.to_csv(R / 'minimax_check.csv', index=False)
for st, g in t.groupby('setting'):
    ex = g[g.kfg < g.kappa]
    print(f'{st}: {len(g):,} comparisons, condition met at the deployment budget on '
          f'{100 * (g.kappa <= g.inv_zinf).mean():.1f} percent; median 1/||z|| {g.inv_zinf.median():.3f}, '
          f'median budget {g.kappa.median():.2f}')
    print(f'   exposed {len(ex):,}: benchmark choice minimax on '
          f'{100 * (ex.reg_bench <= ex.reg_other + 1e-12).mean():.1f} percent')
