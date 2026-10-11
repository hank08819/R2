"""u56_holdout_certificate.py -- the certificate on data it was not computed from.

On a single test set the certificate cannot fail: a pair whose benchmark gap exceeds
sd_mu(r) sd_mu(d) cannot reverse, by the Cauchy-Schwarz inequality.  Here each test set
of the suite is split in two at random.  The first half decides which comparisons are
certified, and the second half says whether they reverse under the same deployment.
The deployment weights are computed once on the whole test set and restricted to each
half, so both halves face the same deployment.  Writes
results/universal/holdout_certificate.csv, one row per comparison.
"""
import warnings, json, itertools, csv; warnings.filterwarnings('ignore')
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
        perm = np.random.default_rng(seed).permutation(len(yte))
        A, B = perm[: len(perm) // 2], perm[len(perm) // 2:]
        for dep, w in deployments(yte).items():
            if w[A].sum() <= 0 or w[B].sum() <= 0:
                continue
            muA, muB = np.full(len(A), 1 / len(A)), np.full(len(B), 1 / len(B))
            nuA, nuB = w[A] / w[A].sum(), w[B] / w[B].sum()
            sdrA = float(np.sqrt(muA @ (nuA / muA - 1) ** 2))
            for i, j in itertools.combinations(range(len(L)), 2):
                dA, dB = L[i, A] - L[j, A], L[i, B] - L[j, B]
                eA = muA @ dA; sA = np.sqrt(muA @ (dA - eA) ** 2)
                if eA == 0 or sA == 0:
                    continue
                rows.append(dict(dataset=NAME.get(str(did), str(did)), seed=seed,
                                 deployment=dep, ratio=abs(eA) / sA / sdrA,
                                 rev_in=int(np.sign(eA) != np.sign(nuA @ dA)),
                                 rev_out=int(np.sign(eA) != np.sign(nuB @ dB)),
                                 bench_flip=int(np.sign(eA) != np.sign(muB @ dB))))
        print(did, seed, len(rows), flush=True)

t = pd.DataFrame(rows)
t.to_csv(R / 'holdout_certificate.csv', index=False)
c = t.ratio >= 1
print(f'{len(t):,} comparisons; certified on the first half {c.sum():,}')
print(f'  reverse on the first half {t.rev_in[c].sum()}, on the second half {t.rev_out[c].sum()}')
print(f'  exposed: {100 * t.rev_out[~c].mean():.1f} percent reverse on the second half')
for th in [.7, .5, .3]:
    m = t.ratio >= th
    print(f'  ratio >= {th}: {m.sum():,} comparisons, {t.rev_out[m].sum()} reverse on the second half')
