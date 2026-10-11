"""u7_scale.py -- how often does R-squared order two regressors the wrong way?

Runs the equal-information protocol over the OpenML-CTR23 curated regression
benchmark: every dataset in the suite, ten learners on a common split, all
forty-five pairs, five deployment measures, three splits.  The criterion is
always R-squared on the full test sample; the deployment measure is where the
prediction is actually used.
"""
import warnings, json, sys, time; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, itertools
from pathlib import Path
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.ensemble import (RandomForestRegressor, ExtraTreesRegressor,
    HistGradientBoostingRegressor, AdaBoostRegressor, BaggingRegressor)
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results' / 'universal'; OUT.mkdir(parents=True, exist_ok=True)
IDS = json.load(open(OUT / 'ctr23_ids.json'))
NMAX, SEEDS = 4000, [0, 1, 2]


def learners(s):
    return {
        'RF': RandomForestRegressor(n_estimators=100, random_state=s, n_jobs=-1),
        'ET': ExtraTreesRegressor(n_estimators=100, random_state=s, n_jobs=-1),
        'HGB': HistGradientBoostingRegressor(random_state=s),
        'AdaB': AdaBoostRegressor(random_state=s),
        'Bag': BaggingRegressor(n_estimators=30, random_state=s, n_jobs=-1),
        'Tree5': DecisionTreeRegressor(max_depth=5, random_state=s),
        'Tree12': DecisionTreeRegressor(max_depth=12, random_state=s),
        'OLS': make_pipeline(StandardScaler(), LinearRegression()),
        'Ridge': make_pipeline(StandardScaler(), Ridge(alpha=10.0)),
        'KNN': make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=10)),
    }


def deployments(y):
    """Density ratios for five ways the same predictions get used."""
    lo, hi = y.min(), y.max()
    out = {
        'top decile of outcome':    (y >= np.quantile(y, .90)).astype(float),
        'top quartile of outcome':  (y >= np.quantile(y, .75)).astype(float),
        'bottom decile of outcome': (y <= np.quantile(y, .10)).astype(float),
        'cost rising in outcome':   (y - lo) / (hi - lo) if hi > lo else None,
    }
    out['relative-error weighted'] = (1.0 / np.maximum(np.abs(y), 1e-6) ** 2
                                      if np.all(np.abs(y) > 1e-6) else None)
    # A weight that is the same on every instance is the benchmark itself, not a
    # deployment: on an outcome that is mostly zero, the top quartile selects all of it.
    return {k: v for k, v in out.items()
            if v is not None and v.sum() > 0 and np.ptp(v) > 0}


rows, skipped = [], []
for i, did in enumerate(IDS):
    t0 = time.time()
    try:
        d = fetch_openml(data_id=did, as_frame=True, parser='auto')
        X = pd.get_dummies(d.data, drop_first=True).astype(float)
        X = SimpleImputer(strategy='median').fit_transform(X)
        y = pd.to_numeric(d.target, errors='coerce').values.astype(float)
        ok = np.isfinite(y) & np.isfinite(X).all(1)
        X, y = X[ok], y[ok]
        if len(y) > NMAX:
            idx = np.random.default_rng(0).choice(len(y), NMAX, replace=False)
            X, y = X[idx], y[idx]
        if len(y) < 200 or X.shape[1] > 300:
            skipped.append((did, 'shape')); continue
    except Exception as e:
        skipped.append((did, type(e).__name__)); continue

    name = d.details.get('name', str(did))
    for s in SEEDS:
        try:
            Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.4, random_state=s)
            pred = {k: m.fit(Xtr, ytr).predict(Xte) for k, m in learners(s).items()}
        except Exception as e:
            skipped.append((did, f'fit:{type(e).__name__}')); continue
        n = len(yte); mu = np.full(n, 1 / n)
        sq = {k: (v - yte) ** 2 for k, v in pred.items()}      # R2 <-> MSE
        for dep, w in deployments(yte).items():
            r = w / (mu @ w)
            sd_r = float(np.sqrt(mu @ (r - 1) ** 2))
            for f, g in itertools.combinations(pred, 2):
                dd = sq[f] - sq[g]
                e_mu, e_nu = float(mu @ dd), float(mu @ (dd * r))
                sd_d = float(np.sqrt(mu @ (dd - e_mu) ** 2))
                rows.append(dict(
                    dataset=name, data_id=did, seed=s, deployment=dep,
                    pair=f'{f}-{g}', sd_r=sd_r, sd_d=sd_d, e_mu=e_mu, e_nu=e_nu,
                    rho=(e_nu - e_mu) / (sd_d * sd_r) if sd_d * sd_r > 0 else 0.0,
                    inverted=int(e_mu != 0 and np.sign(e_mu) != np.sign(e_nu))))
    print(f'[{i+1}/{len(IDS)}] {name:<28} n={len(y):>5} '
          f'rows={len(rows):>7} {time.time()-t0:.0f}s', flush=True)

df = pd.DataFrame(rows)
df.to_csv(OUT / 'scale_r2.csv', index=False)
print(f'\ndatasets used {df.dataset.nunique()}  skipped {skipped}')
print(f'model-pair observations {len(df):,}')
print(f'overall reversal rate {100*df.inverted.mean():.1f}%')
print(df.groupby('deployment').agg(
    n=('inverted', 'size'), sd_r=('sd_r', 'median'),
    rev=('inverted', lambda x: 100 * x.mean())).round(2).to_string())
