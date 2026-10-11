"""u54_geometry_real.py -- the selection regions of one real benchmark, measured.

Figure 2 of the paper draws the geometry as a schematic.  This script draws the same
picture from data: it refits one dataset of the suite, forms the span of the centred
loss differences, projects the five deployment measures onto its two leading
directions, and reports where each lands relative to the pairwise boundaries.

Writes results/universal/geometry_slice.npz, which make_fig_proj.py reads.
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
NAMES = ['RF', 'ET', 'HGB', 'AdaB', 'Bag', 'Tree5', 'Tree12', 'OLS', 'Ridge', 'KNN']


def learners(s):
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
    lo, hi = y.min(), y.max()
    out = {'upper decile':   (y >= np.quantile(y, .90)).astype(float),
           'upper quartile': (y >= np.quantile(y, .75)).astype(float),
           'lower decile':   (y <= np.quantile(y, .10)).astype(float),
           'cost rising in the outcome': (y - lo) / (hi - lo) if hi > lo else None}
    out['relative error'] = (1.0 / np.maximum(np.abs(y), 1e-6) ** 2
                             if np.all(np.abs(y) > 1e-6) else None)
    # A weight that is the same on every instance is the benchmark itself, not a
    # deployment: on an outcome that is mostly zero, the top quartile selects all of it.
    return {k: v for k, v in out.items()
            if v is not None and v.sum() > 0 and np.ptp(v) > 0}


def slice_for(did, seed=0, nmax=4000):
    X, y = fetch_openml(data_id=int(did), return_X_y=True, as_frame=True)
    X = SimpleImputer(strategy='median').fit_transform(X.select_dtypes('number'))
    y = np.asarray(y, float)
    if len(y) > nmax:
        X, _, y, _ = train_test_split(X, y, train_size=nmax, random_state=seed)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.4, random_state=seed)
    L = np.array([(m.fit(Xtr, ytr).predict(Xte) - yte) ** 2 for m in learners(seed)])
    n = len(yte); mu = np.full(n, 1 / n)
    # H_P in the mu inner product: the row space of the model-centred loss matrix
    # scaled by sqrt(mu).  Its two leading directions give the slice we draw.
    Lc = (L - L.mean(0, keepdims=True)) * np.sqrt(mu)
    _, s, Vt = np.linalg.svd(Lc, full_matrices=False)
    e1, e2 = Vt[0], Vt[1]
    pts = {}
    for k, w in deployments(yte).items():
        nu = w / w.sum(); u = (nu / mu - 1) * np.sqrt(mu)
        pts[k] = (float(u @ e1), float(u @ e2))
    lines = []
    for i, j in itertools.combinations(range(len(L)), 2):
        d = (L[i] - L[j]) * np.sqrt(mu)
        e = float((L[i] - L[j]) @ mu)
        dc = d - e * np.sqrt(mu)
        lines.append((float(dc @ e1), float(dc @ e2), -e, i, j))
    risk = L @ mu
    return dict(points=pts, lines=lines, best=int(np.argmin(risk)),
                names=NAMES, spectrum=s.tolist())


if __name__ == '__main__':
    ids = json.load(open(R / 'ctr23_ids.json'))
    name = dict(zip(pd.read_csv(R / 'scale_r2.csv').data_id.astype(str),
                    pd.read_csv(R / 'scale_r2.csv').dataset))
    did = ids[0]
    out = slice_for(did)
    np.savez(R / 'geometry_slice.npz',
             names=np.array(out['names']), best=out['best'],
             labels=np.array(list(out['points'])),
             pts=np.array(list(out['points'].values())),
             lines=np.array(out['lines'], float),
             dataset=name.get(str(did), str(did)))
    print(f"dataset {name.get(str(did), did)}, benchmark choice "
          f"{out['names'][out['best']]}")
    for k, v in out['points'].items():
        print(f'   {k:28s} ({v[0]:+.2f}, {v[1]:+.2f})')
