"""u36_neural.py -- does the reversal survive when the candidates include networks?

The suite as run compares trees, ensembles and linear models.  If orderings only
reverse among those, the finding is about a model family rather than about the
criterion.  This adds neural candidates at two depths and repeats the headline.
"""
import warnings, time, itertools; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from pathlib import Path
import sys; sys.path.insert(0, str(Path(__file__).resolve().parent))
from u7_scale import learners as base_learners, deployments, NMAX, SEEDS, IDS, OUT
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


def learners(s):
    m = base_learners(s)
    m['MLP-1'] = make_pipeline(StandardScaler(), MLPRegressor(
        hidden_layer_sizes=(128,), max_iter=400, early_stopping=True, random_state=s))
    m['MLP-3'] = make_pipeline(StandardScaler(), MLPRegressor(
        hidden_layer_sizes=(256, 128, 64), max_iter=400, early_stopping=True,
        random_state=s))
    return m


rows = []
for i, did in enumerate(IDS):
    t0 = time.time()
    try:
        dd = fetch_openml(data_id=did, as_frame=True, parser='auto')
        X = pd.get_dummies(dd.data, drop_first=True).astype(float)
        X = SimpleImputer(strategy='median').fit_transform(X)
        y = pd.to_numeric(dd.target, errors='coerce').values.astype(float)
        ok = np.isfinite(y) & np.isfinite(X).all(1); X, y = X[ok], y[ok]
        if len(y) > NMAX:
            idx = np.random.default_rng(0).choice(len(y), NMAX, replace=False)
            X, y = X[idx], y[idx]
        if len(y) < 200 or X.shape[1] > 300:
            continue
    except Exception:
        continue
    name = dd.details.get('name', str(did))
    for s in SEEDS:
        try:
            Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.4, random_state=s)
            pred = {k: m.fit(Xtr, ytr).predict(Xte) for k, m in learners(s).items()}
        except Exception as e:
            print(f'   fit fail {name} seed {s}: {type(e).__name__}', flush=True)
            continue
        n = len(yte); mu = np.full(n, 1 / n); ms = sorted(pred)
        sq = {m: (pred[m] - yte) ** 2 for m in ms}
        for dep, w in deployments(yte).items():
            r = w / (mu @ w)
            for f, g in itertools.combinations(ms, 2):
                d = sq[f] - sq[g]
                e_mu, e_nu = float(mu @ d), float(mu @ (d * r))
                neural = ('MLP' in f) + ('MLP' in g)
                rows.append(dict(dataset=name, seed=s, deployment=dep, pair=f'{f}-{g}',
                                 neural=neural, e_mu=e_mu, e_nu=e_nu,
                                 inverted=int(e_mu != 0 and np.sign(e_mu) != np.sign(e_nu))))
    print(f'[{i+1}/{len(IDS)}] {name:<28} rows={len(rows):>6} {time.time()-t0:.0f}s',
          flush=True)

t = pd.DataFrame(rows); t.to_csv(OUT / 'neural.csv', index=False)
print(f'\n{len(t):,} pairs over {t.dataset.nunique()} datasets, '
      f'{t.pair.nunique()} distinct pairings')
print(f'  overall reversal rate            {100*t.inverted.mean():.1f}%')
for k, lab in [(0, 'both candidates classical'), (1, 'one network, one classical'),
               (2, 'both candidates networks')]:
    g = t[t.neural == k]
    if len(g):
        print(f'  {lab:<32} {100*g.inverted.mean():.1f}%   ({len(g):,} pairs)')
