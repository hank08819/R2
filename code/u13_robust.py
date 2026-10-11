"""u13_robust.py -- does the kappa-robust criterion pick better models?

The worst deployment loss inside a chi-squared ball of radius kappa has a closed
form, mean plus kappa times standard deviation of the per-instance loss, so
selecting on it costs nothing beyond what the benchmark already provides.  This
records, for every dataset and split in the suite, each model's benchmark mean and
dispersion and its loss under each deployment measure, which is everything the
comparison against R-squared needs.
"""
import warnings, json, time; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from pathlib import Path
import sys; sys.path.insert(0, str(Path(__file__).resolve().parent))
from u7_scale import learners, deployments, NMAX, SEEDS, IDS, OUT
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

rows = []
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
            continue
    except Exception:
        continue
    name = d.details.get('name', str(did))
    for s in SEEDS:
        try:
            Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.4, random_state=s)
            pred = {k: m.fit(Xtr, ytr).predict(Xte) for k, m in learners(s).items()}
        except Exception:
            continue
        n = len(yte); mu = np.full(n, 1 / n)
        sq = {k: (v - yte) ** 2 for k, v in pred.items()}
        base = {k: (float(mu @ v), float(np.sqrt(mu @ (v - mu @ v) ** 2)))
                for k, v in sq.items()}
        for dep, w in deployments(yte).items():
            r = w / (mu @ w)
            sd_r = float(np.sqrt(mu @ (r - 1) ** 2))
            for k, v in sq.items():
                rows.append(dict(dataset=name, seed=s, deployment=dep, model=k,
                                 sd_r=sd_r, mu_loss=base[k][0], sd_loss=base[k][1],
                                 nu_loss=float(mu @ (v * r))))
    print(f'[{i+1}/{len(IDS)}] {name:<28} rows={len(rows):>6} '
          f'{time.time()-t0:.0f}s', flush=True)

pd.DataFrame(rows).to_csv(OUT / 'robust_raw.csv', index=False)
print('wrote robust_raw.csv', len(rows))
