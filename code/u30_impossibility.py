"""u30_impossibility.py -- no rule reading only the benchmark and a budget can help.

Given two candidates whose benchmark gap is smaller than the budget allows the
covariance to be, two deployment measures inside that budget put the two models in
opposite orders.  A selection rule that sees only the benchmark losses and the
budget returns the same answer to both, so it is wrong for one.  This constructs
the pair on the suite and records how often the construction applies and whether
the two ratios stay non-negative, which is what makes them genuine measures.
"""
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from pathlib import Path
import sys; sys.path.insert(0, str(Path(__file__).resolve().parent))
from u7_scale import learners, deployments, NMAX, SEEDS, IDS, OUT
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import itertools, time

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
        except Exception:
            continue
        ms = sorted(pred); n = len(yte); mu = np.full(n, 1 / n)
        sq = {m: (pred[m] - yte) ** 2 for m in ms}
        f = min(ms, key=lambda m: mu @ sq[m])           # the benchmark's choice
        for dep, w in deployments(yte).items():
            r0 = w / (mu @ w)
            kap = float(np.sqrt(mu @ (r0 - 1) ** 2))
            hit = nonneg = 0
            for g in ms:
                if g == f:
                    continue
                d = sq[f] - sq[g]
                e, sd = float(mu @ d), float(np.sqrt(mu @ (d - mu @ d) ** 2))
                if sd <= 0 or abs(e) >= kap * sd:
                    continue                            # construction does not apply
                hit += 1
                rp = 1 + kap * (d - e) / sd
                rm = 1 - kap * (d - e) / sd
                if rp.min() >= 0 and rm.min() >= 0:
                    nonneg += 1
            rows.append(dict(dataset=name, seed=s, deployment=dep, kappa=kap,
                             rivals=len(ms) - 1, applies=hit, nonneg=nonneg))
    print(f'[{i+1}/{len(IDS)}] {name:<28} rows={len(rows):>5} {time.time()-t0:.0f}s',
          flush=True)
t = pd.DataFrame(rows); t.to_csv(OUT / 'impossibility.csv', index=False)
print(f'\n{len(t)} selection problems')
print(f'the construction applies to at least one rival in '
      f'{100*(t.applies>0).mean():.1f}% of them, to a median of '
      f'{t.applies.median():.0f} of {int(t.rivals.iloc[0])} rivals')
print(f'both constructed ratios stay non-negative in '
      f'{100*(t.nonneg>0).mean():.1f}% of problems')
