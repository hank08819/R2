"""u49_misspec.py -- must the deployment be described exactly for the sign to help?

A practitioner does not guess the sign; the deployment is theirs and they read it
off their own weighting. What they may not do is state that weighting precisely.
So the question is not whether a person can predict the sign but whether a rough
description yields the same one. Here the sign is elicited from a deliberately
misspecified deployment and scored against the true one.
"""
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, itertools, time
from pathlib import Path
import sys; sys.path.insert(0, str(Path(__file__).resolve().parent))
from u7_scale import learners, NMAX, SEEDS, IDS, OUT
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

RHO, TOP = .05, 4


def weights(y):
    """The true deployment, and the coarse descriptions a practitioner might give."""
    lo, hi = y.min(), y.max()
    a = np.abs(y)
    true = {
        'upper decile':   (y >= np.quantile(y, .90)).astype(float),
        'cost rising':    (y - lo) / (hi - lo) if hi > lo else None,
        'relative error': 1.0 / np.maximum(a, 1e-6) ** 2 if (a > 1e-6).all() else None,
    }
    said = {
        # the region named a little wrong
        'upper decile':   [('says upper 15%', (y >= np.quantile(y, .85)).astype(float)),
                           ('says upper quartile', (y >= np.quantile(y, .75)).astype(float)),
                           ('says top third', (y >= np.quantile(y, .667)).astype(float))],
        # the shape of the cost named wrong
        'cost rising':    [('says linear, is quadratic',
                            ((y - lo) / (hi - lo)) ** 2 if hi > lo else None),
                           ('says linear, is a step',
                            (y >= np.median(y)).astype(float)),
                           ('says linear, is a rank', np.argsort(np.argsort(y)) / len(y))],
        'relative error': [('says 1/y^2, is 1/|y|', 1.0 / np.maximum(a, 1e-6)),
                           ('says 1/y^2, is 1/y^3', 1.0 / np.maximum(a, 1e-6) ** 3),
                           ('says 1/y^2, is log', 1.0 / np.maximum(np.log1p(a), .1))],
    }
    return true, said


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
        n = len(yte); mu = np.full(n, 1 / n); ms = sorted(pred)
        sq = {m: (pred[m] - yte) ** 2 for m in ms}
        mu_l = np.array([float(mu @ sq[m]) for m in ms])
        top = set(np.argsort(mu_l)[:TOP].tolist())
        true, said = weights(yte)
        for dep, w in true.items():
            if w is None or w.sum() <= 0:
                continue
            r = w / (mu @ w)
            k = float(np.sqrt(mu @ (r - 1) ** 2))
            nu_l = np.array([float(mu @ (sq[m] * r)) for m in ms])
            best = nu_l.min(); spread = nu_l.max() - best
            if spread <= 0:
                continue
            K = len(ms)
            E = np.zeros((K, K)); S = np.zeros((K, K))
            for a_, b_ in itertools.combinations(range(K), 2):
                d0 = sq[ms[a_]] - sq[ms[b_]]
                e = float(mu @ d0); sd = float(np.sqrt(mu @ (d0 - e) ** 2))
                E[a_, b_], E[b_, a_] = e, -e
                S[a_, b_] = S[b_, a_] = sd
            def choose(wsign):
                rs = wsign / (mu @ wsign)
                sg = np.zeros((K, K))
                for a_, b_ in itertools.combinations(range(K), 2):
                    if a_ in top and b_ in top:
                        d0 = sq[ms[a_]] - sq[ms[b_]]
                        cov = float(mu @ (d0 * rs)) - float(mu @ d0)
                        sg[a_, b_], sg[b_, a_] = np.sign(cov), -np.sign(cov)
                return int(np.argmin((E + sg * RHO * k * S).sum(1)))
            base = (nu_l[int(np.argmin(mu_l))] - best) / spread
            exact = (nu_l[choose(w)] - best) / spread
            for lab, wm in said[dep]:
                if wm is None or wm.sum() <= 0:
                    continue
                rows.append(dict(dataset=name, seed=s, deployment=dep, said=lab,
                                 r2=base, exact=exact,
                                 rough=(nu_l[choose(wm)] - best) / spread))
    print(f'[{i+1}/{len(IDS)}] {name:<28} rows={len(rows):>5} {time.time()-t0:.0f}s',
          flush=True)

t = pd.DataFrame(rows); t.to_csv(OUT / 'misspec.csv', index=False)
print(f'\n{len(t):,} elicitations under a misdescribed deployment\n')
print(f'  {"description given":<30}{"mean excess (%)":>17}{"vs criterion":>14}')
print(f'  {"the criterion, no questions":<30}{100*t.r2.mean():>17.1f}')
print(f'  {"the deployment stated exactly":<30}{100*t.exact.mean():>17.1f}'
      f'{100*(t.exact.mean()-t.r2.mean()):>14.1f}')
for lab, g in t.groupby('said'):
    print(f'  {lab:<30}{100*g.rough.mean():>17.1f}{100*(g.rough.mean()-g.r2.mean()):>14.1f}')
