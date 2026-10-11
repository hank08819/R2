"""u21_mape_rr.py -- regret ratios for every criterion, in one currency.

The deployment loss fixes the units, so each criterion's choice is charged the
excess deployment loss it incurs, divided by the benchmark risk of that same
choice measured with the deployment's own per-instance score.  R-squared and MAPE
are then directly comparable, which they are not when each is normalised by its
own reported risk.
"""
import warnings, time; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from pathlib import Path
import sys; sys.path.insert(0, str(Path(__file__).resolve().parent))
from u7_scale import learners, deployments, NMAX, SEEDS, IDS, OUT
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer


def criteria(y):
    n = len(y); u = np.full(n, 1 / n); a = np.abs(y)
    out = [('R2', lambda p: (p - y) ** 2, u),
           ('MAE', lambda p: np.abs(p - y), u)]
    if (a > 1e-8).all():
        w = 1 / a; out.append(('MAPE', lambda p: np.abs(p - y), w / w.sum()))
        w2 = 1 / a ** 2; out.append(('RSE', lambda p: (p - y) ** 2, w2 / w2.sum()))
    return out


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
        bench = {m: float(mu @ sq[m]) for m in ms}          # common denominator
        crit = criteria(yte)
        pick = {}
        for cname, score, meas in crit:
            v = [float(meas @ score(pred[m])) for m in ms]
            pick[cname] = ms[int(np.argmin(v))]
        for dep, w in deployments(yte).items():
            r = w / (mu @ w)
            risk = {m: float(mu @ (sq[m] * r)) for m in ms}
            best = min(risk.values())
            for cname, _, meas in crit:
                f = pick[cname]
                rr = (risk[f] - best) / bench[f] if bench[f] > 0 else np.nan
                sdr = float(np.sqrt(meas @ ((w / w.sum()) / meas
                            / (meas @ ((w / w.sum()) / meas)) - 1) ** 2))
                rows.append(dict(dataset=name, seed=s, deployment=dep,
                                 criterion=cname, sd_r=sdr, rr=rr,
                                 wrong=int(risk[f] > best + 1e-12)))
    print(f'[{i+1}/{len(IDS)}] {name:<28} rows={len(rows):>6} {time.time()-t0:.0f}s',
          flush=True)
pd.DataFrame(rows).to_csv(OUT / 'mape_rr.csv', index=False)
print('wrote mape_rr.csv', len(rows))
