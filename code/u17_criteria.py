"""u17_criteria.py -- choose the criterion, not a robustification.

Every familiar criterion is a per-instance score averaged under some measure:
R-squared is squared error under the uniform one, MAPE is absolute error under a
measure proportional to the reciprocal of the outcome, and so on.  Corollary 1
then says the criterion to report is the one whose implied measure sits closest to
the use the prediction will be put to.  This computes, for every dataset and
split, which model each criterion selects and what that choice costs under each
deployment measure.
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
    """(name, per-instance score, implied measure) for each reported criterion."""
    n = len(y); u = np.full(n, 1 / n)
    a = np.abs(y)
    out = [('R2 (squared error)', lambda p: (p - y) ** 2, u),
           ('MAE (absolute error)', lambda p: np.abs(p - y), u)]
    if (a > 1e-8).all():
        w = 1.0 / a; w = w / w.sum()
        out.append(('MAPE', lambda p: np.abs(p - y), w))
        w2 = 1.0 / a ** 2; w2 = w2 / w2.sum()
        out.append(('relative squared error', lambda p: (p - y) ** 2, w2))
    return out


rows = []
for i, did in enumerate(IDS):
    t0 = time.time()
    try:
        dd = fetch_openml(data_id=did, as_frame=True, parser='auto')
        X = pd.get_dummies(dd.data, drop_first=True).astype(float)
        X = SimpleImputer(strategy='median').fit_transform(X)
        y = pd.to_numeric(dd.target, errors='coerce').values.astype(float)
        ok = np.isfinite(y) & np.isfinite(X).all(1)
        X, y = X[ok], y[ok]
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
        crit = criteria(yte)
        # what each criterion would select
        chosen = {}
        for cname, score, meas in crit:
            v = [float(meas @ score(pred[m])) for m in ms]
            chosen[cname] = ms[int(np.argmin(v))]
        # what each deployment actually pays, on squared error
        for dep, w in deployments(yte).items():
            r = w / (mu @ w)
            risk = {m: float(mu @ ((pred[m] - yte) ** 2 * r)) for m in ms}
            best = min(risk.values()); spread = max(risk.values()) - best
            for cname, score, meas in crit:
                # mismatch between this criterion's measure and the deployment
                rr = (w / w.sum()) / meas
                sdr = float(np.sqrt(meas @ (rr / (meas @ rr) - 1) ** 2))
                rows.append(dict(dataset=name, seed=s, deployment=dep,
                                 criterion=cname, sd_r=sdr,
                                 excess=(risk[chosen[cname]] - best) / spread
                                        if spread > 0 else 0.,
                                 wrong=int(risk[chosen[cname]] > best + 1e-12)))
    print(f'[{i+1}/{len(IDS)}] {name:<28} rows={len(rows):>6} {time.time()-t0:.0f}s',
          flush=True)

pd.DataFrame(rows).to_csv(OUT / 'criteria_choice.csv', index=False)
print('wrote criteria_choice.csv', len(rows))
