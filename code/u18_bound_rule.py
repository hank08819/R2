"""u18_bound_rule.py -- pick the criterion that minimises the bound of Theorem 2.

Matching the measure alone is not enough: the bias is bounded by a product, and
both factors move when the criterion changes.  The rule the theory actually
prescribes is to minimise that product, the diameter of the candidate set in the
criterion's own metric times the mismatch between the criterion's measure and the
intended use.  Both are available from the predictions the benchmark has produced.
"""
import warnings, time, itertools; warnings.filterwarnings('ignore')
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
        a = np.abs(yte)
        if not (a > 1e-8).all():
            continue
        # criteria sharing the deployment's per-instance score, differing in measure
        MEAS = {'R2 (uniform)': mu}
        for lab, ww in [('relative squared error', 1.0 / a ** 2),
                        ('root-weighted squared error', 1.0 / np.sqrt(a))]:
            MEAS[lab] = ww / ww.sum()
        sq = {m: (pred[m] - yte) ** 2 for m in ms}
        info = {}
        for cname, mm in MEAS.items():
            v = {m: float(mm @ sq[m]) for m in ms}

            def sd_diff(f, g, mm=mm):
                dd = sq[f] - sq[g]
                return float(np.sqrt(mm @ (dd - mm @ dd) ** 2))

            D = max(sd_diff(f, g) for f, g in itertools.combinations(ms, 2))
            info[cname] = (min(v, key=v.get), D, mm)
        for dep, w in deployments(yte).items():
            nu = w / w.sum()
            risk = {m: float(nu @ sq[m]) for m in ms}
            best = min(risk.values()); spread = max(risk.values()) - best
            for cname, (choice, D, mm) in info.items():
                r = nu / mm
                sdr = float(np.sqrt(mm @ (r / (mm @ r) - 1) ** 2))
                rows.append(dict(dataset=name, seed=s, deployment=dep,
                                 criterion=cname, sd_r=sdr, diam=D, bound=D * sdr,
                                 excess=(risk[choice] - best) / spread if spread > 0 else 0.))
    print(f'[{i+1}/{len(IDS)}] {name:<28} rows={len(rows):>6} {time.time()-t0:.0f}s',
          flush=True)
pd.DataFrame(rows).to_csv(OUT / 'bound_rule.csv', index=False)
print('wrote bound_rule.csv', len(rows))
