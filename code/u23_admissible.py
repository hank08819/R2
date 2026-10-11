"""u23_admissible.py -- what a benchmark can settle, before any deployment is named.

Treat the test instances as states of nature and a deployment measure as a prior
over them.  Model selection is then a finite decision problem, and Wald's complete
class theorem applies: the models a criterion could ever be right to choose are
exactly the vertices of the lower convex hull of the loss vectors.  Anything else
is dominated by a mixture and can be ruled out without knowing the deployment;
anything inside is optimal under some deployment and cannot be ruled out at all.
"""
import warnings, time, itertools; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from pathlib import Path
from scipy.optimize import linprog
import sys; sys.path.insert(0, str(Path(__file__).resolve().parent))
from u7_scale import learners, deployments, NMAX, SEEDS, IDS, OUT
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer


def admissible(L, tol=1e-9):
    """Which rows of the loss matrix are optimal under some probability vector.

    Row f qualifies unless some mixture of the other rows is pointwise no worse
    and somewhere better, which is a small linear program in the mixture weights.
    """
    K, n = L.shape
    keep = []
    for f in range(K):
        others = [g for g in range(K) if g != f]
        # min t s.t. sum_g lam_g L_g - L_f <= t, lam in simplex ; f is dominated
        # when the optimum is <= 0 with a strict improvement somewhere.
        A_ub = np.hstack([L[others].T - L[f][:, None], -np.ones((n, 1))])
        res = linprog(c=np.r_[np.zeros(K - 1), 1.0], A_ub=A_ub, b_ub=np.zeros(n),
                      A_eq=np.r_[np.ones(K - 1), 0.0][None, :], b_eq=[1.0],
                      bounds=[(0, None)] * (K - 1) + [(None, None)],
                      method='highs')
        if (not res.success) or res.x[-1] > -tol:
            keep.append(f)                      # not dominated: admissible
    return keep


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
        L = np.vstack([(pred[m] - yte) ** 2 for m in ms])
        A = admissible(L)
        f_r2 = int(np.argmin(L @ mu))
        for dep, w in deployments(yte).items():
            r = w / (mu @ w)
            risk = L @ (mu * r)
            g = int(np.argmin(risk))
            spread = risk.max() - risk.min()
            rows.append(dict(dataset=name, seed=s, deployment=dep, K=len(ms),
                             n_adm=len(A), r2_in=int(f_r2 in A), opt_in=int(g in A),
                             r2_is_opt=int(f_r2 == g),
                             excess=(risk[f_r2]-risk[g])/spread if spread > 0 else 0.))
    print(f'[{i+1}/{len(IDS)}] {name:<28} rows={len(rows):>6} {time.time()-t0:.0f}s',
          flush=True)
pd.DataFrame(rows).to_csv(OUT / 'admissible.csv', index=False)
print('wrote admissible.csv', len(rows))
