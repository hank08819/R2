"""u20_visible.py -- what a benchmark can see of a deployment, and what it encodes.

Refits the suite to recover per-instance losses, which the pair-level files do not
carry, and writes one row per dataset, seed and deployment with

  sd_r       the stated mismatch, sd_mu(r)
  visible    the part of it the candidate set can resolve, ||P(r-1)||
  h          the data entropy of the centred loss matrix
  eff_dim    2**h, an effective number of directions, at most K-1
  encoded    the median share kappa^2/(1+kappa^2) of a pair's loss difference that
             the criterion encodes, over the 45 pairs
  reversed   the share of pairs the criterion orders against the deployment

The regret ratio is not recomputed here: u19 owns it, and the audit joins on
dataset, seed and deployment.  Dataset names follow scale_r2.csv, so the join key
matches the rest of the pipeline rather than the OpenML identifier.
"""
import warnings, json, itertools, csv; warnings.filterwarnings('ignore')
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
IDS = json.load(open(R / 'ctr23_ids.json'))
NMAX, SEEDS = 4000, [0, 1, 2]
NAME = dict(zip(pd.read_csv(R / 'scale_r2.csv').data_id.astype(str),
                pd.read_csv(R / 'scale_r2.csv').dataset))


def learners(s):
    """The ten of u7_scale.py, at library defaults."""
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
    """The five of u7_scale.py; the relative weight needs y away from zero."""
    lo, hi = y.min(), y.max()
    out = {'top decile of outcome':    (y >= np.quantile(y, .90)).astype(float),
           'top quartile of outcome':  (y >= np.quantile(y, .75)).astype(float),
           'bottom decile of outcome': (y <= np.quantile(y, .10)).astype(float),
           'cost rising in outcome':   (y - lo) / (hi - lo) if hi > lo else None}
    out['relative-error weighted'] = (1.0 / np.maximum(np.abs(y), 1e-6) ** 2
                                      if np.all(np.abs(y) > 1e-6) else None)
    # A weight that is the same on every instance is the benchmark itself, not a
    # deployment: on an outcome that is mostly zero, the top quartile selects all of it.
    return {k: v for k, v in out.items()
            if v is not None and v.sum() > 0 and np.ptp(v) > 0}


rows = []
for did in IDS:
    for seed in SEEDS:
        try:
            X, y = fetch_openml(data_id=int(did), return_X_y=True, as_frame=True)
            X = SimpleImputer(strategy='median').fit_transform(X.select_dtypes('number'))
            y = np.asarray(y, float)
            if len(y) > NMAX:
                X, _, y, _ = train_test_split(X, y, train_size=NMAX, random_state=seed)
            Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.4, random_state=seed)
            L = np.array([(m.fit(Xtr, ytr).predict(Xte) - yte) ** 2 for m in learners(seed)])
        except Exception as e:
            print(f'  {did} seed {seed}: {type(e).__name__}', flush=True); continue
        n = len(yte); mu = np.full(n, 1 / n)
        # H is the span of the centred loss differences (Proposition 1); in the mu
        # inner product that is the row space of the loss matrix centred over
        # instances under mu and across models, scaled by sqrt(mu).
        Lt = L - (L @ mu)[:, None]
        Lc = (Lt - Lt.mean(0, keepdims=True)) * np.sqrt(mu)
        _, s_, Vt = np.linalg.svd(Lc, full_matrices=False)
        q = np.linalg.matrix_rank(Lc); B = Vt[:q]
        u_ = s_[:q] / s_[:q].sum(); h = float(-(u_ * np.log2(u_)).sum())
        enc = []
        for i, j in itertools.combinations(range(len(L)), 2):
            dd = L[i] - L[j]; e = mu @ dd; sd = np.sqrt(mu @ (dd - e) ** 2)
            if sd > 0:
                k = abs(e) / sd; enc.append(k * k / (1 + k * k))
        for dep, w in deployments(yte).items():
            nu = w / w.sum(); r = nu / mu; um = (r - 1) * np.sqrt(mu)
            inv = tot = 0
            for i, j in itertools.combinations(range(len(L)), 2):
                dd = L[i] - L[j]; e, en = mu @ dd, nu @ dd
                if e != 0:
                    tot += 1; inv += int(np.sign(e) != np.sign(en))
            rows.append(dict(dataset=NAME.get(str(did), str(did)), seed=seed,
                             deployment=dep,
                             sd_r=float(np.sqrt(mu @ (r - 1) ** 2)),
                             visible=float(np.linalg.norm(B.T @ (B @ um))),
                             h=h, eff_dim=2 ** h,
                             encoded=float(np.median(enc)) * 100,
                             reversed_pct=100 * inv / max(tot, 1)))
    pd.DataFrame(rows).to_csv(R / 'visible_mismatch.csv', index=False)

t = pd.DataFrame(rows)
t.to_csv(R / 'visible_mismatch.csv', index=False)
rr = pd.read_csv(R / 'regret_ratio.csv')
m = t.merge(rr[['dataset', 'seed', 'deployment', 'rr']],
            on=['dataset', 'seed', 'deployment'], how='inner')
from scipy.stats import spearmanr
pos = m[m.rr > 1e-12]
print(f'{len(t)} combinations, {len(m)} joined to a regret ratio\n')
print(f'visible share ||P(r-1)||/sd(r): median {(t.visible/t.sd_r).median():.3f}')
print(f'effective directions 2^h:       median {t.eff_dim.median():.2f} of 9')
print(f'encoded share:                  median {t.encoded.median():.2f} per cent\n')
print(f'regret ratio, the {len(pos)} problems that cost anything:')
print(f'  spearman with ||P(r-1)||  {spearmanr(pos.visible, pos.rr)[0]: .3f}')
print(f'  spearman with sd(r)       {spearmanr(pos.sd_r, pos.rr)[0]: .3f}')
