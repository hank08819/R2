"""u35_inference.py -- standard errors for the bias, and for the reversals it causes.

Every quantity the paper reports is a plug-in estimate on a finite test sample.
The bias is a smooth function of three sample means, so the delta method gives it
an influence function and a standard error in closed form.  Part one checks that
formula against a bootstrap; part two applies it to ask how many of the reversals
found on the suite are distinguishable from none.
"""
import warnings; warnings.filterwarnings('ignore')
import numpy as np

rng = np.random.default_rng(7)


def bias_and_se(d, w):
    """Plug-in bias and its delta-method standard error on one sample.

    With r the weight normalised to average one, B = E_nu[d] - E_mu[d] and the
    influence function is r(d - E_nu[d]) - (d - E_mu[d]).
    """
    n = len(d)
    r = w / w.mean()
    e_mu = d.mean()
    e_nu = float((d * r).mean())
    inf = r * (d - e_nu) - (d - e_mu)
    return e_nu - e_mu, float(np.sqrt((inf ** 2).mean() / n))


print('formula against the bootstrap, 6 settings\n')
print('  n      analytic SE   bootstrap SE   ratio')
for n in [200, 500, 2000]:
    for tail in [False, True]:
        x = rng.standard_normal(n)
        d = .3 * x + rng.standard_normal(n) * .5
        w = np.exp(1.2 * x) if tail else (x > np.quantile(x, .8)).astype(float) + 1e-9
        b, se = bias_and_se(d, w)
        boot = [bias_and_se(d[i], w[i])[0]
                for i in (rng.integers(0, n, n) for _ in range(3000))]
        bse = float(np.std(boot, ddof=1))
        print(f'  {n:<6} {se:11.5f} {bse:14.5f} {se/bse:7.3f}'
              f'   {"heavy tail" if tail else "region"}')


# --------------------------------------------------------- on the suite ---
import pandas as pd, itertools, time
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
        n = len(yte); ms = sorted(pred)
        sq = {m: (pred[m] - yte) ** 2 for m in ms}
        for dep, w in deployments(yte).items():
            r = w / w.mean()
            for f, g in itertools.combinations(ms, 2):
                d = sq[f] - sq[g]
                e_mu = float(d.mean()); e_nu = float((d * r).mean())
                se_mu = float(d.std(ddof=1) / np.sqrt(n))
                se_nu = float(np.sqrt(((r * (d - e_nu)) ** 2).mean() / n))
                rows.append(dict(dataset=name, seed=s, deployment=dep, n=n,
                                 e_mu=e_mu, e_nu=e_nu, se_mu=se_mu, se_nu=se_nu))
    print(f'[{i+1}/{len(IDS)}] {name:<28} rows={len(rows):>6} {time.time()-t0:.0f}s',
          flush=True)

t = pd.DataFrame(rows); t.to_csv(OUT / 'inference.csv', index=False)
z = 1.96
t['rev'] = (np.sign(t.e_mu) != np.sign(t.e_nu)) & (t.e_mu != 0)
t['both_sig'] = (t.e_mu.abs() > z * t.se_mu) & (t.e_nu.abs() > z * t.se_nu)
t['settled'] = t.rev & t.both_sig
print(f'\n{len(t):,} model pairs')
print(f'  reversed, as counted            {100*t.rev.mean():.1f}%')
print(f'  reversed and both gaps distinguishable from zero at 5%: '
      f'{100*t.settled.mean():.1f}%')
print(f'  of the reversals, the share that are: {100*t.settled.sum()/t.rev.sum():.1f}%')
print(f'  pairs whose benchmark gap is itself indistinguishable from zero: '
      f'{100*(t.e_mu.abs() <= z*t.se_mu).mean():.1f}%')
