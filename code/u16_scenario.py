"""u16_scenario.py -- minimax regret over stated scenarios, tested out of sample.

A practitioner cannot name a chi-squared ball, but can name the handful of ways
the prediction will be used.  This selects the model whose regret is smallest
across those scenarios and scores it on a deployment measure held out of that set,
so the test is whether stating some uses helps against a use not stated.
"""
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from pathlib import Path

R = Path(__file__).resolve().parents[1] / 'results' / 'universal'
d = pd.read_csv(R / 'scale_r2.csv')
d['left'] = d.pair.str.split('-').str[0]; d['right'] = d.pair.str.split('-').str[1]

def risks(g):
    """Per-model risks, up to a common shift, from the pairwise gaps."""
    ms = sorted(set(g.left) | set(g.right)); ix = {m: i for i, m in enumerate(ms)}
    K = len(ms); A = np.zeros((K, K)); B = np.zeros((K, K))
    for _, x in g.iterrows():
        i, j = ix[x.left], ix[x.right]
        A[i, j], A[j, i] = x.e_mu, -x.e_mu
        B[i, j], B[j, i] = x.e_nu, -x.e_nu
    return ms, A.sum(1) / K, B.sum(1) / K

rows = []
for (ds, seed), g in d.groupby(['dataset', 'seed']):
    per = {}
    for dep, h in g.groupby('deployment'):
        ms, rmu, rnu = risks(h)
        per[dep] = (ms, rmu, rnu)
    deps = sorted(per)
    if len(deps) < 3:
        continue
    for held in deps:
        seen = [x for x in deps if x != held]
        ms, rmu, _ = per[held]
        reg = np.zeros(len(ms))
        for s_ in seen:                       # worst regret over stated scenarios
            r = per[s_][2]
            reg = np.maximum(reg, r - r.min())
        _, _, rout = per[held]
        best = rout.min(); spread = rout.max() - rout.min()
        j_sc = int(np.argmin(reg))            # scenario rule
        j_r2 = int(np.argmin(rmu))            # ordinary criterion
        rows.append(dict(dataset=ds, seed=seed, held=held, spread=spread,
                         sd_r=float(g[g.deployment == held].sd_r.iloc[0]),
                         exc_r2=(rout[j_r2]-best)/spread if spread > 0 else 0.,
                         exc_sc=(rout[j_sc]-best)/spread if spread > 0 else 0.,
                         wrong_r2=int(rout[j_r2] > best+1e-12),
                         wrong_sc=int(rout[j_sc] > best+1e-12)))
s = pd.DataFrame(rows); s.to_csv(R / 'scenario_eval.csv', index=False)
print(f'\n{len(s):,} held-out selection problems, {s.dataset.nunique()} datasets\n')
print('                       picks wrong (%)   excess: mean   median   90th pct')
for lab, w, e in [('R^2', s.wrong_r2, s.exc_r2), ('scenario minimax regret', s.wrong_sc, s.exc_sc)]:
    print(f'  {lab:<24} {100*w.mean():8.1f} {100*e.mean():14.1f} '
          f'{100*e.median():8.1f} {100*e.quantile(.9):10.1f}')
print(f'\n  improved {int((s.exc_sc<s.exc_r2-1e-12).sum())}   '
      f'worsened {int((s.exc_sc>s.exc_r2+1e-12).sum())}   '
      f'unchanged {int((np.abs(s.exc_sc-s.exc_r2)<=1e-12).sum())}')
