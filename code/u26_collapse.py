"""u26_collapse.py -- how much a stated budget narrows the field.

Against an unrestricted deployment measure nothing can be ruled out.  Against a
budget, a model is out of contention if some rival is at once better on the
benchmark and better in the worst case the budget allows.  This traces how many
models survive that test as the budget widens, which is the amount of information
a practitioner buys by naming one.
"""
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from pathlib import Path

R = Path(__file__).resolve().parents[1] / 'results' / 'universal'
d = pd.read_csv(R / 'scale_r2.csv')
d['left'] = d.pair.str.split('-').str[0]; d['right'] = d.pair.str.split('-').str[1]
KAP = [0.01, .05, .1, .25, .5, 1., 2., 3., 5., 10.]

rows = []
for (ds, se), g in d.groupby(['dataset', 'seed']):
    ms = sorted(set(g.left) | set(g.right)); ix = {m: i for i, m in enumerate(ms)}
    K = len(ms); E = np.zeros((K, K)); S = np.zeros((K, K))
    for _, x in g.iterrows():
        i, j = ix[x.left], ix[x.right]
        E[i, j], E[j, i] = x.e_mu, -x.e_mu
        S[i, j] = S[j, i] = x.sd_d
    rmu = E.sum(1) / K
    rec = dict(dataset=ds, seed=se, K=K)
    for k in KAP:
        W = (E + k * S).max(1)
        front = [f for f in range(K)
                 if not any((rmu[j] <= rmu[f]) and (W[j] <= W[f]) and
                            ((rmu[j] < rmu[f]) or (W[j] < W[f])) for j in range(K))]
        rec[f'n_{k}'] = len(front)
    rows.append(rec)
s = pd.DataFrame(rows); s.to_csv(R / 'collapse.csv', index=False)

print(f'{len(s)} dataset-splits, {s.K.iloc[0]} candidate models each\n')
print('  budget      surviving models: median   mean    max   share with 1')
for k in KAP:
    c = s[f'n_{k}']
    print(f'  {k:6.2f} {c.median():18.0f} {c.mean():7.2f} {c.max():6.0f} '
          f'{100*(c==1).mean():13.1f}%')
print(f'\nunrestricted deployment: all {s.K.iloc[0]} survive '
      f'(admissibility check, 97.0% of problems)')
