"""u45_cases.py -- worked cases for the six-question correction.

Aggregate rates say the rule helps. A practitioner asks what it looks like on one
problem: which model the criterion picks, which the six answers move it to, and
what that move is worth.
"""
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from pathlib import Path

R = Path(__file__).resolve().parents[1] / 'results' / 'universal'
d = pd.read_csv(R / 'scale_r2.csv')
d['left'] = d.pair.str.split('-').str[0]; d['right'] = d.pair.str.split('-').str[1]
RHO, TOP = .05, 4

rows = []
for (ds, se, dep), h in d.groupby(['dataset', 'seed', 'deployment']):
    ms = sorted(set(h.left) | set(h.right)); ix = {m: i for i, m in enumerate(ms)}
    K = len(ms)
    E = np.zeros((K, K)); S = np.zeros((K, K)); N = np.zeros((K, K))
    for _, x in h.iterrows():
        i, j = ix[x.left], ix[x.right]
        E[i, j], E[j, i] = x.e_mu, -x.e_mu
        N[i, j], N[j, i] = x.e_nu, -x.e_nu
        S[i, j] = S[j, i] = x.sd_d
    k = float(h.sd_r.iloc[0])
    rmu, rnu = E.sum(1) / K, N.sum(1) / K
    top = set(np.argsort(rmu)[:TOP].tolist())
    sgn = np.zeros((K, K))
    for i in range(K):
        for j in range(i + 1, K):
            if i in top and j in top:
                s0 = np.sign(N[i, j] - E[i, j]); sgn[i, j], sgn[j, i] = s0, -s0
    f0 = int(np.argmin(rmu)); f1 = int(np.argmin((E + sgn * RHO * k * S).sum(1)))
    g = int(np.argmin(rnu))
    spread = rnu.max() - rnu.min()
    if spread <= 0:
        continue
    rows.append(dict(dataset=ds, seed=se, deployment=dep, sd_r=k,
                     r2=ms[f0], rule=ms[f1], best=ms[g],
                     exc_r2=(rnu[f0] - rnu[g]) / spread,
                     exc_rule=(rnu[f1] - rnu[g]) / spread,
                     fixed=int(f0 != g and f1 == g)))
t = pd.DataFrame(rows); t.to_csv(R / 'cases.csv', index=False)

fix = t[t.fixed == 1].copy()
fix['gain'] = fix.exc_r2 - fix.exc_rule
print(f'{len(t)} problems.  The six answers move the choice to the deployment '
      f'optimum in {len(fix)} of the {int((t.exc_r2>1e-12).sum())} the criterion '
      f'gets wrong.\n')
print('The eight largest gains:\n')
print(f'  {"dataset":<26}{"deployment":<26}{"R2 picks":<9}{"rule picks":<11}'
      f'{"cost avoided":>13}')
for r in fix.sort_values('gain', ascending=False).head(8).itertuples():
    print(f'  {r.dataset[:25]:<26}{r.deployment[:25]:<26}{r.r2:<9}{r.rule:<11}'
          f'{100*r.gain:>12.1f}%')
print(f'\n  median cost avoided over all {len(fix)} corrected problems: '
      f'{100*fix.gain.median():.1f}% of the candidate range')
w = t[(t.exc_rule > t.exc_r2 + 1e-12)]
print(f'  the rule also damages {len(w)} problems, by a median of '
      f'{100*(w.exc_rule-w.exc_r2).median():.1f}%')
