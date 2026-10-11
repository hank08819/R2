"""u27_offfrontier.py -- is the model deployment wants even on the frontier?

Trading benchmark accuracy for worst-case robustness moves along a frontier in
two objectives.  That is only worth doing if the model the deployment measure
actually prefers lies on it.  This checks.
"""
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from pathlib import Path

R = Path(__file__).resolve().parents[1] / 'results' / 'universal'
d = pd.read_csv(R / 'scale_r2.csv')
d['left'] = d.pair.str.split('-').str[0]; d['right'] = d.pair.str.split('-').str[1]

rows = []
for (ds, se), g in d.groupby(['dataset', 'seed']):
    ms = sorted(set(g.left) | set(g.right)); ix = {m: i for i, m in enumerate(ms)}
    K = len(ms); E = np.zeros((K, K)); S = np.zeros((K, K))
    for _, x in g.iterrows():
        i, j = ix[x.left], ix[x.right]
        E[i, j], E[j, i] = x.e_mu, -x.e_mu
        S[i, j] = S[j, i] = x.sd_d
    rmu = E.sum(1) / K
    for dep, h in g.groupby('deployment'):
        k = float(h.sd_r.iloc[0]); N = np.zeros((K, K))
        for _, x in h.iterrows():
            i, j = ix[x.left], ix[x.right]
            N[i, j], N[j, i] = x.e_nu, -x.e_nu
        rnu = N.sum(1) / K
        W = (E + k * S).max(1)
        front = [f for f in range(K) if not any(
            (rmu[j] <= rmu[f]) and (W[j] <= W[f]) and
            ((rmu[j] < rmu[f]) or (W[j] < W[f])) for j in range(K))]
        gstar = int(np.argmin(rnu))
        rows.append(dict(dataset=ds, seed=se, deployment=dep, K=K,
                         n_front=len(front), on_front=int(gstar in front),
                         is_r2=int(gstar == int(np.argmin(rmu)))))
s = pd.DataFrame(rows); s.to_csv(R / 'off_frontier.csv', index=False)
K = int(s.K.iloc[0])
print(f'{len(s)} selection problems, {K} candidates each\n')
print(f'frontier size: median {s.n_front.median():.0f}, mean {s.n_front.mean():.2f}, '
      f'max {s.n_front.max()}')
print(f'the deployment-optimal model lies ON the frontier in '
      f'{100*s.on_front.mean():.1f}% of problems')
q = s[s.is_r2 == 0]
print(f'restricted to the {len(q)} problems where it is not the criterion\'s own '
      f'choice: {100*q.on_front.mean():.1f}%')
print(f'\nif a model were drawn at random from the {K}, it would be on a frontier of '
      f'median size {s.n_front.median():.0f} with probability '
      f'{100*s.n_front.mean()/K:.1f}%')
