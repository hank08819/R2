"""u46_rr_signed.py -- the six-question correction, priced in the paper's currency.

The excess loss of a choice is reported elsewhere as a share of the range the
candidates span, which is convenient but arbitrary. The regret ratio divides it
instead by the benchmark risk the chosen model was credited with, a quantity known
before deployment. This prices the correction the same way.
"""
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from pathlib import Path

R = Path(__file__).resolve().parents[1] / 'results' / 'universal'
d = pd.read_csv(R / 'scale_r2.csv')
d['left'] = d.pair.str.split('-').str[0]; d['right'] = d.pair.str.split('-').str[1]
m = pd.read_csv(R / 'robust_raw.csv')
base = {(r.dataset, r.seed, r.deployment, r.model): (r.mu_loss, r.nu_loss)
        for r in m.itertuples()}
RHO, TOP = .05, 4

rows = []
for (ds, se, dep), h in d.groupby(['dataset', 'seed', 'deployment']):
    ms = sorted(set(h.left) | set(h.right)); ix = {x: i for i, x in enumerate(ms)}
    K = len(ms)
    try:
        mu_l = np.array([base[(ds, se, dep, x)][0] for x in ms])
        nu_l = np.array([base[(ds, se, dep, x)][1] for x in ms])
    except KeyError:
        continue
    if (mu_l <= 0).any():
        continue
    E = np.zeros((K, K)); S = np.zeros((K, K)); N = np.zeros((K, K))
    for _, x in h.iterrows():
        i, j = ix[x.left], ix[x.right]
        E[i, j], E[j, i] = x.e_mu, -x.e_mu
        N[i, j], N[j, i] = x.e_nu, -x.e_nu
        S[i, j] = S[j, i] = x.sd_d
    k = float(h.sd_r.iloc[0])
    top = set(np.argsort(mu_l)[:TOP].tolist())
    sgn = np.zeros((K, K))
    for i in range(K):
        for j in range(i + 1, K):
            if i in top and j in top:
                s0 = np.sign(N[i, j] - E[i, j]); sgn[i, j], sgn[j, i] = s0, -s0
    f0 = int(np.argmin(mu_l)); f1 = int(np.argmin((E + sgn * RHO * k * S).sum(1)))
    best = nu_l.min()
    rows.append(dict(dataset=ds, seed=se, deployment=dep, sd_r=k,
                     rr_r2=(nu_l[f0] - best) / mu_l[f0],
                     # both choices are priced in the benchmark risk of the criterion's
                     # choice, so the two ratios share one denominator
                     rr_rule=(nu_l[f1] - best) / mu_l[f0]))
t = pd.DataFrame(rows); t.to_csv(R / 'rr_signed.csv', index=False)

print(f'{len(t)} selection problems.  Regret ratio: excess deployment loss per unit '
      f'of the benchmark risk\nof the benchmark choice, for both rules.\n')
print(f'  {"":<26}{"criterion":>12}{"six questions":>16}')
for q, lab in [(.5, 'median'), (.75, '75th percentile'), (.9, '90th percentile'),
               (.95, '95th percentile')]:
    print(f'  {lab:<26}{100*t.rr_r2.quantile(q):11.1f}%{100*t.rr_rule.quantile(q):15.1f}%')
print(f'  {"mean":<26}{100*t.rr_r2.mean():11.1f}%{100*t.rr_rule.mean():15.1f}%')
print(f'  {"exceeds 25%":<26}{100*(t.rr_r2>.25).mean():11.1f}%{100*(t.rr_rule>.25).mean():15.1f}%')
print('\nby quartile of the mismatch (mean regret ratio):')
b = t.assign(q=pd.qcut(t.sd_r, 4, duplicates='drop')).groupby('q', observed=True)
for name, g in b:
    print(f'  sd(r) median {g.sd_r.median():5.2f}   criterion {100*g.rr_r2.mean():6.1f}%'
          f'   six questions {100*g.rr_rule.mean():6.1f}%')
