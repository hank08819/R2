"""u19_regret_ratio.py -- what selecting on the wrong criterion costs, as a rate.

The excess loss of the criterion's choice is reported relative to the benchmark
risk of that same choice, so the number reads as a percentage of the error the
benchmark already told the practitioner to expect.  Normalising instead by the
best attainable deployment loss is unstable: that denominator is near zero in the
handful of problems where some model nearly interpolates the deployment region.
"""
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from pathlib import Path

R = Path(__file__).resolve().parents[1] / 'results' / 'universal'
d = pd.read_csv(R / 'robust_raw.csv')

rows = []
for (ds, se, dep), g in d.groupby(['dataset', 'seed', 'deployment']):
    mu, nu = g.mu_loss.values, g.nu_loss.values
    f = int(np.argmin(mu))                       # the criterion's choice
    best = nu.min()
    rows.append(dict(dataset=ds, seed=se, deployment=dep,
                     sd_r=float(g.sd_r.iloc[0]),
                     rr=(nu[f] - best) / mu[f],  # regret, per unit benchmark risk
                     rr_best=(nu[f] - best) / best if best > 0 else np.nan,
                     best=best))
s = pd.DataFrame(rows); s.to_csv(R / 'regret_ratio.csv', index=False)

print(f'{len(s)} selection problems\n')
print('regret ratio = excess deployment loss / benchmark risk of the chosen model')
for q in [.5, .75, .9, .95, .99]:
    print(f'   {int(100*q):2d}th percentile {100*s.rr.quantile(q):8.1f}%')
print(f'   mean {100*s.rr.mean():.1f}%   max {100*s.rr.max():.0f}%')
print(f'   exceeds 10% in {100*(s.rr>.10).mean():.1f}% of problems, '
      f'25% in {100*(s.rr>.25).mean():.1f}%')
print('\nby mismatch quartile:')
print(s.groupby(pd.qcut(s.sd_r, 4, duplicates='drop'), observed=True).agg(
    sd_r=('sd_r', 'median'), mean=('rr', lambda x: 100*x.mean()),
    p90=('rr', lambda x: 100*x.quantile(.9))).round(1).to_string())
hi = s[s.rr_best > 1]
print(f'\nwhy not normalise by the best attainable loss: in the {len(hi)} problems '
      f'where that ratio exceeds one,\nits denominator has median {hi.best.median():.3g} '
      f'against {s[s.rr_best<=1].best.median():.3g} elsewhere.')
