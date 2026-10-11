"""u14_robust_eval.py -- does the kappa-robust criterion actually pick better?

For every selection problem in the suite, compare the model chosen by the ordinary
criterion (kappa = 0, that is R-squared) with the one chosen at a positive budget,
and score both by the loss they actually incur under the deployment measure.  The
budget a practitioner would state is the mismatch of the use they have in mind, so
kappa = sd(r) is reported alongside a sweep that assumes no such knowledge.
"""
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from pathlib import Path

R = Path(__file__).resolve().parents[1] / 'results' / 'universal'
d = pd.read_csv(R / 'robust_raw.csv')
CELL = ['dataset', 'seed', 'deployment']
KAPPA = [0.0, .25, .5, 1.0, 1.5, 2.0, 3.0, 5.0]

rows = []
for key, g in d.groupby(CELL):
    mu, sd, nu = g.mu_loss.values, g.sd_loss.values, g.nu_loss.values
    sd_r = float(g.sd_r.iloc[0])
    best = nu.min(); spread = nu.max() - nu.min()
    rec = dict(dataset=key[0], seed=key[1], deployment=key[2], sd_r=sd_r,
               spread=spread)
    for k in KAPPA + ['stated']:
        kk = sd_r if k == 'stated' else k
        j = int(np.argmin(mu + kk * sd))
        rec[f'exc_{k}'] = (nu[j] - best) / spread if spread > 0 else 0.0
        rec[f'wrong_{k}'] = int(nu[j] > best)
    rows.append(rec)
s = pd.DataFrame(rows); s.to_csv(R / 'robust_eval.csv', index=False)

print(f'{len(s):,} selection problems, {d.dataset.nunique()} datasets\n')
print('kappa      wrong (%)   excess, median (%)   excess, 90th pct (%)')
for k in KAPPA + ['stated']:
    e, w = s[f'exc_{k}'], s[f'wrong_{k}']
    lab = 'sd(r)' if k == 'stated' else f'{k:g}'
    star = '   <- R^2' if k == 0.0 else ''
    print(f'  {lab:<8} {100*w.mean():7.1f}   {100*e.median():14.1f} '
          f'{100*e.quantile(.9):19.1f}{star}')

base, rob = s['exc_0.0'], s['exc_stated']
print(f'\nagainst R^2, with the budget set to the mismatch of the intended use:')
print(f'  mean excess          {100*base.mean():.1f}%  ->  {100*rob.mean():.1f}%')
print(f'  90th percentile      {100*base.quantile(.9):.1f}%  ->  {100*rob.quantile(.9):.1f}%')
print(f'  problems improved    {int((rob < base - 1e-12).sum())}')
print(f'  problems worsened    {int((rob > base + 1e-12).sum())}')
print(f'  unchanged            {int((np.abs(rob-base) <= 1e-12).sum())}')
b = s.groupby(pd.qcut(s.sd_r, 4, duplicates='drop'), observed=True).apply(
    lambda g: pd.Series({'sd_r': g.sd_r.median(),
                         'R2': 100*g['exc_0.0'].mean(),
                         'robust': 100*g['exc_stated'].mean()}))
print('\nby mismatch quartile (mean excess, %):')
print(b.round(1).to_string())
