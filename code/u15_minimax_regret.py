"""u15_minimax_regret.py -- minimax-regret selection, and why the naive rule fails.

Worst-case loss lets the adversary attack each candidate with its own density
ratio.  One ratio in fact applies to all of them, so the object to minimise is the
worst-case regret, whose closed form replaces the dispersion of a model's own loss
by the dispersion of the difference between two models.  That difference cancels
the irreducible error the candidates share, which is what the naive rule spends
its whole budget on.
"""
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from pathlib import Path

R = Path(__file__).resolve().parents[1] / 'results' / 'universal'
d = pd.read_csv(R / 'scale_r2.csv')
d['left'] = d.pair.str.split('-').str[0]
d['right'] = d.pair.str.split('-').str[1]
CELL = ['dataset', 'seed', 'deployment']
KAPPA = [0.0, .25, .5, 1.0, 2.0, 3.0]

rows = []
for key, g in d.groupby(CELL):
    ms = sorted(set(g.left) | set(g.right)); ix = {m: i for i, m in enumerate(ms)}
    K = len(ms)
    Emu = np.zeros((K, K)); Enu = np.zeros((K, K)); SD = np.zeros((K, K))
    for _, x in g.iterrows():
        i, j = ix[x.left], ix[x.right]
        Emu[i, j], Emu[j, i] = x.e_mu, -x.e_mu
        Enu[i, j], Enu[j, i] = x.e_nu, -x.e_nu
        SD[i, j] = SD[j, i] = x.sd_d
    rnu = Enu.sum(1) / K                      # deployment risks up to a shift
    best = rnu.min(); spread = rnu.max() - rnu.min()
    rec = dict(dataset=key[0], seed=key[1], deployment=key[2],
               sd_r=float(g.sd_r.iloc[0]), spread=spread)
    for k in KAPPA + ['stated']:
        kk = float(g.sd_r.iloc[0]) if k == 'stated' else k
        W = (Emu + kk * SD).max(1)            # worst-case regret of each model
        j = int(np.argmin(W))
        rec[f'exc_{k}'] = (rnu[j] - best) / spread if spread > 0 else 0.
        rec[f'wrong_{k}'] = int(rnu[j] > best + 1e-12)
    rows.append(rec)
s = pd.DataFrame(rows); s.to_csv(R / 'minimax_regret.csv', index=False)

print(f'{len(s):,} selection problems\n')
print('kappa      picks a model deployment would not (%)   excess: median   90th pct')
for k in KAPPA + ['stated']:
    e, w = s[f'exc_{k}'], s[f'wrong_{k}']
    lab = 'sd(r)' if k == 'stated' else f'{k:g}'
    star = '   <- R^2' if k == 0.0 else ''
    print(f'  {lab:<8} {100*w.mean():27.1f} {100*e.median():15.1f} '
          f'{100*e.quantile(.9):10.1f}{star}')

b, r = s['exc_0.0'], s['exc_stated']
print(f'\nR^2 -> minimax regret with the budget set to the intended use:')
print(f'  mean excess       {100*b.mean():5.1f}%  ->  {100*r.mean():5.1f}%')
print(f'  90th percentile   {100*b.quantile(.9):5.1f}%  ->  {100*r.quantile(.9):5.1f}%')
print(f'  improved {int((r<b-1e-12).sum())}   worsened {int((r>b+1e-12).sum())}   '
      f'unchanged {int((np.abs(r-b)<=1e-12).sum())}')
