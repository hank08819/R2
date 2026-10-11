"""u9_scale_stats.py -- what the suite-scale run says, at the pair level and at
the level a practitioner actually faces: which single model gets chosen."""
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from pathlib import Path
from scipy.stats import spearmanr

R = Path(__file__).resolve().parents[1] / 'results' / 'universal'
d = pd.read_csv(R / 'scale_r2.csv')
d['left'] = d.pair.str.split('-').str[0]
d['right'] = d.pair.str.split('-').str[1]
CELL = ['dataset', 'seed', 'deployment']

print(f'datasets              {d.dataset.nunique()}')
print(f'deployment measures   {d.deployment.nunique()}')
print(f'splits per dataset    {d.seed.nunique()}')
print(f'model-pair records    {len(d):,}')
print(f'selection problems    {d.groupby(CELL).ngroups:,}')
print(f'\npair reversal rate    {100*d.inverted.mean():.1f}%')

per = d.groupby('dataset').inverted.mean().mul(100)
print(f'by dataset            median {per.median():.1f}%  range {per.min():.1f}-{per.max():.1f}%'
      f'  with at least one reversal {int((per>0).sum())}/{len(per)}')
print('\nby deployment measure')
print(d.groupby('deployment').agg(pairs=('inverted', 'size'),
      sd_r=('sd_r', 'median'), rev=('inverted', lambda x: 100*x.mean())
      ).round(1).to_string())

q = pd.qcut(d.sd_r, 5, duplicates='drop')
print('\nby mismatch quintile')
print(d.groupby(q, observed=True).agg(sd_r=('sd_r', 'median'),
      rev=('inverted', lambda x: 100*x.mean())).round(2).to_string())
rho, p = spearmanr(d.sd_r, d.inverted)
print(f'\nspearman(sd_r, reversed) over all records  {rho:.3f}  (p={p:.1e})')

# ---- selection level: pairwise gaps recover the ordering of the risks ---
rows = []
for key, g in d.groupby(CELL):
    models = sorted(set(g.left) | set(g.right))
    idx = {m: i for i, m in enumerate(models)}
    K = len(models)
    Amu = np.zeros((K, K)); Anu = np.zeros((K, K))
    for _, x in g.iterrows():
        i, j = idx[x.left], idx[x.right]
        Amu[i, j], Amu[j, i] = x.e_mu, -x.e_mu
        Anu[i, j], Anu[j, i] = x.e_nu, -x.e_nu
    rmu, rnu = Amu.sum(1) / K, Anu.sum(1) / K       # risks up to a common shift
    f, gs = int(np.argmin(rmu)), int(np.argmin(rnu))
    spread = rnu.max() - rnu.min()
    rows.append(dict(dataset=key[0], seed=key[1], deployment=key[2],
                     sd_r=g.sd_r.iloc[0], wrong=int(f != gs),
                     regret=rnu[f] - rnu[gs],
                     rel=(rnu[f] - rnu[gs]) / spread if spread > 0 else 0.0))
sel = pd.DataFrame(rows); sel.to_csv(R / 'scale_selection.csv', index=False)
w = sel[sel.wrong == 1]
print(f'\nselection: R2 picks a model deployment would not, in '
      f'{100*sel.wrong.mean():.1f}% of {len(sel):,} problems')
print(f'  when it does, the excess loss is {100*w.rel.median():.0f}% of the range '
      f'spanned by the candidates (median), {100*w.rel.quantile(.9):.0f}% at the 90th pct')
print(f'  datasets where it happens at least once: '
      f'{sel.groupby("dataset").wrong.max().sum()}/{sel.dataset.nunique()}')
