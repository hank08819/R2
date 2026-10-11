"""u57_cluster_intervals.py -- intervals for the two headline rates, resampled by dataset.

Comparisons within one dataset share its models, splits and deployments, so they are
not independent.  This resamples the 35 datasets with replacement, 5,000 times, and
recomputes the pair reversal rate and the selection error rate on each draw.  Writes
results/universal/cluster_intervals.csv.
"""
import numpy as np, pandas as pd
from pathlib import Path

R = Path(__file__).resolve().parents[1] / 'results' / 'universal'
sc = pd.read_csv(R / 'scale_r2.csv')
sel = pd.read_csv(R / 'scale_selection.csv')
rng = np.random.default_rng(0)
ds = np.array(sorted(sc.dataset.unique()))
g1 = sc.groupby('dataset').inverted.agg(['sum', 'size']).reindex(ds)
g2 = sel.groupby('dataset').wrong.agg(['sum', 'size']).reindex(ds, fill_value=0)
a, b = [], []
for _ in range(5000):
    s = rng.choice(ds, len(ds))
    a.append(100 * g1.loc[s, 'sum'].sum() / g1.loc[s, 'size'].sum())
    b.append(100 * g2.loc[s, 'sum'].sum() / g2.loc[s, 'size'].sum())
out = pd.DataFrame([dict(rate='pair reversal', point=100 * sc.inverted.mean(),
                         lo=np.quantile(a, .025), hi=np.quantile(a, .975)),
                    dict(rate='selection error', point=100 * sel.wrong.mean(),
                         lo=np.quantile(b, .025), hi=np.quantile(b, .975))])
out.to_csv(R / 'cluster_intervals.csv', index=False)
print(f'{len(ds)} datasets')
print(out.round(1).to_string(index=False))
