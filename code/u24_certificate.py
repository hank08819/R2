"""u24_certificate.py -- what a benchmark settles, and the budget up to which it does.

Without a constraint on the deployment measure a benchmark settles nothing: every
model that is best at even one instance is optimal under the point mass there, so
none can be ruled out.  Constrain the deployment to a chi-squared budget and the
question has an exact answer, a single scalar per selection problem, the smallest
budget at which any rival can overtake the benchmark's choice.
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
    K = len(ms)
    E = np.zeros((K, K)); S = np.zeros((K, K))
    for _, x in g.iterrows():
        i, j = ix[x.left], ix[x.right]
        E[i, j], E[j, i] = x.e_mu, -x.e_mu
        S[i, j] = S[j, i] = x.sd_d
    f = int(np.argmin(E.sum(1)))                       # the benchmark's choice
    margin = np.array([E[j, f] for j in range(K)])     # E_mu[l_g] - E_mu[l_f] >= 0
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.where(np.arange(K) == f, np.inf, margin / np.where(S[f] > 0, S[f], np.nan))
    kstar = float(np.nanmin(ratio))
    for dep, h in g.groupby('deployment'):
        sd_r = float(h.sd_r.iloc[0])
        Enu = np.zeros((K, K))
        for _, x in h.iterrows():
            i, j = ix[x.left], ix[x.right]
            Enu[i, j], Enu[j, i] = x.e_nu, -x.e_nu
        gstar = int(np.argmin(Enu.sum(1)))
        rows.append(dict(dataset=ds, seed=se, deployment=dep, kstar=kstar,
                         sd_r=sd_r, certified=int(sd_r <= kstar),
                         flipped=int(gstar != f)))
s = pd.DataFrame(rows); s.to_csv(R / 'certificate.csv', index=False)

cov = s[s.certified == 1]
print(f'{len(s)} selection problems, {s.dataset.nunique()} datasets\n')
print('kappa* , the budget up to which the benchmark choice is guaranteed:')
for q in [.1, .25, .5, .75, .9]:
    print(f'   {int(100*q):2d}th percentile {s.kstar.quantile(q):8.3f}')
print(f'   zero in {100*(s.kstar<=1e-12).mean():.1f}% of problems '
      f'(a rival ties on the benchmark)')
print(f'\ncertificate covers the actual mismatch in {100*s.certified.mean():.1f}% '
      f'of problems ({len(cov)} of {len(s)})')
print(f'VIOLATIONS among those certified: {int(cov.flipped.sum())}  '
      f'(must be 0)')
print(f'flip rate where NOT certified: {100*s[s.certified==0].flipped.mean():.1f}%')
