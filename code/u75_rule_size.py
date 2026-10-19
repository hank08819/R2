"""u75_rule_size.py -- what the six answers need besides the six signs.

The direction rule of u44 forms E_mu[d] + eta * rho_bar * sd_mu(r) * sd_mu(d) for the six
pairs among the four leading candidates.  The sign eta is the elicited answer; the size
sd_mu(r) is, in u44, the mismatch the deployment actually exhibits.  A reviewer asked how
much of the rule's gain comes from that size rather than from the six answers.  This
reruns the rule with the size (i) read from the deployment, as in u44, (ii) fixed at the
suite median of sd_mu(r) over the selection problems, the same number for every problem,
(iii) fixed at one, and (iv) half and (v) twice the true size.  rho_bar = 0.05 throughout.
Writes results/universal/rule_size.csv, one row per selection problem.
"""
import numpy as np, pandas as pd
from pathlib import Path

R = Path(__file__).resolve().parents[1] / 'results' / 'universal'
RHO = 0.05
d = pd.read_csv(R / 'scale_r2.csv')
d['left'] = d.pair.str.split('-').str[0]; d['right'] = d.pair.str.split('-').str[1]
MED = float(d.groupby(['dataset', 'seed', 'deployment']).sd_r.first().median())

rows = []
for (ds, se, dep), h in d.groupby(['dataset', 'seed', 'deployment']):
    ms = sorted(set(h.left) | set(h.right)); ix = {m: i for i, m in enumerate(ms)}; K = len(ms)
    E = np.zeros((K, K)); S = np.zeros((K, K)); N = np.zeros((K, K))
    for _, x in h.iterrows():
        i, j = ix[x.left], ix[x.right]
        E[i, j], E[j, i] = x.e_mu, -x.e_mu
        N[i, j], N[j, i] = x.e_nu, -x.e_nu
        S[i, j] = S[j, i] = x.sd_d
    k = float(h.sd_r.iloc[0])
    sgn = np.sign(N - E)                                   # the elicited answer, one per pair
    rnu = N.sum(1) / K; best = rnu.min(); spread = rnu.max() - best
    top = np.argsort(E.sum(1))[:4]                          # the six pairs among the leading four
    mask = np.zeros((K, K)); mask[np.ix_(top, top)] = 1
    rec = dict(dataset=ds, seed=se, deployment=dep, sd_r=k, size_median=MED)
    f0 = int(np.argmin(E.sum(1)))
    rec['exc_criterion'] = (rnu[f0] - best) / spread if spread > 0 else 0.
    for name, size in [('true', k), ('median', MED), ('one', 1.0), ('half', 0.5 * k), ('double', 2 * k)]:
        est = E + sgn * RHO * size * S * mask
        f = int(np.argmin(est.sum(1)))
        rec[f'exc_{name}'] = (rnu[f] - best) / spread if spread > 0 else 0.
    rows.append(rec)
t = pd.DataFrame(rows); t.to_csv(R / 'rule_size.csv', index=False)
print(f'{len(t)} selection problems; suite median sd_r = {MED:.3f}')
for c in ['criterion', 'true', 'median', 'one', 'half', 'double']:
    e = t[f'exc_{c}']
    print(f'  size {c:10s} mean excess {100*e.mean():5.1f}%   90th pct {100*e.quantile(.9):5.1f}%')
