"""u29_moo_ratio.py -- the second objective as a rate, and at a realistic magnitude.

Worst-case regret in absolute terms is nearly collinear with benchmark risk, so a
frontier built on the two carries little information.  Expressed as a rate, per
unit of the candidate's own benchmark risk, the two objectives separate.  The
budget is also charged at the correlation deployments actually exhibit rather than
at the worst the budget permits, which the suite puts near a tenth.
"""
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from pathlib import Path

R = Path(__file__).resolve().parents[1] / 'results' / 'universal'
p = pd.read_csv(R / 'scale_r2.csv')
p['left'] = p.pair.str.split('-').str[0]; p['right'] = p.pair.str.split('-').str[1]
m = pd.read_csv(R / 'robust_raw.csv')
RHO_TYP = float(p.rho.abs().median())
LAM = [1.0, .5, .25, RHO_TYP]
EPS = [0.0, .02, .05, .10, .20, .35]

base = {(r.dataset, r.seed, r.deployment, r.model): (r.mu_loss, r.nu_loss)
        for r in m.itertuples()}
rows = []
for (ds, se, dep), h in p.groupby(['dataset', 'seed', 'deployment']):
    ms = sorted(set(h.left) | set(h.right)); ix = {x: i for i, x in enumerate(ms)}
    K = len(ms)
    try:
        mu_l = np.array([base[(ds, se, dep, x)][0] for x in ms])
        nu_l = np.array([base[(ds, se, dep, x)][1] for x in ms])
    except KeyError:
        continue
    if (mu_l <= 0).any():
        continue
    k = float(h.sd_r.iloc[0])
    E = np.zeros((K, K)); S = np.zeros((K, K))
    for _, x in h.iterrows():
        i, j = ix[x.left], ix[x.right]
        E[i, j], E[j, i] = x.e_mu, -x.e_mu
        S[i, j] = S[j, i] = x.sd_d
    best = nu_l.min(); spread = nu_l.max() - best
    span = mu_l.max() - mu_l.min()
    rec = dict(dataset=ds, seed=se, deployment=dep, sd_r=k)
    for lam in LAM:
        W_abs = (E + lam * k * S).max(1)
        W_rel = W_abs / mu_l                       # the regret as a rate
        for tag, W in [('abs', W_abs), ('rel', W_rel)]:
            for e in EPS:
                ok = np.where(mu_l <= mu_l.min() + e * span)[0]
                f = int(ok[np.argmin(W[ok])])
                rec[f'{tag}_{lam:.3f}_{e}'] = ((nu_l[f]-best)/spread
                                               if spread > 0 else 0.)
    rows.append(rec)
s = pd.DataFrame(rows); s.to_csv(R / 'moo_ratio.csv', index=False)

print(f'{len(s)} selection problems.  typical |rho| = {RHO_TYP:.3f}\n')
print('mean excess (%), rows = budget charged at lam*kappa, cols = accuracy given up')
for tag, lab in [('abs', 'absolute regret'), ('rel', 'regret ratio')]:
    print(f'\n  {lab}')
    print('    lam    ' + ''.join(f'{e:>8.2f}' for e in EPS))
    for lam in LAM:
        cells = ''.join(f'{100*s[f"{tag}_{lam:.3f}_{e}"].mean():8.1f}' for e in EPS)
        print(f'   {lam:5.3f}  {cells}')
b = s['abs_1.000_0.0']
print(f'\n  baseline (R^2, eps=0): {100*b.mean():.1f}%')
best = min(((t, l, e, s[f'{t}_{l:.3f}_{e}'].mean())
            for t in ['abs', 'rel'] for l in LAM for e in EPS), key=lambda z: z[3])
print(f'  best cell: {best[0]}, lam={best[1]:.3f}, eps={best[2]} -> {100*best[3]:.1f}%')
