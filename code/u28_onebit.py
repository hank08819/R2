"""u28_onebit.py -- one bit of direction, where a budget on magnitude cannot help.

A budget bounds how large Cov(d, r) can be and says nothing about its sign, which
is what decides an ordering.  The sign is a question a practitioner can answer
without naming a measure: sort the instances by the loss difference between two
models and ask which end the deployment weights more heavily.  This supplies that
one bit per pair from the true deployment and asks how much it buys.
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
    for dep, h in g.groupby('deployment'):
        k = float(h.sd_r.iloc[0])
        E = np.zeros((K, K)); S = np.zeros((K, K)); N = np.zeros((K, K))
        for _, x in h.iterrows():
            i, j = ix[x.left], ix[x.right]
            E[i, j], E[j, i] = x.e_mu, -x.e_mu
            N[i, j], N[j, i] = x.e_nu, -x.e_nu
            S[i, j] = S[j, i] = x.sd_d
        C = N - E                                  # the covariance, per pair
        sgn = np.sign(C)
        est = E + sgn * k * S                      # one elicited bit per pair
        rmu, rnu = E.sum(1) / K, N.sum(1) / K
        f_r2 = int(np.argmin(rmu))
        f_1b = int(np.argmin(est.sum(1)))
        gstar = int(np.argmin(rnu)); best = rnu.min()
        spread = rnu.max() - rnu.min()
        rows.append(dict(dataset=ds, seed=se, deployment=dep, sd_r=k,
                         exc_r2=(rnu[f_r2]-best)/spread if spread > 0 else 0.,
                         exc_1b=(rnu[f_1b]-best)/spread if spread > 0 else 0.,
                         wrong_r2=int(f_r2 != gstar), wrong_1b=int(f_1b != gstar),
                         r2_was_right=int(f_r2 == gstar)))
s = pd.DataFrame(rows); s.to_csv(R / 'onebit.csv', index=False)

print(f'{len(s)} selection problems\n')
print('                        picks wrong (%)   mean excess (%)   90th pct (%)')
for lab, w, e in [('R^2 (no elicitation)', s.wrong_r2, s.exc_r2),
                  ('one sign bit per pair', s.wrong_1b, s.exc_1b)]:
    print(f'  {lab:<22} {100*w.mean():10.1f} {100*e.mean():16.1f} '
          f'{100*e.quantile(.9):14.1f}')
print(f'\n  improved {int((s.exc_1b<s.exc_r2-1e-12).sum())}   '
      f'worsened {int((s.exc_1b>s.exc_r2+1e-12).sum())}   '
      f'unchanged {int((abs(s.exc_1b-s.exc_r2)<=1e-12).sum())}')
q = s[s.r2_was_right == 0]
print(f'\nrestricted to the {len(q)} problems where the criterion errs:')
print(f'  R^2 mean excess {100*q.exc_r2.mean():.1f}%  ->  '
      f'one bit {100*q.exc_1b.mean():.1f}%')
print(f'  the bit finds the deployment optimum in {100*(1-q.wrong_1b.mean()):.1f}% '
      f'of them')
