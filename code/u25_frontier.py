"""u25_frontier.py -- accuracy and robustness as two objectives, not one.

Ranking by the criterion alone is the corner of a frontier.  The other axis is
the worst regret a stated mismatch budget can inflict, which Proposition 2 makes
a closed form in the pairwise gaps.  Giving up a stated fraction of benchmark
accuracy buys a position further along that frontier; this measures what the
trade buys on the deployment measures the suite actually contains.
"""
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from pathlib import Path

R = Path(__file__).resolve().parents[1] / 'results' / 'universal'
d = pd.read_csv(R / 'scale_r2.csv')
d['left'] = d.pair.str.split('-').str[0]; d['right'] = d.pair.str.split('-').str[1]
EPS = [0.0, .02, .05, .10, .20, .35, .50, 1.0]

rows = []
for (ds, se), g in d.groupby(['dataset', 'seed']):
    ms = sorted(set(g.left) | set(g.right)); ix = {m: i for i, m in enumerate(ms)}
    K = len(ms); E = np.zeros((K, K)); S = np.zeros((K, K))
    for _, x in g.iterrows():
        i, j = ix[x.left], ix[x.right]
        E[i, j], E[j, i] = x.e_mu, -x.e_mu
        S[i, j] = S[j, i] = x.sd_d
    rmu = E.sum(1) / K                                  # benchmark risks, up to a shift
    span = rmu.max() - rmu.min()
    for dep, h in g.groupby('deployment'):
        k = float(h.sd_r.iloc[0])
        Enu = np.zeros((K, K))
        for _, x in h.iterrows():
            i, j = ix[x.left], ix[x.right]
            Enu[i, j], Enu[j, i] = x.e_nu, -x.e_nu
        rnu = Enu.sum(1) / K
        best = rnu.min(); spread = rnu.max() - rnu.min()
        W = (E + k * S).max(1)                          # worst-case regret at budget k
        rec = dict(dataset=ds, seed=se, deployment=dep, sd_r=k,
                   n_frontier=0)
        # Pareto set in (benchmark risk, worst-case regret)
        front = [f for f in range(K)
                 if not any((rmu[j] <= rmu[f]) and (W[j] <= W[f]) and
                            ((rmu[j] < rmu[f]) or (W[j] < W[f])) for j in range(K))]
        rec['n_frontier'] = len(front)
        for e in EPS:
            ok = np.where(rmu <= rmu.min() + e * span)[0]
            f = int(ok[np.argmin(W[ok])])
            rec[f'exc_{e}'] = (rnu[f] - best) / spread if spread > 0 else 0.
            rec[f'wrong_{e}'] = int(rnu[f] > best + 1e-12)
        rows.append(rec)
s = pd.DataFrame(rows); s.to_csv(R / 'frontier.csv', index=False)

print(f'{len(s)} selection problems.  Pareto frontier holds a median of '
      f'{s.n_frontier.median():.0f} of {K} models '
      f'(range {s.n_frontier.min()}-{s.n_frontier.max()}).\n')
print('accuracy tolerance     picks wrong (%)   mean excess (%)   90th pct (%)')
for e in EPS:
    w, x = s[f'wrong_{e}'], s[f'exc_{e}']
    tag = '   <- R^2' if e == 0 else ''
    print(f'  {e:5.2f} of the span {100*w.mean():12.1f} {100*x.mean():17.1f} '
          f'{100*x.quantile(.9):14.1f}{tag}')
b = s['exc_0.0']
for e in EPS[1:]:
    x = s[f'exc_{e}']
    print(f'\n  eps={e:g}: improves {int((x<b-1e-12).sum())}, '
          f'worsens {int((x>b+1e-12).sum())}, unchanged {int((abs(x-b)<=1e-12).sum())}')
