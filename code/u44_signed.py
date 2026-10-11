"""u44_signed.py -- one bit of direction, charged at a realistic magnitude.

A budget bounds how large the covariance can be and leaves its sign free, which is
why hedging fails.  The sign is a question a practitioner can answer without naming
a measure.  Pairing that bit with the worst-case magnitude overshoots, because the
correlation deployments actually exhibit is far below one.  This pairs it with a
fixed typical correlation instead, and sweeps that constant.
"""
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from pathlib import Path

R = Path(__file__).resolve().parents[1] / 'results' / 'universal'
d = pd.read_csv(R / 'scale_r2.csv')
d['left'] = d.pair.str.split('-').str[0]; d['right'] = d.pair.str.split('-').str[1]
RHO = [0.0, .025, .05, .077, .10, .15, .25, .50, 1.0]

rows = []
for (ds, se, dep), h in d.groupby(['dataset', 'seed', 'deployment']):
    ms = sorted(set(h.left) | set(h.right)); ix = {m: i for i, m in enumerate(ms)}
    K = len(ms)
    E = np.zeros((K, K)); S = np.zeros((K, K)); N = np.zeros((K, K))
    for _, x in h.iterrows():
        i, j = ix[x.left], ix[x.right]
        E[i, j], E[j, i] = x.e_mu, -x.e_mu
        N[i, j], N[j, i] = x.e_nu, -x.e_nu
        S[i, j] = S[j, i] = x.sd_d
    k = float(h.sd_r.iloc[0])
    sgn = np.sign(N - E)                       # the elicited bit, one per pair
    rnu = N.sum(1) / K
    best = rnu.min(); spread = rnu.max() - best
    rec = dict(dataset=ds, seed=se, deployment=dep, sd_r=k)
    for rho in RHO:
        est = E + sgn * rho * k * S
        f = int(np.argmin(est.sum(1)))
        rec[f'e{rho}'] = (rnu[f] - best) / spread if spread > 0 else 0.
        rec[f'w{rho}'] = int(rnu[f] > best + 1e-12)
    rows.append(rec)
s = pd.DataFrame(rows); s.to_csv(R / 'signed.csv', index=False)

print(f'{len(s)} selection problems.  One elicited sign per pair, magnitude charged '
      f'at a fixed correlation.\n')
print('  assumed |rho|   picks wrong (%)   mean excess (%)   90th pct (%)   improved / damaged')
b = s['e0.0']
for rho in RHO:
    e, w = s[f'e{rho}'], s[f'w{rho}']
    tag = '   <- the criterion' if rho == 0 else ''
    imp = int((e < b - 1e-12).sum()); dam = int((e > b + 1e-12).sum())
    print(f'  {rho:<15.3f}{100*w.mean():10.1f}{100*e.mean():18.1f}'
          f'{100*e.quantile(.9):15.1f}      {imp:4d} / {dam}{tag}')
q = s[b > 1e-12]
print(f'\nrestricted to the {len(q)} problems the criterion gets wrong:')
for rho in [.05, .077, .10, .15]:
    print(f'  |rho|={rho:<6} mean excess {100*q[f"e{rho}"].mean():5.1f}%  '
          f'against {100*q["e0.0"].mean():.1f}%   '
          f'fully corrected {100*(q[f"e{rho}"]<1e-12).mean():.1f}%')


# ------------------------------------------ how many questions does it need? ---
print('\n' + '=' * 74)
print('eliciting the sign only for pairs among the leading candidates\n')
rows = []
for (ds, se, dep), h in d.groupby(['dataset', 'seed', 'deployment']):
    ms = sorted(set(h.left) | set(h.right)); ix = {m: i for i, m in enumerate(ms)}
    K = len(ms)
    E = np.zeros((K, K)); S = np.zeros((K, K)); N = np.zeros((K, K))
    for _, x in h.iterrows():
        i, j = ix[x.left], ix[x.right]
        E[i, j], E[j, i] = x.e_mu, -x.e_mu
        N[i, j], N[j, i] = x.e_nu, -x.e_nu
        S[i, j] = S[j, i] = x.sd_d
    k = float(h.sd_r.iloc[0]); rho = .05
    rmu = E.sum(1) / K; rnu = N.sum(1) / K
    order = np.argsort(rmu)
    best = rnu.min(); spread = rnu.max() - best
    rec = dict(dataset=ds, seed=se, deployment=dep)
    for m in [2, 3, 4, 10]:
        top = set(order[:m].tolist())
        sgn = np.zeros((K, K))
        for i in range(K):
            for j in range(K):
                if i in top and j in top and i != j:
                    sgn[i, j] = np.sign(N[i, j] - E[i, j])
        est = E + sgn * rho * k * S
        f = int(np.argmin(est.sum(1)))
        rec[f'm{m}'] = (rnu[f] - best) / spread if spread > 0 else 0.
        rec[f'q{m}'] = m * (m - 1) // 2
    rec['r2'] = (rnu[int(np.argmin(rmu))] - best) / spread if spread > 0 else 0.
    rows.append(rec)
t = pd.DataFrame(rows); t.to_csv(R / 'signed_top.csv', index=False)      # the six answers: m4
print('  leaders asked about   questions   mean excess (%)   90th pct (%)   improved / damaged')
print(f'  {"none (the criterion)":<21}{0:>10}{100*t.r2.mean():>18.1f}'
      f'{100*t.r2.quantile(.9):>15.1f}')
for m in [2, 3, 4, 10]:
    e = t[f'm{m}']
    print(f'  {"top "+str(m):<21}{m*(m-1)//2:>10}{100*e.mean():>18.1f}'
          f'{100*e.quantile(.9):>15.1f}      '
          f'{int((e < t.r2-1e-12).sum()):4d} / {int((e > t.r2+1e-12).sum())}')


# ----------------------------------------- what if the practitioner is wrong? ---
print('\n' + '=' * 74)
print('robustness to a practitioner who answers some questions wrongly\n')
rng = np.random.default_rng(17)
res = []
for p in [0.0, .1, .2, .3, .4, .5]:
    acc = []
    for _ in range(20):
        tot = []
        for (ds, se, dep), h in d.groupby(['dataset', 'seed', 'deployment']):
            ms = sorted(set(h.left) | set(h.right)); ix = {m: i for i, m in enumerate(ms)}
            K = len(ms)
            E = np.zeros((K, K)); S = np.zeros((K, K)); N = np.zeros((K, K))
            for _, x in h.iterrows():
                i, j = ix[x.left], ix[x.right]
                E[i, j], E[j, i] = x.e_mu, -x.e_mu
                N[i, j], N[j, i] = x.e_nu, -x.e_nu
                S[i, j] = S[j, i] = x.sd_d
            k = float(h.sd_r.iloc[0])
            rmu = E.sum(1) / K; rnu = N.sum(1) / K
            top = set(np.argsort(rmu)[:4].tolist())
            sgn = np.zeros((K, K))
            for i in range(K):
                for j in range(i + 1, K):
                    if i in top and j in top:
                        s0 = np.sign(N[i, j] - E[i, j])
                        if rng.random() < p:
                            s0 = -s0
                        sgn[i, j], sgn[j, i] = s0, -s0
            f = int(np.argmin((E + sgn * .05 * k * S).sum(1)))
            best = rnu.min(); spread = rnu.max() - best
            tot.append((rnu[f] - best) / spread if spread > 0 else 0.)
        acc.append(np.mean(tot))
    res.append((p, 100 * np.mean(acc)))
base = 100 * t.r2.mean()
pd.DataFrame(res, columns=['p_wrong', 'mean_excess_pct']).to_csv(R / 'signed_noise.csv', index=False)
print(f'  answers wrong with probability   mean excess (%)')
for p, v in res:
    tag = '   <- still better than the criterion' if v < base else '   <- worse'
    print(f'  {p:<32.1f}{v:8.1f}{tag}')
print(f'  {"the criterion itself":<32}{base:8.1f}')
