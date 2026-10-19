"""u83_conditional_shift.py -- what reweighting misses on the three real shifts, and an allowance for it.

For the benchmark's choice f in a deployment Q with stated weight r, the true gap to a rival g
splits exactly as
    E_Q[d_fg] = E_mu[r d_fg] + Delta_fg,
the first part what the reweighted benchmark sees, the second what it cannot: a change in how
the outcome follows from the inputs, plus sampling error.  Per selection problem this reads
    D = max_g (true gap - reweighted gap) / benchmark risk of f,
the realized conditional excess on the regret-ratio scale.  With an allowance beta, step 2
certifies when rr_ucb + beta <= eps; beta_crit = eps - rr_ucb is the most a certificate can absorb.
The allowance is learned from other deployments: beta_k is the 90th percentile of D over every
other deployment of the same case, seed and weight (prereg_commute_addendum.md; pre-registered
for the commute and weather cases, applied after the fact to income and Beijing).

Reads acs_problems_u78b.csv, beijing_problems_u79b.csv, commute_problems.csv, weather_problems.csv.
Writes conditional_shift.csv (one row per problem) and conditional_shift_summary.csv.
"""
import json
import numpy as np, pandas as pd
from pathlib import Path

R = Path(__file__).resolve().parents[1] / 'results' / 'universal'
CASES = {'census income': ('acs_problems_u78b.csv', ['state', 'year']),
         'Beijing': ('beijing_problems_u79b.csv', ['year', 'month']),
         'census commute': ('commute_problems.csv', ['state', 'year']),
         'Shifts weather': ('weather_problems.csv', ['cell'])}
EPS, Q = (0.05, 0.10), 0.90

rows, summ = [], []
for case, (fn, place) in CASES.items():
    if not (R / fn).exists():
        print(f'{case}: {fn} missing'); continue
    p = pd.read_csv(R / fn)
    th_h, th_t = np.array([json.loads(v) for v in p.th_hat]), np.array([json.loads(v) for v in p.th_true])
    p['D'] = (th_t - th_h).max(1) / p.m_f
    p['place'] = p[place].astype(str).agg(' '.join, axis=1)
    beta = np.empty(len(p))
    for _, g in p.groupby(['seed', 'weight']):
        for i in g.index:
            beta[p.index.get_loc(i)] = np.quantile(g.D[g.place != p.place[i]], Q)
    p['beta'] = beta; p['case'] = case
    rows.append(p[['case', 'place', 'seed', 'weight', 'opt_true', 'rr_true', 'rr_ucb_j', 'D', 'beta']])
    for w, u in p.groupby('weight'):
        for e in EPS:
            plain, allow, bad = u.rr_ucb_j <= e, u.rr_ucb_j + u.beta <= e, u.rr_true > e
            summ.append(dict(case=case, weight=w, eps=e, n=len(u), n_bad=int(bad.sum()),
                             D_median=u.D.median(), D_90=u.D.quantile(.9), beta_median=u.beta.median(),
                             cert_plain=plain.mean(), wrong_plain=int((plain & bad).sum()),
                             cert_allow=allow.mean(), wrong_allow=int((allow & bad).sum())))
out = pd.concat(rows); out.to_csv(R / 'conditional_shift.csv', index=False)
s = pd.DataFrame(summ); s.to_csv(R / 'conditional_shift_summary.csv', index=False)
pd.set_option('display.width', 220)
print(s.round(4).to_string(index=False))
