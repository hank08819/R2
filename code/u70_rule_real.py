"""u70_rule_real.py -- does the direction rule help on the two real shifts, given correct
answers?  And can a pair's direction be read from the two ends of the sorted loss
difference, one record each or a group each?

The runs mirror u59 (census: 52 state-years, three weights) and u61 (Beijing: 24 months,
two weights), seeds 0, 1, 2, with the same random draws, so each deployment is the one
of Table S10.  For each deployment and weight r:
  criterion        the benchmark choice f0
  reweighted       argmin of the reweighted benchmark risk E_mu[r l_f], all K candidates
  rule, rho_bar    the direction rule over the four leaders, with the true pair signs
                   sign Cov_mu(r, d), at rho_bar in {0.01, 0.02, 0.05}
and the realized regret ratio of each on the deployment's own labels, all divided by the
benchmark choice's risk mu[f0], the paper's denominator.  For the six leading pairs, the
share whose sign is recovered by (i) one record from each end (5th-15th and 85th-95th
percentiles), (ii) the mean weight of the bottom and top tenths.  For all K(K-1)/2 pairs,
|corr_mu(r, d)|, the quantity the rule fixes at rho_bar.
Writes results/universal/rule_real.csv and rule_real_corr.csv.
    python u70_rule_real.py census|beijing
"""

import itertools, importlib.util, warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from pathlib import Path
HERE = Path(__file__).resolve().parent; R = HERE.parent / 'results' / 'universal'
def mod(name, fn):
    s = importlib.util.spec_from_file_location(name, HERE / fn); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
u59 = mod('u59', 'u59_acs_real_shift.py')
from folktables.load_acs import state_list
src = open(HERE / 'u61_beijing_real_shift.py').read().split('rows, probs = [], []')[0]
bj = {'__file__': str(HERE / 'u61_beijing_real_shift.py')}; exec(compile(src, 'u61_head', 'exec'), bj)
RHOS, TOP = (0.01, 0.02, 0.05), 4


def evaluate(case, seed, dep_id, Lb, dep, r, rng, corr_rows):
    r = r / r.mean()
    mu = Lb.mean(1); top = list(np.argsort(mu)[:TOP]); f0 = top[0]
    rr = lambda f: float((dep[f] - dep.min()) / mu[f0])
    sd_r = float(np.sqrt(np.mean((r - 1) ** 2)))
    K = len(Lb)
    for a, b in itertools.combinations(range(K), 2):
        d = Lb[a] - Lb[b]; sd = d.std()
        if sd > 0:
            corr_rows.append(dict(case=case, seed=seed, deployment=dep_id, pair=f'{a}-{b}', leading=int(a in top and b in top),
                                  abs_corr=abs(float(np.mean((r - 1) * (d - d.mean())) / (sd_r * sd)))))
    sign = {}; one, grp = [], []
    for a, b in itertools.combinations(top, 2):
        d = Lb[a] - Lb[b]; s = int(np.sign(np.mean((r - 1) * d))) or 1; sign[(a, b)] = s
        q = np.quantile(d, [.05, .1, .15, .85, .9, .95])
        lo = rng.choice(np.where((d >= q[0]) & (d <= q[2]))[0]); hi = rng.choice(np.where((d >= q[3]) & (d <= q[5]))[0])
        one.append(int((1 if r[hi] > r[lo] else -1) == s))
        grp.append(int((1 if r[d >= q[4]].mean() > r[d <= q[1]].mean() else -1) == s))
    row = dict(case=case, seed=seed, deployment=dep_id, sd_r=sd_r, rr_criterion=rr(f0),
               rr_reweighted=rr(int(np.argmin((Lb * r).mean(1)))), one_record=np.mean(one), group=np.mean(grp))
    for rho in RHOS:
        score = np.zeros(TOP)
        for i, a in enumerate(top):
            for b in top:
                if a == b: continue
                d = Lb[a] - Lb[b]; s = sign.get((a, b), -sign.get((b, a), 0))
                score[i] += d.mean() + s * rho * sd_r * d.std()
        row[f'rr_rule_{rho}'] = rr(top[int(np.argmin(score))])
    return row


import sys
which = sys.argv[1]
rows, corr_rows = [], []
pick = np.random.default_rng(12345)          # record draws for the one-record check only
if which == 'census':
    for S in [0, 1, 2]:
        rng = np.random.default_rng(S)        # the draws of u59.one
        Xca, yca = u59.load('CA', '2018'); Fca = u59.full(Xca, 'CA', '2018')
        idx = rng.permutation(len(yca))
        tr, te, pb = idx[:u59.N_TRAIN], idx[u59.N_TRAIN:u59.N_TRAIN + u59.N_TEST], idx[u59.N_TRAIN + u59.N_TEST:u59.N_TRAIN + u59.N_TEST + u59.N_POOL]
        models = u59.learners(S)
        for m in models: m.fit(Fca[tr], yca[tr])
        Lb = np.array([(m.predict(Fca[te]) - yca[te]) ** 2 for m in models])
        for st, yr in [(s, '2018') for s in state_list if s not in ('CA', 'PR')] + [('CA', y) for y in ('2019', '2021', '2022')]:
            Xd, yd = u59.load(st, yr); Fd = u59.full(Xd, st, yr)
            j = rng.permutation(len(yd)); ti, pool = j[:u59.N_TRUTH], j[u59.N_TRUTH:u59.N_TRUTH + u59.N_POOL]
            if len(pool) < 1000: h = len(yd) // 2; ti, pool = j[:h], j[h:]
            dep = np.array([(m.predict(Fd[ti]) - yd[ti]) ** 2 for m in models]).mean(1)
            W = {'classifier': u59.classifier_weight(Xca[pb], Xd[pool], Xca[te], S)[0],
                 'marginals': u59.marginal_weight(Xca[te], Xd[pool]),
                 'birthplace': u59.classifier_weight(Xca[pb], Xd[pool], Xca[te], S, birth_state=True)[0]}
            u59._cache.clear()
            for w, r in W.items():
                rows.append(evaluate(f'census-{w}', S, f'{st}-{yr}', Lb, dep, r, pick, corr_rows))
        print('census seed', S, 'done', flush=True)
else:
    d, X = bj['load'](); F = bj['full'](X); y = d.y.values; bench = d.year <= 2012
    for S in [0, 1, 2]:
        rng = np.random.default_rng(S)        # the draws of u61
        days = np.unique(d.day[bench])
        test_days = set(rng.choice(days, size=len(days) * 2 // 5, replace=False)); is_test = d.day.isin(test_days).values
        trb = np.where(bench & ~is_test)[0]; te_all = np.where(bench & is_test)[0]; te_all = te_all[rng.permutation(len(te_all))]
        pool_b, teb = te_all[: len(te_all) // 2], te_all[len(te_all) // 2:]
        models = bj['learners'](S)
        for m in models: m.fit(F[trb], y[trb])
        Lbb = np.array([(m.predict(F[teb]) - y[teb]) ** 2 for m in models])
        for yr in (2013, 2014):
            for mo in range(1, 13):
                ix = np.where((d.year == yr) & (d.month == mo))[0]; dd = np.unique(d.day[ix])
                half = set(rng.choice(dd, size=len(dd) // 2, replace=False))
                lab = ix[~d.day[ix].isin(half).values]; unl = ix[d.day[ix].isin(half).values]
                dep = np.array([(m.predict(F[lab]) - y[lab]) ** 2 for m in models]).mean(1)
                W = {'calendar': (d.month.values[teb] == mo).astype(float),
                     'classifier': bj['classifier_weight'](X[pool_b], X[unl], X[teb], S)}
                for w, r in W.items():
                    rows.append(evaluate(f'beijing-{w}', S, f'{yr}-{mo:02d}', Lbb, dep, r, pick, corr_rows))
        print('beijing seed', S, 'done', flush=True)
t = pd.DataFrame(rows); t.to_csv(R / f'rule_real_{which}.csv', index=False)
pd.DataFrame(corr_rows).to_csv(R / f'rule_real_corr_{which}.csv', index=False)
cols = ['rr_criterion', 'rr_reweighted'] + [f'rr_rule_{x}' for x in RHOS]
print(t.groupby('case')[cols + ['one_record', 'group', 'sd_r']].mean().round(4).to_string())
for c, u in t.groupby('case'):
    for k in cols[1:]:
        print(c, k, 'improved', int((u[k] < u.rr_criterion - 1e-12).sum()), 'damaged', int((u[k] > u.rr_criterion + 1e-12).sum()), 'of', len(u))
