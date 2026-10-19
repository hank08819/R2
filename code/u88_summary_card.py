"""u88_summary_card.py -- the summary card a practitioner reads to answer one of the six questions.

Census income, California 2018 benchmark (as u59), seed 0, the ten classical learners.  For the
leading pair (the benchmark's choice f and its runner-up g), sort the benchmark test records by the
loss difference d = l_f - l_g.  The bottom tenth is where f does best against g, the top tenth where
g does best.  Each end is summarized by plain attributes.  The question is which end the
deployment serves more.  Every other state in 2018 is scanned under the census-marginals weight; the card shows the state
whose two ends differ most in mean weight.  There the
answer is read from the mean weight of each tenth and checked against sign Cov(r, d) and against
the deployment's own labels.
Writes results/universal/summary_card.csv.
"""
import importlib.util, warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from pathlib import Path
HERE = Path(__file__).resolve().parent; R = HERE.parent / 'results' / 'universal'
s = importlib.util.spec_from_file_location('u59', HERE / 'u59_acs_real_shift.py'); u59 = importlib.util.module_from_spec(s); s.loader.exec_module(u59)
FEAT = u59.FEAT
NAMES = ['random forest', 'extra trees', 'gradient boosting', 'AdaBoost', 'bagging', 'tree depth 5',
         'tree depth 12', 'least squares', 'ridge', 'nearest neighbors']


def describe(X, y):
    f = pd.DataFrame(X, columns=FEAT)
    return {'mean age': f.AGEP.mean(), "bachelor's or more (%)": 100 * (f.SCHL >= 21).mean(),
            'mean weekly hours': f.WKHP.mean(), 'women (%)': 100 * (f.SEX == 2).mean(),
            'self-employed (%)': 100 * f.COW.isin([6, 7]).mean(),
            'median income ($)': float(np.exp(np.median(y)))}


if __name__ == '__main__':
    seed = 0; rng = np.random.default_rng(seed)
    X, y = u59.load('CA', '2018')
    idx = rng.permutation(len(y))
    tr, te = idx[:u59.N_TRAIN], idx[u59.N_TRAIN:u59.N_TRAIN + u59.N_TEST]
    models = u59.u58.learners(seed)
    for m in models:
        m.fit(X[tr], y[tr])
    Lb = np.array([(m.predict(X[te]) - y[te]) ** 2 for m in models])
    order = np.argsort(Lb.mean(1)); f, g = int(order[0]), int(order[1])
    d = Lb[f] - Lb[g]
    lo, hi = d <= np.quantile(d, .1), d >= np.quantile(d, .9)
    from folktables.load_acs import state_list
    scan = []
    for st in [x for x in state_list if x not in ('CA', 'PR')]:
        Xd, yd = u59.load(st, '2018')
        jdx = np.random.default_rng(seed).permutation(len(yd)); half = len(yd) // 2
        truth, pool = jdx[:half], jdx[half:]
        r = u59.marginal_weight(X[te], Xd[pool])
        dep = np.array([(m.predict(Xd[truth]) - yd[truth]) ** 2 for m in models]).mean(1)
        scan.append(dict(state=st, w_lo=float(r[lo].mean()), w_hi=float(r[hi].mean()),
                         sign_cov=float(np.sign(np.mean((r - 1) * (d - d.mean())))),
                         dep_gap_pct=100 * float((dep[f] - dep[g]) / Lb[f].mean()),
                         reweighted_gap_pct=100 * float(np.mean(r * d) / Lb[f].mean())))
    scan = pd.DataFrame(scan); scan['spread'] = (scan.w_hi - scan.w_lo).abs()
    scan.to_csv(R / 'summary_card_scan.csv', index=False)
    best = scan.sort_values('spread').iloc[-1].state       # the state whose two ends differ most
    Xd, yd = u59.load(best, '2018')
    jdx = np.random.default_rng(seed).permutation(len(yd)); half = len(yd) // 2
    truth, pool = jdx[:half], jdx[half:]
    r = u59.marginal_weight(X[te], Xd[pool])
    dep = np.array([(m.predict(Xd[truth]) - yd[truth]) ** 2 for m in models]).mean(1)
    rows = []
    for end, mask in (('f ahead (bottom tenth of d)', lo), ('g ahead (top tenth of d)', hi)):
        rows.append({'end': end, 'records': int(mask.sum()), **describe(X[te][mask], y[te][mask]),
                     'mean weight': float(r[mask].mean())})
    out = pd.DataFrame(rows)
    out['state'] = best; out['f'] = NAMES[f]; out['g'] = NAMES[g]
    out['answer'] = 'g end' if r[hi].mean() > r[lo].mean() else 'f end'
    out['sign_cov'] = float(np.sign(np.mean((r - 1) * (d - d.mean()))))
    out['bench_gap_pct'] = 100 * float(d.mean() / Lb[f].mean())
    out['dep_gap_pct'] = 100 * float((dep[f] - dep[g]) / Lb[f].mean())
    out['reweighted_gap_pct'] = 100 * float(np.mean(r * d) / Lb[f].mean())
    print(scan.sort_values('spread').tail(8).to_string())
    print('ends answer = sign_cov in', int(((scan.w_hi > scan.w_lo) == (scan.sign_cov > 0)).sum()), 'of', len(scan))
    out.to_csv(R / 'summary_card.csv', index=False)
    pd.set_option('display.width', 250); print(out.T.to_string())
