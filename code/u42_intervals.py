"""u42_intervals.py -- interval estimates for the three structural results.

The reversal rates carry standard errors from the influence function of the bias.
The three results that describe the structure of a benchmark rather than a single
comparison do not, and they are the ones the paper leans on hardest, so we resample
them: the collapse of the candidate set with the budget, the certified budget
itself, and the share of frontiers containing the model the deployment prefers.
"""
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from pathlib import Path

R = Path(__file__).resolve().parents[1] / 'results' / 'universal'
rng = np.random.default_rng(5)
B = 5000


def boot(v, stat, b=B):
    v = np.asarray(v)
    d = [stat(v[rng.integers(0, len(v), len(v))]) for _ in range(b)]
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


out = []
c = pd.read_csv(R / 'collapse.csv')
for k in [10., 1., .25]:
    col = c[f'n_{k}'].values
    lo, hi = boot(col, np.median)
    out.append(dict(quantity=f'models left at budget {k:g}, median',
                    est=float(np.median(col)), lo=lo, hi=hi, n=len(col)))
    lo, hi = boot(col, lambda v: 100 * (v == 1).mean())
    out.append(dict(quantity=f'decided outright at budget {k:g} (%)',
                    est=100 * (col == 1).mean(), lo=lo, hi=hi, n=len(col)))

ks = pd.read_csv(R / 'certificate.csv').kstar.values
lo, hi = boot(ks, np.median)
out.append(dict(quantity='certified budget, median', est=float(np.median(ks)),
                lo=lo, hi=hi, n=len(ks)))

f = pd.read_csv(R / 'off_frontier.csv')
q = f[f.is_r2 == 0]
lo, hi = boot(q.on_front.values, lambda v: 100 * v.mean())
out.append(dict(quantity='deployment optimum on the frontier, criterion wrong (%)',
                est=100 * q.on_front.mean(), lo=lo, hi=hi, n=len(q)))
chance = 100 * f.n_front.mean() / f.K.iloc[0]
lo2, hi2 = boot(f.n_front.values, lambda v: 100 * v.mean() / f.K.iloc[0])
out.append(dict(quantity='same, expected under a random pick (%)', est=chance,
                lo=lo2, hi=hi2, n=len(f)))

ce = pd.read_csv(R / 'crowding.csv')
for s, gg in ce.groupby('setting'):
    sl = np.polyfit(np.log(gg.K), np.log(gg.kstar), 1)[0]
    idx = np.arange(len(gg))
    d = []
    for _ in range(2000):
        j = rng.integers(0, len(gg), len(gg))
        if len(set(gg.K.values[j])) < 3:
            continue
        d.append(np.polyfit(np.log(gg.K.values[j]), np.log(gg.kstar.values[j]), 1)[0])
    out.append(dict(quantity=f'crowding slope, {s}', est=sl,
                    lo=float(np.percentile(d, 2.5)), hi=float(np.percentile(d, 97.5)),
                    n=len(gg)))

t = pd.DataFrame(out); t.to_csv(R / 'intervals.csv', index=False)
print(f'{B} bootstrap resamples\n')
print(f'  {"quantity":<52}{"estimate":>10}{"95% interval":>22}')
for r in t.itertuples():
    print(f'  {r.quantity:<52}{r.est:>10.2f}   [{r.lo:.2f}, {r.hi:.2f}]')
