"""u62_beijing_spot_check.py -- the regret test on a few measured hours of each deployment month.

The benchmark's choice is fixed (the model already deployed), so the spot-check sample
plays no part in choosing it.  Hours of one day are dependent, so the multiplier
bootstrap is drawn per day (the '_iid' columns keep the hour-level version).  Its losses and the
rivals' are taken on n measured hours from the half of the month that u61 used only for
its weather; the other half is the truth.  Writes results/universal/beijing_spot_check.csv.
"""
import time, importlib.util, warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from pathlib import Path
HERE = Path(__file__).resolve().parent
def mod(name, fn):
    s = importlib.util.spec_from_file_location(name, HERE / fn); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
u58 = mod('u58', 'u58_regret_test.py')
R = HERE.parent / 'results' / 'universal'
src = open(HERE / 'u61_beijing_real_shift.py').read().split('rows, probs = [], []')[0]
ns = {'__file__': str(HERE / 'u61_beijing_real_shift.py')}; exec(compile(src, 'u61_head', 'exec'), ns)
load, learners, full = ns['load'], ns['learners'], ns['full']
SIZES, NREP, NBOOT, ALPHA = [50, 100, 200, 300], 40, 1000, 0.05


def fixed_test(L, f, risk, rng, cl=None):
    """The test with the deployed choice f fixed.  With cluster labels cl (the day of each
    hour), the influence functions are summed within a day and the multipliers drawn per
    day, since hours of one day are not independent."""
    D = L[f] - np.delete(L, f, axis=0)
    theta = D.mean(1); psi = D - theta[:, None]; n = L.shape[1]
    if cl is None:
        S = psi; G = n
    else:
        u, inv = np.unique(cl, return_inverse=True); G = len(u)
        S = np.zeros((psi.shape[0], G)); np.add.at(S.T, inv, psi.T)
    se = np.sqrt((S ** 2).sum(1) * G / max(G - 1, 1)) / n; ok = se > 0
    z = theta[ok] / se[ok]
    zmax = ((rng.standard_normal((NBOOT, S.shape[1])) @ S[ok].T) / n / se[ok]).max(1)
    c = float(np.quantile(zmax, 1 - ALPHA))
    up = theta.copy(); up[ok] = theta[ok] + c * se[ok]
    return dict(p=float((zmax >= z.max()).mean()), reject=int(z.max() > c), rr_ucb=max(0.0, float(up.max())) / risk)


d, X = load(); F = full(X); y = d.y.values; bench = d.year <= 2012
rows = []; t0 = time.time()
for seed in [0, 1, 2]:
    rng = np.random.default_rng(seed)
    days = np.unique(d.day[bench])
    test_days = set(rng.choice(days, size=len(days) * 2 // 5, replace=False))
    is_test = d.day.isin(test_days).values
    tr = np.where(bench & ~is_test)[0]; te_all = np.where(bench & is_test)[0]
    te = te_all[rng.permutation(len(te_all))]; te = te[len(te) // 2:]
    models = learners(seed)
    for m in models: m.fit(F[tr], y[tr])
    Lb = np.array([(m.predict(F[te]) - y[te]) ** 2 for m in models])
    f = int(np.argmin(Lb.mean(1))); risk = float(Lb[f].mean())
    for yr in (2013, 2014):
        for mo in range(1, 13):
            idx = np.where((d.year == yr) & (d.month == mo))[0]
            ddays = np.unique(d.day[idx])
            half = set(rng.choice(ddays, size=len(ddays) // 2, replace=False))
            lab = idx[~d.day[idx].isin(half).values]; spot = idx[d.day[idx].isin(half).values]
            dep = np.array([(m.predict(F[lab]) - y[lab]) ** 2 for m in models]).mean(1)
            Ls = np.array([(m.predict(F[spot]) - y[spot]) ** 2 for m in models])
            opt = int(dep[f] <= dep.min() + 1e-15); rr = float((dep[f] - dep.min()) / risk)
            for n in SIZES:
                if n > Ls.shape[1]: continue
                for b in range(NREP):
                    s = rng.choice(Ls.shape[1], n, replace=False)
                    o_iid = fixed_test(Ls[:, s], f, risk, rng)
                    o_day = fixed_test(Ls[:, s], f, risk, rng, cl=d.day.values[spot][s])
                    rows.append(dict(year=yr, month=mo, seed=seed, n=n, rep=b, opt_true=opt, rr_true=rr,
                                     n_days=len(np.unique(d.day.values[spot][s])),
                                     **o_day, **{k + '_iid': v for k, v in o_iid.items()}))
    print(f'seed {seed} done {time.time()-t0:.0f}s', flush=True)
t = pd.DataFrame(rows); t.to_csv(R / 'beijing_spot_check.csv', index=False)
print('\nmeasured hours | flags a right choice | catches a wrong choice | bound holds')
for n, u in t.groupby('n'):
    print(f'  {n:5d}          {100*u.reject[u.opt_true == 1].mean():5.1f}%               {100*u.reject[u.opt_true == 0].mean():5.1f}%'
          f'               {100*(u.rr_true <= u.rr_ucb + 1e-12).mean():5.1f}%')
w = t[t.opt_true == 0]
print('\ncatch rate by true regret ratio:')
for n, u in w.groupby('n'):
    g = u.groupby(pd.cut(u.rr_true, [0, .05, .2, np.inf]), observed=True).reject.mean()
    print(f'  n={n:4d} ', ' '.join(f'{k}:{100*v:5.1f}%' for k, v in g.items()))
