"""u60_acs_spot_check.py -- the deployment regret test run on a few labeled deployment cases.

Before deployment the test can only reweight the benchmark, so it cannot see a change
in how features map to the outcome.  After deployment, a small labeled sample of the
deployment is enough to run the same test on the deployment losses themselves: the
benchmark's choice is fixed (it is the model already deployed), every rival is compared
with it at once, and the regret ratio keeps the benchmark risk of the choice as its
denominator.  Here the labeled spot-check sample is drawn from the deployment pool that
u59 used only for its features, so it is disjoint from the truth sample that scores it.

Writes results/universal/acs_spot_check.csv.
"""
import time, importlib.util, warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('u59', HERE / 'u59_acs_real_shift.py')
u59 = importlib.util.module_from_spec(spec); spec.loader.exec_module(u59)
from folktables.load_acs import state_list
R = u59.R
SIZES = [100, 250, 500, 1000, 2000]
NREP, ALPHA, NBOOT = 20, 0.05, 1000


def fixed_test(L, f, risk, rng):
    """The regret test with the choice f fixed and equal weights (deployment losses)."""
    K, n = L.shape
    D = L[f] - np.delete(L, f, axis=0)
    theta = D.mean(1); psi = D - theta[:, None]
    se = psi.std(1, ddof=1) / np.sqrt(n); ok = se > 0
    z = theta[ok] / se[ok]
    zmax = ((rng.standard_normal((NBOOT, n)) @ psi[ok].T) / n / se[ok]).max(1)
    c = float(np.quantile(zmax, 1 - ALPHA))
    upper = theta.copy(); upper[ok] = theta[ok] + c * se[ok]
    return dict(p=float((zmax >= z.max()).mean()), reject=int(z.max() > c),
                rr_ucb=max(0.0, float(upper.max())) / risk)


rows = []
deps = [(s, '2018') for s in state_list if s not in ('CA', 'PR')] + [('CA', y) for y in ('2019', '2021', '2022')]
t0 = time.time()
for seed in [0, 1, 2]:
    rng = np.random.default_rng(seed)
    Xca, yca = u59.load('CA', '2018'); Fca = u59.full(Xca, 'CA', '2018')
    idx = rng.permutation(len(yca))
    tr, te = idx[:u59.N_TRAIN], idx[u59.N_TRAIN:u59.N_TRAIN + u59.N_TEST]
    models = u59.learners(seed)
    for m in models:
        m.fit(Fca[tr], yca[tr])
    Lb = np.array([(m.predict(Fca[te]) - yca[te]) ** 2 for m in models])
    f = int(np.argmin(Lb.mean(1))); risk = float(Lb[f].mean())
    for st, yr in deps:
        Xd, yd = u59.load(st, yr); Fd = u59.full(Xd, st, yr)
        jdx = rng.permutation(len(yd))
        truth, pool = jdx[:u59.N_TRUTH], jdx[u59.N_TRUTH:u59.N_TRUTH + u59.N_POOL]
        if len(pool) < 1000:
            half = len(yd) // 2; truth, pool = jdx[:half], jdx[half:]
        dep = np.array([(m.predict(Fd[truth]) - yd[truth]) ** 2 for m in models]).mean(1)
        Lp = np.array([(m.predict(Fd[pool]) - yd[pool]) ** 2 for m in models])
        rr_true = float((dep[f] - dep.min()) / risk)
        opt_true = int(dep[f] <= dep.min() + 1e-15)
        for n in SIZES:
            if n > Lp.shape[1]:
                continue
            for b in range(NREP):
                s = rng.choice(Lp.shape[1], n, replace=False)
                o = fixed_test(Lp[:, s], f, risk, rng)
                rows.append(dict(state=st, year=yr, seed=seed, n=n, rep=b, opt_true=opt_true,
                                 rr_true=rr_true, **o))
    print(f'seed {seed} done {time.time()-t0:.0f}s', flush=True)

t = pd.DataFrame(rows); t.to_csv(R / 'acs_spot_check.csv', index=False)
print('\nlabeled deployment cases | rejects right choice | catches wrong choice | bound holds | median p (wrong)')
for n, u in t.groupby('n'):
    print(f'  {n:6d}                  {100*u.reject[u.opt_true == 1].mean():5.1f}%'
          f'              {100*u.reject[u.opt_true == 0].mean():5.1f}%'
          f'           {100*(u.rr_true <= u.rr_ucb + 1e-12).mean():5.1f}%'
          f'        {u.p[u.opt_true == 0].median():.3f}')
w = t[t.opt_true == 0]
print('\ncatch rate by true regret ratio (wrong choices only):')
for n, u in w.groupby('n'):
    g = u.groupby(pd.cut(u.rr_true, [0, .01, .05, .2, np.inf]), observed=True).reject.mean()
    print(f'  n={n:5d} ', ' '.join(f'{str(k)}:{100*v:5.1f}%' for k, v in g.items()))
