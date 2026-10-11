"""u58_regret_test.py -- a deployment regret test, from benchmark losses and a stated weight.

The test asks, before deployment, whether the benchmark's choice is still the best model
under a stated deployment weight, and how much it can lose if not.  It needs the
per-instance losses of the candidates on the benchmark and the weight w the intended use
puts on each instance; no deployment data.

For the benchmark's choice f* and each rival g, with d = l_f* - l_g and r = w / mean(w),
    theta_g  = sum(w d) / sum(w)                  the deployment gap, by reweighting
    psi_g(i) = r_i (d_i - theta_g)                its influence function (the E_nu term of
                                                  the paper's psi)
    se_g     = sd(psi_g) / sqrt(n)
H0: f* is deployment-optimal, theta_g <= 0 for every rival.  T = max_g theta_g / se_g is
referred to a multiplier bootstrap of the joint psi's, which respects the correlation
between rivals; Bonferroni is reported as the no-bootstrap fallback.  The same critical
value c gives a simultaneous upper bound theta_g <= theta_g_hat + c se_g, hence a
one-sided upper confidence bound on the regret ratio
    RR = max(0, max_g theta_g) / E_mu[l_f*].
The effective sample size of the weight, (sum w)^2 / sum w^2, is reported and flagged
below 30.

Validation on the suite, two ways.
  Resampling: each full test set is treated as the population.  Benchmarks of half its
  size are drawn with replacement, the test is run on each, and the truth is the
  population regret of the model the sample chose.  This gives size, power and coverage.
  Split halves: as u56, the test runs on one half of each test set and is scored against
  the other half.  The second half is itself noisy, so its error rates are upper bounds.
Writes results/universal/regret_test.csv and regret_test_ab.csv.
"""
import warnings, json, time; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from pathlib import Path
from scipy.stats import norm
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.ensemble import (RandomForestRegressor, ExtraTreesRegressor,
    HistGradientBoostingRegressor, AdaBoostRegressor, BaggingRegressor)
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / 'results' / 'universal'
IDS = json.load(open(R / 'ctr23_ids.json'))
NMAX, SEEDS, ALPHA, NBOOT, NSUB = 4000, [0, 1, 2], 0.05, 1000, 40


def learners(s):
    """The ten of u7_scale.py, at library defaults."""
    return [RandomForestRegressor(n_estimators=100, random_state=s, n_jobs=-1),
            ExtraTreesRegressor(n_estimators=100, random_state=s, n_jobs=-1),
            HistGradientBoostingRegressor(random_state=s),
            AdaBoostRegressor(random_state=s),
            BaggingRegressor(n_estimators=30, random_state=s, n_jobs=-1),
            DecisionTreeRegressor(max_depth=5, random_state=s),
            DecisionTreeRegressor(max_depth=12, random_state=s),
            make_pipeline(StandardScaler(), LinearRegression()),
            make_pipeline(StandardScaler(), Ridge(alpha=10.0)),
            make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=10))]


def deployments(y):
    """The five of u7_scale.py, without the constant weight."""
    lo, hi = y.min(), y.max()
    out = {'top decile of outcome':    (y >= np.quantile(y, .90)).astype(float),
           'top quartile of outcome':  (y >= np.quantile(y, .75)).astype(float),
           'bottom decile of outcome': (y <= np.quantile(y, .10)).astype(float),
           'cost rising in outcome':   (y - lo) / (hi - lo) if hi > lo else None}
    out['relative-error weighted'] = (1.0 / np.maximum(np.abs(y), 1e-6) ** 2
                                      if np.all(np.abs(y) > 1e-6) else None)
    return {k: v for k, v in out.items()
            if v is not None and v.sum() > 0 and np.ptp(v) > 0}


def regret_test(L, w, alpha=ALPHA, nboot=NBOOT, rng=None):
    """The deployment regret test.  L: (K, n) benchmark losses; w: (n,) stated weight."""
    rng = rng or np.random.default_rng(0)
    K, n = L.shape
    f = int(np.argmin(L.mean(1)))
    riv = [g for g in range(K) if g != f]
    D = L[f] - L[riv]                                   # (K-1, n)
    r = w / w.mean()
    theta = (D * r).mean(1)
    psi = r * (D - theta[:, None])                      # (K-1, n)
    se = psi.std(1, ddof=1) / np.sqrt(n)
    ok = se > 0
    z = np.where(ok, theta / np.where(ok, se, 1), -np.inf)
    T = float(z.max())
    # multiplier bootstrap of max_g (theta_g_hat - theta_g) / se_g under the joint psi's
    xi = rng.standard_normal((nboot, n))
    Zb = (xi @ psi[ok].T) / n / se[ok]
    c = float(np.quantile(Zb.max(1), 1 - alpha))
    p_boot = float((Zb.max(1) >= T).mean())
    # moment selection: rivals far behind on the deployment cannot be the binding ones,
    # so the critical value is taken over the rest (threshold sqrt(2 log log n))
    zk = z[ok]
    sel = zk > -np.sqrt(2 * np.log(np.log(n)))
    if sel.any():
        zsel = Zb[:, sel].max(1)
        c_gms = float(np.quantile(zsel, 1 - alpha)); p_gms = float((zsel >= T).mean())
    else:
        c_gms, p_gms = np.inf, 1.0
    p_bonf = float(min(1.0, (K - 1) * norm.sf(T)))
    upper = np.where(ok, theta + c * se, theta)
    risk = float(L[f].mean())
    return dict(choice=f, T=T, crit=c, p_boot=p_boot, p_bonf=p_bonf,
                reject=int(T > c), crit_gms=c_gms, p_gms=p_gms, reject_gms=int(T > c_gms), rr_hat=max(0.0, float(theta.max())) / risk,
                rr_ucb=max(0.0, float(upper.max())) / risk,
                neff=float(w.sum() ** 2 / (w ** 2).sum()))


def truth(L, w, f):
    """Deployment regret ratio of model f, computed on the sample given as truth."""
    nu = w / w.sum()
    loss = L @ nu
    return float((loss[f] - loss.min()) / L[f].mean()), int(loss[f] <= loss.min() + 1e-15)


if __name__ == '__main__':
    rows, ab = [], []
    t0 = time.time()
    for i, did in enumerate(IDS):
        try:
            dd = fetch_openml(data_id=did, as_frame=True, parser='auto')
            X = pd.get_dummies(dd.data, drop_first=True).astype(float)
            X = SimpleImputer(strategy='median').fit_transform(X)
            y = pd.to_numeric(dd.target, errors='coerce').values.astype(float)
            ok = np.isfinite(y) & np.isfinite(X).all(1); X, y = X[ok], y[ok]
            if len(y) > NMAX:
                idx = np.random.default_rng(0).choice(len(y), NMAX, replace=False)
                X, y = X[idx], y[idx]
            if len(y) < 200 or X.shape[1] > 300:
                continue
        except Exception:
            continue
        name = dd.details.get('name', str(did))
        for s in SEEDS:
            Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.4, random_state=s)
            L = np.array([(m.fit(Xtr, ytr).predict(Xte) - yte) ** 2 for m in learners(s)])
            n = len(yte)
            perm = np.random.default_rng(s).permutation(n)
            A, B = perm[: n // 2], perm[n // 2:]
            for dep, w in deployments(yte).items():
                rng = np.random.default_rng(1000 * s + len(rows))
                # resampling: the full test set is the population
                for b in range(NSUB):
                    idx = rng.integers(0, n, n // 2)
                    if w[idx].sum() <= 0 or np.ptp(w[idx]) == 0:
                        continue
                    o = regret_test(L[:, idx], w[idx], rng=rng)
                    rr_pop, opt_pop = truth(L, w, o['choice'])
                    rows.append(dict(dataset=name, seed=s, deployment=dep, rep=b, n=n // 2,
                                     rr_pop=rr_pop, opt_pop=opt_pop, **o))
                # split halves, as u56
                if w[A].sum() > 0 and w[B].sum() > 0 and np.ptp(w[A]) > 0:
                    o = regret_test(L[:, A], w[A], rng=rng)
                    rr_b, opt_b = truth(L[:, B], w[B], o['choice'])
                    ab.append(dict(dataset=name, seed=s, deployment=dep, n=len(A),
                                   rr_B=rr_b, opt_B=opt_b, **o))
        print(f'[{i+1}/{len(IDS)}] {name:<28} {len(rows):>6} {time.time()-t0:.0f}s', flush=True)

    t = pd.DataFrame(rows); t.to_csv(R / 'regret_test.csv', index=False)
    a = pd.DataFrame(ab); a.to_csv(R / 'regret_test_ab.csv', index=False)
    small = t.neff < 30
    print(f'\nresampling: {len(t):,} tests over {t.groupby(["dataset","seed","deployment"]).ngroups} problems'
          f'; effective size below 30 in {100*small.mean():.1f}%')
    for lab, m in [('all', ~small | small), ('neff >= 30', ~small)]:
        u = t[m]
        print(f'  [{lab}] size: rejects when the choice is deployment-optimal  '
              f'{100*u.reject[u.opt_pop == 1].mean():.1f}%  (Bonferroni '
              f'{100*(u.p_bonf[u.opt_pop == 1] < ALPHA).mean():.1f}%)')
        print(f'  [{lab}] power: rejects when it is not                       '
              f'{100*u.reject[u.opt_pop == 0].mean():.1f}%')
        print(f'  [{lab}] with moment selection: size {100*u.reject_gms[u.opt_pop == 1].mean():.1f}%, '
              f'power {100*u.reject_gms[u.opt_pop == 0].mean():.1f}%')
        for lo, hi in [(0, .01), (.01, .05), (.05, .2), (.2, np.inf)]:
            v = u[(u.opt_pop == 0) & (u.rr_pop > lo) & (u.rr_pop <= hi)]
            print(f'      true regret ratio in ({lo:.2f},{hi:.2f}]: n={len(v):6d}, power {100*v.reject.mean():5.1f}%')
        print(f'  [{lab}] coverage of the 95% upper bound on RR             '
              f'{100*(u.rr_pop <= u.rr_ucb + 1e-12).mean():.1f}%')
    print(f'\nsplit halves: {len(a)} problems')
    print(f'  rejects when the second half says the choice is optimal     {100*a.reject[a.opt_B == 1].mean():.1f}%')
    print(f'  rejects when the second half says it is not                 {100*a.reject[a.opt_B == 0].mean():.1f}%')
    print(f'  with moment selection: {100*a.reject_gms[a.opt_B == 1].mean():.1f}% and {100*a.reject_gms[a.opt_B == 0].mean():.1f}%')
    print(f'  second-half regret ratio within the upper bound             {100*(a.rr_B <= a.rr_ucb + 1e-12).mean():.1f}%')
    print(f'  median upper bound on RR {100*a.rr_ucb.median():.1f}%, '
          f'90th percentile {100*a.rr_ucb.quantile(.9):.1f}%')
