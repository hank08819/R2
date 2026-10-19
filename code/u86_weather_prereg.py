"""u86_weather_prereg.py -- the pre-registered weather case: a published shift benchmark outside the census.

Tests the decision-tree confirm rule on the Shifts Weather Prediction data (Malinin et al. 2021),
as fixed in results/universal/prereg_weather.md before any model was fitted.  The outcome is the
measured air temperature; the features are the 123 forecast and climate features of the canonical
partition (meta-data, the precipitation class and the climate label are left out).

Benchmark: climate type 'mild temperate', September to November 2018 (cells of u85).  Each other
(climate type, calendar month) cell with at least MIN_ROWS sampled rows is a deployment: a new
place, a new time, or both.  Candidates: the ten classical learners, the MLP and the FT-Transformer.

Per seed and deployment it records, as u80 does for the commute case,
  step 1  every pair: certified or not, reversed in the deployment or not;
  step 2  the selection-corrected regret test from the benchmark alone (u77), per weight;
  step 4  20 spot-check draws each of 500 and 1,000 labeled deployment records.
Writes weather_pairs.csv, weather_problems.csv, weather_spot.csv.

    python u86_weather_prereg.py          the pre-registered run
    python u86_weather_prereg.py pilot    one seed, two deployments; prints sizes only
"""
import sys, json, time, itertools, importlib.util, warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingClassifier

HERE = Path(__file__).resolve().parent
def mod(name, fn):
    s = importlib.util.spec_from_file_location(name, HERE / fn); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
u58 = mod('u58', 'u58_regret_test.py'); u65 = mod('u65', 'u65_deep_learners.py')
u77 = mod('u77', 'u77_joint_cluster.py')
R = HERE.parent / 'results' / 'universal'
DATA = HERE.parent / 'data' / 'shifts_weather' / 'weather_cells.csv.gz'

META = ['fact_time', 'fact_latitude', 'fact_longitude', 'fact_temperature', 'fact_cwsm_class', 'climate', 'cell', 'source']
TARGET = 'fact_temperature'
BENCH = ['mild temperate|2018-09', 'mild temperate|2018-10', 'mild temperate|2018-11']
CLF_FEAT = ['climate_temperature', 'climate_pressure', 'topography_bathymetry', 'sun_elevation', 'gfs_humidity', 'gfs_wind_speed']
RAKE_FEAT = ['climate_temperature', 'sun_elevation', 'topography_bathymetry']
N_TRAIN, N_TEST, N_POOL, N_TRUTH, CLIP, MIN_ROWS = 10_000, 10_000, 20_000, 20_000, 20.0, 3_000
SIZES, NREP, NBOOT, ALPHA = [500, 1000], 20, 1000, 0.05


def spot_test(L, f, risk, rng):
    """The spot-check of u80 (u60): choice f fixed, equal weights on labeled deployment losses."""
    D = L[f] - np.delete(L, f, axis=0)
    theta = D.mean(1); psi = D - theta[:, None]; n = L.shape[1]
    se = psi.std(1, ddof=1) / np.sqrt(n); ok = se > 0
    z = theta[ok] / se[ok]
    zmax = ((rng.standard_normal((NBOOT, n)) @ psi[ok].T) / n / se[ok]).max(1)
    c = float(np.quantile(zmax, 1 - ALPHA))
    up = theta.copy(); up[ok] = theta[ok] + c * se[ok]
    return dict(reject=int(z.max() > c), rr_ucb=max(0.0, float(up.max())) / risk)


def load():
    d = pd.read_csv(DATA)
    d['climate'] = d.climate.astype(str).str.lower()
    d['cell'] = d.climate + '|' + d.cell.str.split('|').str[1]
    feat = [c for c in d.columns if c not in META]
    d[feat] = d[feat].astype(float).fillna(d[feat].median())
    return d, feat


def classifier_weight(Pb, Pd, T, seed):
    clf = HistGradientBoostingClassifier(max_iter=200, random_state=seed)
    clf.fit(np.vstack([Pb, Pd]), np.r_[np.zeros(len(Pb)), np.ones(len(Pd))])
    p = np.clip(clf.predict_proba(T)[:, 1], 1e-6, 1 - 1e-6)
    r = p / (1 - p) * len(Pb) / len(Pd); r = r / r.mean()
    r = np.clip(r, 1 / CLIP, CLIP)
    return r / r.mean()


def marginal_weight(T, Pd, iters=50):
    """Rake the benchmark test set to the deployment's quintile shares of each raking feature,
    quintiles cut on the benchmark test set."""
    cuts = [np.quantile(T[:, k], [.2, .4, .6, .8]) for k in range(T.shape[1])]
    tb = [np.digitize(T[:, k], c) for k, c in enumerate(cuts)]
    db = [np.digitize(Pd[:, k], c) for k, c in enumerate(cuts)]
    w = np.ones(len(T))
    for _ in range(iters):
        for t, dd in zip(tb, db):
            target = np.bincount(dd, minlength=5) / len(dd)
            cur = np.bincount(t, weights=w, minlength=5) / w.sum()
            w = w * np.divide(target, cur, out=np.ones_like(cur), where=cur > 0)[t]
    return w / w.mean()


def run(seeds, deps, d, feat):
    ci = {c: feat.index(c) for c in feat}
    pairs, probs, spots = [], [], []
    t0 = time.time()
    B = d[d.cell.isin(BENCH)]
    Xb, yb = B[feat].values, B[TARGET].values
    for seed in seeds:
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(yb))
        tr, te, pb = idx[:N_TRAIN], idx[N_TRAIN:N_TRAIN + N_TEST], idx[N_TRAIN + N_TEST:N_TRAIN + N_TEST + N_POOL]
        models = u65.learners13(seed, len(feat), u58.learners)[:12]
        for m in models:
            m.fit(Xb[tr], yb[tr])
        Lb = np.array([(m.predict(Xb[te]) - yb[te]) ** 2 for m in models])
        f = int(np.argmin(Lb.mean(1))); risk = float(Lb[f].mean())
        for c in deps:
            D = d[d.cell == c]; Xd, yd = D[feat].values, D[TARGET].values
            jdx = rng.permutation(len(yd))
            truth, pool = jdx[:N_TRUTH], jdx[N_TRUTH:N_TRUTH + N_POOL]
            if len(pool) < 1000:
                half = len(yd) // 2; truth, pool = jdx[:half], jdx[half:]
            dep = np.array([(m.predict(Xd[truth]) - yd[truth]) ** 2 for m in models]).mean(1)
            rr_true = float((dep[f] - dep.min()) / risk); opt_true = int(dep[f] <= dep.min() + 1e-15)
            cf = [ci[k] for k in CLF_FEAT]; rf = [ci[k] for k in RAKE_FEAT]
            weights = {'classifier': classifier_weight(Xb[pb][:, cf], Xd[pool][:, cf], Xb[te][:, cf], seed),
                       'classifier_all': classifier_weight(Xb[pb], Xd[pool], Xb[te], seed),
                       'marginals': marginal_weight(Xb[te][:, rf], Xd[pool][:, rf])}
            for var, r in weights.items():
                sd_r = float(np.sqrt(np.mean((r - 1) ** 2)))
                for i, j in itertools.combinations(range(len(models)), 2):
                    dd = Lb[i] - Lb[j]; e_mu, sd_d = float(dd.mean()), float(dd.std())
                    ratio = abs(e_mu) / sd_d / sd_r if sd_d > 0 and sd_r > 0 else np.inf
                    pairs.append(dict(cell=c, seed=seed, weight=var, pair=f'{i}-{j}', ratio=ratio,
                                      certified=int(ratio >= 1),
                                      reversed_true=int(np.sign(e_mu) != np.sign(dep[i] - dep[j]))))
                o = u77.regret_test_joint(Lb, r, rng=np.random.default_rng(1000 + seed))
                assert o['choice_j'] == f
                probs.append(dict(cell=c, seed=seed, weight=var, sd_r=sd_r, choice=f,
                                  best_true=int(np.argmin(dep)), opt_true=opt_true, rr_true=rr_true, m_f=risk,
                                  th_hat=json.dumps([float(((Lb[f] - Lb[g]) * r).mean()) for g in range(len(models))]),
                                  th_true=json.dumps([float(dep[f] - dep[g]) for g in range(len(models))]), **o))
            Lp = np.array([(m.predict(Xd[pool]) - yd[pool]) ** 2 for m in models])
            for n in SIZES:
                if n > Lp.shape[1]:
                    continue
                for b in range(NREP):
                    s = rng.choice(Lp.shape[1], n, replace=False)
                    spots.append(dict(cell=c, seed=seed, n=n, rep=b, opt_true=opt_true, rr_true=rr_true,
                                      **spot_test(Lp[:, s], f, risk, rng)))
            print(f'  seed {seed} {c} done {time.time() - t0:.0f}s', flush=True)
    return pd.DataFrame(pairs), pd.DataFrame(probs), pd.DataFrame(spots)


if __name__ == '__main__':
    d, feat = load()
    n = d.cell.value_counts()
    deps = sorted(c for c in n.index if c not in BENCH and n[c] >= MIN_ROWS)
    print(f'{len(feat)} features; benchmark rows {n[BENCH].sum()}; {len(deps)} deployments', flush=True)
    if len(sys.argv) > 1 and sys.argv[1] == 'pilot':
        t, p, s = run([0], deps[:2], d, feat)
        print('pilot sizes:', len(t), len(p), len(s))
    elif len(sys.argv) > 1 and sys.argv[1].startswith('seed='):     # one seed per process; merged below
        k = int(sys.argv[1][5:])
        t, p, s = run([k], deps, d, feat)
        for x, nm in ((t, 'pairs'), (p, 'problems'), (s, 'spot')):
            x.to_csv(R / f'weather_{nm}_seed{k}.csv', index=False)
        print('SEED_DONE', k, len(t), len(p), len(s))
    elif len(sys.argv) > 1 and sys.argv[1] == 'merge':
        for nm in ('pairs', 'problems', 'spot'):
            pd.concat([pd.read_csv(R / f'weather_{nm}_seed{k}.csv') for k in (0, 1, 2)]).to_csv(R / f'weather_{nm}.csv', index=False)
        print('WEATHER_DONE')
    else:
        t, p, s = run([0, 1, 2], deps, d, feat)
        t.to_csv(R / 'weather_pairs.csv', index=False); p.to_csv(R / 'weather_problems.csv', index=False)
        s.to_csv(R / 'weather_spot.csv', index=False)
        print('WEATHER_DONE', len(t), len(p), len(s))
