"""u61_beijing_real_shift.py -- the certificate and the regret test, forward in time.

Beijing PM2.5 (UCI 381), hourly, 2010-2014.  Thirteen learners (the ten, an MLP, an FT-Transformer and a ridge on frozen LLM embeddings; u65) predict log(1 + PM2.5) from
the weather and the calendar.  The benchmark is 2010-2012: whole days are assigned at
random to training or to the benchmark test, so neighbouring hours never straddle the
two.  Each month of 2013 and 2014 is a deployment, 24 in all, scored afterwards on its
own measured PM2.5.

Weights, two ways, both fixed before any deployment label is read.
  calendar   : the stated use "deployed in month m" alone; the benchmark test hours of
               month m carry the weight, the others none.
  classifier : a domain classifier on the weather features, trained on a benchmark pool
               and on the unlabeled half of the deployment month; r = odds ratio,
               clipped to [1/20, 20] and renormalised.
The labeled half of each deployment month is the truth; the other half only lends its
features to the classifier.

Writes results/universal/beijing_pairs.csv and beijing_problems.csv.
"""
import time, itertools, importlib.util, warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingClassifier

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
R = ROOT / 'results' / 'universal'
spec = importlib.util.spec_from_file_location('u58', HERE / 'u58_regret_test.py')
u58 = importlib.util.module_from_spec(spec); spec.loader.exec_module(u58)
spec = importlib.util.spec_from_file_location('u65', HERE / 'u65_deep_learners.py')
u65 = importlib.util.module_from_spec(spec); spec.loader.exec_module(u65)
regret_test = u58.regret_test
learners = lambda s: u65.learners13(s, NTAB, u58.learners)
CLIP, BINS = 20.0, [0, .1, .25, .5, .75, 1, np.inf]


def load():
    d = pd.read_csv(ROOT / 'data' / 'beijing' / 'PRSA_data_2010.1.1-2014.12.31.csv')
    d = d.dropna(subset=['pm2.5']).reset_index(drop=True)
    d['y'] = np.log1p(d['pm2.5'])
    d['day'] = pd.to_datetime(d[['year', 'month', 'day']]).dt.dayofyear + 1000 * d.year
    wind = pd.get_dummies(d.cbwd, prefix='wd', dtype=float)
    X = pd.concat([d[['month', 'hour', 'DEWP', 'TEMP', 'PRES', 'Iws', 'Is', 'Ir']].astype(float), wind], axis=1)
    return d, X.values


NTAB = 12


def full(X):
    """The table with each hour's LLM embedding (u64) appended; only the LLM learner reads the latter."""
    return np.hstack([X, np.load(ROOT / 'data' / 'real_emb' / 'beijing.npy').astype(float)])


def classifier_weight(Xb_pool, Xd_pool, Xtest, seed):
    clf = HistGradientBoostingClassifier(max_iter=200, random_state=seed)
    feats = slice(2, None)                       # weather only: the calendar is the other variant
    clf.fit(np.vstack([Xb_pool[:, feats], Xd_pool[:, feats]]),
            np.r_[np.zeros(len(Xb_pool)), np.ones(len(Xd_pool))])
    p = np.clip(clf.predict_proba(Xtest[:, feats])[:, 1], 1e-6, 1 - 1e-6)
    r = p / (1 - p) * len(Xb_pool) / len(Xd_pool)
    r = np.clip(r / r.mean(), 1 / CLIP, CLIP)
    return r / r.mean()


rows, probs = [], []
d, X = load(); F = full(X)
y = d.y.values
bench = d.year <= 2012
t0 = time.time()
for seed in [0, 1, 2]:
    rng = np.random.default_rng(seed)
    days = np.unique(d.day[bench])
    test_days = set(rng.choice(days, size=len(days) * 2 // 5, replace=False))
    is_test = d.day.isin(test_days).values
    tr = np.where(bench & ~is_test)[0]
    te_all = np.where(bench & is_test)[0]
    te = te_all[rng.permutation(len(te_all))]
    pool_b, te = te[: len(te) // 2], te[len(te) // 2:]      # classifier pool / benchmark test
    models = learners(seed)
    for m in models:
        m.fit(F[tr], y[tr])
    Lb = np.array([(m.predict(F[te]) - y[te]) ** 2 for m in models])
    for yr in (2013, 2014):
        for mo in range(1, 13):
            idx = np.where((d.year == yr) & (d.month == mo))[0]
            ddays = np.unique(d.day[idx])
            half = set(rng.choice(ddays, size=len(ddays) // 2, replace=False))
            lab = idx[~d.day[idx].isin(half).values]; unl = idx[d.day[idx].isin(half).values]
            dep = np.array([(m.predict(F[lab]) - y[lab]) ** 2 for m in models]).mean(1)
            weights = {'calendar': (d.month.values[te] == mo).astype(float),
                       'classifier': classifier_weight(X[pool_b], X[unl], X[te], seed)}
            for var, w in weights.items():
                r = w / w.mean()
                sd_r = float(np.sqrt(np.mean((r - 1) ** 2)))
                neff = float(r.sum() ** 2 / (r ** 2).sum())
                for i, j in itertools.combinations(range(len(models)), 2):
                    dd = Lb[i] - Lb[j]
                    e_mu, sd_d = float(dd.mean()), float(dd.std())
                    e_hat, e_true = float((dd * r).mean()), float(dep[i] - dep[j])
                    ratio = abs(e_mu) / sd_d / sd_r if sd_d > 0 and sd_r > 0 else np.inf
                    rows.append(dict(year=yr, month=mo, seed=seed, weight=var, pair=f'{i}-{j}',
                                     e_mu=e_mu, e_nu_hat=e_hat, e_nu_true=e_true, sd_d=sd_d, sd_r=sd_r,
                                     n_eff=neff, ratio=ratio, certified=int(ratio >= 1),
                                     reversed_true=int(np.sign(e_mu) != np.sign(e_true)),
                                     reversed_pred=int(np.sign(e_mu) != np.sign(e_hat))))
                o = regret_test(Lb, r, rng=np.random.default_rng(seed))
                f = o['choice']
                probs.append(dict(year=yr, month=mo, seed=seed, weight=var, sd_r=sd_r, n_eff=neff,
                                  n_truth=len(lab), best_true=int(np.argmin(dep)),
                                  opt_true=int(dep[f] <= dep.min() + 1e-15),
                                  rr_true=float((dep[f] - dep.min()) / Lb[f].mean()),
                                  **{k: v for k, v in o.items()}))
    print(f'seed {seed} done {time.time()-t0:.0f}s', flush=True)

t, p = pd.DataFrame(rows), pd.DataFrame(probs)
t.to_csv(R / 'beijing_pairs.csv', index=False); p.to_csv(R / 'beijing_problems.csv', index=False)
print(f'\n{len(t):,} comparisons, {len(p)} selection problems')
for var, u in t.groupby('weight'):
    c = u.certified == 1
    print(f'  {var:10s} certified {c.sum():5d}: reverse in deployment '
          f'{100*u.reversed_true[c].mean() if c.any() else float("nan"):5.1f}% | exposed {100*u.reversed_true[~c].mean():5.1f}%'
          f' | reweighted sign right {100*(u.reversed_pred == u.reversed_true).mean():5.1f}%')
    g = u.groupby(pd.cut(u.ratio, BINS, right=False), observed=False).reversed_true
    print('     by ratio bin:', ' '.join(f'{100*v:5.1f}%({n})' for v, n in zip(g.mean(), g.size())))
for var, u in p.groupby('weight'):
    print(f'  {var:10s} R^2 wrong in {100*(1-u.opt_true).mean():.1f}%; effective size below 30 in {(u.n_eff < 30).sum()}')
    print(f'  {var:10s} regret test: rejects {100*u.reject[u.opt_true == 1].mean():.1f}% of right choices,'
          f' {100*u.reject[u.opt_true == 0].mean() if (u.opt_true == 0).any() else float("nan"):.1f}% of wrong ones;'
          f' bound holds {100*(u.rr_true <= u.rr_ucb + 1e-12).mean():.1f}%')
    w = u[u.opt_true == 0]
    if len(w):
        print(f'     wrong choices: true regret ratio median {100*w.rr_true.median():.1f}%, max {100*w.rr_true.max():.1f}%;'
              f' p-values median {w.p_boot.median():.3f}')
