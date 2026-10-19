"""u78_acs_regret_joint.py -- u59 rerun with the joint regret test of u77 added (same seeds, same samples).

u59: the certificate and the regret test under a real shift.

Thirteen learners (the ten, an MLP, an FT-Transformer and a ridge on frozen LLM embeddings; u65) are trained on California 2018 census records (ACS, through folktables)
to predict log personal income, and ranked by R^2 on a California 2018 test set: the
benchmark.  Each other state in 2018, and California in 2019, 2021 and 2022, is a
deployment.  Before deployment only unlabeled features of the deployment population
are used, to weight the benchmark test set.  After the fact, the models are scored on
labeled records of the deployment, which decides whether each check was right.

The shift is real: besides who lives where, the same person can earn differently in
another state or year.  The reweighting covers the first part and not the second, so
this case tests the method beyond the setting the identity covers exactly.

Weights, two ways.
  classifier : a domain classifier on coarsened features (occupation major group, place
               of birth by region), trained on a California pool and a deployment pool
               that are disjoint from every sample used below; r = odds ratio, clipped
               to [1/20, 20] and renormalised.
  marginals  : the California test set raked to the deployment's marginals of age
               band, education and weekly hours, which a published table would give.

Disjoint samples per seed: California train, benchmark test, classifier pool; per
deployment, a classifier pool and a labeled truth sample.

Writes results/universal/acs_pairs.csv (one row per pair, deployment, weight variant
and seed) and acs_problems.csv (one row per selection problem).
    python u59_acs_real_shift.py            full run
    python u59_acs_real_shift.py pilot      one seed, four deployments
"""
import json, sys, time, itertools, importlib.util, warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from pathlib import Path
from folktables import ACSDataSource, BasicProblem, adult_filter
from folktables.load_acs import state_list
from sklearn.ensemble import HistGradientBoostingClassifier

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / 'results' / 'universal'
DATA = ROOT / 'data' / 'acs'
spec = importlib.util.spec_from_file_location('u58', Path(__file__).with_name('u58_regret_test.py'))
u58 = importlib.util.module_from_spec(spec); spec.loader.exec_module(u58)
spec = importlib.util.spec_from_file_location('u65', Path(__file__).with_name('u65_deep_learners.py'))
u65 = importlib.util.module_from_spec(spec); spec.loader.exec_module(u65)
regret_test = u58.regret_test
spec77 = importlib.util.spec_from_file_location('u77', Path(__file__).with_name('u77_joint_cluster.py'))
u77 = importlib.util.module_from_spec(spec77); spec77.loader.exec_module(u77)
learners = lambda s: u65.learners13(s, len(FEAT), u58.learners)

FEAT = ['AGEP', 'COW', 'SCHL', 'MAR', 'OCCP', 'POBP', 'RELP', 'WKHP', 'SEX', 'RAC1P']
PROB = BasicProblem(features=FEAT, target='PINCP', target_transform=lambda x: np.log(x),
                    group='SEX', preprocess=adult_filter,
                    postprocess=lambda x: np.nan_to_num(x, nan=-1))
N_TRAIN, N_TEST, N_POOL, N_TRUTH, CLIP = 10_000, 10_000, 20_000, 20_000, 20.0
RELSHIPP_TO_RELP = {20: 0, 21: 1, 23: 1, 22: 13, 24: 13, 25: 2, 26: 3, 27: 4, 28: 5, 29: 6,
                    30: 7, 31: 8, 32: 9, 33: 10, 34: 12, 35: 14, 36: 15, 37: 16, 38: 17}
_cache = {}


def load(state, year):
    key = (state, year)
    if key not in _cache:
        ds = ACSDataSource(survey_year=year, horizon='1-Year', survey='person', root_dir=str(DATA))
        df = ds.get_data(states=[state], download=False)
        if 'RELP' not in df and 'RELSHIPP' in df:
            # from 2019 the relationship item is RELSHIPP with new codes; map back to RELP
            df['RELP'] = df.RELSHIPP.map(RELSHIPP_TO_RELP)
        X, y, _ = PROB.df_to_numpy(df)
        _cache[key] = (X.astype(float), y.astype(float))
    return _cache[key]


def full(X, st, yr):
    """The table with the row's LLM embedding (u64) appended; only the LLM learner reads the latter."""
    return np.hstack([X, np.load(ROOT / 'data' / 'real_emb' / f'acs_{st}_{yr}.npy').astype(float)])


def coarse(X, birth_state=False):
    """Classifier features: occupation major group and birthplace region, not raw codes.
    With birth_state, a US birthplace keeps its state, the rest their region."""
    f = pd.DataFrame(X, columns=FEAT)
    f['OCCP'] = (f.OCCP // 1000).clip(lower=0)
    f['POBP'] = np.where(f.POBP < 100, f.POBP if birth_state else 0, 100 + f.POBP // 100)
    return pd.get_dummies(f.astype(int), columns=['COW', 'MAR', 'OCCP', 'POBP', 'RELP', 'SEX', 'RAC1P'],
                          dtype=float)


def classifier_weight(Xpool_b, Xpool_d, Xtest, seed, birth_state=False):
    A, B, T = (coarse(z, birth_state) for z in (Xpool_b, Xpool_d, Xtest))
    cols = sorted(set(A.columns) | set(B.columns) | set(T.columns))
    A, B, T = (z.reindex(columns=cols, fill_value=0.0) for z in (A, B, T))
    clf = HistGradientBoostingClassifier(max_iter=200, random_state=seed)
    clf.fit(np.vstack([A, B]), np.r_[np.zeros(len(A)), np.ones(len(B))])
    p = np.clip(clf.predict_proba(T)[:, 1], 1e-6, 1 - 1e-6)
    r = p / (1 - p) * len(A) / len(B)
    r = r / r.mean()
    clipped = float(((r < 1 / CLIP) | (r > CLIP)).mean())
    r = np.clip(r, 1 / CLIP, CLIP)
    return r / r.mean(), clipped


def bands(X):
    f = pd.DataFrame(X, columns=FEAT)
    age = np.digitize(f.AGEP, [25, 35, 45, 55, 65])
    edu = np.digitize(f.SCHL, [16, 18, 21, 22])
    hrs = np.digitize(f.WKHP, [35, 45])
    return age, edu, hrs


def marginal_weight(Xtest, Xpool_d, iters=50):
    """Rake the benchmark test set to the deployment's marginals (raking, IPF)."""
    tb, db = bands(Xtest), bands(Xpool_d)
    w = np.ones(len(Xtest))
    for _ in range(iters):
        for t, d in zip(tb, db):
            target = np.bincount(d, minlength=t.max() + 1 if t.size else 1) / len(d)
            cur = np.bincount(t, weights=w, minlength=len(target)) / w.sum()
            ratio = np.divide(target[:len(cur)], cur, out=np.ones_like(cur), where=cur > 0)
            w = w * ratio[t]
    return w / w.mean()


def one(seed, deployments, pair_rows, prob_rows):
    rng = np.random.default_rng(seed)
    Xca, yca = load('CA', '2018'); Fca = full(Xca, 'CA', '2018')
    idx = rng.permutation(len(yca))
    tr, te, pb = idx[:N_TRAIN], idx[N_TRAIN:N_TRAIN + N_TEST], idx[N_TRAIN + N_TEST:N_TRAIN + N_TEST + N_POOL]
    models = learners(seed)
    for m in models:
        m.fit(Fca[tr], yca[tr])
    Lb = np.array([(m.predict(Fca[te]) - yca[te]) ** 2 for m in models])
    occ_train = set(Xca[tr, FEAT.index('OCCP')].astype(int))
    for st, yr in deployments:
        Xd, yd = load(st, yr); Fd = full(Xd, st, yr)
        jdx = rng.permutation(len(yd))
        truth, pool = jdx[:N_TRUTH], jdx[N_TRUTH:N_TRUTH + N_POOL]
        if len(pool) < 1000:                       # small states: split what there is
            half = len(yd) // 2
            truth, pool = jdx[:half], jdx[half:]
        Ld = np.array([(m.predict(Fd[truth]) - yd[truth]) ** 2 for m in models])
        unseen = float(np.mean([o not in occ_train for o in Xd[truth, FEAT.index('OCCP')].astype(int)]))
        dep_loss = Ld.mean(1)
        weights = {}
        weights['classifier'], clipped = classifier_weight(Xca[pb], Xd[pool], Xca[te], seed)
        weights['marginals'] = marginal_weight(Xca[te], Xd[pool])
        weights['birthplace'], clipped_b = classifier_weight(Xca[pb], Xd[pool], Xca[te], seed, birth_state=True)
        for var, r in weights.items():
            sd_r = float(np.sqrt(np.mean((r - 1) ** 2)))
            neff = float(r.sum() ** 2 / (r ** 2).sum())
            for i, j in itertools.combinations(range(len(models)), 2):
                d = Lb[i] - Lb[j]
                e_mu = float(d.mean()); sd_d = float(d.std())
                e_hat = float((d * r).mean())
                e_true = float(dep_loss[i] - dep_loss[j])
                ratio = abs(e_mu) / sd_d / sd_r if sd_d > 0 and sd_r > 0 else np.inf
                pair_rows.append(dict(state=st, year=yr, seed=seed, weight=var, pair=f'{i}-{j}',
                                      e_mu=e_mu, e_nu_hat=e_hat, e_nu_true=e_true, sd_d=sd_d,
                                      sd_r=sd_r, n_eff=neff, ratio=ratio,
                                      certified=int(ratio >= 1),
                                      reversed_true=int(np.sign(e_mu) != np.sign(e_true)),
                                      reversed_pred=int(np.sign(e_mu) != np.sign(e_hat))))
            o = regret_test(Lb, r, rng=np.random.default_rng(seed))
            o.update(u77.regret_test_joint(Lb, r, rng=np.random.default_rng(1000 + seed)))
            o.update(m_f=float(Lb[o['choice_j']].mean()), th_hat=json.dumps([float(((Lb[o['choice_j']] - Lb[g]) * r).mean()) for g in range(len(Lb))]),
                     th_true=json.dumps([float(dep_loss[o['choice_j']] - dep_loss[g]) for g in range(len(Lb))]))
            f = o['choice']
            rr_true = float((dep_loss[f] - dep_loss.min()) / Lb[f].mean())
            prob_rows.append(dict(state=st, year=yr, seed=seed, weight=var, sd_r=sd_r, n_eff=neff,
                                  clipped={'classifier': clipped, 'birthplace': clipped_b}.get(var, 0.0),
                                  unseen_occ=unseen, choice=f, best_true=int(np.argmin(dep_loss)),
                                  opt_true=int(dep_loss[f] <= dep_loss.min() + 1e-15),
                                  rr_true=rr_true, **{k: v for k, v in o.items() if k != 'choice'}))
        print(f'  seed {seed} {st} {yr}: sd_r {prob_rows[-3]["sd_r"]:.2f}/{prob_rows[-2]["sd_r"]:.2f}/{prob_rows[-1]["sd_r"]:.2f}'
              f'  n_eff {prob_rows[-3]["n_eff"]:.0f}/{prob_rows[-2]["n_eff"]:.0f}/{prob_rows[-1]["n_eff"]:.0f}'
              f'  clipped {clipped:.3f}  unseen occ {unseen:.3f}', flush=True)


def summary(t, p, tag):
    print(f'\n[{tag}] {len(t):,} comparisons, {len(p)} selection problems')
    bins = [0, .1, .25, .5, .75, 1, np.inf]
    for var, u in t.groupby('weight'):
        c = u.certified == 1
        print(f'  {var:10s} certified {c.sum():5d}: reverse in deployment {100*u.reversed_true[c].mean() if c.any() else float("nan"):5.1f}%'
              f' | exposed {100*u.reversed_true[~c].mean():5.1f}%'
              f' | reweighted sign right {100*(u.reversed_pred == u.reversed_true).mean():5.1f}%')
        g = u.groupby(pd.cut(u.ratio, bins, right=False), observed=False).reversed_true
        print('     by ratio bin:', ' '.join(f'{100*v:5.1f}%({n})' for v, n in zip(g.mean(), g.size())))
    for var, u in p.groupby('weight'):
        flag = u.n_eff < 30
        print(f'  {var:10s} effective size below 30 in {flag.sum()} problems; among the wrong choices {flag[u.opt_true == 0].sum()} of {(u.opt_true == 0).sum()}')
        print(f'  {var:10s} regret test: rejects {100*u.reject[u.opt_true == 1].mean():.1f}% of right choices,'
              f' {100*u.reject[u.opt_true == 0].mean():.1f}% of wrong ones; bound holds'
              f' {100*(u.rr_true <= u.rr_ucb + 1e-12).mean():.1f}%; R^2 wrong in {100*(1-u.opt_true).mean():.1f}%')


if __name__ == '__main__':
    pilot = len(sys.argv) > 1 and sys.argv[1] == 'pilot'
    if pilot:
        deps, seeds = [('TX', '2018'), ('NY', '2018'), ('WV', '2018'), ('CA', '2022')], [0]
    else:
        deps = [(s, '2018') for s in state_list if s not in ('CA', 'PR')] + \
               [('CA', y) for y in ('2019', '2021', '2022')]
        seeds = [0, 1, 2]
    pr, qr = [], []
    t0 = time.time()
    for s in seeds:
        one(s, deps, pr, qr)
        print(f'seed {s} done {time.time()-t0:.0f}s', flush=True)
    t, p = pd.DataFrame(pr), pd.DataFrame(qr)
    suffix = '_pilot' if pilot else ''
    t.to_csv(R / f'acs_pairs_u78b{suffix}.csv', index=False)
    p.to_csv(R / f'acs_problems_u78b{suffix}.csv', index=False)
    summary(t, p, 'pilot' if pilot else 'full')
