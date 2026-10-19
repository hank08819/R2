"""u80_commute_prereg.py -- the pre-registered third real shift: census commute time.

Tests the decision-tree confirm rule (a step-2 "certified" verdict for a new place or time
passes to the step-4 spot-check) on a case fixed in results/universal/prereg_commute.md
before any model was fitted.  Same ACS files, states, years, seeds, sample sizes and weights
as the income case (u59); the outcome is log(1 + travel time to work), the candidates the
ten classical learners, the MLP and the FT-Transformer.

Per seed and deployment it records
  step 1  every pair: certified or not, reversed in the deployment or not;
  step 2  the selection-corrected regret test from the benchmark alone (u77), per weight;
  step 4  20 spot-check draws each of 500 and 1,000 labeled deployment records (u60 test).
Writes commute_pairs.csv, commute_problems.csv, commute_spot.csv.

    python u80_commute_prereg.py          the pre-registered run
    python u80_commute_prereg.py pilot    one seed, two deployments; prints sizes only
"""
import sys, json, time, itertools, importlib.util, warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from pathlib import Path
from folktables import ACSDataSource, BasicProblem
from folktables.load_acs import state_list

HERE = Path(__file__).resolve().parent
def mod(name, fn):
    s = importlib.util.spec_from_file_location(name, HERE / fn); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
u59 = mod('u59', 'u59_acs_real_shift.py'); u77 = mod('u77', 'u77_joint_cluster.py')
R, DATA = u59.R, u59.DATA

FEAT = ['AGEP', 'COW', 'SCHL', 'MAR', 'OCCP', 'POBP', 'RELP', 'WKHP', 'SEX', 'RAC1P', 'JWTR', 'DIS', 'CIT', 'MIG']
u59.FEAT = FEAT                         # coarse() and bands() read the module's feature list


def commute_filter(df):
    d = df[(df.AGEP > 16) & (df.PWGTP >= 1) & (df.ESR == 1)]
    return d[d.JWMNP > 0]


PROB = BasicProblem(features=FEAT, target='JWMNP', target_transform=lambda x: np.log1p(x),
                    group='SEX', preprocess=commute_filter, postprocess=lambda x: np.nan_to_num(x, nan=-1))
JWTRNS_TO_JWTR = {3: 4, 4: 5, 5: 3}
N_TRAIN, N_TEST, N_POOL, N_TRUTH = u59.N_TRAIN, u59.N_TEST, u59.N_POOL, u59.N_TRUTH
SIZES, NREP, NBOOT, ALPHA = [500, 1000], 20, 1000, 0.05
_cache = {}


def load(state, year):
    if (state, year) not in _cache:
        df = ACSDataSource(survey_year=year, horizon='1-Year', survey='person', root_dir=str(DATA)).get_data(states=[state], download=False)
        if 'RELP' not in df and 'RELSHIPP' in df:
            df['RELP'] = df.RELSHIPP.map(u59.RELSHIPP_TO_RELP)
        if 'JWTR' not in df and 'JWTRNS' in df:
            df['JWTR'] = df.JWTRNS.map(lambda v: JWTRNS_TO_JWTR.get(v, v))
        X, y, _ = PROB.df_to_numpy(df)
        _cache[(state, year)] = (X.astype(float), y.astype(float))
    return _cache[(state, year)]


def learners(seed):
    return u59.u65.learners13(seed, len(FEAT), u59.u58.learners)[:12]      # without the LLM ridge


def spot_test(L, f, risk, rng):
    """The u60 spot-check: choice f fixed, equal weights on labeled deployment losses."""
    D = L[f] - np.delete(L, f, axis=0)
    theta = D.mean(1); psi = D - theta[:, None]; n = L.shape[1]
    se = psi.std(1, ddof=1) / np.sqrt(n); ok = se > 0
    z = theta[ok] / se[ok]
    zmax = ((rng.standard_normal((NBOOT, n)) @ psi[ok].T) / n / se[ok]).max(1)
    c = float(np.quantile(zmax, 1 - ALPHA))
    up = theta.copy(); up[ok] = theta[ok] + c * se[ok]
    return dict(reject=int(z.max() > c), rr_ucb=max(0.0, float(up.max())) / risk)


def run(seeds, deps):
    pairs, probs, spots = [], [], []
    t0 = time.time()
    for seed in seeds:
        rng = np.random.default_rng(seed)
        Xca, yca = load('CA', '2018')
        idx = rng.permutation(len(yca))
        tr, te, pb = idx[:N_TRAIN], idx[N_TRAIN:N_TRAIN + N_TEST], idx[N_TRAIN + N_TEST:N_TRAIN + N_TEST + N_POOL]
        models = learners(seed)
        for m in models:
            m.fit(Xca[tr], yca[tr])
        Lb = np.array([(m.predict(Xca[te]) - yca[te]) ** 2 for m in models])
        f = int(np.argmin(Lb.mean(1))); risk = float(Lb[f].mean())
        for st, yr in deps:
            Xd, yd = load(st, yr)
            jdx = rng.permutation(len(yd))
            truth, pool = jdx[:N_TRUTH], jdx[N_TRUTH:N_TRUTH + N_POOL]
            if len(pool) < 1000:
                half = len(yd) // 2; truth, pool = jdx[:half], jdx[half:]
            dep = np.array([(m.predict(Xd[truth]) - yd[truth]) ** 2 for m in models]).mean(1)
            rr_true = float((dep[f] - dep.min()) / risk); opt_true = int(dep[f] <= dep.min() + 1e-15)
            weights = {'classifier': u59.classifier_weight(Xca[pb], Xd[pool], Xca[te], seed)[0],
                       'marginals': u59.marginal_weight(Xca[te], Xd[pool]),
                       'birthplace': u59.classifier_weight(Xca[pb], Xd[pool], Xca[te], seed, birth_state=True)[0]}
            for var, r in weights.items():
                sd_r = float(np.sqrt(np.mean((r - 1) ** 2)))
                for i, j in itertools.combinations(range(len(models)), 2):
                    d = Lb[i] - Lb[j]; e_mu, sd_d = float(d.mean()), float(d.std())
                    ratio = abs(e_mu) / sd_d / sd_r if sd_d > 0 and sd_r > 0 else np.inf
                    pairs.append(dict(state=st, year=yr, seed=seed, weight=var, pair=f'{i}-{j}', ratio=ratio,
                                      certified=int(ratio >= 1),
                                      reversed_true=int(np.sign(e_mu) != np.sign(dep[i] - dep[j]))))
                o = u77.regret_test_joint(Lb, r, rng=np.random.default_rng(1000 + seed))
                assert o['choice_j'] == f
                probs.append(dict(state=st, year=yr, seed=seed, weight=var, sd_r=sd_r, choice=f,
                                  best_true=int(np.argmin(dep)), opt_true=opt_true, rr_true=rr_true, m_f=risk,
                                  th_hat=json.dumps([float(((Lb[f] - Lb[g]) * r).mean()) for g in range(len(models))]),
                                  th_true=json.dumps([float(dep[f] - dep[g]) for g in range(len(models))]), **o))
            Lp = np.array([(m.predict(Xd[pool]) - yd[pool]) ** 2 for m in models])
            for n in SIZES:
                if n > Lp.shape[1]:
                    continue
                for b in range(NREP):
                    s = rng.choice(Lp.shape[1], n, replace=False)
                    spots.append(dict(state=st, year=yr, seed=seed, n=n, rep=b, opt_true=opt_true, rr_true=rr_true,
                                      **spot_test(Lp[:, s], f, risk, rng)))
            print(f'  seed {seed} {st} {yr} done {time.time() - t0:.0f}s', flush=True)
    return pd.DataFrame(pairs), pd.DataFrame(probs), pd.DataFrame(spots)


if __name__ == '__main__':
    pilot = len(sys.argv) > 1 and sys.argv[1] == 'pilot'
    if pilot:
        t, p, s = run([0], [('TX', '2018'), ('CA', '2022')])
        print('pilot sizes:', len(t), len(p), len(s), 'rows of CA 2018:', len(load('CA', '2018')[1]))
    else:
        deps = [(s, '2018') for s in state_list if s not in ('CA', 'PR')] + [('CA', y) for y in ('2019', '2021', '2022')]
        t, p, s = run([0, 1, 2], deps)
        t.to_csv(R / 'commute_pairs.csv', index=False); p.to_csv(R / 'commute_problems.csv', index=False)
        s.to_csv(R / 'commute_spot.csv', index=False)
        print('COMMUTE_DONE', len(t), len(p), len(s))
