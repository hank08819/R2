"""u67_signed_budget_test.py -- the selection certificate from a budget and signs, with
sampling error.  Stage 2 of u66.

For the benchmark's choice f, each rival g and d_g = l_f - l_g, the worst deployment gap
over {r >= 0, E r = 1, sd(r) <= rho, eta_k Cov(r, a_k) >= 0} is the convex dual of u66,
    U_g = min_{e, b >= 0}  e - b . E a + s |(d_g - e + A b)_+|,   s = sqrt(1 + rho^2).
At the optimum the envelope theorem gives its influence function,
    phi_g(i) = -b . (a_i - E a) + s ((z_i)_+^2 - |z_+|^2) / (2 |z_+|),   z = d_g - e + A b,
and a multiplier bootstrap of max_g phi_g / se_g gives one critical value c for all
rivals at once.  The choice is certified at level 1 - alpha when U_g + c se_g <= 0 for
every rival: f is then best under every deployment the stated budget and answers allow.

Questions.  'choice': one per rival, whether the use weights more the instances on
which f loses to g (the sign of Cov(r, l_f - l_g)).  'six': the six pairs among the
four leaders, as in the paper.  Answers come from the true weight (a practitioner who
knows the direction, not the size); 'flip' corrupts each with probability 0.2.

Validation, as u58.  Resampling: the full test set is the population; benchmarks of a
quarter and a half of it are drawn with replacement, and the truth is whether the
sample's choice is optimal on the population.  Budget stated as 0.5, 1 or 2 times the
true sd(r).  Split halves: certify on one half, score on the other (post-selection).
Writes results/universal/signed_budget_test.csv and signed_budget_test_ab.csv.
"""
import sys, json, time, zlib, warnings, importlib.util; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from pathlib import Path
from multiprocessing import Pool
from scipy.optimize import minimize

HERE = Path(__file__).resolve().parent
R = HERE.parent / 'results' / 'universal'
def mod(name, fn):
    s = importlib.util.spec_from_file_location(name, HERE / fn); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
u58 = mod('u58', 'u58_regret_test.py'); u66 = mod('u66', 'u66_signed_budget.py')
ALPHA, NBOOT, NSUB, TOP, FLIP = 0.05, 1000, 10, 4, 0.2


def dual(d, A, rho):
    """Worst gap U and its per-instance influence function."""
    s = np.sqrt(1 + rho ** 2); n, m = A.shape; Ea = A.mean(0) if m else np.zeros(0)
    def obj(x):
        e, b = x[0], x[1:]
        zp = np.clip(d - e + A @ b, 0, None); nz = np.sqrt(np.mean(zp ** 2))
        if nz <= 0: return e - b @ Ea, np.r_[1.0, -Ea]
        gz = s * zp / (nz * n)
        return e - b @ Ea + s * nz, np.r_[1.0 - gz.sum(), -Ea + A.T @ gz]
    res = minimize(obj, np.r_[np.median(d), np.zeros(m)], jac=True, method='L-BFGS-B',
                   bounds=[(None, None)] + [(0, None)] * m, options={'maxiter': 2000, 'ftol': 1e-13, 'gtol': 1e-10})
    e, b = res.x[0], res.x[1:]
    zp = np.clip(d - e + A @ b, 0, None); nz = np.sqrt(np.mean(zp ** 2))
    phi = -(A - Ea) @ b + (s * (zp ** 2 - nz ** 2) / (2 * nz) if nz > 0 else 0.0)
    return float(res.fun), phi


def answers(L, r, f, kind, rng):
    K = L.shape[0]
    if kind == 'choice':
        pairs = [(f, g) for g in range(K) if g != f]
    else:
        top = list(np.argsort(L.mean(1))[:TOP]); pairs = [(top[i], top[j]) for i in range(TOP) for j in range(i + 1, TOP)]
    out = []
    for a, b in pairs:
        eta = 1.0 if np.mean((r - r.mean()) * (L[a] - L[b])) >= 0 else -1.0
        if rng is not None and rng.random() < FLIP: eta = -eta
        out.append((a, b, eta))
    return out


def certify(L, ans, rho, rng):
    """Selection certificate on the sample L for its benchmark choice."""
    K, n = L.shape
    f = int(np.argmin(L.mean(1))); risk = float(L[f].mean())
    A = np.column_stack([e * (L[a] - L[b]) for a, b, e in ans]) if ans else np.zeros((n, 0))
    U, Phi = [], []
    for g in range(K):
        if g == f: continue
        u, phi = dual(L[f] - L[g], A, rho); U.append(u); Phi.append(phi)
    U, Phi = np.array(U), np.array(Phi)
    se = Phi.std(1, ddof=1) / np.sqrt(n); se = np.where(se > 0, se, 1e-12)
    Zb = (rng.standard_normal((NBOOT, n)) @ Phi.T) / n / se
    c = float(np.quantile(Zb.max(1), 1 - ALPHA))
    upper = U + c * se
    return dict(choice=f, cert=int(upper.max() <= 0), cert_plugin=int(U.max() <= 0),
                rr_ucb=max(0.0, float(upper.max())) / risk, crit=c)


def variants(Lpop, r_pop, Ls, rho0, rng):
    """All stated budgets and answer sets for one sample; answers asked about its choice."""
    f = int(np.argmin(Ls.mean(1))); out = []
    for kind, flip, mult in [('choice', 0, .5), ('choice', 0, 1), ('choice', 0, 2), ('choice', 1, 1),
                             ('six', 0, 1), ('none', 0, 1)]:
        if kind == 'none': ans = []
        else:
            # direction asked about the population (the practitioner's knowledge), for the
            # sample's own leaders
            ans = answers(Lpop, r_pop, f, kind, rng if flip else None) if kind == 'choice' else \
                  [(a, b, (1.0 if np.mean((r_pop - 1) * (Lpop[a] - Lpop[b])) >= 0 else -1.0))
                   for a, b, _ in answers(Ls, np.ones(Ls.shape[1]), f, 'six', None)]
        o = certify(Ls, ans, mult * rho0, rng)
        out.append(dict(questions=kind, flipped=flip, budget=mult, **o))
    return out


def job(arg):
    did, s = arg
    out = u66.losses(did, s)
    if out is None: return [], []
    L, y, name = out
    rows, ab = [], []
    n = L.shape[1]; perm = np.random.default_rng(s).permutation(n); A_, B_ = perm[: n // 2], perm[n // 2:]
    for dep, w in u58.deployments(y).items():
        rng = np.random.default_rng(zlib.crc32(f'{did}-{s}-{dep}'.encode()))
        r = w / w.mean(); rho0 = float(np.sqrt(np.mean((r - 1) ** 2)))
        for frac in (4, 2):
            for b in range(NSUB):
                idx = rng.integers(0, n, n // frac)
                for o in variants(L, r, L[:, idx], rho0, rng):
                    rr, opt = u58.truth(L, w, o['choice'])
                    rows.append(dict(dataset=name, seed=s, deployment=dep, n=n // frac, frac=1 / frac, rep=b, rho0=rho0,
                                     rr_pop=rr, opt_pop=opt, **o))
        # split halves: answers and budget from the first half's view, truth from the second
        if w[A_].sum() > 0 and w[B_].sum() > 0 and np.ptp(w[A_]) > 0:
            rA = w[A_] / w[A_].mean(); rhoA = float(np.sqrt(np.mean((rA - 1) ** 2)))
            for o in variants(L[:, A_], rA, L[:, A_], rhoA, rng):
                rr, opt = u58.truth(L[:, B_], w[B_], o['choice'])
                ab.append(dict(dataset=name, seed=s, deployment=dep, n=len(A_), rr_B=rr, opt_B=opt, **o))
    return rows, ab


if __name__ == '__main__':
    ids = json.load(open(R / 'ctr23_ids.json'))
    pilot = len(sys.argv) > 1 and sys.argv[1] == 'pilot'
    if pilot: ids = ids[:3]
    nproc = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    jobs = [(d, s) for d in ids for s in u58.SEEDS]
    t0 = time.time(); rows, ab = [], []
    with Pool(nproc) as p:
        for k, (r_, a_) in enumerate(p.imap_unordered(job, jobs)):
            rows += r_; ab += a_
            print(f'[{k+1}/{len(jobs)}] {len(rows)} {time.time()-t0:.0f}s', flush=True)
    t, a = pd.DataFrame(rows), pd.DataFrame(ab)
    sfx = '_pilot' if pilot else ''
    t.to_csv(R / f'signed_budget_test{sfx}.csv', index=False); a.to_csv(R / f'signed_budget_test_ab{sfx}.csv', index=False)
    print(f'\nresampling: {len(t):,} tests, {t.groupby(["dataset","seed","deployment"]).ngroups} problems')
    print('questions flipped budget  n   | certifies right choices | certifies WRONG choices (must stay <= 5%) | plug-in wrong')
    for (q, fl, bu, n), u in t.groupby(['questions', 'flipped', 'budget', 'frac']):
        print(f'{q:7s} {fl}  {bu:4.1f}  {n:4.2f} | {100*u.cert[u.opt_pop==1].mean():6.1f}%'
              f'                 | {100*u.cert[u.opt_pop==0].mean():6.2f}%  ({u.cert[u.opt_pop==0].sum()} of {(u.opt_pop==0).sum()})'
              f'                   | {100*u.cert_plugin[u.opt_pop==0].mean():6.2f}%')
    print(f'\nsplit halves: {len(a)}')
    for (q, fl, bu), u in a.groupby(['questions', 'flipped', 'budget']):
        print(f'{q:7s} {fl} {bu:4.1f} | certifies right {100*u.cert[u.opt_B==1].mean():6.1f}% | certifies wrong {100*u.cert[u.opt_B==0].mean():6.2f}%'
              f' ({u.cert[u.opt_B==0].sum()} of {(u.opt_B==0).sum()})')
