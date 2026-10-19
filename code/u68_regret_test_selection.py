"""u68_regret_test_selection.py -- the deployment regret test with the benchmark choice
made on the same sample, and the regret ratio's denominator estimated.

u58 fixes the chosen model f* and bootstraps the maximum over its K-1 rivals.  Because f*
is itself the sample argmin, that critical value does not cover the selected row
(a reviewer's example: ten models, deployment on half the population, coverage 92.7
percent in the limit).  The repair reads the chosen row from a band that holds for every
row at once:
    for all ordered pairs (f, g):  theta_fg <= theta_fg_hat + c se_fg,
    for all f:                     m_f      >= m_f_hat      - c se_mf,
with one c, the 1 - alpha quantile of the multiplier bootstrap of the largest absolute
value of all K(K-1) + K standardized deviations: one two-sided band, so that the
'certified' and 'exceeds' verdicts rest on the same event and the whole three-way
verdict is wrong with probability at most alpha.  On that event, whichever row the data select,
    RR(f*) = max_g theta_f*g / m_f*  <=  max(0, max_g upper_f*g) / lower_f*  = RR_ucb,
so the bound covers at level 1 - alpha whatever the selection rule.  The benchmark
risk m_f = E_mu[l_f] is the population benchmark risk, estimated, not the sample's.

Decision with a tolerance eps stated in advance:
    certified      RR_ucb <= eps
    exceeds        the lower band shows RR > eps  (some rival's lower bound on
                   theta_f*g exceeds eps times the upper bound on m_f*)
    undetermined   otherwise.

Validation on the suite (losses cached by u66), every test run with both versions on
the same draws: resampling (benchmarks of a quarter and a half of the test set, truth on
the full set) and split halves.  The weight is a fixed function of the outcome defined
on the population (the decile thresholds are not re-estimated on the sample).
Also the reviewer's example, simulated.
Writes results/universal/regret_test_sel.csv, regret_test_sel_ab.csv, regret_test_sel_cex.csv.
"""
import sys, json, time, zlib, warnings, importlib.util; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from pathlib import Path
from multiprocessing import Pool

HERE = Path(__file__).resolve().parent
R = HERE.parent / 'results' / 'universal'
def mod(name, fn):
    s = importlib.util.spec_from_file_location(name, HERE / fn); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
u58 = mod('u58', 'u58_regret_test.py'); u66 = mod('u66', 'u66_signed_budget.py')
ALPHA, NBOOT, NSUB = 0.05, 1000, 20
EPS = [0.01, 0.02, 0.05, 0.10, 0.20]


def regret_test_sel(L, w, alpha=ALPHA, nboot=NBOOT, rng=None, eps=EPS):
    rng = rng or np.random.default_rng(0)
    K, n = L.shape
    r = w / w.mean()
    m_hat = L.mean(1); f = int(np.argmin(m_hat))
    I, J = np.where(~np.eye(K, dtype=bool))                  # all ordered pairs
    D = L[I] - L[J]                                          # (P, n)
    theta = (D * r).mean(1)
    psi_t = r * (D - theta[:, None])
    psi_m = L - m_hat[:, None]
    Psi = np.vstack([psi_t, -psi_m])                         # one-sided: theta up, m down
    se = Psi.std(1, ddof=1) / np.sqrt(n)
    flat = se <= 1e-12 * max(1.0, float(np.abs(L).max()))     # known exactly up to rounding
    se = np.where(flat, 0.0, se)
    Zb = (rng.standard_normal((nboot, n)) @ Psi.T) / n / np.where(flat, 1.0, se)
    Zb[:, flat] = 0.0
    c1 = float(np.quantile(Zb.max(1), 1 - alpha))          # one-sided, kept for the record
    # two-sided band for the 'exceeds' verdict
    c2 = float(np.quantile(np.abs(Zb).max(1), 1 - alpha))
    c = c2                                                  # one two-sided band serves both verdicts
    P = len(I)
    up_t, lo_t = theta + c * se[:P], theta - c2 * se[:P]
    lo_m, up_m = m_hat - c * se[P:], m_hat + c2 * se[P:]
    row = I == f
    num_up = max(0.0, float(up_t[row].max())); num_lo = float(lo_t[row].max())
    rr_ucb = num_up / lo_m[f] if lo_m[f] > 0 else np.inf
    out = dict(choice=f, crit_sel=c, crit_onesided=c1, rr_ucb_sel=rr_ucb, rr_lcb_sel=max(0.0, num_lo) / up_m[f])
    for e in eps:
        out[f'cert_{e}'] = int(rr_ucb <= e)
        out[f'exceed_{e}'] = int(num_lo > e * up_m[f])
    return out


def truth_pop(L, w, f):
    """Population regret ratio with the population benchmark risk as denominator."""
    nu = (L * (w / w.sum())).sum(1)
    return float((nu[f] - nu.min()) / L[f].mean()), int(nu[f] <= nu.min() + 1e-15)


def job(arg):
    did, s = arg
    out = u66.losses(did, s)
    if out is None: return [], []
    L, y, name = out
    n = L.shape[1]; perm = np.random.default_rng(s).permutation(n); A, B = perm[: n // 2], perm[n // 2:]
    rows, ab = [], []
    for dep, w in u58.deployments(y).items():
        rng = np.random.default_rng(zlib.crc32(f'{did}-{s}-{dep}'.encode()))
        for frac in (4, 2):
            for b in range(NSUB):
                idx = rng.integers(0, n, n // frac)
                if w[idx].sum() <= 0 or np.ptp(w[idx]) == 0: continue
                Ls, ws = L[:, idx], w[idx]
                try:
                    old = u58.regret_test(Ls, ws, rng=rng)
                except ValueError:                   # every rival tied with the choice on the draw
                    continue
                new = regret_test_sel(Ls, ws, rng=rng)
                rr, opt = truth_pop(L, w, new['choice'])
                rows.append(dict(dataset=name, seed=s, deployment=dep, frac=1 / frac, n=len(idx), rep=b,
                                 rr_pop=rr, opt_pop=opt, rr_ucb_old=old['rr_ucb'], reject_old=old['reject'],
                                 neff=old['neff'], **new))
        if w[A].sum() > 0 and w[B].sum() > 0 and np.ptp(w[A]) > 0:
            old = u58.regret_test(L[:, A], w[A], rng=rng); new = regret_test_sel(L[:, A], w[A], rng=rng)
            nuB = (L[:, B] * (w[B] / w[B].sum())).sum(1); f = new['choice']
            ab.append(dict(dataset=name, seed=s, deployment=dep, n=len(A),
                           rr_B=float((nuB[f] - nuB.min()) / L[f, B].mean()), opt_B=int(nuB[f] <= nuB.min() + 1e-15),
                           rr_ucb_old=old['rr_ucb'], reject_old=old['reject'], **new))
    return rows, ab


def counterexample(reps=2000, n=4000, seed=1):
    rng = np.random.default_rng(seed); rows = []
    for t in range(reps):
        grp = rng.random(n) < .5
        L = np.zeros((10, n)); L[0] = np.where(grp, 0, 10)
        L[1:] = np.where(grp, rng.choice([1., 3.], size=(9, n)), 0)
        w = grp.astype(float)
        old = u58.regret_test(L, w, rng=rng); new = regret_test_sel(L, w, rng=rng)
        rows.append(dict(rep=t, choice=new['choice'], rr_true=2.0, rr_ucb_old=old['rr_ucb'], rr_ucb_sel=new['rr_ucb_sel']))
    return pd.DataFrame(rows)


if __name__ == '__main__':
    nproc = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    cx = counterexample(); cx.to_csv(R / 'regret_test_sel_cex.csv', index=False)
    print(f'reviewer example, {len(cx)} draws: coverage old {100*(cx.rr_true <= cx.rr_ucb_old + 1e-12).mean():.1f}%,'
          f' selection-adjusted {100*(cx.rr_true <= cx.rr_ucb_sel + 1e-12).mean():.1f}%', flush=True)
    ids = json.load(open(R / 'ctr23_ids.json'))
    jobs = [(d, s) for d in ids for s in u58.SEEDS]; rows, ab = [], []; t0 = time.time()
    with Pool(nproc) as p:
        for k, (r_, a_) in enumerate(p.imap_unordered(job, jobs)):
            rows += r_; ab += a_; print(f'[{k+1}/{len(jobs)}] {len(rows)} {time.time()-t0:.0f}s', flush=True)
    t, a = pd.DataFrame(rows), pd.DataFrame(ab)
    t.to_csv(R / 'regret_test_sel.csv', index=False); a.to_csv(R / 'regret_test_sel_ab.csv', index=False)
    print(f'\nresampling: {len(t):,} tests, {t.groupby(["dataset","seed","deployment"]).ngroups} problems')
    for fr, u in t.groupby('frac'):
        pos = u.rr_pop > 0
        print(f' benchmark = {fr:.2f} of test set (n median {u.n.median():.0f})')
        for v in ('old', 'sel'):
            col = 'rr_ucb_old' if v == 'old' else 'rr_ucb_sel'
            print(f'   {v}: coverage all {100*(u.rr_pop <= u[col] + 1e-12).mean():5.1f}%, where regret > 0'
                  f' {100*(u.rr_pop[pos] <= u[col][pos] + 1e-12).mean():5.1f}%, median bound {100*u[col].median():5.1f}%')
        for lab, m in [('regret > 0, n_eff >= 30', pos & (u.neff >= 30)), ('regret > 0, n_eff < 30', pos & (u.neff < 30))]:
            print(f'   sel coverage {lab}: {100*(u.rr_pop[m] <= u.rr_ucb_sel[m] + 1e-12).mean():5.1f}% ({m.sum()})')
        print('   tolerance | certified | wrongly certified (RR > eps) | exceeds | wrongly exceeds | undetermined')
        for e in EPS:
            bad = u.rr_pop > e
            print(f'   {100*e:5.0f}%    | {100*u[f"cert_{e}"].mean():6.1f}%  | {100*u[f"cert_{e}"][bad].mean() if bad.any() else 0:6.2f}% of {bad.sum()}'
                  f'           | {100*u[f"exceed_{e}"].mean():5.1f}%  | {100*u[f"exceed_{e}"][~bad].mean():5.2f}%'
                  f'          | {100*(1-u[f"cert_{e}"]-u[f"exceed_{e}"]).mean():5.1f}%')
    print(f'\nsplit halves: {len(a)}; second half is itself a sample, so this is an agreement check')
    for v, col in (('old', 'rr_ucb_old'), ('sel', 'rr_ucb_sel')):
        print(f'  {v}: second-half regret ratio within the bound {100*(a.rr_B <= a[col] + 1e-12).mean():.1f}%, where > 0 '
              f'{100*(a.rr_B[a.rr_B > 0] <= a[col][a.rr_B > 0] + 1e-12).mean():.1f}%')
    for e in EPS:
        bad = a.rr_B > e
        print(f'  eps {100*e:3.0f}%: certified {100*a[f"cert_{e}"].mean():5.1f}%, of those second half exceeds eps {100*a[f"cert_{e}"][bad].sum()/max(1,a[f"cert_{e}"].sum()):5.1f}%')
