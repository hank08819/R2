"""u76_regret_test_joint.py -- one two-sided band for both verdicts of the regret test.

u68 certifies from a one-sided band (critical value c, the 1 - alpha quantile of the
largest standardized deviation) and declares 'exceeds' from a two-sided band (c2, the
quantile of the largest absolute deviation).  Each band holds with probability 1 - alpha,
but the two wrong verdicts live on different events, so together they are bounded only
by 2 alpha.  This reruns the same draws with the two-sided band used for both verdicts:
one event, one alpha, slightly fewer certifications.  Same draws as u68 (same seeds and
resamples), same population truth.  Writes results/universal/regret_test_joint.csv.
"""
import sys, json, time, zlib, warnings, importlib.util; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from pathlib import Path
from multiprocessing import Pool

HERE = Path(__file__).resolve().parent
R = HERE.parent / 'results' / 'universal'
def mod(name, fn):
    s = importlib.util.spec_from_file_location(name, HERE / fn); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
u58 = mod('u58', 'u58_regret_test.py'); u66 = mod('u66', 'u66_signed_budget.py'); u68 = mod('u68', 'u68_regret_test_selection.py')
ALPHA, NBOOT, NSUB, EPS = u68.ALPHA, u68.NBOOT, u68.NSUB, u68.EPS


def regret_test_joint(L, w, alpha=ALPHA, nboot=NBOOT, rng=None, eps=EPS):
    """As u68.regret_test_sel, returning both the one-sided (u68) and the joint verdicts
    from the same bootstrap draw."""
    rng = rng or np.random.default_rng(0)
    K, n = L.shape
    r = w / w.mean()
    m_hat = L.mean(1); f = int(np.argmin(m_hat))
    I, J = np.where(~np.eye(K, dtype=bool))
    D = L[I] - L[J]
    theta = (D * r).mean(1)
    psi_t = r * (D - theta[:, None])
    psi_m = L - m_hat[:, None]
    Psi = np.vstack([psi_t, -psi_m])
    se = Psi.std(1, ddof=1) / np.sqrt(n)
    flat = se <= 1e-12 * max(1.0, float(np.abs(L).max()))
    se = np.where(flat, 0.0, se)
    Zb = (rng.standard_normal((nboot, n)) @ Psi.T) / n / np.where(flat, 1.0, se)
    Zb[:, flat] = 0.0
    c = float(np.quantile(Zb.max(1), 1 - alpha))
    c2 = float(np.quantile(np.abs(Zb).max(1), 1 - alpha))
    P = len(I); row = I == f
    out = dict(choice=f, crit_sel=c, crit_joint=c2)
    for tag, cc in (('sel', c), ('joint', c2)):
        up_t, lo_t = theta + cc * se[:P], theta - c2 * se[:P]
        lo_m, up_m = m_hat - cc * se[P:], m_hat + c2 * se[P:]
        num_up = max(0.0, float(up_t[row].max())); num_lo = float(lo_t[row].max())
        rr_ucb = num_up / lo_m[f] if lo_m[f] > 0 else np.inf
        out[f'rr_ucb_{tag}'] = rr_ucb
        for e in eps:
            out[f'cert_{tag}_{e}'] = int(rr_ucb <= e)
            out[f'exceed_{tag}_{e}'] = int(num_lo > e * up_m[f])
    return out


def job(arg):
    did, s = arg
    out = u66.losses(did, s)
    if out is None: return []
    L, y, name = out
    n = L.shape[1]; rows = []
    for dep, w in u58.deployments(y).items():
        rng = np.random.default_rng(zlib.crc32(f'{did}-{s}-{dep}'.encode()))
        for frac in (4, 2):
            for b in range(NSUB):
                idx = rng.integers(0, n, n // frac)
                if w[idx].sum() <= 0 or np.ptp(w[idx]) == 0: continue
                Ls, ws = L[:, idx], w[idx]
                try:                                  # same random stream as u68
                    u58.regret_test(Ls, ws, rng=rng)
                except ValueError:
                    continue
                o = regret_test_joint(Ls, ws, rng=rng)
                rr, opt = u68.truth_pop(L, w, o['choice'])
                rows.append(dict(dataset=name, seed=s, deployment=dep, frac=1 / frac, n=len(idx), rep=b,
                                 rr_pop=rr, opt_pop=opt, **o))
    return rows


if __name__ == '__main__':
    nproc = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    ids = json.load(open(R / 'ctr23_ids.json'))
    jobs = [(d, s) for d in ids for s in u58.SEEDS]; rows = []; t0 = time.time()
    with Pool(nproc) as p:
        for k, r_ in enumerate(p.imap_unordered(job, jobs)):
            rows += r_; print(f'[{k+1}/{len(jobs)}] {len(rows)} {time.time()-t0:.0f}s', flush=True)
    t = pd.DataFrame(rows); t.to_csv(R / 'regret_test_joint.csv', index=False)
    for fr, u in t.groupby('frac'):
        pos = u.rr_pop > 0
        print(f'\nbenchmark = {fr:.2f} of test set, {len(u)} draws')
        for tag in ('sel', 'joint'):
            ucb = u[f'rr_ucb_{tag}']
            print(f'  {tag:5s}: coverage where regret > 0 {100*(u.rr_pop[pos] <= ucb[pos] + 1e-12).mean():.1f}%, median bound {100*ucb.median():.1f}%')
        print('  tol | certified sel / joint | wrongly certified | exceeds sel / joint | wrongly exceeds | undetermined sel / joint')
        for e in EPS:
            bad = u.rr_pop > e
            cs, cj = u[f'cert_sel_{e}'], u[f'cert_joint_{e}']; xs, xj = u[f'exceed_sel_{e}'], u[f'exceed_joint_{e}']
            print(f'  tol{100*e:3.0f}% | {100*cs.mean():5.1f} / {100*cj.mean():5.1f} | {100*cs[bad].mean():5.2f} / {100*cj[bad].mean():5.2f}'
                  f' | {100*xs.mean():5.1f} / {100*xj.mean():5.1f} | {100*xs[~bad].mean():5.2f} / {100*xj[~bad].mean():5.2f}'
                  f' | {100*(1-cs-xs).mean():5.1f} / {100*(1-cj-xj).mean():5.1f}')
