"""u74_regret_test_split.py -- a sample-splitting baseline for the deployment regret test.

The test of u68 chooses and tests on the same benchmark sample, and pays for that with
one band over every ordered pair.  The standard alternative splits the sample: choose on
half A, then test the chosen row on half B.  The choice is independent of B, so the band
on B needs to cover only the K-1 comparisons of that one row and the one risk, not all
K(K-1)+K.  It is narrower, but it has half the data, and the choice is made on half the
data too.

Same draws as u68 (same seeds, same resamples), same population truth, same alpha.  For
each draw both tests are run: u68 on the whole draw, the split test with A, B the two
halves of the draw.  Writes results/universal/regret_test_split.csv.
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


def split_test(LA, LB, wB, alpha=ALPHA, nboot=NBOOT, rng=None, eps=EPS):
    """Choose on A; bound the chosen row's regret ratio on B with a band over that row only."""
    rng = rng or np.random.default_rng(0)
    f = int(np.argmin(LA.mean(1)))
    K, n = LB.shape
    r = wB / wB.mean()
    m_hat = LB.mean(1)
    J = np.array([g for g in range(K) if g != f])
    D = LB[f] - LB[J]                                        # (K-1, n)
    theta = (D * r).mean(1)
    psi_t = r * (D - theta[:, None])
    psi_m = (LB[f] - m_hat[f])[None, :]
    Psi = np.vstack([psi_t, -psi_m])
    se = Psi.std(1, ddof=1) / np.sqrt(n)
    flat = se <= 1e-12 * max(1.0, float(np.abs(LB).max()))
    se = np.where(flat, 0.0, se)
    Zb = (rng.standard_normal((nboot, n)) @ Psi.T) / n / np.where(flat, 1.0, se)
    Zb[:, flat] = 0.0
    c = float(np.quantile(Zb.max(1), 1 - alpha))
    c2 = float(np.quantile(np.abs(Zb).max(1), 1 - alpha))
    P = K - 1
    up_t, lo_t = theta + c2 * se[:P], theta - c2 * se[:P]            # one two-sided band, as in u68
    lo_m, up_m = m_hat[f] - c2 * se[P], m_hat[f] + c2 * se[P]
    num_up = max(0.0, float(up_t.max())); num_lo = float(lo_t.max())
    rr_ucb = num_up / lo_m if lo_m > 0 else np.inf
    out = dict(choice_split=f, rr_ucb_split=rr_ucb)
    for e in eps:
        out[f'cert_split_{e}'] = int(rr_ucb <= e)
        out[f'exceed_split_{e}'] = int(num_lo > e * up_m)
    return out


def job(arg):
    did, s = arg
    out = u66.losses(did, s)
    if out is None: return []
    L, y, name = out
    n = L.shape[1]; rows = []
    for dep, w in u58.deployments(y).items():
        rng = np.random.default_rng(zlib.crc32(f'{did}-{s}-{dep}'.encode()))
        rng_s = np.random.default_rng(zlib.crc32(f'{did}-{s}-{dep}-split'.encode()))   # own stream, so u68's draws are untouched
        for frac in (4, 2):
            for b in range(NSUB):
                idx = rng.integers(0, n, n // frac)
                if w[idx].sum() <= 0 or np.ptp(w[idx]) == 0: continue
                Ls, ws = L[:, idx], w[idx]
                h = len(idx) // 2; A, B = np.arange(h), np.arange(h, len(idx))
                try:                                  # same random stream as u68, so the same draws
                    u58.regret_test(Ls, ws, rng=rng)
                except ValueError:
                    continue
                sel = u68.regret_test_sel(Ls, ws, rng=rng)
                if ws[B].sum() > 0 and np.ptp(ws[B]) > 0:
                    spl = split_test(Ls[:, A], Ls[:, B], ws[B], rng=rng_s)
                else:                                 # half B carries no usable weight: no verdict
                    spl = dict(choice_split=int(np.argmin(Ls[:, A].mean(1))), rr_ucb_split=np.inf,
                               **{f'cert_split_{e}': 0 for e in EPS}, **{f'exceed_split_{e}': 0 for e in EPS})
                rr_sel, _ = u68.truth_pop(L, w, sel['choice'])
                rr_spl, _ = u68.truth_pop(L, w, spl['choice_split'])
                rows.append(dict(dataset=name, seed=s, deployment=dep, frac=1 / frac, n=len(idx), rep=b,
                                 rr_pop_sel=rr_sel, rr_pop_split=rr_spl, **sel, **spl))
    return rows


if __name__ == '__main__':
    nproc = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    ids = json.load(open(R / 'ctr23_ids.json'))
    jobs = [(d, s) for d in ids for s in u58.SEEDS]; rows = []; t0 = time.time()
    with Pool(nproc) as p:
        for k, r_ in enumerate(p.imap_unordered(job, jobs)):
            rows += r_; print(f'[{k+1}/{len(jobs)}] {len(rows)} {time.time()-t0:.0f}s', flush=True)
    t = pd.DataFrame(rows); t.to_csv(R / 'regret_test_split.csv', index=False)
    for fr, u in t.groupby('frac'):
        print(f'\nbenchmark = {fr:.2f} of test set, {len(u)} draws')
        for v in ('sel', 'split'):
            rr = u[f'rr_pop_{v}']; ucb = u['rr_ucb_sel' if v == 'sel' else 'rr_ucb_split']; pos = rr > 0
            print(f'  {v:5s}: mean true regret ratio of the choice {100*rr.mean():.2f}%, coverage where > 0 '
                  f'{100*(rr[pos] <= ucb[pos] + 1e-12).mean():.1f}%')
        print('  tol | certified sel / split | wrongly certified sel / split | exceeds sel / split | undetermined sel / split')
        for e in EPS:
            cs, cp = u[f'cert_{e}'], u[f'cert_split_{e}']; xs, xp = u[f'exceed_{e}'], u[f'exceed_split_{e}']
            bs, bp = u.rr_pop_sel > e, u.rr_pop_split > e
            print(f'  tol{100*e:3.0f}% | {100*cs.mean():5.1f} / {100*cp.mean():5.1f} | {100*cs[bs].mean():5.2f} / {100*cp[bp].mean():5.2f}'
                  f' | {100*xs.mean():5.1f} / {100*xp.mean():5.1f} | {100*(1-cs-xs).mean():5.1f} / {100*(1-cp-xp).mean():5.1f}')
