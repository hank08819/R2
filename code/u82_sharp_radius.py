"""u82_sharp_radius.py -- the stability radius with the deployment weight kept nonnegative.

The radius kappa_fg = |E d| / sd(d) is the smallest sd_mu(r) at which some weight r with
E_mu r = 1 can tie the pair, if r may go negative.  A deployment weight cannot.  The sharp
radius keeps r >= 0:
    kappa+_fg = min sd_mu(r)  subject to  r >= 0,  E_mu r = 1,  E_mu[r d] = 0
(infinite when d never favours the loser).  It is never below kappa_fg, and any budget above it
admits a nonnegative weight that ties the pair.  By the KKT conditions the minimiser is
r = max(0, a + b d), so a and b are found by two nested bisections.

On the suite (losses cached by u66, deployments of u58) it counts the comparisons each radius
certifies, and checks both on held-out halves as u56 does: certify on one random half of each
test set, look for a reversal on the other, with the deployment weight computed once on the
whole test set.  Writes results/universal/sharp_radius.csv.
"""
import importlib.util, warnings, itertools; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from pathlib import Path

HERE = Path(__file__).resolve().parent
def mod(name, fn):
    s = importlib.util.spec_from_file_location(name, HERE / fn); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
u58 = mod('u58', 'u58_regret_test.py'); u66 = mod('u66', 'u66_signed_budget.py')
R = HERE.parent / 'results' / 'universal'


def _r(d, b):
    """For slope b, the weight max(0, a + b d) with mean one."""
    lo, hi = -b * d.max() - 1.0, 1.0 - b * d.min() + 1.0
    for _ in range(100):
        a = (lo + hi) / 2
        if np.maximum(0, a + b * d).mean() > 1: hi = a
        else: lo = a
    return np.maximum(0, (lo + hi) / 2 + b * d)


def sharp_radius(d):
    """kappa+ for a pair whose benchmark gap E d is not zero; d is oriented inside."""
    d = d if d.mean() < 0 else -d
    if d.max() <= 0:
        return np.inf
    lo, hi = 0.0, 1.0
    while (_r(d, hi) * d).mean() < 0:
        hi *= 2
        if hi > 1e12: return np.inf
    for _ in range(100):
        b = (lo + hi) / 2
        if (_r(d, b) * d).mean() < 0: lo = b
        else: hi = b
    r = _r(d, hi)
    return float(np.sqrt(np.mean((r - 1) ** 2)))


def kappa(d):
    return abs(d.mean()) / d.std() if d.std() > 0 else np.inf


def job(fn):
    did, s = fn.stem.split('_'); out = u66.losses(int(did), int(s))
    if out is None: return []
    rows = []
    L, y, name = out
    n = L.shape[1]
    half = np.random.default_rng(1000 + int(s)).permutation(n)
    A, B = half[: n // 2], half[n // 2:]
    pairs = list(itertools.combinations(range(L.shape[0]), 2))
    rad = {(i, j): (kappa(L[i] - L[j]), sharp_radius(L[i] - L[j]), kappa((L[i] - L[j])[A]), sharp_radius((L[i] - L[j])[A]))
           for i, j in pairs if (L[i] - L[j]).mean() != 0 and (L[i] - L[j])[A].mean() != 0}
    for dep, w in u58.deployments(y).items():
        r = w / w.mean(); rA, rB = r[A] / r[A].mean(), r[B] / r[B].mean()
        sd_r, sd_rA = np.sqrt(np.mean((r - 1) ** 2)), np.sqrt(np.mean((rA - 1) ** 2))
        for (i, j), (k_full, s_full, k_A, s_A) in rad.items():
            d = L[i] - L[j]
            dA, dB = d[A], d[B]
            rows.append(dict(dataset=name, seed=int(s), deployment=dep, pair=f'{i}-{j}', sd_r=sd_r,
                             kappa=k_full, sharp=s_full,
                             rev_full=int(np.sign(d.mean()) != np.sign((r * d).mean())),
                             sd_rA=sd_rA, kappa_A=k_A, sharp_A=s_A,
                             rev_out=int(np.sign(dA.mean()) != np.sign((rB * dB).mean()))))

    return rows


if __name__ == '__main__':
    ex = np.array([-10., -3, -3, 1])
    print(f'four-point example: kappa {kappa(ex):.3f}, sharp {sharp_radius(ex):.3f}')
    from multiprocessing import Pool
    with Pool(24) as pool:
        parts = pool.map(job, sorted((R / 'losses').glob('*.npz')))
    rows = [x for part in parts for x in part]
    t = pd.DataFrame(rows); t.to_csv(R / 'sharp_radius.csv', index=False)
    for tag, k, sr, rev in (('full test set', 'kappa', 'sd_r', 'rev_full'), ('held out', 'kappa_A', 'sd_rA', 'rev_out')):
        sh = 'sharp' if tag == 'full test set' else 'sharp_A'
        c0, c1 = t[k] >= t[sr], t[sh] >= t[sr]
        print(f'\n[{tag}] {len(t):,} comparisons')
        print(f'  kappa   certifies {c0.sum():6d}; reversed {t[rev][c0].sum()} ({100 * t[rev][c0].mean():.2f}%)')
        print(f'  sharp   certifies {c1.sum():6d}; reversed {t[rev][c1].sum()} ({100 * t[rev][c1].mean():.2f}%)')
        print(f'  added by sharp {(c1 & ~c0).sum()}; reversed among them {t[rev][c1 & ~c0].sum()}; exposed by both reverse {100 * t[rev][~c1].mean():.1f}%')
        print(f'  sharp < kappa anywhere: {(t[sh] < t[k] - 1e-9).sum()}')
