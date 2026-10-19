"""u73_leaderboard_regret_test.py -- the deployment regret test on the four leaderboards.

For each board of u72 and each of its deployments, runs certify.regret_test (the
selection-adjusted test of Supplementary Note 8) at tolerances 0.05 and 0.10.  The
instances are the board's tasks.  Two boards weight their tasks unequally (mu), so the
test is written for a weighted benchmark: every mean becomes a mu-weighted sum, the
influence functions are the same, the standard error is sqrt(n/(n-1) sum mu^2 psi^2) and
the multipliers act on mu psi.  With mu uniform this is certify.regret_test exactly, which
is asserted on the two uniform boards.  The point estimate must equal the exact regret
ratio of u72; this is asserted too.
Writes results/universal/leaderboards/leaderboards_regret_test.csv.
"""
import importlib.util, sys, warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from pathlib import Path

HERE = Path(__file__).resolve().parent
R = HERE.parent / 'results' / 'universal' / 'leaderboards'


def mod(name, path):
    s = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m); return m


u72 = mod('u72', HERE / 'u72_leaderboards.py')
cert = mod('certify', HERE.parents[1] / 'universal_metric' / 'nmi-3-code' / 'certify.py')

def regret_test_w(L, mu, nu, tol=0.10, alpha=0.05, nboot=2000, seed=0):
    """certify.regret_test with benchmark weights mu (sum 1) and deployment weights nu."""
    K, n = L.shape
    r = np.divide(nu, mu, out=np.zeros_like(nu), where=mu > 0)
    neff = float(nu.sum() ** 2 / (nu ** 2).sum())             # Kish, as in certify: the weights mu r = nu
    m = L @ mu; f = int(np.argmin(m))
    I, J = np.where(~np.eye(K, dtype=bool))
    D = L[I] - L[J]
    theta = D @ nu
    Psi = np.vstack([r * (D - theta[:, None]), -(L - m[:, None])])
    se = np.sqrt(n / (n - 1) * ((mu * Psi) ** 2).sum(1))
    flat = se <= 1e-12 * max(1.0, float(np.abs(L).max()))     # known exactly up to rounding, as in certify
    se = np.where(flat, 0.0, se)
    Zb = (np.random.default_rng(seed).standard_normal((nboot, n)) @ (mu * Psi).T) / np.where(flat, 1.0, se)
    Zb[:, flat] = 0.0
    c = float(np.quantile(Zb.max(1), 1 - alpha)); c2 = float(np.quantile(np.abs(Zb).max(1), 1 - alpha))
    P = len(I); row = I == f
    up = max(0.0, float((theta + c2 * se[:P])[row].max())); lo = float((theta - c2 * se[:P])[row].max())   # one two-sided band
    m_lo, m_up = m[f] - c2 * se[P + f], m[f] + c2 * se[P + f]
    rr_ucb = up / m_lo if m_lo > 0 else np.inf
    verdict = 'certified' if rr_ucb <= tol else ('exceeds' if lo > tol * m_up else 'undetermined')
    return dict(verdict=verdict, tol=tol, rr_hat=max(0.0, float(theta[row].max())) / m[f],
                rr_ucb=rr_ucb, neff=neff)


rows = []
for board in ['openllm', 'medical', 'finance', 'livebench']:
    t = u72.load(board)
    S = t.pivot_table(index='model', columns='task', values='score', aggfunc='first').dropna(axis=0)
    L = 1.0 - S.values; n = L.shape[1]
    mu, dep = u72.measures(board, t, list(S.columns), L)
    sel = pd.read_csv(R / f'{board}_selection.csv').set_index('deployment')
    uniform = np.allclose(mu, 1.0 / n)
    for name, nu in dep.items():
        out = {tol: regret_test_w(L, mu, nu, tol=tol) for tol in (0.05, 0.10)}
        if uniform:                                     # the weighted test is certify's test
            o = cert.regret_test(L, nu, tol=0.10)
            assert o['verdict'] == out[0.10]['verdict'] and np.isclose(o['rr_ucb'], out[0.10]['rr_ucb'], rtol=1e-9) and np.isclose(o['neff'], out[0.10]['neff']), (board, name, o, out[0.10])
        exact = float(sel.loc[name, 'regret_ratio'])
        assert abs(out[0.10]['rr_hat'] - exact) < 1e-9, (board, name, out[0.10]['rr_hat'], exact)
        rows.append(dict(board=board, deployment=name, tasks=n, neff=out[0.10]['neff'],
                         rr_exact=exact, rr_ucb=out[0.10]['rr_ucb'],
                         verdict_05=out[0.05]['verdict'], verdict_10=out[0.10]['verdict']))
d = pd.DataFrame(rows); d.to_csv(R / 'leaderboards_regret_test.csv', index=False)
pd.set_option('display.width', 200)
print(d.round(3).to_string(index=False))
print()
print(d.groupby('board')[['verdict_05', 'verdict_10']].agg(lambda x: x.value_counts().to_dict()).to_string())
