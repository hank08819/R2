"""u77_joint_cluster.py -- the deployment regret test of u76, with an optional cluster bootstrap.

Same test as u76.regret_test_joint: one two-sided band over all K(K-1) weighted pair
differences and all K benchmark risks, the chosen row read from it, so the verdict holds
whatever row the sample selects.  With cluster labels cl (the day of each Beijing hour),
the influence functions are summed within a cluster and the multipliers drawn per
cluster.  Used by u78 (census) and u79 (Beijing), which run it from the benchmark alone
under the stated weights and score it on the deployment's own labels.
"""
import numpy as np

ALPHA, NBOOT, EPS = 0.05, 1000, [0.01, 0.02, 0.05, 0.10, 0.20]


def regret_test_joint(L, w, cl=None, alpha=ALPHA, nboot=NBOOT, rng=None, eps=EPS):
    rng = rng or np.random.default_rng(0)
    K, n = L.shape
    r = w / w.mean()
    m_hat = L.mean(1); f = int(np.argmin(m_hat))
    I, J = np.where(~np.eye(K, dtype=bool))
    D = L[I] - L[J]
    theta = (D * r).mean(1)
    Psi = np.vstack([r * (D - theta[:, None]), -(L - m_hat[:, None])])
    if cl is None:
        S, G = Psi, n
    else:
        _, inv = np.unique(cl, return_inverse=True); G = inv.max() + 1
        S = np.zeros((Psi.shape[0], G)); np.add.at(S.T, inv, Psi.T)
    se = np.sqrt((S ** 2).sum(1) * G / max(G - 1, 1)) / n
    flat = se <= 1e-12 * max(1.0, float(np.abs(L).max()))
    se = np.where(flat, 0.0, se)
    Zb = (rng.standard_normal((nboot, S.shape[1])) @ S.T) / n / np.where(flat, 1.0, se)
    Zb[:, flat] = 0.0
    c = float(np.quantile(np.abs(Zb).max(1), 1 - alpha))
    P = len(I); row = I == f
    up_t, lo_t = theta + c * se[:P], theta - c * se[:P]
    lo_m, up_m = m_hat - c * se[P:], m_hat + c * se[P:]
    num_up = max(0.0, float(up_t[row].max())); num_lo = float(lo_t[row].max())
    out = dict(choice_j=f, crit_j=c, rr_ucb_j=num_up / lo_m[f] if lo_m[f] > 0 else np.inf)
    for e in eps:
        out[f'cert_j_{e}'] = int(out['rr_ucb_j'] <= e)
        out[f'exceed_j_{e}'] = int(num_lo > e * up_m[f])
    return out
