"""u66_signed_budget.py -- a regret bound from a budget and six signs, no weight needed.

Information ladder for the benchmark's choice f and each rival g, d_g = l_f - l_g:
  weight known          : the deployment regret test of u58.
  budget only           : every deployment with sd_mu(r) <= rho.  Worst gap
                          U_g = E d_g + rho sd(d_g)                   (Cauchy-Schwarz)
  budget and six signs  : also eta_ab Cov_mu(r, l_a - l_b) >= 0 for the six pairs among
                          the four leading models.  With u = r - 1 and C the cone the
                          signs cut out,  sup_{u in C, |u| <= rho} <u, v> = rho |P_C v|
                          (Moreau), so U_g = E d_g + rho |P_C d~_g|, computed from the
                          covariance of the K losses by non-negative least squares.
Neither bound imposes r >= 0, so both are conservative; the exact budget-only value with
r >= 0 is the one-dimensional dual  inf_eta { eta + sqrt(1+rho^2) |(d - eta)_+| }.

Stage 1 (this file, 'plugin'): the population check on the suite's 498 problems --
does the true gap lie under each bound (it must), how tight is each bound against the
realized regret, and how often do the signs certify the choice outright.
Per-instance test losses are cached in results/universal/losses/ for later stages.
"""
import sys, json, time, warnings, importlib.util; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from pathlib import Path
from scipy.optimize import nnls, minimize_scalar

HERE = Path(__file__).resolve().parent
R = HERE.parent / 'results' / 'universal'
CACHE = R / 'losses'; CACHE.mkdir(exist_ok=True)
spec = importlib.util.spec_from_file_location('u58', HERE / 'u58_regret_test.py')
u58 = importlib.util.module_from_spec(spec); spec.loader.exec_module(u58)
TOP = 4


def losses(did, s):
    """Per-instance test losses of the ten learners, exactly as u58 computes them."""
    f = CACHE / f'{did}_{s}.npz'
    if f.exists():
        z = np.load(f, allow_pickle=True); return z['L'], z['y'], str(z['name'])
    from sklearn.datasets import fetch_openml
    from sklearn.model_selection import train_test_split
    from sklearn.impute import SimpleImputer
    dd = fetch_openml(data_id=did, as_frame=True, parser='auto')
    X = pd.get_dummies(dd.data, drop_first=True).astype(float)
    X = SimpleImputer(strategy='median').fit_transform(X)
    y = pd.to_numeric(dd.target, errors='coerce').values.astype(float)
    ok = np.isfinite(y) & np.isfinite(X).all(1); X, y = X[ok], y[ok]
    if len(y) > u58.NMAX:
        idx = np.random.default_rng(0).choice(len(y), u58.NMAX, replace=False); X, y = X[idx], y[idx]
    if len(y) < 200 or X.shape[1] > 300:
        return None
    name = dd.details.get('name', str(did))
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.4, random_state=s)
    L = np.array([(m.fit(Xtr, ytr).predict(Xte) - yte) ** 2 for m in u58.learners(s)])
    np.savez_compressed(f, L=L, y=yte, name=name)
    return L, yte, name


def signs(L, r, top):
    """The six answers: for each pair (a, b) among the leaders, the sign of Cov(r, l_a - l_b)."""
    out = []
    for i in range(len(top)):
        for j in range(i + 1, len(top)):
            a, b = top[i], top[j]
            c = float(np.mean((r - r.mean()) * (L[a] - L[b])))
            out.append((a, b, 1.0 if c >= 0 else -1.0))
    return out


def cone_norms(L, f, answers, w=None):
    """|P_C d~_g| for every rival g of f, from the (weighted) covariance of the losses.
    The cone is {u : eta (l_a - l_b)~ . u >= 0} over the answers."""
    K = L.shape[0]
    G = np.cov(L, aweights=w, bias=True)
    ev, V = np.linalg.eigh(G); Rt = (V * np.sqrt(np.clip(ev, 0, None))).T     # G = Rt' Rt
    A = np.zeros((K, len(answers)))
    for k, (a, b, eta) in enumerate(answers):
        A[a, k], A[b, k] = eta, -eta
    RA = Rt @ A
    out, coef = np.zeros(K), np.zeros((K, K))
    for g in range(K):
        if g == f: continue
        c = np.zeros(K); c[f], c[g] = 1, -1
        lam, _ = nnls(RA, -(Rt @ c)) if len(answers) else (np.zeros(0), 0)
        p = c + A @ lam if len(answers) else c
        out[g] = np.sqrt(max(p @ G @ p, 0)); coef[g] = p
    return out, coef


def exact_budget(d, rho):
    """sup E[r d] over r >= 0, E r = 1, sd(r) <= rho: the chi-square dual."""
    s = np.sqrt(1 + rho ** 2)
    obj = lambda e: e + s * np.sqrt(np.mean(np.clip(d - e, 0, None) ** 2))
    lo, hi = d.min() - 1e-9, d.max()
    res = minimize_scalar(obj, bounds=(lo - (hi - lo), hi), method='bounded', options={'xatol': 1e-10})
    return min(float(res.fun), float(d.max()))


def exact_signed(d, A, rho):
    """sup E[r d] over r >= 0, E r = 1, sd(r) <= rho and Cov(r, a_k) >= 0 for the signed
    columns a_k of A (n x m).  Convex dual in (eta, beta >= 0):
        eta - beta . E a + sqrt(1+rho^2) |(d - eta + A beta)_+|."""
    from scipy.optimize import minimize
    s = np.sqrt(1 + rho ** 2); m = A.shape[1]; Ea = A.mean(0)
    def obj(x):
        e, b = x[0], x[1:]
        z = d - e + A @ b; zp = np.clip(z, 0, None); nz = np.sqrt(np.mean(zp ** 2))
        val = e - b @ Ea + s * nz
        if nz <= 0: return val, np.r_[1.0, -Ea]
        gz = s * zp / (nz * len(d))
        return val, np.r_[1.0 - gz.sum(), -Ea + A.T @ gz]
    x0 = np.r_[np.median(d), np.zeros(m)]
    res = minimize(obj, x0, jac=True, method='L-BFGS-B', bounds=[(None, None)] + [(0, None)] * m,
                   options={'maxiter': 2000, 'ftol': 1e-13, 'gtol': 1e-10})
    return min(float(res.fun), float(d.max()))


def plugin(L, w):
    """Population check for one problem: true weight w, true budget, true answers."""
    K, n = L.shape
    r = w / w.mean(); rho = float(np.sqrt(np.mean((r - 1) ** 2)))
    mu = L.mean(1); nu = (L * r).mean(1)
    f = int(np.argmin(mu)); top = list(np.argsort(mu)[:TOP]); risk = mu[f]
    ans = signs(L, r, top)
    cn, _ = cone_norms(L, f, ans)
    Asg = np.column_stack([e * (L[a] - L[b]) for a, b, e in ans])
    ans_f = [(f, g, 1.0 if np.mean((r - 1) * (L[f] - L[g])) >= 0 else -1.0) for g in range(K) if g != f]
    Af = np.column_stack([e * (L[a] - L[b]) for a, b, e in ans_f])
    gap_true, U_b, U_s, U_x, U_xs, U_xf, binder = [], [], [], [], [], [], []
    for g in range(K):
        if g == f: continue
        d = L[f] - L[g]
        gap_true.append(float((d * r).mean()))
        U_b.append(d.mean() + rho * d.std()); U_s.append(d.mean() + rho * cn[g]); U_x.append(exact_budget(d, rho))
        U_xs.append(exact_signed(d, Asg, rho)); binder.append(g)
        U_xf.append(exact_signed(d, Af, rho))
    gap_true, U_b, U_s, U_x, U_xs, U_xf = map(np.array, (gap_true, U_b, U_s, U_x, U_xs, U_xf))
    # robust choice from the signs: the model whose worst signed regret is least
    worst = []
    for h in range(K):
        cnh, _ = cone_norms(L, h, ans)
        worst.append(max(0.0, max(L[h].mean() - L[g].mean() + rho * cnh[g] for g in range(K) if g != h)))
    h = int(np.argmin(worst))
    return dict(K=K, n=n, rho=rho, choice=f, rr_true=float((nu[f] - nu.min()) / risk), opt_true=int(nu[f] <= nu.min() + 1e-15),
                viol_budget=int((gap_true > U_b + 1e-9).any()), viol_signed=int((gap_true > U_s + 1e-9).any()),
                viol_exact=int((gap_true > U_x + 1e-9).any()), viol_xs=int((gap_true > U_xs + 1e-7).any()),
                rr_xsigned=max(0, U_xs.max()) / risk, cert_xs=int(U_xs.max() <= 0),
                viol_xf=int((gap_true > U_xf + 1e-7).any()), rr_xchoice=max(0, U_xf.max()) / risk, cert_xf=int(U_xf.max() <= 0),
                bind_rank=int(np.argsort(np.argsort(mu))[binder[int(np.argmax(U_x))]]),
                bind_rank_xs=int(np.argsort(np.argsort(mu))[binder[int(np.argmax(U_xs))]]),
                rr_budget=max(0, U_b.max()) / risk, rr_signed=max(0, U_s.max()) / risk, rr_exact=max(0, U_x.max()) / risk,
                certified=int(U_s.max() <= 0), n_favour=int(sum(1 for a, b, e in ans if (a == f and e < 0) or (b == f and e > 0))),
                robust=h, rr_robust_true=float((nu[h] - nu.min()) / risk),
                rr_robust_bound=worst[h] / L[h].mean())


if __name__ == '__main__':
    ids = json.load(open(R / 'ctr23_ids.json'))
    if len(sys.argv) > 1 and sys.argv[1] == 'pilot': ids = ids[:3]
    rows = []; t0 = time.time()
    for i, did in enumerate(ids):
        for s in u58.SEEDS:
            try:
                out = losses(did, s)
            except Exception as e:
                print('skip', did, e); out = None
            if out is None: break
            L, y, name = out
            for dep, w in u58.deployments(y).items():
                rows.append(dict(dataset=name, seed=s, deployment=dep, **plugin(L, w)))
        print(f'[{i+1}/{len(ids)}] {did} {len(rows)} {time.time()-t0:.0f}s', flush=True)
    t = pd.DataFrame(rows); t.to_csv(R / 'signed_budget_plugin.csv', index=False)
    print(f'\n{len(t)} problems')
    print(f'violations (must be 0): budget {t.viol_budget.sum()}, signed {t.viol_signed.sum()}, exact {t.viol_exact.sum()}')
    w = t[t.opt_true == 0]
    print(f'violations exact+signed {t.viol_xs.sum()}')
    print(f'violations exact+choice signs {t.viol_xf.sum()}; certifies {100*t.cert_xf.mean():.1f}%, truly optimal {100*t.opt_true[t.cert_xf==1].mean():.1f}%')
    for c in ['rr_budget', 'rr_exact', 'rr_signed', 'rr_xsigned', 'rr_xchoice']:
        ratio = (w[c] / w.rr_true)
        print(f'{c:10s}: median bound {100*t[c].median():6.2f}% | wrong choices: median bound/true {ratio.median():6.1f}x')
    print(f'certified by the signs: {100*t.certified.mean():.1f}% of problems; of those, truly optimal {100*t.opt_true[t.certified==1].mean():.1f}%')
    print(f'exact+signed certifies {100*t.cert_xs.mean():.1f}%; of those truly optimal {100*t.opt_true[t.cert_xs==1].mean():.1f}%')
    print('binding rival benchmark rank (exact):', t.bind_rank.value_counts().sort_index().to_dict(), ' (exact+signed):', t.bind_rank_xs.value_counts().sort_index().to_dict())
    print(f'robust choice: mean true RR {100*t.rr_robust_true.mean():.2f}% vs benchmark choice {100*t.rr_true.mean():.2f}%')
