"""u31_unittest.py -- does the selection machinery work where the answer is known?

Three constructed problems with hand-checkable answers: one where the criterion is
right and any budget must agree, one where the criterion is wrong and the budget
rule is built to catch it, and one where the deployment is a point mass.  If the
implementation cannot recover these, the negative results on real data mean
nothing.
"""
import warnings; warnings.filterwarnings('ignore')
import numpy as np

def risks(L, mu):                      return L @ mu
def pair_tab(L, mu):
    K = len(L); E = np.zeros((K, K)); S = np.zeros((K, K))
    for i in range(K):
        for j in range(K):
            if i == j: continue
            d = L[i] - L[j]
            E[i, j] = mu @ d
            S[i, j] = np.sqrt(mu @ (d - mu @ d) ** 2)
    return E, S

def pick_r2(L, mu):        return int(np.argmin(risks(L, mu)))
def pick_wc(L, mu, k):
    E, S = pair_tab(L, mu); return int(np.argmin((E + k * S).max(1)))

def report(name, L, mu, r, k, want_r2, want_wc):
    nu = mu * r
    f0, f1 = pick_r2(L, mu), pick_wc(L, mu, k)
    g = int(np.argmin(L @ nu))
    ok = (f0 == want_r2) and (f1 == want_wc)
    print(f'  {name}')
    print(f'    benchmark risks   {np.round(L @ mu, 4)}')
    print(f'    deployment risks  {np.round(L @ nu, 4)}')
    print(f'    R^2 picks {f0} (expect {want_r2}) | worst-case picks {f1} '
          f'(expect {want_wc}) | deployment wants {g}')
    print(f'    -> {"PASS" if ok else "FAIL"}')
    return ok

n = 1000
mu = np.full(n, 1 / n)
half = np.arange(n) < n // 2
allok = []

# 1. model 0 uniformly better: every rule must choose it
L = np.vstack([np.full(n, 1.0), np.full(n, 2.0)])
allok.append(report('uniformly better model', L, mu, np.ones(n), 1.0, 0, 0))

# 2. model 0 wins on the benchmark by a hair; model 1 wins on the half the
#    deployment weights.  The criterion must err and the deployment must want 1.
L = np.vstack([np.where(half, 1.0, 0.0), np.where(half, 0.9, 0.2)])
r = np.where(half, 2.0, 0.0)
k = float(np.sqrt(mu @ (r - 1) ** 2))
allok.append(report('criterion errs, deployment wants the other', L, mu, r, k, 0, 0))

# 3. two models: the worst-case rule provably cannot disagree with the criterion,
#    since the pair contributes -m + kS to one and +m + kS to the other.
rng = np.random.default_rng(0)
dis = 0
for _ in range(2000):
    L = rng.random((2, 200))
    kk = float(rng.uniform(0, 5))
    if pick_r2(L, np.full(200, 1 / 200)) != pick_wc(L, np.full(200, 1 / 200), kk):
        dis += 1
print(f'  two-model disagreements between R^2 and worst-case regret over 2000 '
      f'draws: {dis}  -> {"PASS" if dis == 0 else "FAIL"}')
allok.append(dis == 0)

# 4. with three models disagreement becomes possible: give the criterion's choice
#    a large dispersion against a third model.
L = np.vstack([np.where(half, 0.0, 1.0),                 # best on the benchmark
               np.full(200 * 0 + n, 0.52),               # flat, slightly worse
               np.where(half, 1.6, 0.0)])                # drags 0's worst case up
E, S = pair_tab(L, mu)
f0, f1 = pick_r2(L, mu), pick_wc(L, mu, 2.0)
print(f'  three models: benchmark risks {np.round(L @ mu, 3)}, '
      f'R^2 picks {f0}, worst-case at kappa=2 picks {f1}  '
      f'-> {"PASS" if f0 != f1 else "FAIL (no disagreement possible)"}')
allok.append(f0 != f1)

# 5. point mass dispersion
r = np.zeros(n); r[0] = n
sd = float(np.sqrt(mu @ (r - 1) ** 2))
print(f'  point mass sd(r) = {sd:.1f} against sqrt(n-1) = {np.sqrt(n-1):.1f}  '
      f'-> {"PASS" if abs(sd - np.sqrt(n-1)) < 1e-6 else "FAIL"}')
allok.append(abs(sd - np.sqrt(n - 1)) < 1e-6)

# 6. the impossibility construction, on a pair with a genuine benchmark gap
L = np.vstack([np.where(half, 0.0, 1.0), np.where(half, 0.9, 0.3)])
d = L[0] - L[1]; e = float(mu @ d); sd = float(np.sqrt(mu @ (d - e) ** 2))
k = 1.5 * abs(e) / sd
rp = 1 + k * (d - e) / sd; rm = 1 - k * (d - e) / sd
sp, sm = float(mu @ (d * rp)), float(mu @ (d * rm))
ok = (sp * sm < 0) and rp.min() >= -1e-12 and rm.min() >= -1e-12
print(f'  impossibility: E_mu[d]={e:+.4f} -> E_nu+[d]={sp:+.4f}, '
      f'E_nu-[d]={sm:+.4f}; both ratios non-negative: '
      f'{rp.min()>=-1e-12 and rm.min()>=-1e-12}  -> {"PASS" if ok else "FAIL"}')
allok.append(ok)

print(f'\n{sum(allok)}/{len(allok)} passed')
raise SystemExit(0 if all(allok) else 1)
