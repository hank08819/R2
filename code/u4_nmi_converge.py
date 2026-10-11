"""Convergence check: along nu_t = (1-t)mu + t*nu the relative remainder must
vanish linearly in t if the influence function and the machinery are right."""
import warnings; warnings.filterwarnings('ignore')
import numpy as np, itertools
from pathlib import Path
import sys; sys.path.insert(0, str(Path(__file__).resolve().parent))
from u2_nmi_remainder import joint, NMI, IF_NMI, load, MODELS, OUT
from sklearn.model_selection import train_test_split

X, y = load('mammography')
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.4, random_state=0, stratify=y)
pred = {k: m.fit(Xtr, ytr).predict(Xte).astype(int) for k, m in MODELS().items()}
n = len(yte); mu = np.full(n, 1 / n)
w = np.where(yte == 1, 50.0, 1.0); r = w / (mu @ w)
IFs = {k: IF_NMI(joint(yte, yh, mu))[yte.astype(int), yh] for k, yh in pred.items()}
base = {k: NMI(joint(yte, yh, mu)) for k, yh in pred.items()}

rows = []
print('   t     sd(r_t)   |R2|/|first| med    ratio / t')
for t in [1.0, 3e-1, 1e-1, 3e-2, 1e-2, 1e-3, 1e-4]:
    rt = 1 - t + t * r
    sdt = float(np.sqrt(mu @ (rt - 1) ** 2))
    nt = {k: NMI(joint(yte, yh, mu * rt)) for k, yh in pred.items()}
    rel = []
    for f, g in itertools.combinations(pred, 2):
        d = IFs[f] - IFs[g]
        first = float(mu @ (d * rt)) - float(mu @ d)
        exact = (nt[f] - nt[g]) - (base[f] - base[g])
        rel.append(abs(exact - first) / max(abs(first), 1e-15))
    m = float(np.median(rel))
    rows.append(dict(t=t, sd_rt=sdt, rel_med=m, ratio=m / t))
    print(f'  {t:6.0e}  {sdt:7.4f}   {m:12.5f}    {m/t:10.3f}')

import pandas as pd
pd.DataFrame(rows).to_csv(OUT / 'nmi_converge.csv', index=False)
print('\nlimiting ratio %.1f  =>  relative remainder ~ %.1f * sd(r)'
      % (rows[-1]['ratio'], rows[-1]['ratio'] / rows[0]['sd_rt']))
