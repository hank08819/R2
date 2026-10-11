"""u3_nmi_sweep.py -- where does the first-order expansion for NMI hold?

Theory says the remainder is second order in the mismatch while the covariance
term is first order, so the relative remainder should grow linearly in sd(r).
This traces that out and reads off the mismatch at which the approximation stops
being usable.
"""
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, itertools
from pathlib import Path
import sys; sys.path.insert(0, str(Path(__file__).resolve().parent))
from u2_nmi_remainder import joint, NMI, IF_NMI, load, MODELS, OUT
from sklearn.model_selection import train_test_split

rows = []
for name in ['mammography', 'sick']:
    X, y = load(name)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.4, random_state=0, stratify=y)
    pred = {k: m.fit(Xtr, ytr).predict(Xte).astype(int) for k, m in MODELS().items()}
    n = len(yte); mu = np.full(n, 1 / n)
    for C in [1.2, 1.5, 2, 3, 5, 8, 12, 20, 35, 50]:
        w = np.where(yte == 1, float(C), 1.0); r = w / (mu @ w)
        sd_r = float(np.sqrt(mu @ (r - 1) ** 2))
        nmu, nnu, IFs = {}, {}, {}
        for k, yh in pred.items():
            p = joint(yte, yh, mu)
            nmu[k], nnu[k] = NMI(p), NMI(joint(yte, yh, mu * r))
            IFs[k] = IF_NMI(p)[yte.astype(int), yh]
        rel, ok, np_, na = [], [], 0, 0
        for f, g in itertools.combinations(pred, 2):
            gm, gn = nmu[f] - nmu[g], nnu[f] - nnu[g]
            d = IFs[f] - IFs[g]
            first = float(mu @ (d * r)) - float(mu @ d)
            exact = gn - gm
            rel.append(abs(exact - first) / max(abs(first), 1e-12))
            ok.append(np.sign(gm) != np.sign(gn))
            np_ += int(np.sign(gm) != np.sign(gm + first)); na += int(np.sign(gm) != np.sign(gn))
        rows.append(dict(dataset=name, cost=C, sd_r=sd_r,
                         rel_med=float(np.median(rel)), rel_max=float(np.max(rel)),
                         rev_pred=np_, rev_actual=na))

df = pd.DataFrame(rows); df.to_csv(OUT / 'nmi_sweep.csv', index=False)
for ds, g in df.groupby('dataset'):
    print(f'\n{ds}')
    print('  sd(r)   |R2|/|first| med    max     ratio/sd(r)   rev pred/actual')
    for _, x in g.iterrows():
        print(f'  {x.sd_r:5.2f}  {x.rel_med:12.3f} {x.rel_max:7.2f}   '
              f'{x.rel_med/x.sd_r:10.3f}      {int(x.rev_pred):3d} / {int(x.rev_actual)}')
