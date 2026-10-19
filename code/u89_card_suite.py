"""u89_card_suite.py -- the summary card for one of the six questions, on one dataset of the suite.

Concrete compressive strength (OpenML 44959), prepared as in u58 (one-hot, median imputation,
40 percent held out, seed 0), the ten learners.  Deployment: the top decile of strength, as for a
structure that must carry the most load.  For the leading pair (the benchmark's choice f and its
runner-up g), the test records are sorted by d = l_f - l_g.  The bottom tenth is where f does best
against g, the top tenth where g does best.  Each end is summarized by its mix and its strength.
The practitioner's question: which end does your deployment weight more?  The true answer is the
sign of Cov(r, d); the mean weight of each tenth gives the card's answer.
Writes results/universal/card_suite.csv.
"""
import importlib.util, warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.datasets import fetch_openml
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
HERE = Path(__file__).resolve().parent; R = HERE.parent / 'results' / 'universal'
s = importlib.util.spec_from_file_location('u58', HERE / 'u58_regret_test.py'); u58 = importlib.util.module_from_spec(s); s.loader.exec_module(u58)
NAMES = ['random forest', 'extra trees', 'gradient boosting', 'AdaBoost', 'bagging', 'tree depth 5',
         'tree depth 12', 'least squares', 'ridge', 'nearest neighbors']

if __name__ == '__main__':
    dd = fetch_openml(data_id=44959, as_frame=True, parser='auto')
    cols = list(dd.data.columns)
    X = SimpleImputer(strategy='median').fit_transform(pd.get_dummies(dd.data, drop_first=True).astype(float))
    y = pd.to_numeric(dd.target).values.astype(float)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.4, random_state=0)
    L = np.array([(m.fit(Xtr, ytr).predict(Xte) - yte) ** 2 for m in u58.learners(0)])
    w = u58.deployments(yte)['top decile of outcome']; r = w / w.mean()
    order = np.argsort(L.mean(1)); f, g = int(order[0]), int(order[1])
    d = L[f] - L[g]; m = len(d) // 10; srt = np.argsort(d)
    lo, hi = srt[:m], srt[-m:]
    T = pd.DataFrame(Xte, columns=cols)
    rows = []
    for end, ix in (('f ahead', lo), ('g ahead', hi)):
        rows.append({'end': end, 'records': len(ix), 'cement (kg/m3)': T.cement.values[ix].mean(),
                     'water (kg/m3)': T.water.values[ix].mean(), 'age (days)': np.median(T.age.values[ix]),
                     'strength (MPa)': yte[ix].mean(), 'in top decile (%)': 100 * w[ix].mean(),
                     'mean weight': r[ix].mean()})
    out = pd.DataFrame(rows)
    nu = r / r.sum()
    out['f'] = NAMES[f]; out['g'] = NAMES[g]
    out['card_answer'] = 'g end' if r[hi].mean() > r[lo].mean() else 'f end'
    out['sign_cov'] = float(np.sign(np.mean((r - 1) * (d - d.mean()))))
    out['bench_gap_pct'] = 100 * d.mean() / L[f].mean()
    out['dep_gap_pct'] = 100 * float(d @ nu) / L[f].mean()
    out['dep_best'] = NAMES[int(np.argmin(L @ nu))]
    out['test_n'] = len(yte)
    out.to_csv(R / 'card_suite.csv', index=False)
    pd.set_option('display.width', 250); print(out.T.to_string())
