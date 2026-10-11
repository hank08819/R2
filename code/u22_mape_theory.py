"""u22_mape_theory.py -- MAPE's implied measure, and the closed form of its mismatch.

Ordering by MAPE is ordering by absolute error under the measure proportional to
the reciprocal of the outcome.  Its density ratio to any deployment therefore
carries an extra factor of the outcome, and the dispersion of that ratio is a
coefficient of variation.  This checks the closed form against a direct
computation on two datasets from the suite.
"""
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.datasets import fetch_openml

OUT = Path(__file__).resolve().parents[1] / 'results' / 'universal'
rows = []
for did in [44963, 44970]:
    d = fetch_openml(data_id=did, as_frame=True, parser='auto')
    y = pd.to_numeric(d.target, errors='coerce').values.astype(float)
    y = y[np.isfinite(y)]; y = y[np.abs(y) > 1e-8]
    a = np.abs(y); n = len(y)
    mu = np.full(n, 1 / n)
    mM = (1 / a) / (1 / a).sum()                      # MAPE's implied measure
    for lab, w in [('unweighted', np.ones(n)),
                   ('percentage, $w\\propto 1/|y|$', 1 / a),
                   ('upper decile of the outcome',
                    (y >= np.quantile(y, .9)).astype(float) + 1e-12),
                   ('rising in the outcome, $w\\propto|y|$', a)]:
        nu = w * mu; nu = nu / nu.sum()
        r = nu / mM; r = r / (mM @ r)
        direct = float(np.sqrt(mM @ (r - 1) ** 2))
        z = w * a
        closed = float(np.sqrt(mM @ (z - mM @ z) ** 2) / (mM @ z))
        rows.append(dict(dataset=d.details['name'], n=n, deployment=lab,
                         direct=direct, closed=closed, err=abs(direct - closed)))
t = pd.DataFrame(rows); t.to_csv(OUT / 'mape_theory.csv', index=False)
print(t.to_string(index=False))
print(f'\nlargest discrepancy between the two routes: {t.err.max():.2e}')
