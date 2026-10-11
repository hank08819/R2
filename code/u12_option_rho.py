"""u12_option_rho.py -- why the widest-mismatch option panels reverse least.

The factorization in the Methods says the bias is sd(d) * sd(r) * rho, so a wide mismatch is harmless
where it fails to line up with what separates the models.  The five option panels
disagree sharply in sd(r); this measures the other two factors on the same panels,
using the seven learners common to all five, and asks whether the product tracks
the reversal rate where sd(r) alone does not.
"""
import warnings; warnings.filterwarnings('ignore')
import os, sys, itertools, numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from v4_analysis import DATASETS, DATA, load_panel, build_test, otm_mask

# Option predictions are commercial data and are not released; point OPTION_PRED at them.
PRED = Path(os.environ.get('OPTION_PRED',
            Path(__file__).resolve().parents[2] / 'JF-code-data' / 'predictions'))
OUT = Path(__file__).resolve().parents[1] / 'results' / 'universal'
MODELS = ['RF-snap', 'XGBoost-snap', 'CatBoost-snap',
          'RF-rich', 'XGBoost-rich', 'CatBoost-rich', 'CASUAL-LSTM-rich']

rows = []
for name, cfg in DATASETS.items():
    d = PRED / f'{name.lower()}_ng_preds'
    te = build_test(load_panel(DATA / cfg['file']), cfg['regimes'])
    n = len(te)
    iv = te['IV'].values
    mu = np.full(n, 1 / n)
    w = np.abs(te['Vega'].values) * ((te['regime'].values == cfg['transition'])
                                     & otm_mask(te['moneyness'].values,
                                                te['option_type'].values))
    r = w / (mu @ w)
    sd_r = float(np.sqrt(mu @ (r - 1) ** 2))
    P = {m: np.load(d / f'{m}.npy') for m in MODELS}
    assert all(len(v) == n for v in P.values()), f'{name}: prediction length mismatch'
    sq = {m: (np.clip(v, .01, 4.) - iv) ** 2 for m, v in P.items()}
    for f, g in itertools.combinations(MODELS, 2):
        dd = sq[f] - sq[g]
        e_mu, e_nu = float(mu @ dd), float(mu @ (dd * r))
        sd_d = float(np.sqrt(mu @ (dd - e_mu) ** 2))
        rows.append(dict(panel=name, pair=f'{f}|{g}', sd_r=sd_r, sd_d=sd_d,
                         rho=(e_nu - e_mu) / (sd_d * sd_r) if sd_d * sd_r > 0 else 0.,
                         e_mu=e_mu, e_nu=e_nu,
                         inverted=int(e_mu != 0 and np.sign(e_mu) != np.sign(e_nu))))
    print(f'  {name} done', flush=True)

df = pd.DataFrame(rows); df.to_csv(OUT / 'option_rho.csv', index=False)
s = df.groupby('panel').agg(sd_r=('sd_r', 'first'), sd_d=('sd_d', 'median'),
                            absrho=('rho', lambda x: x.abs().median()),
                            rev=('inverted', lambda x: 100 * x.mean()))
s['product'] = s.sd_d * s.sd_r * s.absrho
print('\n', s.round(4).to_string())
print('\nspearman with reversal rate:')
for c in ['sd_r', 'sd_d', 'absrho', 'product']:
    print(f'  {c:<8} {s[c].corr(s.rev, method="spearman"):+.2f}')
