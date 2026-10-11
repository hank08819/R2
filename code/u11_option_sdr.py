"""u11_option_sdr.py -- the density-ratio dispersion of the option-pricing row.

The criterion is R-squared over the full test panel, so the benchmark measure is
uniform there.  The deployment measure is what a desk actually pays: the
vega-weighted pricing error on out-of-the-money contracts inside the transition
window.  Both masks and the vega column come from the same loader the rest of the
option pipeline uses, so this row is built the same way as every other number in
that pipeline.
"""
import warnings; warnings.filterwarnings('ignore')
import sys, numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from v4_analysis import DATASETS, DATA, load_panel, build_test, otm_mask

OUT = Path(__file__).resolve().parents[1] / 'results' / 'universal'
rows = []
for name, cfg in DATASETS.items():
    df = load_panel(DATA / cfg['file'])
    te = build_test(df, cfg['regimes'])
    n = len(te)
    mu = np.full(n, 1 / n)                                   # criterion measure
    inwin = (te['regime'].values == cfg['transition'])
    otm = otm_mask(te['moneyness'].values, te['option_type'].values)
    w = np.abs(te['Vega'].values) * (inwin & otm)             # deployment weight
    r = w / (mu @ w)                                          # mean one under mu
    sd_r = float(np.sqrt(mu @ (r - 1) ** 2))
    rows.append(dict(panel=name, n_test=n, n_transition_otm=int((inwin & otm).sum()),
                     share=100 * (inwin & otm).mean(), sd_r=sd_r))
    print(f'  {name:<5} n={n:>8,}  transition OTM {(inwin&otm).sum():>7,} '
          f'({100*(inwin&otm).mean():4.1f}%)  sd(r)={sd_r:6.2f}', flush=True)

d = pd.DataFrame(rows)
d.to_csv(OUT / 'option_sdr.csv', index=False)
print(f"\nmean across panels {d.sd_r.mean():.2f}   median {d.sd_r.median():.2f}")
