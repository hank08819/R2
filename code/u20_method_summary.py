"""u20_method_summary.py -- the numbers the method section reports, in one place."""
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from pathlib import Path
R = Path(__file__).resolve().parents[1] / 'results' / 'universal'

out = {}
n = pd.read_csv(R / 'robust_eval.csv')          # worst-case loss rule
out['naive_p90_r2'] = 100 * n['exc_0.0'].quantile(.9)
out['naive_p90_stated'] = 100 * n['exc_stated'].quantile(.9)
out['naive_med_stated'] = 100 * n['exc_stated'].median()

m = pd.read_csv(R / 'minimax_regret.csv')       # worst-case regret on the ball
out['ball_mean_r2'] = 100 * m['exc_0.0'].mean()
out['ball_mean_stated'] = 100 * m['exc_stated'].mean()
out['ball_better'] = int((m.exc_stated < m['exc_0.0'] - 1e-12).sum())
out['ball_worse'] = int((m.exc_stated > m['exc_0.0'] + 1e-12).sum())

s = pd.read_csv(R / 'scenario_eval.csv')        # scenarios, held out
out['scen_mean_r2'] = 100 * s.exc_r2.mean()
out['scen_mean'] = 100 * s.exc_sc.mean()
out['scen_better'] = int((s.exc_sc < s.exc_r2 - 1e-12).sum())
out['scen_worse'] = int((s.exc_sc > s.exc_r2 + 1e-12).sum())

c = pd.read_csv(R / 'criteria_choice.csv')      # choosing the criterion
g = c[c.deployment == 'relative-error weighted']
for lab, key in [('relative squared error', 'rel'), ('MAPE', 'mape'),
                 ('R2 (squared error)', 'r2')]:
    h = g[g.criterion == lab]
    out[f'rel_{key}_sdr'] = h.sd_r.median()
    out[f'rel_{key}_exc'] = 100 * h.excess.mean()
t = c[c.deployment == 'top decile of outcome']
for lab, key in [('MAPE', 'mape'), ('R2 (squared error)', 'r2')]:
    h = t[t.criterion == lab]
    out[f'top_{key}_sdr'] = h.sd_r.median()
    out[f'top_{key}_exc'] = 100 * h.excess.mean()

pd.Series(out).to_csv(R / 'method_summary.csv', header=['value'])
for k, v in out.items():
    print(f'  {k:<22} {v:8.2f}')
