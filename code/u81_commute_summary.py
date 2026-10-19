"""u81_commute_summary.py -- the pre-registered outcomes of the commute-time case (u80).

Outcomes as fixed in results/universal/prereg_commute.md, per choice, eps in {5%, 10%},
spot-check size n in {500, 1000}:
  without the rule  deployed if step 2 certifies; else, unless step 2 finds 'exceeds',
                    deployed when the step-4 draw does not flag;
  with the rule     every non-'exceeds' choice goes to step 4; deployed when not flagged.
  1. wrong pass   among choices with true regret ratio > eps, share deployed (mean over draws);
  2. false alarm  among choices that are the deployment's best, share not deployed;
  3. coverage and wrong certifications of the benchmark-only test (as Table S12).
Criterion: the rule is supported if its wrong pass is below the no-rule wrong pass at eps 5%
and n 1,000, and its false alarm is at most 5%.  Writes commute_summary.csv.
"""
import numpy as np, pandas as pd
from pathlib import Path

R = Path(__file__).resolve().parents[1] / 'results' / 'universal'
p = pd.read_csv(R / 'commute_problems.csv'); s = pd.read_csv(R / 'commute_spot.csv'); t = pd.read_csv(R / 'commute_pairs.csv')
key = ['state', 'year', 'seed']
flag = s.groupby(key + ['n']).reject.mean().rename('flag').reset_index()

rows = []
for eps in (0.05, 0.10):
    for n in (500, 1000):
        q = p.merge(flag[flag.n == n], on=key, how='left')
        cert, exc, fl = q[f'cert_j_{eps}'] == 1, q[f'exceed_j_{eps}'] == 1, q.flag
        dep_no = np.where(cert, 1.0, np.where(exc, 0.0, 1 - fl))
        dep_rule = np.where(exc, 0.0, 1 - fl)
        bad, best = q.rr_true > eps, q.opt_true == 1
        for w, m in [('all', np.ones(len(q), bool))] + [(w, (q.weight == w).values) for w in sorted(q.weight.unique())]:
            rows.append(dict(eps=eps, n=n, weight=w, n_choices=int(m.sum()), n_bad=int((bad & m).sum()), n_best=int((best & m).sum()),
                             wrong_pass_no_rule=dep_no[bad & m].mean() if (bad & m).any() else np.nan,
                             wrong_pass_rule=dep_rule[bad & m].mean() if (bad & m).any() else np.nan,
                             false_alarm_no_rule=1 - dep_no[best & m].mean(), false_alarm_rule=1 - dep_rule[best & m].mean(),
                             step2_wrong_cert=int((cert & bad & m).sum()),
                             step2_cover=float((q.rr_true <= q.rr_ucb_j + 1e-12)[m].mean())))
out = pd.DataFrame(rows); out.to_csv(R / 'commute_summary.csv', index=False)
pd.set_option('display.width', 200)
print(out.round(4).to_string(index=False))

print(f"\nselection error: {100 * (p.drop_duplicates(key).opt_true == 0).mean():.1f}% of {p.drop_duplicates(key).shape[0]} choices;"
      f" regret ratio where wrong: median {100 * p.drop_duplicates(key).query('opt_true == 0').rr_true.median():.1f}%,"
      f" max {100 * p.drop_duplicates(key).rr_true.max():.1f}%")
for w, u in t.groupby('weight'):
    c = u.certified == 1
    print(f"step 1 {w:10s}: certified {c.sum()}, reversed {u.reversed_true[c].sum()}; exposed reversed {100 * u.reversed_true[~c].mean():.1f}%")
a = out[(out.eps == 0.05) & (out.n == 1000) & (out.weight == 'all')].iloc[0]
if a.n_bad == 0:
    print('\nCRITERION: untested at eps 5% (no choice exceeds it)')
else:
    ok = (a.wrong_pass_rule < a.wrong_pass_no_rule) and (a.false_alarm_rule <= 0.05)
    print(f"\nCRITERION at eps 5%, n 1,000: wrong pass {a.wrong_pass_rule:.3f} with the rule vs {a.wrong_pass_no_rule:.3f} without;"
          f" false alarm {a.false_alarm_rule:.3f}  ->  {'MET' if ok else 'NOT MET'}")
