"""u34_llm_analysis.py -- the identity on a public language-model leaderboard.

The leaderboard's headline number averages six task groups equally, whatever the
number of questions in each.  That equal weighting is the benchmark measure.  A
reader who cares about the questions rather than the groups, or about one group
in particular, is deploying under a different one, and Theorem 1 gives the gap
between the two orderings exactly.
"""
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, itertools, re
from pathlib import Path

R = Path(__file__).resolve().parents[1] / 'results' / 'universal'
d = pd.read_csv(R / 'llm_raw.csv')

AGG = {'leaderboard_bbh', 'leaderboard_gpqa', 'leaderboard_math_hard',
       'leaderboard_musr', 'leaderboard'}
d = d[~d.task.isin(AGG)]

def group(t):
    for g in ['bbh', 'math', 'gpqa', 'musr', 'mmlu_pro', 'ifeval']:
        if t.startswith('leaderboard_' + g):
            return g
    return 'other'

d['group'] = d.task.map(group)
d = d[d.group != 'other']
# Keep only models evaluated on every subtask: equal counts are not enough,
# because a missing cell becomes a NaN that np.argmin would silently select.
W = d.pivot(index='model', columns='task', values='score')
tasks = [t for t in W.columns if W[t].notna().sum() >= .8 * len(W)]
W = W[tasks].dropna(axis=0)
models = sorted(W.index)
d = d[d.model.isin(models) & d.task.isin(tasks)]
tasks = sorted(tasks)
G = d.drop_duplicates('task').set_index('task')['group']
N = d.drop_duplicates('task').set_index('task')['n'].astype(float)
S = d.pivot(index='model', columns='task', values='score').loc[models, tasks]
L = 1.0 - S.values                                    # per-task error, the score
assert np.isfinite(L).all(), 'missing cells remain in the score matrix'

# the leaderboard's own measure: six groups equally, subtasks equally within
gsz = G.loc[tasks].value_counts()
mu = np.array([1.0 / (len(gsz) * gsz[G[t]]) for t in tasks]); mu /= mu.sum()
n = N.loc[tasks].values

DEPLOY = {'by question count': n / n.sum()}
for g in sorted(gsz.index):
    w = np.array([1.0 if G[t] == g else 0.0 for t in tasks])
    DEPLOY[f'the {g} group alone'] = w / w.sum()
hard = (L.mean(0) >= np.quantile(L.mean(0), .75)).astype(float)
DEPLOY['the hardest quartile of tasks'] = hard / hard.sum()

rows = []
for dep, nu in DEPLOY.items():
    r = nu / mu
    sd_r = float(np.sqrt(mu @ (r - 1) ** 2))
    for i, j in itertools.combinations(range(len(models)), 2):
        dd = L[i] - L[j]
        e_mu, e_nu = float(mu @ dd), float(nu @ dd)
        sd_d = float(np.sqrt(mu @ (dd - e_mu) ** 2))
        rows.append(dict(deployment=dep, pair=f'{models[i]}|{models[j]}',
                         sd_r=sd_r, sd_d=sd_d, e_mu=e_mu, e_nu=e_nu,
                         cov=e_nu - e_mu,
                         rho=(e_nu - e_mu) / (sd_d * sd_r) if sd_d * sd_r > 0 else 0.,
                         inverted=int(e_mu != 0 and np.sign(e_mu) != np.sign(e_nu)),
                         identity_err=abs((e_nu - e_mu) - float(mu @ ((dd - e_mu) * (r - 1))))
                                      / max(abs(e_nu - e_mu), 1e-300)))
t = pd.DataFrame(rows); t.to_csv(R / 'llm_pairs.csv', index=False)

print(f'{len(models)} models, {len(tasks)} subtasks in {len(gsz)} groups, '
      f'{len(models)*(len(models)-1)//2} pairs per deployment, {len(t)} rows\n')
print(f'identity holds to a relative {t.identity_err.max():.1e}\n')
print(f'{"deployment":<32}{"sd(r)":>8}{"reversed (%)":>14}{"median |rho|":>14}')
for dep, g in t.groupby('deployment'):
    print(f'  {dep:<30}{g.sd_r.iloc[0]:8.2f}{100*g.inverted.mean():14.1f}'
          f'{g.rho.abs().median():14.3f}')
print(f'\noverall reversal rate {100*t.inverted.mean():.1f}%')
# selection: does the leaderboard's top model stay on top?
sel = []
for dep, nu in DEPLOY.items():
    f = int(np.argmin(L @ mu)); g = int(np.argmin(L @ nu))
    rr = (float(nu @ L[f]) - float(nu @ L[g])) / float(mu @ L[f])
    sel.append(dict(deployment=dep, leader=models[f], best=models[g],
                    same=int(f == g), regret_ratio=rr))
s = pd.DataFrame(sel); s.to_csv(R / 'llm_selection.csv', index=False)
print(f'\nthe leaderboard leader is also best under the deployment in '
      f'{int(s.same.sum())} of {len(s)} deployments')
print(s[['deployment', 'same', 'regret_ratio']].to_string(index=False))
