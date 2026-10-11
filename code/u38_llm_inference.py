"""u38_llm_inference.py -- which of the leaderboard's orderings survive its own noise.

Each subtask score is a proportion over a known number of questions, so the gap
between two models carries a standard error.  Treating the two models as
independent overstates that error, since they answer the same questions and their
errors are positively correlated, so the count of resolved comparisons below is
conservative.
"""
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, itertools
from pathlib import Path

R = Path(__file__).resolve().parents[1] / 'results' / 'universal'
d = pd.read_csv(R / 'llm_raw.csv')
AGG = {'leaderboard_bbh', 'leaderboard_gpqa', 'leaderboard_math_hard',
       'leaderboard_musr', 'leaderboard'}
d = d[~d.task.isin(AGG)]


def grp(t):
    for g in ['bbh', 'math', 'gpqa', 'musr', 'mmlu_pro', 'ifeval']:
        if t.startswith('leaderboard_' + g):
            return g


d['group'] = d.task.map(grp); d = d[d.group.notna()]
W = d.pivot(index='model', columns='task', values='score')
tasks = [t for t in W.columns if W[t].notna().sum() >= .8 * len(W)]
W = W[tasks].dropna(axis=0)
models = list(W.index)
G = d.drop_duplicates('task').set_index('task')['group']
N = d.drop_duplicates('task').set_index('task')['n'].loc[tasks].values.astype(float)
gsz = G.loc[tasks].value_counts()
mu = np.array([1.0 / (len(gsz) * gsz[G[t]]) for t in tasks]); mu /= mu.sum()
P = W.values                                   # accuracy per subtask
L = 1.0 - P
V = P * (1 - P) / N                            # variance of each subtask score

DEP = {'by question count': N / N.sum()}
for g in sorted(gsz.index):
    w = np.array([1.0 if G[t] == g else 0.0 for t in tasks]); DEP[f'the {g} group alone'] = w / w.sum()
hard = (L.mean(0) >= np.quantile(L.mean(0), .75)).astype(float)
DEP['the hardest quartile of tasks'] = hard / hard.sum()

rows = []
for dep, nu in DEP.items():
    for i, j in itertools.combinations(range(len(models)), 2):
        dd = L[i] - L[j]
        vd = V[i] + V[j]                       # independent models: conservative
        e_mu, e_nu = float(mu @ dd), float(nu @ dd)
        se_mu = float(np.sqrt((mu ** 2 * vd).sum()))
        se_nu = float(np.sqrt((nu ** 2 * vd).sum()))
        rows.append(dict(deployment=dep, e_mu=e_mu, e_nu=e_nu,
                         se_mu=se_mu, se_nu=se_nu))
t = pd.DataFrame(rows); t.to_csv(R / 'llm_inference.csv', index=False)
z = 1.96
t['rev'] = (np.sign(t.e_mu) != np.sign(t.e_nu)) & (t.e_mu != 0)
t['sm'] = t.e_mu.abs() > z * t.se_mu
t['sn'] = t.e_nu.abs() > z * t.se_nu
both = t[t.sm & t.sn]
print(f'{len(t):,} pair-and-deployment combinations, {len(models)} models\n')
print(f'  the leaderboard resolves            {100*t.sm.mean():.1f}% of pairs')
print(f'  both measures resolve               {100*(t.sm&t.sn).mean():.1f}%')
print(f'  reversal rate, all pairs            {100*t.rev.mean():.1f}%')
print(f'  reversal rate, resolved pairs       {100*both.rev.mean():.1f}%')
print('\nby deployment, among resolved pairs:')
for dep, g in both.groupby('deployment'):
    print(f'  {dep:<32}{100*g.rev.mean():6.1f}%   ({len(g):,} pairs)')
