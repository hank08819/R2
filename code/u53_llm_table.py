"""u53_llm_table.py -- the 77 language models, by name, with how far each one moves.

The leaderboard analysis names no model in the main text beyond the dozen Figure 1 has
room for.  This writes the whole list: rank under the leaderboard's stated aggregation,
rank when the same published scores are weighted by question count, and the widest rank
a model reaches across the eight deployment measures.  Emits a LaTeX longtable to
results/universal/llm_models.tex and the same content as a CSV.
"""
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
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
W = d.pivot(index='model', columns='task', values='score')
tasks = sorted(t for t in W.columns if W[t].notna().sum() >= .8 * len(W))
models = sorted(W[tasks].dropna(axis=0).index)
S = d.pivot(index='model', columns='task', values='score').loc[models, tasks]
L = 1.0 - S.values
G = d.drop_duplicates('task').set_index('task')['group']
N = d.drop_duplicates('task').set_index('task')['n'].astype(float).loc[tasks].values
gsz = G.loc[tasks].value_counts()
mu = np.array([1.0 / (len(gsz) * gsz[G[t]]) for t in tasks]); mu /= mu.sum()

DEPLOY = {'by question count': N / N.sum()}
for g in sorted(gsz.index):
    w = np.array([1.0 if G[t] == g else 0.0 for t in tasks])
    DEPLOY[f'the {g} group alone'] = w / w.sum()
hard = (L.mean(0) >= np.quantile(L.mean(0), .75)).astype(float)
DEPLOY['the hardest quartile of tasks'] = hard / hard.sum()

rank = lambda v: pd.Series(v).rank(method='min').astype(int).values
r_mu = rank(L @ mu)
r_q = rank(L @ DEPLOY['by question count'])
ranks = np.vstack([rank(L @ nu) for nu in DEPLOY.values()])

t = pd.DataFrame(dict(model=models, rank=r_mu, by_question=r_q,
                      best=ranks.min(0), worst=ranks.max(0))).sort_values('rank')
t.to_csv(R / 'llm_models.csv', index=False)

esc = lambda s: s.replace('_', r'\_').replace('&', r'\&')
half = (len(t) + 1) // 2
left, right = t.iloc[:half].values, t.iloc[half:].values
lines = []
for i in range(half):
    a = left[i]
    row = f'{a[1]} & \\texttt{{{esc(a[0])}}} & {a[2]} & {a[3]}--{a[4]}'
    if i < len(right):
        b = right[i]
        row += f' & {b[1]} & \\texttt{{{esc(b[0])}}} & {b[2]} & {b[3]}--{b[4]}'
    else:
        row += ' & & & & '
    lines.append(row + r' \\')
head = (r'\begin{tabular}{rl@{\hskip 4pt}rc@{\hskip 10pt}rl@{\hskip 4pt}rc}' '\n' r'\toprule' '\n'
        r'\textbf{\#} & \textbf{Model} & \textbf{q} & \textbf{range} & '
        r'\textbf{\#} & \textbf{Model} & \textbf{q} & \textbf{range} \\' '\n' r'\midrule')
(R / 'llm_models.tex').write_text(head + '\n' + '\n'.join(lines) + '\n'
                                  + r'\bottomrule' + '\n' + r'\end{tabular}' + '\n')

print(f'{len(t)} models written to llm_models.tex and llm_models.csv')
print(f'widest range: {(t.worst - t.best).max()} places '
      f'({t.loc[(t.worst - t.best).idxmax(), "model"]})')
print(f'models whose rank is the same under every deployment: {int((t.worst == t.best).sum())}')
print(t.head(8).to_string(index=False))
