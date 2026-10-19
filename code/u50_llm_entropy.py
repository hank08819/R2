"""u50_llm_entropy.py -- how many directions the leaderboard's candidate set exposes.

Data entropy reads a spectrum as an effective number of directions.  Applied to the
matrix whose row space is H_P (Proposition 1), it says how much of the 38-dimensional space
the leaderboard actually uses, which direction carries most of it, and how far the
eight deployment measures lie from that direction.

Filtering follows u34_llm_analysis.py exactly: group-level aggregates dropped, the
subtasks scored for at least eighty percent of models kept, then the models scored on
every one of those.  Writes results/universal/llm_entropy.csv, one row per quantity.
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
L = 1.0 - S.values                       # per-task error, the per-instance score
G = d.drop_duplicates('task').set_index('task')['group']
N = d.drop_duplicates('task').set_index('task')['n'].astype(float).loc[tasks].values

# The matrix whose row space is H_P in the mu inner product (Proposition 1): losses
# centred over subtasks under the leaderboard's own measure mu, then across models,
# scaled by sqrt(mu).  No per-subtask standardisation, so the spectrum is that of H_P.
gsz = G.loc[tasks].value_counts()
mu = np.array([1.0 / (len(gsz) * gsz[G[t]]) for t in tasks]); mu /= mu.sum()
Lt = L - (L @ mu)[:, None]
D = (Lt - Lt.mean(0, keepdims=True)) * np.sqrt(mu)
_, s, Vt = np.linalg.svd(D, full_matrices=False)
s = s[s > 1e-10]
u = s / s.sum()
h = float(-(u * np.log2(u)).sum())
energy = s ** 2 / (s ** 2).sum()

# the eight deployments of u34
DEPLOY = {'by question count': N / N.sum()}
for g in sorted(gsz.index):
    w = np.array([1.0 if G[t] == g else 0.0 for t in tasks])
    DEPLOY[f'the {g} group alone'] = w / w.sum()
hard = (L.mean(0) >= np.quantile(L.mean(0), .75)).astype(float)
DEPLOY['the hardest quartile of tasks'] = hard / hard.sum()

v1 = Vt[0]
cos = {k: abs(float((((nu / mu - 1) * np.sqrt(mu)) @ v1) / (np.linalg.norm((nu / mu - 1) * np.sqrt(mu)) * np.linalg.norm(v1))))
       for k, nu in DEPLOY.items()}

# Who wins, and what the certificate settles, under the same eight deployments.
import itertools
winners = {k: models[int(np.argmin(L @ nu))] for k, nu in DEPLOY.items()}
cert = certrev = exposed = exprev = 0
for k, nu in DEPLOY.items():
    rr = nu / mu
    sd_rk = float(np.sqrt(mu @ (rr - 1) ** 2))
    for i, j in itertools.combinations(range(len(models)), 2):
        dd = L[i] - L[j]
        e = mu @ dd
        sd = np.sqrt(mu @ (dd - e) ** 2)
        if e == 0 or sd == 0:
            continue
        rev = int(np.sign(e) != np.sign(nu @ dd))
        if abs(e) / sd >= sd_rk:
            cert += 1; certrev += rev
        else:
            exposed += 1; exprev += rev

rows = [dict(quantity='distinct_winners', value=len(set(winners.values()))),
        dict(quantity='certified_pairs', value=cert),
        dict(quantity='certified_reversals', value=certrev),
        dict(quantity='exposed_pairs', value=exposed),
        dict(quantity='exposed_reversal_pct', value=100 * exprev / max(exposed, 1)),
        dict(quantity='models', value=len(models)),
        dict(quantity='subtasks', value=len(tasks)),
        dict(quantity='h_bits', value=h),
        dict(quantity='effective_directions', value=2 ** h),
        dict(quantity='leading_energy_pct', value=100 * energy[0]),
        dict(quantity='first_five_energy_pct', value=100 * energy[:5].sum()),
        dict(quantity='h_over_log2q', value=h / np.log2(len(s))),
        dict(quantity='max_cos_deployment', value=max(cos.values()))]
for g in sorted(gsz.index):
    m = np.array([G[t] == g for t in tasks])
    rows.append(dict(quantity=f'loading_{g}', value=float((v1[m] ** 2).sum())))
    rows.append(dict(quantity=f'loading_per_subtask_{g}', value=float((v1[m] ** 2).sum() / m.sum())))
pd.DataFrame(rows).to_csv(R / 'llm_entropy.csv', index=False)

print(f'{len(models)} models on {len(tasks)} subtasks')
print(f'h(D) = {h:.2f} bits, 2^h = {2 ** h:.1f} of {len(s)}, h/log2 q = {h / np.log2(len(s)):.2f}')
print(f'leading direction {100 * energy[0]:.1f} per cent, first five {100 * energy[:5].sum():.1f}')
for g in sorted(gsz.index):
    m = np.array([G[t] == g for t in tasks])
    print(f'  {g:9s} {m.sum():2d} subtasks, squared loading {(v1[m] ** 2).sum():.2f}')
print('largest |cos| between a deployment and the leading direction: '
      f'{max(cos.values()):.2f} ({max(cos, key=cos.get)})')
print(f'\neight deployments crown {len(set(winners.values()))} distinct models:')
for k, v in winners.items():
    print(f'   {k:32s} {v}')
print(f'certificate: {cert:,} comparisons certified, {certrev} reverse; '
      f'{exposed:,} exposed, {100 * exprev / max(exposed, 1):.1f} percent reverse')
