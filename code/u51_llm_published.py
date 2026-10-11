"""u51_llm_published.py -- our rebuilt aggregation against the published headline.

The benchmark measure of this paper is the leaderboard's stated rule, six task groups
weighted equally and subtasks equally within each, computed from the raw per-subtask
scores.  The published headline additionally rescales each subtask against its
random-guess baseline and clips at zero before averaging.  The two differ in scale, and
this script measures whether they differ in order.

It downloads the archived leaderboard table, keeps the models the analysis uses, and
reports the rank correlation between the two orderings.  Writes a snapshot of the
published averages to results/universal/llm_published.csv and the two correlations to
results/universal/llm_rank_agreement.csv, so the comparison can be rerun offline.
"""
import warnings; warnings.filterwarnings('ignore')
import json, time, urllib.request, urllib.parse
import numpy as np, pandas as pd
from pathlib import Path
from scipy.stats import spearmanr, kendalltau

R = Path(__file__).resolve().parents[1] / 'results' / 'universal'
SNAP = R / 'llm_published.csv'
API = ('https://datasets-server.huggingface.co/rows?dataset='
       + urllib.parse.quote('open-llm-leaderboard/contents', safe='')
       + '&config=default&split=train&offset={}&length=100')


def download():
    """The archived leaderboard table, one row per submission."""
    rows, off = [], 0
    while True:
        req = urllib.request.Request(API.format(off), headers={'User-Agent': 'py'})
        for attempt in range(6):          # the endpoint rate-limits; back off and retry
            try:
                d = json.loads(urllib.request.urlopen(req, timeout=60).read().decode())
                break
            except Exception as e:
                if attempt == 5:
                    raise
                time.sleep(5 * (attempt + 1))
        batch = d.get('rows', [])
        rows += [{'fullname': r['row']['fullname'], 'published': r['row']['Average ⬆️']}
                 for r in batch]
        off += len(batch)
        if off >= d['num_rows_total'] or not batch:
            break
    t = pd.DataFrame(rows).dropna().drop_duplicates('fullname')
    t.to_csv(SNAP, index=False)
    print(f'downloaded {len(t)} published averages')
    return t


pub = pd.read_csv(SNAP) if SNAP.exists() else download()

# our benchmark measure, rebuilt exactly as u34_llm_analysis.py does
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
G = d.drop_duplicates('task').set_index('task')['group']
gsz = G.loc[tasks].value_counts()
mu = np.array([1.0 / (len(gsz) * gsz[G[t]]) for t in tasks]); mu /= mu.sum()
ours = pd.DataFrame({'fullname': models, 'rebuilt': S.values @ mu})

m = ours.merge(pub, on='fullname', how='inner')
sp = spearmanr(m.rebuilt, m.published)[0]
kt = kendalltau(m.rebuilt, m.published)[0]
same = m.loc[m.rebuilt.idxmax(), 'fullname'] == m.loc[m.published.idxmax(), 'fullname']
pd.DataFrame([dict(matched=len(m), spearman=sp, kendall=kt, same_first=int(same),
                   first_rebuilt=m.loc[m.rebuilt.idxmax(), 'fullname'],
                   first_published=m.loc[m.published.idxmax(), 'fullname'])]
             ).to_csv(R / 'llm_rank_agreement.csv', index=False)

print(f'{len(models)} models analysed, {len(m)} matched to a published average')
print(f'Spearman {sp:.3f}   Kendall tau {kt:.3f}   same model first: {bool(same)}')
print(f'  rebuilt first   {m.loc[m.rebuilt.idxmax(), "fullname"]}')
print(f'  published first {m.loc[m.published.idxmax(), "fullname"]}')
