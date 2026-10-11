"""u33_llm_fetch.py -- per-subtask scores for a spread of models on a public leaderboard.

The Open LLM Leaderboard reports one number per model, the mean of six task-group
scores.  That mean is a measure over tasks, and it is the benchmark measure of
Theorem 1.  A user who cares about one group, or about the tasks in proportion to
their size, is deploying under a different one.  This collects the per-subtask
scores needed to compare the two.
"""
import json, time, urllib.request, urllib.parse
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / 'results' / 'universal'
BASE = 'https://huggingface.co/api/datasets/open-llm-leaderboard/results/tree/main/'
RAW = 'https://huggingface.co/datasets/open-llm-leaderboard/results/resolve/main/'
ORGS = ['meta-llama', 'mistralai', 'Qwen', 'google', 'microsoft', 'HuggingFaceH4',
        'tiiuae', '01-ai', 'allenai', 'databricks', 'CohereForAI', 'stabilityai',
        'EleutherAI', 'bigscience', 'upstage', 'NousResearch', 'teknium',
        'openchat', 'WizardLMTeam', 'deepseek-ai']
MAXPER = 4

def api(u, t=40):
    r = urllib.request.urlopen(urllib.request.Request(u, headers={'User-Agent': 'py'}),
                               timeout=t)
    return json.loads(r.read().decode())

def score(d):
    """The subtask's headline number, whatever the task calls it."""
    for k in ('acc_norm,none', 'acc,none', 'exact_match,none',
              'prompt_level_strict_acc,none'):
        if k in d and isinstance(d[k], (int, float)):
            return float(d[k])
    return None

rows, seen = [], 0
for org in ORGS:
    try:
        models = [x['path'] for x in api(BASE + org)
                  if x.get('type') == 'directory']
    except Exception as e:
        print(f'  {org}: {type(e).__name__}', flush=True); continue
    for mp in models[:MAXPER]:
        try:
            files = [x['path'] for x in api(BASE + urllib.parse.quote(mp))
                     if x['path'].endswith('.json')]
            if not files:
                continue
            f = sorted(files)[-1]                       # the most recent run
            d = json.loads(urllib.request.urlopen(urllib.request.Request(
                RAW + urllib.parse.quote(f), headers={'User-Agent': 'py'}), timeout=40).read())
            ns = d.get('n-samples', {})
            got = 0
            for task, v in d.get('results', {}).items():
                s = score(v)
                if s is None or task in ('leaderboard',):
                    continue
                n = ns.get(task, {}).get('effective')
                if not n:
                    continue
                rows.append(dict(model=mp, task=task, score=s, n=int(n)))
                got += 1
            seen += 1
            print(f'  {mp:<46} {got} subtasks', flush=True)
        except Exception as e:
            print(f'  {mp}: {type(e).__name__}', flush=True)
        time.sleep(.3)

import pandas as pd
t = pd.DataFrame(rows)
t.to_csv(OUT / 'llm_raw.csv', index=False)
print(f'\n{seen} models, {t.model.nunique()} with results, '
      f'{t.task.nunique()} distinct subtasks, {len(t)} rows')
