"""u71_fetch_leaderboards.py -- per-task scores for five further leaderboards.

Rules fixed in data/leaderboards/protocol.md before any analysis.  Writes one raw file per
board, data/leaderboards/<board>/raw.csv, with the columns of llm_raw.csv plus the group:
model, task, score (0-1, higher is better), n (items, where known), group.

    python u71_fetch_leaderboards.py medical | finance | asr | livebench | mteb
"""
import json, sys, time, urllib.request, urllib.parse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import pandas as pd

D = Path(__file__).resolve().parents[1] / 'data' / 'leaderboards'
UA = {'User-Agent': 'py'}


def get(u, t=60, tries=4):
    for k in range(tries):
        try:
            return urllib.request.urlopen(urllib.request.Request(u, headers=UA), timeout=t).read()
        except Exception:
            if k == tries - 1:
                raise
            time.sleep(2 * (k + 1))


def save(board, rows):
    t = pd.DataFrame(rows); t.to_csv(D / board / 'raw.csv', index=False)
    print(f'{board}: {t.model.nunique()} models, {t.task.nunique()} tasks, {len(t)} rows')


# ------------------------------------------------------------ Open Medical-LLM Leaderboard
MED_TREE = 'https://huggingface.co/api/datasets/openlifescienceai/results/tree/main/'
MED_RAW = 'https://huggingface.co/datasets/openlifescienceai/results/resolve/main/'
MED = {'medmcqa': ('medmcqa', 4183), 'medqa_4options': ('medqa', 1273), 'pubmedqa': ('pubmedqa', 500)}
MMLU_MED = ['anatomy', 'clinical_knowledge', 'college_biology', 'college_medicine',
            'medical_genetics', 'professional_medicine']


def med_model(mp):
    try:
        return _med_model(mp)
    except Exception as e:
        print(f'  {mp}: {type(e).__name__}, not fetched', flush=True)
        return mp, None


def _med_model(mp):
    files = sorted(x['path'] for x in json.loads(get(MED_TREE + urllib.parse.quote(mp)))
                   if x.get('type') == 'file' and x['path'].endswith('.json'))
    if not files:
        return mp, None
    d = json.loads(get(MED_RAW + urllib.parse.quote(files[-1])))       # the latest result file
    return mp, {k: v.get('acc,none') for k, v in d.get('results', {}).items()}


def medical():
    (D / 'medical').mkdir(exist_ok=True)
    sizes = json.loads(get('https://datasets-server.huggingface.co/size?dataset=cais/mmlu'))
    nmmlu = {s['config']: s['num_rows'] for s in sizes['size']['splits'] if s['split'] == 'test'}
    n = dict({k: v[1] for k, v in MED.items()}, **{f'mmlu_{s}': nmmlu[s] for s in MMLU_MED})
    grp = dict({k: v[0] for k, v in MED.items()}, **{f'mmlu_{s}': 'mmlu_medical' for s in MMLU_MED})
    orgs = [x['path'] for x in json.loads(get(MED_TREE)) if x.get('type') == 'directory']
    models = []
    for o in orgs:
        try:
            items = json.loads(get(MED_TREE + urllib.parse.quote(o)))
            if any(x.get('type') == 'file' and x['path'].endswith('.json') for x in items):
                models.append(o)                      # results kept at the top of the folder
            models += [x['path'] for x in items if x.get('type') == 'directory'
                       and not x['path'].split('/')[-1].startswith('.')]   # no hidden folders
        except Exception as e:
            print(f'  {o}: {type(e).__name__}', flush=True)
    with ThreadPoolExecutor(8) as ex:
        got = list(ex.map(med_model, models))
    rows, skipped = [], 0
    for mp, r in got:
        if not r or any(not isinstance(r.get(t), (int, float)) for t in n):
            skipped += 1; print(f'  {mp}: missing {[t for t in n if not r or not isinstance(r.get(t), (int, float))]}'); continue
        rows += [dict(model=mp, task=t, score=float(r[t]), n=n[t], group=grp[t]) for t in n]
    print(f'  {len(models)} model folders, {skipped} without all nine tasks')
    save('medical', rows)


# ------------------------------------------------- Open FinLLM Leaderboard, MultiFinBen text
def finance():
    d = json.loads((D / 'finance' / 'scores.json').read_text())
    ds = [x for x in d['datasets'] if x['modality'] == 'Text']
    rows, skipped = [], 0
    for m in d['models']:
        if 'Text' not in m['caps']:                  # vision- and audio-only models score 0 here
            continue
        s = m['scores']
        if any(not isinstance(s.get(x['key']), (int, float)) for x in ds):
            skipped += 1; continue
        rows += [dict(model=m['name'], task=x['key'], score=s[x['key']] / 100, n=None,
                      group=x['task'] if x['task'] in ('IE', 'QA', 'TA', 'TG') else 'other',
                      lang=x['lang']) for x in ds]
    print(f'  {skipped} models without all {len(ds)} text scores')
    save('finance', rows)


# ------------------------------------------------------------- Open ASR Leaderboard, English
ASR_SETS = ['AMI-Cleaned', 'Earnings22-Cleaned-AA-chunked', 'Gigaspeech-Cleaned', 'LS Clean',
            'LS Other', 'SPGISpeech', 'URGENT2024', 'Voice Arena Monsoon', 'Voxpopuli-AA-Cleaned']


def asr():
    t = pd.read_csv(D / 'open_asr' / 'english_short_latest.csv')
    cols = [f'{s} WER' for s in ASR_SETS]
    t = t.dropna(subset=cols)
    rows = [dict(model=r.model, task=s, score=1 - r[f'{s} WER'] / 100, n=None, group=s)
            for _, r in t.iterrows() for s in ASR_SETS]
    save('open_asr', rows)


# ------------------------------------------------------------------- LiveBench 2025-11-25
def livebench():
    t = pd.read_csv(D / 'livebench' / 'table_2025_11_25.csv')
    cat = json.loads((D / 'livebench' / 'categories_2025_11_25.json').read_text())
    g = {task: c for c, ts in cat.items() for task in ts}
    tasks = [c for c in t.columns if c != 'model' and c in g]
    t = t.dropna(subset=tasks)
    rows = [dict(model=r.model, task=k, score=r[k] / 100, n=None, group=g[k])
            for _, r in t.iterrows() for k in tasks]
    save('livebench', rows)


if __name__ == '__main__':
    {'medical': medical, 'finance': finance, 'asr': asr, 'livebench': livebench}[sys.argv[1]]()
