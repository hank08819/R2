"""u64_llm_embed.py -- frozen LLM embeddings of the rows of the two real-shift cases.

Each row is written as a short English sentence and passed through a frozen decoder
(Qwen2.5-1.5B, no fine-tuning); the last hidden layer is mean-pooled over the tokens.
The embeddings are reduced to 256 principal components fitted on the benchmark rows
only (California 2018; Beijing 2010-2012), so no deployment row shapes the projection.
The learner that uses them is a ridge regression on these 256 columns (u58.learners_real).

ACS categories with a small code list are written as words; occupation, birthplace and
household relationship stay numeric codes, as in the census files.

Run on a GPU machine:  python u64_llm_embed.py embed <gpu> <ngpu>   (one per GPU)
                       python u64_llm_embed.py project
Reads data/real_rows/*.npz (u63), writes data/real_emb/<name>.npy (float16, n x 256).
"""
import sys, time, glob
import numpy as np
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
ROWS, RAW, OUT = ROOT / 'data' / 'real_rows', ROOT / 'data' / 'real_emb_raw', ROOT / 'data' / 'real_emb'
MODEL, K, BATCH = 'Qwen/Qwen2.5-1.5B', 256, 256

COW = {1: 'private for-profit employee', 2: 'private not-for-profit employee', 3: 'local government employee',
       4: 'state government employee', 5: 'federal government employee', 6: 'self-employed, not incorporated',
       7: 'self-employed, incorporated', 8: 'unpaid family worker', 9: 'unemployed'}
SCHL = {1: 'no schooling', 2: 'nursery school', 3: 'kindergarten', **{k: f'grade {k-3}' for k in range(4, 15)},
        15: 'grade 12, no diploma', 16: 'high school diploma', 17: 'GED', 18: 'some college, under one year',
        19: 'some college, no degree', 20: "associate's degree", 21: "bachelor's degree", 22: "master's degree",
        23: 'professional degree', 24: 'doctorate'}
MAR = {1: 'married', 2: 'widowed', 3: 'divorced', 4: 'separated', 5: 'never married'}
SEX = {1: 'male', 2: 'female'}
RACE = {1: 'White', 2: 'Black', 3: 'American Indian', 4: 'Alaska Native', 5: 'American Indian or Alaska Native',
        6: 'Asian', 7: 'Native Hawaiian or Pacific Islander', 8: 'some other race', 9: 'two or more races'}


def acs_text(X):
    # FEAT = AGEP COW SCHL MAR OCCP POBP RELP WKHP SEX RAC1P
    g = lambda m, v: m.get(int(v), f'code {int(v)}')
    return [f'Person, age {int(a)}, {g(SEX, s)}, {g(RACE, r)}, {g(MAR, m)}. Education: {g(SCHL, e)}. '
            f'Works as a {g(COW, c)}, {int(h)} hours a week, occupation code {int(o)}. '
            f'Birthplace code {int(p)}; household relationship code {int(rl)}.'
            for a, c, e, m, o, p, rl, h, s, r in X]


def beijing_text(X, cols):
    wd = {'wd_NE': 'northeast', 'wd_NW': 'northwest', 'wd_SE': 'southeast', 'wd_cv': 'calm and variable'}
    out = []
    for row in X:
        v = dict(zip(cols, row)); w = next((wd[c] for c in wd if v.get(c, 0) == 1), 'unknown')
        out.append(f'Beijing, month {int(v["month"])}, hour {int(v["hour"])}. Dew point {v["DEWP"]:.0f} C, '
                   f'temperature {v["TEMP"]:.0f} C, pressure {v["PRES"]:.0f} hPa. Wind {w}, '
                   f'cumulated speed {v["Iws"]:.1f} m/s. Hours of snow {int(v["Is"])}, hours of rain {int(v["Ir"])}.')
    return out


def texts(f):
    z = np.load(f, allow_pickle=True)
    return beijing_text(z['X'], list(z['cols'])) if 'cols' in z else acs_text(z['X'])


def embed(gpu, ngpu):
    import torch
    from transformers import AutoTokenizer, AutoModel
    RAW.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(MODEL); tok.padding_side = 'right'
    net = AutoModel.from_pretrained(MODEL, torch_dtype=torch.bfloat16).to(f'cuda:{gpu}').eval()
    files = sorted(glob.glob(str(ROWS / '*.npz')))[gpu::ngpu]
    t0 = time.time()
    for f in files:
        name = Path(f).stem; out = RAW / f'{name}.npy'
        if out.exists(): continue
        T = texts(f); E = np.empty((len(T), net.config.hidden_size), np.float16)
        with torch.no_grad():
            for i in range(0, len(T), BATCH):
                b = tok(T[i:i + BATCH], return_tensors='pt', padding=True).to(f'cuda:{gpu}')
                h = net(**b).last_hidden_state.float(); m = b['attention_mask'].unsqueeze(-1).float()
                E[i:i + BATCH] = ((h * m).sum(1) / m.sum(1)).cpu().numpy().astype(np.float16)
        np.save(out, E); print(f'gpu {gpu} {name}: {len(T):,} rows, {time.time()-t0:.0f}s', flush=True)


def project():
    OUT.mkdir(parents=True, exist_ok=True)
    for case, bench in (('acs', lambda: np.load(RAW / 'acs_CA_2018.npy')),
                        ('beijing', lambda: np.load(RAW / 'beijing.npy')[np.load(ROWS / 'beijing.npz')['year'] <= 2012])):
        B = bench().astype(np.float32)
        rng = np.random.default_rng(0); B = B[rng.choice(len(B), min(len(B), 60_000), replace=False)]
        mu = B.mean(0); _, s, Vt = np.linalg.svd(B - mu, full_matrices=False); V = Vt[:K].T
        print(f'{case}: {K} components keep {100*(s[:K]**2).sum()/(s**2).sum():.1f}% of the benchmark variance')
        for f in sorted(RAW.glob(f'{case}*.npy')):
            np.save(OUT / f.name, ((np.load(f).astype(np.float32) - mu) @ V).astype(np.float16))


if __name__ == '__main__':
    if sys.argv[1] == 'embed': embed(int(sys.argv[2]), int(sys.argv[3]))
    else: project()
