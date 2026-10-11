"""u52_llm_two_readings.py -- two readings of one published table, and where they differ.

The Open LLM Leaderboard publishes, for every model, a raw score per task group and a
headline average.  The headline rescales each subtask against its random-guess baseline
and clips at zero before averaging the six groups; averaging the raw scores under the
same group weights does not.  Both numbers are the leaderboard's own, and the choice
between them is a choice of measure in the sense of equation (1): the headline weights a
subtask by roughly one over one minus its baseline.

This script measures how far that choice carries.  It writes
results/universal/llm_two_readings.csv.

Run u51_llm_published.py first; this script reuses the archived table it downloads.
"""
import warnings; warnings.filterwarnings('ignore')
import json, time, urllib.request, urllib.parse, itertools
import numpy as np, pandas as pd
from pathlib import Path
from scipy.stats import spearmanr, kendalltau

R = Path(__file__).resolve().parents[1] / 'results' / 'universal'
SNAP = R / 'llm_contents.csv'
GROUPS = ['BBH', 'GPQA', 'IFEval', 'MATH Lvl 5', 'MMLU-PRO', 'MUSR']
COLS = ['fullname', 'Average ⬆️'] + GROUPS + [g + ' Raw' for g in GROUPS]
API = ('https://datasets-server.huggingface.co/rows?dataset='
       + urllib.parse.quote('open-llm-leaderboard/contents', safe='')
       + '&config=default&split=train&offset={}&length=100')


def download():
    rows, off = [], 0
    while True:
        req = urllib.request.Request(API.format(off), headers={'User-Agent': 'py'})
        for attempt in range(6):                 # the endpoint rate-limits
            try:
                d = json.loads(urllib.request.urlopen(req, timeout=60).read().decode())
                break
            except Exception:
                if attempt == 5:
                    raise
                time.sleep(5 * (attempt + 1))
        batch = d.get('rows', [])
        rows += [{k: r['row'].get(k) for k in COLS} for r in batch]
        off += len(batch)
        if off >= d['num_rows_total'] or not batch:
            break
    t = pd.DataFrame(rows)
    t.to_csv(SNAP, index=False)
    return t


t = pd.read_csv(SNAP) if SNAP.exists() else download()
t = t.dropna(subset=COLS).drop_duplicates('fullname').reset_index(drop=True)

raw = t[[g + ' Raw' for g in GROUPS]].values          # the six raw group scores
stated = raw.mean(1)                                  # groups weighted equally
published = t['Average ⬆️'].values                    # the headline, as published

# What separates the two: the headline is affine in the raw score within each group,
# with a slope and an offset that do not depend on the model, so ordering by it is
# ordering by a reweighting of the same raw scores.
slope, fit = [], []
for g in GROUPS:
    x, y = t[g + ' Raw'].values, t[g].values
    ok = y > 1e-9                                     # scores clipped at zero carry no fit
    a = np.polyfit(x[ok], y[ok], 1)
    pred = np.polyval(a, x[ok])
    slope.append(a[0])
    fit.append(1 - ((y[ok] - pred) ** 2).sum() / ((y[ok] - y[ok].mean()) ** 2).sum())
slope = np.array(slope)
mu = np.ones(len(GROUPS)) / len(GROUPS)
nu = slope / slope.sum()
r = nu / mu
sd_r = float(np.sqrt(mu @ (r - 1) ** 2))

sp = spearmanr(stated, published)[0]
kt = kendalltau(stated, published)[0]

# How many of the leaderboard's own comparisons the two readings order oppositely,
# over every pair of archived models; pairs tied under either reading are skipped.
L = 1.0 - raw                                        # error, so lower is better
inv = n = 0
for i in range(len(t) - 1):
    da, db = stated[i] - stated[i + 1:], published[i] - published[i + 1:]
    ok = (da != 0) & (db != 0)
    n += int(ok.sum())
    inv += int((np.sign(da[ok]) != np.sign(db[ok])).sum())
rev = 100 * inv / n

# What that mismatch leaves undecided.  A pair is decided within a budget when its
# stability radius exceeds it (Proposition 3); otherwise either order is attainable,
# and a model's rank is an interval rather than a number.
risk = L @ mu
order = np.argsort(risk)
intervals = []
for pos in range(1, 21):
    f = order[pos - 1]
    dd = L[f] - L
    e = dd @ mu
    sd = np.sqrt(((dd - e[:, None]) ** 2) @ mu)
    decided = np.abs(e) > sd_r * sd
    ahead = e > 0
    intervals.append(dict(rank=pos, model=t.fullname[f],
                          best=1 + int((ahead & decided).sum()),
                          worst=int(ahead.sum()) + int(((~ahead) & (~decided)).sum())))
iv = pd.DataFrame(intervals)
iv.to_csv(R / 'llm_rank_intervals.csv', index=False)
top100 = order[:100]
opn = tot = 0
for i in range(len(top100)):
    dd = L[top100[i]] - L[top100[i + 1:]]
    e = dd @ mu
    sd = np.sqrt(((dd - e[:, None]) ** 2) @ mu)
    tot += len(e); opn += int((np.abs(e) <= sd_r * sd).sum())
open_top100 = 100 * opn / tot

# How far the crowding law of Theorem 2 reaches at this resolution.  Six group scores
# leave few directions, and a leaderboard fills with fine-tunes of one base whose gap
# and dispersion shrink together, so the median radius stops falling.
rng2 = np.random.default_rng(1)


def kstar(idx):
    f = idx[np.argmin(risk[idx])]
    dd = L[f] - L[idx]
    e = dd @ mu
    sd = np.sqrt(((dd - e[:, None]) ** 2) @ mu)
    m = (e < 0) & (sd > 0)
    return np.min(-e[m] / sd[m]) if m.any() else np.nan


Ks = [2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987, 1597, 2584, len(t)]
med = [float(np.nanmedian([kstar(rng2.choice(len(L), K, replace=False))
                           for _ in range(400 if K < 1000 else 120)])) for K in Ks]
pd.DataFrame(dict(K=Ks, kstar=med)).to_csv(R / 'llm_crowding_full.csv', index=False)
slope_full = float(np.polyfit(np.log(Ks), np.log(med), 1)[0])
f0 = int(np.argmin(risk))
dd = L[f0] - L
e = dd @ mu
sd = np.sqrt(((dd - e[:, None]) ** 2) @ mu)
m = (e < 0) & (sd > 0)
rat = np.sort(-e[m] / sd[m])
lo = rat[:max(10, int(.30 * len(rat)))]
tail = float(np.polyfit(np.log(lo), np.log(np.arange(1, len(lo) + 1) / len(rat)), 1)[0])

top_stated = [t.fullname[i] for i in np.argsort(-stated)[:5]]
top_pub = [t.fullname[i] for i in np.argsort(-published)[:5]]
swap = int(top_stated[:5] != top_pub[:5])

pd.DataFrame([dict(models=len(t), pairs=n, reversed_pct=rev, spearman=sp, kendall=kt,
                   sd_r=sd_r, reweight_fit=float(np.corrcoef(-(L @ nu), published)[0, 1]),
                   top5_differ=swap, open_top100_pct=open_top100,
                   rank19_best=int(iv.best.iloc[18]), rank19_worst=int(iv.worst.iloc[18]),
                   rank1_best=int(iv.best.iloc[0]), rank1_worst=int(iv.worst.iloc[0]),
                   kstar_K2=med[0], kstar_K34=med[Ks.index(34)], kstar_Kmax=med[-1],
                   crowd_slope_full=slope_full, crowd_slope_pred=-1 / tail,
                   top_stated='|'.join(top_stated), top_published='|'.join(top_pub))
              ]).to_csv(R / 'llm_two_readings.csv', index=False)

print(f'{len(t)} archived models, {n:,} comparisons')
print(f'the two readings order {rev:.1f} percent of them oppositely')
print(f'Spearman {sp:.4f}   Kendall {kt:.4f}')
print(f'implied weight per group {np.round(r, 3)}, dispersion sd_mu(r) = {sd_r:.2f}')
print(f'reweighting reproduces the headline ordering at r = '
      f'{np.corrcoef(-(L @ nu), published)[0, 1]:.4f}; per-group fit '
      f'{np.round(fit, 3)}')
print(f'\nat that mismatch {open_top100:.0f} percent of the pairs among the top hundred '
      f'models are open')
print(iv.to_string(index=False))
print(f'\ncrowding at this resolution: kappa* {med[0]:.2f} at K=2, {med[Ks.index(34)]:.2f} at '
      f'K=34, {med[-1]:.2f} at K={Ks[-1]}; slope {slope_full:.2f} against {-1 / tail:.2f} predicted')
print('\ntop five under the stated group weights:')
for m in top_stated:
    print('   ', m)
print('top five as published:')
for m in top_pub:
    print('   ', m)
