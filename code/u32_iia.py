"""u32_iia.py -- the worst-case rule's disagreements come from irrelevant models.

Ranking by a criterion assigns each model a number, so dropping some other model
cannot change which of the rest is chosen.  Minimax regret over a budget ranks a
model by its worst comparison, which is a property of the whole set, so dropping a
model that was never in contention can change the answer.  This counts how often
that happens.
"""
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from pathlib import Path

R = Path(__file__).resolve().parents[1] / 'results' / 'universal'
d = pd.read_csv(R / 'scale_r2.csv')
d['left'] = d.pair.str.split('-').str[0]; d['right'] = d.pair.str.split('-').str[1]

rows = []
for (ds, se, dep), h in d.groupby(['dataset', 'seed', 'deployment']):
    ms = sorted(set(h.left) | set(h.right)); ix = {x: i for i, x in enumerate(ms)}
    K = len(ms); E = np.zeros((K, K)); S = np.zeros((K, K))
    for _, x in h.iterrows():
        i, j = ix[x.left], ix[x.right]
        E[i, j], E[j, i] = x.e_mu, -x.e_mu
        S[i, j] = S[j, i] = x.sd_d
    k = float(h.sd_r.iloc[0])
    W = lambda idx: (E[np.ix_(idx, idx)] + k * S[np.ix_(idx, idx)]).max(1)
    full = list(range(K))
    w_full = full[int(np.argmin(W(full)))]
    r_full = full[int(np.argmin((E + 0 * S).max(1)))]      # the criterion's choice
    # Each rule is tested by dropping only models it did not itself choose:
    # removing the winner is not an independence violation.
    flips_w = drops_w = flips_r = drops_r = 0
    for y in range(K):
        sub = [i for i in full if i != y]
        if y != w_full:
            drops_w += 1
            if sub[int(np.argmin(W(sub)))] != w_full:
                flips_w += 1
        if y != r_full:
            drops_r += 1
            Es = E[np.ix_(sub, sub)]
            if sub[int(np.argmin(Es.max(1)))] != r_full:
                flips_r += 1
    rows.append(dict(dataset=ds, seed=se, deployment=dep,
                     drops_w=drops_w, drops_r=drops_r,
                     flips_w=flips_w, flips_r=flips_r,
                     any_w=int(flips_w > 0), any_r=int(flips_r > 0)))
s = pd.DataFrame(rows); s.to_csv(R / 'iia.csv', index=False)
print(f'{len(s)} selection problems, dropping each model that was not chosen\n')
print(f'  minimax regret over a budget: the choice changes in '
      f'{100*s.flips_w.sum()/s.drops_w.sum():.1f}% of drops, and at least once in '
      f'{100*s.any_w.mean():.1f}% of problems')
print(f'  the criterion itself:         {100*s.flips_r.sum()/s.drops_r.sum():.1f}% '
      f'of drops, {100*s.any_r.mean():.1f}% of problems  (must be 0)')
