"""Figure 1: two benchmarks, reordered the same way, from the stored result files.

a, the twelve leading models of the Open LLM Leaderboard and how many places each
   moves when the published per-subtask scores are weighted by question count
   instead of by task group (llm_raw.csv, as in u6_figures.fig10).
b, the share of the 2,926 model pairs each deployment reverses (llm_pairs.csv).
c, the share of pairs reversed on each of the 35 OpenML-CTR23 datasets (scale_r2.csv).
d, the reversal rate by decile of the mismatch sd_mu(r) (scale_r2.csv).

The panel designs follow the editorial redesign of 2026-10-17 (FIGURE1_UPDATE/);
the data, type size and colours follow the other figures of the paper.
"""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, NullFormatter
from pathlib import Path

_H = Path(__file__).resolve().parents[1]
R = next(p for p in (_H / 'optionJF' / 'results' / 'universal', _H / 'results' / 'universal') if p.exists())
plt.rcParams.update({'font.size': 9, 'axes.titlesize': 9, 'axes.labelsize': 9,
                     'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 8,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'pdf.fonttype': 42, 'ps.fonttype': 42})
INK, RED, GREY, LIGHT, FAINT = 'C0', 'C3', 'C7', '#7FB3D5', '#E6E6E6'

# ------------------------------------------------------------------ data
d = pd.read_csv(R / 'llm_raw.csv')
AGG = {'leaderboard_bbh', 'leaderboard_gpqa', 'leaderboard_math_hard', 'leaderboard_musr', 'leaderboard'}
d = d[~d.task.isin(AGG)]


def grp(t):
    for g in ['bbh', 'math', 'gpqa', 'musr', 'mmlu_pro', 'ifeval']:
        if t.startswith('leaderboard_' + g):
            return g


d['group'] = d.task.map(grp); d = d[d.group.notna()]
W = d.pivot(index='model', columns='task', values='score')
tasks = [t for t in W.columns if W[t].notna().sum() >= .8 * len(W)]
W = W[tasks].dropna(axis=0)
G = d.drop_duplicates('task').set_index('task')['group']
gsz = G.loc[tasks].value_counts()
mu = np.array([1. / (len(gsz) * gsz[G[t]]) for t in tasks]); mu /= mu.sum()
n = d.drop_duplicates('task').set_index('task')['n'].loc[tasks].values.astype(float)
lead = pd.Series(W.values @ mu, index=W.index)
byq = pd.Series(W.values @ (n / n.sum()), index=W.index)
rl = lead.rank(ascending=False).astype(int); rq = byq.rank(ascending=False).astype(int)
top = rl.sort_values().index[:12]
NMODEL = len(W)

SHORT = {'allenai/Llama-3.1-Tulu-3-70B-DPO': 'Tulu-3-70B-DPO', 'allenai/Llama-3.1-Tulu-3-70B': 'Tulu-3-70B',
         'upstage/solar-pro-preview-instruct': 'Solar-Pro-Preview', 'allenai/Llama-3.1-Tulu-3-70B-SFT': 'Tulu-3-70B-SFT',
         'deepseek-ai/DeepSeek-R1-Distill-Qwen-14B': 'DeepSeek-R1-Distill-14B', 'tiiuae/Falcon3-10B-Instruct': 'Falcon3-10B-Instruct',
         'HuggingFaceH4/zephyr-orpo-141b-A35b-v0.1': 'Zephyr-ORPO-141B', 'Qwen/QwQ-32B-Preview': 'QwQ-32B-Preview',
         '01-ai/Yi-1.5-34B-Chat': 'Yi-1.5-34B-Chat', 'microsoft/Phi-3-medium-128k-instruct': 'Phi-3-medium-128k',
         'NousResearch/DeepHermes-3-Mistral-24B-Preview': 'DeepHermes-3-24B', 'CohereForAI/aya-expanse-32b': 'Aya-Expanse-32B'}

t = pd.read_csv(R / 'llm_pairs.csv')
g = t.groupby('deployment').agg(inv=('inverted', lambda x: 100 * x.mean())).sort_values('inv', ascending=False)
NICE = {'musr': 'MuSR', 'gpqa': 'GPQA', 'ifeval': 'IFEval', 'bbh': 'BBH', 'math': 'MATH', 'mmlu_pro': 'MMLU-Pro'}
blab = [NICE.get(x.replace('the ', '').replace(' group alone', ''), x.replace('the ', '').replace(' of tasks', ''))
        for x in g.index]
NPAIR = len(t) // t.deployment.nunique()

sc = pd.read_csv(R / 'scale_r2.csv')
per = sc.groupby('dataset').inverted.mean().mul(100).sort_values(ascending=False)
pooled = 100 * sc.inverted.mean()
q = pd.qcut(sc.sd_r, 10, duplicates='drop')
bn = sc.groupby(q, observed=True).agg(x=('sd_r', 'median'), y=('inverted', 'mean'))

# ------------------------------------------------------------------ figure
fig, ((a, b), (c, dd)) = plt.subplots(2, 2, figsize=(7.2, 5.9),
                                      gridspec_kw=dict(width_ratios=[1.85, 1], height_ratios=[1.08, 1],
                                                       wspace=.5, hspace=.50))

# a: rank changes, one row per leading model
L, RT, Z, SC = 0, 100, 64, 2.0             # table edges, the zero of the rank-change column, units per place
a.axis('off'); a.set_xlim(L, RT); a.set_ylim(-1.0, 13.6)
hdr = dict(fontsize=7, color='0.35', va='bottom')
a.text(L + .2, 12.75, 'rank   model', ha='left', **hdr)
a.text(Z, 12.75, 'places moved', ha='center', **hdr)
a.text(RT - .2, 12.75, 'before → after', ha='right', **hdr)
a.plot([L + .1, RT - .1], [12.55, 12.55], color='0.75', lw=.7)
a.plot([Z, Z], [-.2, 12.3], color='0.7', lw=.7, ls=(0, (3, 3)), zorder=1)
for i, m in enumerate(top, start=1):
    y = 12.5 - i
    before, after = rl[m], rq[m]; v = after - before
    big = abs(v) >= 3
    a.plot([L + .1, RT - .1], [y - .5, y - .5], color=FAINT, lw=.4, zorder=0)
    a.text(L + .2, y, f'{before:2d}', ha='left', va='center', fontsize=7.2, color='0.35')
    a.text(L + 5.5, y, SHORT.get(m, m.split('/')[-1]), ha='left', va='center', fontsize=7.4,
           color=RED if big else 'black', fontweight='bold' if big else 'normal')
    col = INK if v < 0 else (RED if v > 0 else GREY)
    a.plot([Z, Z + SC * v], [y, y], color=col, lw=2.0 if big else 1.4, solid_capstyle='round', zorder=2)
    a.plot([Z + SC * v], [y], 'o', ms=4.6 if big else 3.6, color=col, mec='white', mew=.6, zorder=3)
    if v:
        a.text(Z + SC * v + (-.9 if v < 0 else .9), y, f'{v:+d}', ha='right' if v < 0 else 'left', va='center',
               fontsize=7.2, color=col, fontweight='bold' if big else 'normal')
    else:
        a.text(Z + .9, y, '0', ha='left', va='center', fontsize=7.2, color=GREY)
    a.text(RT - .2, y, f'{before} → {after}', ha='right', va='center', fontsize=7.5,
           color=RED if big else '0.35', fontweight='bold' if big else 'normal')
a.text(Z - 1, -.55, '← moves up', ha='right', va='center', fontsize=7.5, color=INK)
a.text(Z + 1, -.55, 'moves down →', ha='left', va='center', fontsize=7.5, color=RED)
a.set_title(f'(a) the leading 12 of {NMODEL} models, questions equal', loc='left', fontsize=8.6)

# b: pairs each deployment reverses
yb = np.arange(len(g))
cols = [LIGHT if blab[i] == 'by question count' else INK for i in range(len(g))]   # light: the weighting of panel a
b.barh(yb, g.inv.values, color=cols, height=.62)
for yy, v in zip(yb, g.inv.values):
    b.text(v + .5, yy, f'{v:.1f}', va='center', fontsize=7.5)
b.invert_yaxis(); b.set_yticks(yb); b.set_yticklabels(blab, fontsize=8)
b.tick_params(axis='y', length=0); b.spines['left'].set_visible(False)
b.set_xlim(0, 32); b.set_xticks([0, 10, 20, 30])
b.set_xlabel('pairs ordered opposite to the deployment (%)', fontsize=8)
b.set_title(f'(b) pairs each deployment reverses, of {NPAIR:,}', loc='left', fontsize=8.6)

# c: every dataset affected
yc = np.arange(len(per))
c.barh(yc, per.values, color=INK, height=.76)
c.invert_yaxis(); c.set_yticks([]); c.set_ylim(len(per) - .3, -1.6)
c.spines['left'].set_visible(False)
c.axvline(pooled, color=RED, lw=1.1, ls='--')
c.text(pooled + .7, len(per) - 1.6, f'all comparisons {pooled:.1f}%', color=RED, fontsize=7.5, va='center')
c.set_xlim(0, 34); c.set_xticks([0, 10, 20, 30])
c.set_xlabel('pairs whose ordering reverses (%)'); c.set_ylabel(f'{len(per)} datasets', labelpad=2)
c.set_title('(c) every dataset affected', loc='left', fontsize=8.6)

# d: the rate rises with mismatch
dd.plot(bn.x, 100 * bn.y, 'o-', color=INK, ms=4.2, mfc='white', mew=1.1, lw=1.4)
dd.set_xscale('log'); dd.set_xticks([.3, .5, 1, 2, 3, 5])
dd.xaxis.set_major_formatter(FormatStrFormatter('%g')); dd.xaxis.set_minor_formatter(NullFormatter())
dd.set_xlim(.24, 5.6); dd.set_ylim(0, 27); dd.set_yticks([0, 5, 10, 15, 20, 25])
for yy in (5, 10, 15, 20, 25):
    dd.axhline(yy, color=FAINT, lw=.5, zorder=0)
top_y = 100 * bn.y.iloc[-1]
dd.text(bn.x.iloc[-1], top_y + 1.3, f'{top_y:.1f}%', ha='center', fontsize=7.5, color=INK)
dd.set_xlabel(r'mismatch $\mathrm{sd}_\mu(r)$, median of each decile'); dd.set_ylabel('reversal rate (%)')
dd.set_title('(d) the rate rises with mismatch', loc='left', fontsize=8.6)

fig.savefig('fig_two_benchmarks.pdf', bbox_inches='tight', pad_inches=.03)
fig.savefig('fig_two_benchmarks.png', dpi=300, bbox_inches='tight', pad_inches=.03)
print(f'{NMODEL} models; {NPAIR} pairs; {len(per)} datasets, pooled {pooled:.1f}%; {len(sc)} comparisons')
for m in top:
    print(f'  {rl[m]:2d} -> {rq[m]:2d}  {m}')
