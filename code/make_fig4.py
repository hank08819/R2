"""Figure 4: what the wrong choice costs, and how far a benchmark settles.

Drawn in its own style, distinct from Figures 1 to 3: bold panel letters without
titles, no spines, a dashed grid on a white background; blue and red as elsewhere.

a, histogram of the regret ratio over the 498 selection problems, log-spaced bins,
   with the zero-regret share as its own bar (regret_ratio.csv).
b, survival curves of the regret ratio by quartile of the visible mismatch
   (regret_ratio.csv joined with visible_mismatch.csv), ninetieth percentile marked.
c, median ranking-stability radius against the number of candidates, with the fitted
   power law (crowding.csv).
d, slope chart from the predicted to the measured exponent, with resampled intervals
   (intervals.csv).

Same data and numbers as u6_figures.fig11, the Figure 4 of drafts up to V7.
"""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, NullFormatter
from pathlib import Path

_H = Path(__file__).resolve().parents[1]
R = next(p for p in (_H / 'optionJF' / 'results' / 'universal', _H / 'results' / 'universal') if p.exists())
PANEL, GRID, TXT = 'white', '0.82', '#222222'
TEAL, VERM = '#1E90FF', 'C3'                                   # bright blue and red
QCOL = ['#BFE3FF', '#7CC4FF', '#1E90FF', '#0047AB']          # quartiles, light to dark bright blue
plt.rcParams.update({'font.size': 9, 'axes.labelsize': 9, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
                     'legend.fontsize': 8, 'text.color': TXT, 'axes.labelcolor': TXT,
                     'xtick.color': TXT, 'ytick.color': TXT, 'pdf.fonttype': 42, 'ps.fonttype': 42})


def style(ax):
    ax.set_facecolor(PANEL)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.grid(True, color=GRID, lw=.6, ls=(0, (3, 3))); ax.set_axisbelow(True)
    ax.tick_params(length=0, pad=4)


def letter(ax, s):
    ax.text(-0.16, 1.06, s, transform=ax.transAxes, fontsize=12, fontweight='bold', va='bottom', ha='left')


# ------------------------------------------------------------------ data
s = pd.read_csv(R / 'regret_ratio.csv')
rr = 100 * s.rr.values
zero = 100 * (rr <= 0).mean(); pos = rr[rr > 0]
q90, q95 = np.percentile(rr, 90), np.percentile(rr, 95)

vm = pd.read_csv(R / 'visible_mismatch.csv')
sv = s.merge(vm[['dataset', 'seed', 'deployment', 'visible']], on=['dataset', 'seed', 'deployment'], how='inner')
sv['q'] = pd.qcut(sv.visible, 4, labels=False, duplicates='drop')
qstat = sv.groupby('q').agg(x=('visible', 'median'), m=('rr', 'mean'), p=('rr', lambda z: z.quantile(.9)))

c = pd.read_csv(R / 'crowding.csv')
NAME = {'regression suite': 'regression suite', 'language models': 'Open LLM Leaderboard'}
COL = {'regression suite': TEAL, 'language models': VERM}
PRED = {'regression suite': -0.88, 'language models': -0.83}
iv = pd.read_csv(R / 'intervals.csv').set_index('quantity')
fits = {k: np.polyfit(np.log(h.K), np.log(h.kstar), 1) for k, h in c.groupby('setting')}

# ------------------------------------------------------------------ figure
fig, ((a, b), (cc, d)) = plt.subplots(2, 2, figsize=(7.2, 5.9), gridspec_kw=dict(wspace=.36, hspace=.48))
for ax, L in zip((a, b, cc, d), 'abcd'):
    style(ax); letter(ax, L)

# a: histogram of the regret ratio, log-spaced bins, zero as its own bar
edges = np.geomspace(1e-2, 1e3, 21)
cnt, _ = np.histogram(pos, bins=edges)
a.bar(edges[:-1], 100 * cnt / len(rr), width=np.diff(edges), align='edge', color=TEAL, edgecolor=PANEL, lw=.6)
a.bar([2.2e-3], [zero], width=1.8e-3, color='0.55', edgecolor=PANEL, lw=.6)
a.text(5e-3, 57, f'{zero:.1f}% cost nothing', ha='left', va='center', fontsize=7.5, color='0.35')
for q, lab, side in [(q90, '90th', 'right'), (q95, '95th', 'left')]:
    a.axvline(q, color=VERM, lw=1.1, ls=(0, (4, 2)))
    a.text(q * (0.85 if side == 'right' else 1.18), 40, f'{lab}\n{q:.0f}%', color=VERM, fontsize=7.5, va='top', ha=side, linespacing=1.1)
a.set_xscale('log'); a.set_xlim(1.2e-3, 1.2e3); a.set_ylim(0, 65)
a.set_xticks([1e-2, 1e-1, 1, 10, 100, 1000]); a.xaxis.set_major_formatter(FormatStrFormatter('%g')); a.xaxis.set_minor_formatter(NullFormatter())
a.set_xlabel('regret ratio of the benchmark\'s choice (%)'); a.set_ylabel('problems (%)')

# b: survival curves by quartile of the visible mismatch
for qi, (_, row) in enumerate(qstat.iterrows()):
    z = np.sort(100 * sv.rr[sv.q == qi].values); sur = 100 * (1 - np.arange(len(z)) / len(z))
    keep = z > 0
    b.step(z[keep], sur[keep], where='post', color=QCOL[qi], lw=1.7, label=f'{row.x:.2f}')
    b.plot([1e-2, z[keep][0]], [sur[keep][0]] * 2, color=QCOL[qi], lw=1.7)
    b.plot(100 * row.p, 10, 'o', color=QCOL[qi], ms=5, mec='white', mew=.8, zorder=5)
b.axhline(10, color='0.6', lw=.8, ls=(0, (4, 2)))
b.text(1.3e-2, 11.5, '90th percentile', fontsize=7.5, color='0.35')
b.set_xscale('log'); b.set_xlim(1e-2, 6e2); b.set_ylim(0, 80)
b.set_xticks([1e-2, 1e-1, 1, 10, 100]); b.xaxis.set_major_formatter(FormatStrFormatter('%g')); b.xaxis.set_minor_formatter(NullFormatter())
b.legend(title='visible mismatch,\nquartile median', frameon=False, loc='upper right', title_fontsize=7.5, handlelength=1.4)
b.set_xlabel('regret ratio (%)'); b.set_ylabel('at least this costly (%)')

# c: radius against candidates, points and the fitted power law
for k, h in c.groupby('setting'):
    h = h.sort_values('K'); sl = fits[k]; xx = np.geomspace(h.K.min() * .85, h.K.max() * 1.18, 40)
    cc.plot(xx, np.exp(np.polyval(sl, np.log(xx))), color=COL[k], lw=1.6, alpha=.9, zorder=2)
    cc.scatter(h.K, h.kstar, s=34, color=COL[k], edgecolor='white', lw=.8, zorder=3, label=NAME[k])
    cc.text(xx[-1] * 1.12, np.exp(np.polyval(sl, np.log(xx[-1]))), f'$K^{{{sl[0]:.2f}}}$',
            color=COL[k], fontsize=8.5, ha='left', va='center')
cc.set_xscale('log'); cc.set_yscale('log'); cc.set_xlim(1.6, 120); cc.set_ylim(.02, 1.5)
cc.set_xticks([2, 5, 10, 20, 50, 100]); cc.xaxis.set_major_formatter(FormatStrFormatter('%g')); cc.xaxis.set_minor_formatter(NullFormatter())
cc.set_yticks([.03, .1, .3, 1]); cc.yaxis.set_major_formatter(FormatStrFormatter('%g')); cc.yaxis.set_minor_formatter(NullFormatter())
cc.legend(frameon=False, loc='upper right', handletextpad=.3)
cc.set_xlabel('$K$, candidates in the set'); cc.set_ylabel(r'median radius $\kappa^\star$')

# d: slope chart, predicted to measured, with resampled intervals
for k, dx, va, dy in (('regression suite', -0.05, 'bottom', .015), ('language models', 0.05, 'top', -.015)):
    lo, hi = iv.loc[f'crowding slope, {k}', 'lo'], iv.loc[f'crowding slope, {k}', 'hi']; m = fits[k][0]
    d.plot([0, 1 + dx], [PRED[k], m], color=COL[k], lw=1.8, zorder=2)
    d.plot([1 + dx, 1 + dx], [lo, hi], color=COL[k], lw=5, alpha=.25, solid_capstyle='butt', zorder=1)
    d.scatter([0, 1 + dx], [PRED[k], m], s=40, color=COL[k], edgecolor='white', lw=.8, zorder=4)
    d.text(-0.09, PRED[k], f'{PRED[k]:.2f}', ha='right', va='center', fontsize=7.5, color=COL[k])
    d.plot([], [], '-', color=COL[k], lw=1.8, label=NAME[k])
    d.text(1.12, m + dy, f'{m:.2f}  [{lo:.2f}, {hi:.2f}]', ha='left', va=va, fontsize=7.5, color=COL[k])
d.legend(frameon=False, loc='upper left', handlelength=1.4)
d.set_xlim(-0.5, 1.95); d.set_xticks([0, 1]); d.set_xticklabels(['predicted', 'measured'])
d.set_ylim(-1.15, -0.4); d.set_yticks([-1.1, -0.9, -0.7, -0.5])
d.set_ylabel(r'exponent of $\kappa^\star$ in $K$')

fig.savefig('fig_cost_crowding.pdf', bbox_inches='tight', pad_inches=.03)
fig.savefig('fig_cost_crowding.png', dpi=300, bbox_inches='tight', pad_inches=.03)
print(f'{len(rr)} problems; zero regret {zero:.1f}%; 90th {q90:.1f}%, 95th {q95:.1f}%')
print('quartiles (median visible, mean rr, 90th rr):', [(round(r.x, 2), round(100 * r.m, 1), round(100 * r.p, 1)) for _, r in qstat.iterrows()])
for k in fits:
    print(f'{NAME[k]}: exponent {fits[k][0]:.3f}, interval [{iv.loc[f"crowding slope, {k}", "lo"]:.2f}, {iv.loc[f"crowding slope, {k}", "hi"]:.2f}], predicted {PRED[k]}')
