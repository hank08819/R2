"""Figure 3: four public leaderboards and two real shifts, from the stored result files.

a, on each board, the deployments under which the leader stays best (u72).
b, the leader's ranking-stability radius against the mildest deployment's mismatch (u72).
c, on the two real shifts, the share of comparisons reversed by the ratio of radius
   to mismatch (u59, u61).
d, after deployment, the share of right choices the labeled spot-check flags, by its
   sample size (u60, u62).
"""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

_H = Path(__file__).resolve().parents[1]
R = next(p for p in (_H / 'optionJF' / 'results' / 'universal', _H / 'results' / 'universal') if p.exists())
plt.rcParams.update({'font.size': 9, 'axes.titlesize': 9, 'axes.labelsize': 9,
                     'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 8,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'pdf.fonttype': 42, 'ps.fonttype': 42})
INK, RED, GREY, GREEN, AMBER = 'C0', 'C3', 'C7', '#2E7D32', '#B26A00'

# ------------------------------------------------------------------ data
BOARDS = [('openllm', 'Open LLM'), ('medical', 'Medical-LLM'), ('finance', 'FinLLM, text'), ('livebench', 'LiveBench')]
sel = {k: pd.read_csv(R / 'leaderboards' / f'{k}_selection.csv') for k, _ in BOARDS}
best = {k: int(s.same.sum()) for k, s in sel.items()}
nd = {k: len(s) for k, s in sel.items()}
kstar = {k: float(s.kappa_star.iloc[0]) for k, s in sel.items()}
mild = {k: float(s.sd_r.min()) for k, s in sel.items()}

BINS = [0, .1, .25, .5, .75, 1, np.inf]
LAB = ['<0.1', '0.1–0.25', '0.25–0.5', '0.5–0.75', '0.75–1', '≥1']


def by_ratio(p):
    g = pd.DataFrame({'b': pd.cut(p.ratio, BINS, right=False), 'rev': p.reversed_true}).groupby('b', observed=False).rev
    return 100 * g.mean().values, g.size().values


ap, bp = pd.read_csv(R / 'acs_pairs.csv'), pd.read_csv(R / 'beijing_pairs.csv')
cen_rate, cen_n = by_ratio(ap); bj_rate, bj_n = by_ratio(bp)
cen_cert = int((ap.ratio >= 1).sum()); cen_cert_rev = int(ap.reversed_true[ap.ratio >= 1].sum())
bj_lo = 100 * bp.reversed_true[bp.ratio < 0.1].mean(); bj_hi = 100 * bp.reversed_true[bp.ratio >= 0.1].mean()

sa, sb = pd.read_csv(R / 'acs_spot_check.csv'), pd.read_csv(R / 'beijing_spot_check.csv')
ff_cen = 100 * sa[sa.opt_true == 1].groupby('n').reject.mean()
ff_bj = 100 * sb[sb.opt_true == 1].groupby('n').reject.mean()
ff_bj_iid = 100 * sb[sb.opt_true == 1].groupby('n').reject_iid.mean()
sel_err_cen = 100 * (1 - pd.read_csv(R / 'acs_problems.csv').opt_true.mean())
sel_err_bj = 100 * (1 - pd.read_csv(R / 'beijing_problems.csv').opt_true.mean())

# ------------------------------------------------------------------ figure
fig, ((a, b), (c, d)) = plt.subplots(2, 2, figsize=(7.2, 6.0))

# a: where the leader stays best
y = np.arange(len(BOARDS))[::-1]
for i, (k, name) in enumerate(BOARDS):
    a.barh(y[i], best[k], color=GREEN, height=0.6)
    a.barh(y[i], nd[k] - best[k], left=best[k], color=RED, height=0.6, alpha=0.85)
    a.text(nd[k] + 0.25, y[i], f'best in {best[k]} of {nd[k]}', va='center', fontsize=8,
           color=GREEN if best[k] > nd[k] / 2 else RED)
a.set_yticks(y); a.set_yticklabels([n for _, n in BOARDS])
a.set_xlim(0, 12.6); a.set_xticks(range(0, 13, 2)); a.set_xlabel('deployments')
from matplotlib.patches import Patch
a.legend(handles=[Patch(color=GREEN, label='leader stays best'), Patch(color=RED, alpha=0.85, label='another model is best')],
         frameon=False, loc='lower right', fontsize=7.5, bbox_to_anchor=(1.02, -0.02))
a.set_ylim(-1.3, 3.6)
a.set_title('(a) leader stays best in 9 of 34 deployments', loc='left', fontsize=8.6)

# b: the leader's margin against the mildest mismatch
for i, (k, name) in enumerate(BOARDS):
    col = GREEN if kstar[k] >= mild[k] else RED
    b.plot([kstar[k], mild[k]], [y[i], y[i]], color=col, lw=1.2, zorder=1)
    b.plot(kstar[k], y[i], 'o', color=INK, ms=6, zorder=2)
    b.plot(mild[k], y[i], 's', color=GREY, ms=6, zorder=2)
b.set_xscale('log'); b.set_xlim(0.012, 2.2)
b.set_yticks(y); b.set_yticklabels([n for _, n in BOARDS])
b.set_xlabel(r'$\kappa^\star$ and $\mathrm{sd}_\mu(r)$, log scale')
b.plot([], [], 'o', color=INK, label=r"leader's ranking-stability radius $\kappa^\star$")
b.plot([], [], 's', color=GREY, label=r'mildest mismatch $\mathrm{sd}_\mu(r)$')
b.plot([], [], '-', color=GREEN, lw=1.2, label='radius covers it')
b.plot([], [], '-', color=RED, lw=1.2, label='radius falls short')
b.legend(frameon=False, loc='upper left', fontsize=7, bbox_to_anchor=(0.0, 1.02), ncol=2, columnspacing=1.0, handletextpad=0.5)
b.set_ylim(-0.7, 4.9)
b.set_title('(b) radius against mildest mismatch', loc='left', fontsize=8.6)

# c: the certificate on two real shifts
x = np.arange(len(LAB))
c.plot(x, cen_rate, marker='o', color=INK, lw=1.4, ms=4.5, label=f'census, 52 deployments')
ok = bj_n > 0
c.plot(x[ok], bj_rate[ok], marker='s', color=RED, lw=1.4, ms=4.5, label='Beijing, 24 months')
c.axvspan(4.5, 5.5, color=GREEN, alpha=0.10, lw=0)
c.text(5.0, 18.5, 'certified', ha='center', fontsize=8, color=GREEN, fontweight='bold')
c.text(5.0, 15.5, f'{cen_cert_rev} of {cen_cert:,}', ha='center', fontsize=7.5, color=GREEN)
c.text(1.1, 11.0, 'Beijing: no comparison\nreaches the certificate', fontsize=7.2, color=RED, ha='left', va='bottom', linespacing=1.1)
c.set_xticks(x); c.set_xticklabels(LAB, rotation=30, ha='right')
c.set_xlabel(r'radius / mismatch, $\kappa_{fg}\,/\,\mathrm{sd}_\mu(r)$')
c.set_ylabel('comparisons reversed (%)'); c.set_ylim(0, 24)
c.legend(frameon=False, loc='upper right', fontsize=7.5, bbox_to_anchor=(0.86, 1.0))
c.set_title('(c) real shifts: the ratio still orders risk', loc='left', fontsize=8.6)

# d: the spot-check after deployment
d.plot(ff_cen.index, ff_cen.values, marker='o', color=INK, lw=1.4, ms=4.5, label='census records')
d.plot(ff_bj.index, ff_bj.values, marker='s', color=RED, lw=1.4, ms=4.5, label='Beijing hours, resampled by day')
d.plot(ff_bj_iid.index, ff_bj_iid.values, marker='^', color=GREY, lw=1.2, ms=4.5, ls='--', label='Beijing hours, as if independent')
d.axhline(5, color='0.5', lw=0.8, ls=(0, (4, 3))); d.text(2400, 5.4, 'error level 5%', fontsize=7.2, color='0.4', ha='right')
d.set_xscale('log'); d.set_xlim(40, 2600)
d.set_xlabel('labeled deployment cases in the spot-check'); d.set_ylabel('right choices flagged (%)')
d.set_ylim(0, 17)
d.legend(frameon=False, loc='upper left', fontsize=7.5, bbox_to_anchor=(0.0, 1.02))
d.set_title('(d) spot-check after deployment holds its level', loc='left', fontsize=8.6)

fig.tight_layout(h_pad=1.6, w_pad=2.0)
fig.savefig('fig_boards.pdf', bbox_inches='tight')
fig.savefig('fig_boards.png', dpi=220, bbox_inches='tight')
for k, name in BOARDS:
    print(f'{name:14s} best in {best[k]} of {nd[k]}, kappa* {kstar[k]:.3f}, mildest sd_r {mild[k]:.3f}')
print(f'census certified {cen_cert}, reversed {cen_cert_rev}; selection error {sel_err_cen:.1f}%')
print(f'beijing reversed <0.1: {bj_lo:.1f}%, >=0.1: {bj_hi:.1f}%; selection error {sel_err_bj:.1f}%')
print('spot-check false flags, max %: census', round(ff_cen.max(), 2), 'beijing by day', round(ff_bj.max(), 2), 'as iid', round(ff_bj_iid.max(), 2))
