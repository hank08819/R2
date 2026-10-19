"""Figure 5: the decision tree in one picture, from the stored result files.

a, the decision tree: stability radius, regret test, direction, spot-check, with the
   evidence for each step read from its result file.
b, reversal rate against the ratio of radius to mismatch, on the suite, the
   leaderboard and the half of each test set the certificate never saw.
c, the budget rules against the six answers; d, the correction by quartile of the mismatch;
e, what the six answers buy in the tail of the cost, over the selection problems.
"""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from pathlib import Path

_H = Path(__file__).resolve().parents[1]
R = next(p for p in (_H / 'optionJF' / 'results' / 'universal', _H / 'results' / 'universal') if p.exists())
plt.rcParams.update({'font.size': 9, 'axes.titlesize': 9, 'axes.labelsize': 9,
                     'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 8,
                     'axes.spines.top': False, 'axes.spines.right': False})
INK, RED, GREY, GREEN = 'C0', 'C3', 'C7', '#2E7D32'

# ------------------------------------------------------------------ data
BINS = [0, .1, .25, .5, .75, 1, np.inf]
LAB = ['<0.1', '0.1–0.25', '0.25–0.5', '0.5–0.75', '0.75–1', '≥1']


def by_ratio(ratio, rev):
    b = pd.cut(ratio, BINS, right=False)
    g = pd.DataFrame({'b': b, 'rev': rev}).groupby('b', observed=False).rev
    return 100 * g.mean().values, g.size().values


sc = pd.read_csv(R / 'scale_r2.csv')
sc_ratio = sc.e_mu.abs() / sc.sd_d / sc.sd_r
le = pd.read_csv(R / 'llm_pairs.csv')
le_ratio = le.e_mu.abs() / le.sd_d / le.sd_r
ho = pd.read_csv(R / 'holdout_certificate.csv')
series = [('regression suite', *by_ratio(sc_ratio, sc.inverted), INK, 'o'),
          ('Open LLM Leaderboard', *by_ratio(le_ratio, le.inverted), RED, 's'),
          ('suite, unseen half', *by_ratio(ho.ratio, ho.rev_out), GREY, '^')]
n_cert = int((sc_ratio >= 1).sum() + (le_ratio >= 1).sum())
rev_cert = int(sc.inverted[sc_ratio >= 1].sum() + le.inverted[le_ratio >= 1].sum())
ho_cert = int((ho.ratio >= 1).sum()); ho_rev = int(ho.rev_out[ho.ratio >= 1].sum())

rs = pd.read_csv(R / 'rr_signed.csv')
stats = [('mean', np.mean), ('75th', lambda v: np.percentile(v, 75)),
         ('90th', lambda v: np.percentile(v, 90)), ('95th', lambda v: np.percentile(v, 95))]
crit = [100 * f(rs.rr_r2.values) for _, f in stats]
six = [100 * f(rs.rr_rule.values) for _, f in stats]

# decision-tree evidence, each number from its result file
drt = pd.read_csv(R / 'regret_test_sel.csv'); dh = drt[drt.frac == 0.5]
drt_cov = 100 * (dh.rr_pop <= dh.rr_ucb_sel + 1e-12)[dh.rr_pop > 0].mean()
drt_wrong = 100 * max(u[f'cert_{e}'][u.rr_pop > e].mean() for _, u in drt.groupby('frac')
                      for e in (0.01, 0.02, 0.05, 0.1, 0.2))
spot = lambda t, col: max(100 * u[col][u.opt_true == 1].mean() for _, u in t.groupby('n'))
sp_acs = spot(pd.read_csv(R / 'acs_spot_check.csv'), 'reject')
sp_bj = spot(pd.read_csv(R / 'beijing_spot_check.csv'), 'reject')
rl = pd.concat([pd.read_csv(R / f'rule_real_{w}.csv') for w in ('census', 'beijing')])
rl_bj = rl[rl.case.str.startswith('beijing')]
bj_crit, bj_rule = 100 * rl_bj.rr_criterion.mean(), 100 * rl_bj['rr_rule_0.05'].mean()

# ------------------------------------------------------------------ figure
fig = plt.figure(figsize=(7.2, 7.5))
a = fig.add_axes([0.015, 0.565, 0.97, 0.425]); a.set_axis_off()
a.set_xlim(-0.05, 7.30); a.set_ylim(-0.12, 3.75)
gs = fig.add_gridspec(2, 2, left=0.09, right=0.985, bottom=0.055, top=0.515, wspace=0.42, hspace=0.78)
a.set_title('(a) the AI governance decision tree (AIG-DT) for deployment', loc='left')
BLUE, AMBER, DARK, MUTE = '#1F4E79', '#B26A00', '0.12', '0.40'
GLIGHT, RLIGHT, ALIGHT, BLIGHT = '#E8F3E8', '#FBE9E9', '#FFF4E0', '#EAF1F8'


def rbox(x, y, w, h, fc, ec, lw=1.0, ls='-', rs=0.08):
    a.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f'round,pad=0,rounding_size={rs}',
                               fc=fc, ec=ec, lw=lw, ls=ls))


def node(k, x, y, w, h, title, question, evidence):
    rbox(x, y, w, h, 'white', BLUE, lw=1.2)
    hb = 0.30
    a.add_patch(FancyBboxPatch((x, y + h - hb), w, hb, boxstyle='round,pad=0,rounding_size=0.08',
                               fc=BLUE, ec=BLUE, lw=1.2))
    a.add_patch(plt.Rectangle((x, y + h - hb), w, 0.12, fc=BLUE, ec=BLUE, lw=0))
    a.add_patch(plt.Circle((x + 0.17, y + h - hb / 2), 0.105, fc='white', ec='white'))
    a.text(x + 0.17, y + h - hb / 2, k, ha='center', va='center', fontsize=7.6, color=BLUE, fontweight='bold')
    a.text(x + 0.32, y + h - hb / 2, title, ha='left', va='center', fontsize=7.0, color='white', fontweight='bold')
    a.text(x + w / 2, y + h - hb - 0.11, question, ha='center', va='top', fontsize=7.6, color=DARK, linespacing=1.3)


def pill(x, y, w, h, text, fc, ec, fs=7.0, ls='-'):
    rbox(x, y, w, h, fc, ec, lw=1.1, ls=ls, rs=0.12)
    a.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=fs, color=ec,
           fontweight='bold', linespacing=1.2)


def arr(x0, y0, x1, y1, color='0.4', ls='-', rad=0.0):
    a.annotate('', xy=(x1, y1), xytext=(x0, y0),
               arrowprops=dict(arrowstyle='-|>,head_length=0.35,head_width=0.18', color=color, lw=1.1,
                               shrinkA=0, shrinkB=0, ls=ls, connectionstyle=f'arc3,rad={rad}'))


def tag(x, y, text, color, ha='center'):
    a.text(x, y, text, fontsize=5.9, color=color, ha=ha, va='center', fontweight='bold')


# two entry paths: the weight is known, or only its direction
def inbox(y, title, body):
    rbox(0.0, y, 0.86, 1.00, BLIGHT, BLUE, lw=1.1)
    two = '\\n' in title
    a.text(0.43, y + (0.82 if two else 0.86), title, ha='center', va='center', fontsize=6.4, color=BLUE, fontweight='bold', linespacing=1.05)
    a.text(0.43, y + (0.33 if two else 0.40), body, ha='center', va='center', fontsize=5.6, color=DARK, linespacing=1.15)


W, H = 1.30, 1.00
YA, YB = 1.65, 0.10                                         # weight-known lane, direction-only lane
X1, X2, X4 = 1.10, 2.75, 4.80
inbox(YA, 'Weight known', 'benchmark losses\nand the stated\nweight $r$')
inbox(YB, 'Size and\ndirection', 'benchmark losses,\na size of mismatch\nand a direction')
node('1', X1, YA, W, H, 'Stability radius',
     'does the lead cover\nthe mismatch?\n$\\kappa_{fg}\\geq\\mathrm{sd}_\\mu(r)$', '')
node('2', X2, YA, W, H, 'Regret test (DRT)',
     'is the cost within\na stated tolerance $\\varepsilon$?', '')
node('3', X1, YB, W, H, 'Six questions',
     'for six leading pairs,\nwhich end gets\nmore weight?', '')
YS = (YA + YB) / 2 + 0.05
node('4', X4, YS, W, H, 'Spot-check',
     'the same test on\na few labeled cases', '')

ma, mb = YA + 0.42, YB + 0.42                              # arrow heights inside each lane
arr(0.86, ma, X1, ma); arr(0.86, mb, X1, mb, AMBER)
arr(X1 + W, ma, X2, ma, RED); tag((X1 + W + X2) / 2, ma + 0.13, 'no', RED)
arr(X2 + W, ma, X4 + 0.25, YS + H, '0.4', rad=-0.15)
tag(X2 + W + 0.42, ma + 0.20, 'undetermined', '0.4')
arr(X1 + W, mb, X4, YS + 0.35, AMBER, ls='--', rad=0.10)
tag(X1 + W + 0.70, mb - 0.20, 'validated on the suite', AMBER)

TOP = 2.82
pill(X1 + 0.13, TOP, W - 0.26, 0.44, 'Ranking\ncertified', GLIGHT, GREEN, fs=6.6)
arr(X1 + W / 2, YA + H, X1 + W / 2, TOP, GREEN); tag(X1 + W / 2 + 0.07, (YA + H + TOP) / 2, 'yes', GREEN, 'left')
pill(X2 - 0.02, TOP, 0.64, 0.44, 'Regret\nwithin $\\varepsilon$', GLIGHT, GREEN, fs=6.3)
pill(X2 + 0.68, TOP, 0.64, 0.44, 'Regret\nexceeds $\\varepsilon$', RLIGHT, RED, fs=6.3)
arr(X2 + 0.30, YA + H, X2 + 0.30, TOP, GREEN); arr(X2 + 1.00, YA + H, X2 + 1.00, TOP, RED)
pill(X4 + W + 0.25, YS + 0.28, 0.95, 0.44, 'Verdict on\nreal labels', GLIGHT, GREEN, fs=6.6)
arr(X4 + W, YS + 0.50, X4 + W + 0.25, YS + 0.50, GREEN)
# a certified regret verdict for a new place or time is confirmed by the spot-check
arr(X2 + 0.30, TOP + 0.44, X4 + 0.85, YS + H, GREEN, ls='--', rad=-0.45)
tag(X4 + 0.75, TOP + 0.55, 'new place or time:\nconfirm at step 4', GREEN, 'left')


# b: the ratio decides
b = fig.add_subplot(gs[0, 0])
x = np.arange(len(LAB))
for name, rate, n, col, m in series:
    b.plot(x, rate, marker=m, color=col, lw=1.4, ms=4.5, label=name)
b.axvspan(4.5, 5.5, color=GREEN, alpha=0.10, lw=0)
b.text(5.0, 24, 'certified', ha='center', fontsize=8, color=GREEN, fontweight='bold')
b.text(5.0, 13.5, f'unseen half:\n{ho_rev} of {ho_cert:,}', ha='center', fontsize=7.5, color=GREEN,
       linespacing=1.1)
b.set_xticks(x); b.set_xticklabels(LAB, rotation=20, ha='right', fontsize=7.2)
b.set_xlabel(r'radius / mismatch, $\kappa_{fg}\,/\,\mathrm{sd}_\mu(r)$')
b.set_ylabel('reversed (%)')
b.set_ylim(0, 45)
b.legend(frameon=False, loc='upper center', fontsize=7.2, bbox_to_anchor=(0.56, 1.02))
b.set_title('(b) the ratio decides', loc='left')

# c: what a budget buys, and what a direction buys (u42, u46)
ms = pd.read_csv(R / 'method_summary.csv', index_col=0)['value']
st = pd.read_csv(R / 'signed_top.csv'); fr = pd.read_csv(R / 'frontier.csv')
c = fig.add_subplot(gs[0, 1])
rules = ['worst-case\nregret', 'stated\nscenarios', 'stated\ntolerance', 'the criterion', 'six\nanswers']
vals = [round(v, 1) for v in (ms.ball_mean_stated, ms.scen_mean, 100 * fr['exc_0.02'].mean(),
                              100 * st.r2.mean(), 100 * st.m4.mean())]
cols = [GREY, GREY, GREY, INK, RED]
xr = np.arange(len(vals))
c.bar(xr, vals, color=cols, width=0.68)
for i, v in enumerate(vals):
    c.text(i, v + 0.35, f'{v}', ha='center', fontsize=8, color=cols[i], fontweight='bold')
c.axhline(vals[3], color=INK, lw=0.8, ls=(0, (4, 3)))
c.set_xticks(xr); c.set_xticklabels(rules, fontsize=7.0)
c.set_ylabel('mean excess (%)')
c.set_ylim(0, 19.5)
c.text(0.97, 0.95, 'budget rules', transform=c.transAxes, fontsize=8, va='top', ha='right', color=GREY)
c.set_title('(c) budget rules against six answers', loc='left')

# d: the correction by quartile of the mismatch (u44)
rsq = rs.groupby(pd.qcut(rs.sd_r, 4, duplicates='drop'), observed=True)[['rr_r2', 'rr_rule']].mean() * 100
d = fig.add_subplot(gs[1, 0])
q = np.arange(4); w = 0.38
critq = [round(v, 1) for v in rsq.rr_r2]; sixq = [round(v, 1) for v in rsq.rr_rule]
d.bar(q - w / 2, critq, w, color=INK, label='the criterion')
d.bar(q + w / 2, sixq, w, color=RED, label='with six answers')
for i in range(4):
    d.text(q[i] - w / 2, critq[i] + 0.5, f'{critq[i]}', ha='center', fontsize=7.5, color=INK)
    d.text(q[i] + w / 2, sixq[i] + 0.5, f'{sixq[i]}', ha='center', fontsize=7.5, color=RED)
d.set_xticks(q); d.set_xticklabels(['narrowest', '2nd', '3rd', 'widest'])
d.set_xlabel(r'quartile of the mismatch $\mathrm{sd}_\mu(r)$')
d.set_ylabel('regret ratio (%)')
d.set_ylim(0, 26); d.legend(frameon=False, loc='upper left')
d.set_title('(d) the gain grows with the mismatch', loc='left')

# e: what the questions buy, in the tail
e = fig.add_subplot(gs[1, 1])
w = 0.38; q = np.arange(len(stats))
e.bar(q - w / 2, crit, w, color=INK, label='the criterion')
e.bar(q + w / 2, six, w, color=RED, label='with six answers')
for i in range(len(stats)):
    e.text(q[i] - w / 2, crit[i] + 1.2, f'{crit[i]:.1f}', ha='center', fontsize=7.5, color=INK)
    e.text(q[i] + w / 2, six[i] + 1.2, f'{six[i]:.1f}', ha='center', fontsize=7.5, color=RED)
e.set_xticks(q); e.set_xticklabels([s for s, _ in stats])
e.set_xlabel('over the selection problems')
e.set_ylabel('regret ratio (%)')
e.legend(frameon=False, loc='upper left')
e.set_title('(e) the answers cut the costly tail', loc='left')

fig.savefig('fig_summary.pdf', bbox_inches='tight')
fig.savefig('fig_summary.png', dpi=220, bbox_inches='tight')
print(f'certified {rev_cert} of {n_cert}; unseen half {ho_rev} of {ho_cert}')
for name, rate, n, *_ in series:
    print(f'{name:20s}', ' '.join(f'{r:5.1f}({k})' for r, k in zip(rate, n)))
print('criterion', np.round(crit, 1), 'six', np.round(six, 1))
