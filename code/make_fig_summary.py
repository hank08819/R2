"""Figure 5: the test in one picture, from the stored result files.

a, the decision tree: stability radius, regret test, direction, spot-check, with the
   evidence for each step read from its result file.
b, reversal rate against the ratio of radius to mismatch, on the suite, the
   leaderboard and the half of each test set the certificate never saw.
c, what the six answers buy in the tail of the cost, over the selection problems.
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
          ('language models', *by_ratio(le_ratio, le.inverted), RED, 's'),
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
fig = plt.figure(figsize=(7.2, 7.4))
a = fig.add_axes([0.015, 0.515, 0.97, 0.465]); a.set_axis_off()   # 6.98 x 3.44 in
a.set_xlim(-0.05, 7.30); a.set_ylim(-0.30, 3.33)                  # one unit = 0.95 in both ways
gs = fig.add_gridspec(1, 2, left=0.085, right=0.985, bottom=0.07, top=0.43, wspace=0.32)
a.set_title('(a) the AI governance decision tree (AIG-DT), from benchmark to deployment', loc='left')
BLUE, AMBER, DARK, MUTE = '#1F4E79', '#B26A00', '0.12', '0.40'
GLIGHT, RLIGHT, ALIGHT, BLIGHT = '#E8F3E8', '#FBE9E9', '#FFF4E0', '#EAF1F8'


def rbox(x, y, w, h, fc, ec, lw=1.0, ls='-', rs=0.08):
    a.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f'round,pad=0,rounding_size={rs}',
                               fc=fc, ec=ec, lw=lw, ls=ls))


def node(k, x, y, w, h, title, question, evidence):
    rbox(x, y, w, h, 'white', BLUE, lw=1.2)
    hb = 0.30                                              # header band
    a.add_patch(FancyBboxPatch((x, y + h - hb), w, hb, boxstyle='round,pad=0,rounding_size=0.08',
                               fc=BLUE, ec=BLUE, lw=1.2))
    a.add_patch(plt.Rectangle((x, y + h - hb), w, 0.12, fc=BLUE, ec=BLUE, lw=0))
    a.add_patch(plt.Circle((x + 0.17, y + h - hb / 2), 0.105, fc='white', ec='white'))
    a.text(x + 0.17, y + h - hb / 2, k, ha='center', va='center', fontsize=7.6, color=BLUE, fontweight='bold')
    a.text(x + 0.32, y + h - hb / 2, title, ha='left', va='center', fontsize=7.0, color='white', fontweight='bold')
    a.text(x + w / 2, y + h - hb - 0.11, question, ha='center', va='top', fontsize=7.0, color=DARK, linespacing=1.3)
    a.plot([x + 0.10, x + w - 0.10], [y + 0.48, y + 0.48], color='0.82', lw=0.8)
    a.text(x + w / 2, y + 0.25, evidence, ha='center', va='center', fontsize=6.0, color=MUTE,
           style='italic', linespacing=1.25)


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


# inputs
rbox(0.0, 1.20, 0.70, 1.10, BLIGHT, BLUE, lw=1.1)
a.text(0.35, 2.17, 'Inputs', ha='center', va='center', fontsize=7.4, color=BLUE, fontweight='bold')
a.text(0.35, 1.62, 'benchmark\nlosses\n\nstated use\nweight $r$', ha='center', va='center', fontsize=6.4,
       color=DARK, linespacing=1.15)

W, H, Y = 1.24, 1.50, 1.00
X = [1.00, 2.62, 4.24, 5.86]
node('1', X[0], Y, W, H, 'Stability radius',
     'does the lead cover\nthe mismatch?\n$\\kappa_{fg}\\geq\\mathrm{sd}_\\mu(r)$',
     f'{rev_cert} of {n_cert:,} reversed\nunseen half: {ho_rev} of {ho_cert:,}')
node('2', X[1], Y, W, H, 'Regret test',
     'is the cost within a\ntolerance $\\varepsilon$ fixed\nin advance?',
     f'bound holds in {drt_cov:.1f}%\nwrongly certified ≤ {drt_wrong:.1f}%')
node('3', X[2], Y, W, H, 'Direction',
     'can the user say\nwhich end the use\nweights more?',
     f'suite: cost cut by a third\nreal shifts: {bj_crit:.1f}% → {bj_rule:.1f}%')
node('4', X[3], Y, W, H, 'Spot-check',
     'after deployment, the\nsame test on a few\nlabeled cases',
     f'false alarms\n≤ {sp_acs:.1f}% (census)\n≤ {sp_bj:.1f}% (Beijing)')

ym = Y + H - 0.60                                          # spine height
arr(0.70, 1.75, X[0], 1.75)
for k, lab, col in [(0, 'no', RED), (1, 'open', '0.4'), (2, 'no', '0.4')]:
    arr(X[k] + W, ym, X[k + 1], ym, col)
    tag((X[k] + W + X[k + 1]) / 2, ym + 0.14, lab, col)

# outcomes above
TOP = 2.72
for k, txt in [(0, 'Certified\ndeploy'), (1, 'Within $\\varepsilon$\ndeploy'), (3, 'Verdict on\nreal labels')]:
    pill(X[k] + 0.12, TOP, W - 0.24, 0.48, txt, GLIGHT, GREEN)
    arr(X[k] + W / 2, Y + H, X[k] + W / 2, TOP, GREEN)
tag(X[0] + W / 2 + 0.08, (Y + H + TOP) / 2, 'yes', GREEN, 'left')
tag(X[1] + W / 2 + 0.08, (Y + H + TOP) / 2, 'certified', GREEN, 'left')

# outcomes below
BOT = 0.06
pill(X[1] + 0.12, BOT, W - 0.24, 0.48, 'Exceeds $\\varepsilon$\ndo not deploy', RLIGHT, RED)
arr(X[1] + W / 2, Y, X[1] + W / 2, BOT + 0.48, RED)
tag(X[1] + W / 2 + 0.08, (Y + BOT + 0.48) / 2, 'exceeds', RED, 'left')
pill(X[2] + 0.02, BOT, W - 0.04, 0.48, 'six answers re-rank\nsuite-validated only', ALIGHT, AMBER, fs=6.3, ls='--')
arr(X[2] + W / 2, Y, X[2] + W / 2, BOT + 0.48, AMBER)
tag(X[2] + W / 2 + 0.08, (Y + BOT + 0.48) / 2, 'yes', AMBER, 'left')
arr(X[2] + W - 0.02, BOT + 0.24, X[3] + W / 2, Y, AMBER, ls='--', rad=0.25)

a.text(3.62, -0.20, 'Steps 1 to 3 need only the benchmark and a stated weight; every step can be rerun by a third party.'
       '   EU AI Act Arts. 9, 15 · NIST AI RMF', ha='center', va='center', fontsize=6.2, color='0.45')

# b: the ratio decides
b = fig.add_subplot(gs[0, 0])
x = np.arange(len(LAB))
for name, rate, n, col, m in series:
    b.plot(x, rate, marker=m, color=col, lw=1.4, ms=4.5, label=name)
b.axvspan(4.5, 5.5, color=GREEN, alpha=0.10, lw=0)
b.text(5.0, 38, 'certified', ha='center', fontsize=8, color=GREEN, fontweight='bold')
b.text(5.0, 31, f'{rev_cert} of {n_cert:,}', ha='center', fontsize=7.5, color=GREEN)
b.text(5.0, 25.5, f'unseen half:\n{ho_rev} of {ho_cert:,}', ha='center', fontsize=7.5, color=GREY,
       linespacing=1.1)
b.set_xticks(x); b.set_xticklabels(LAB, rotation=30, ha='right')
b.set_xlabel(r'radius / mismatch, $\kappa_{fg}\,/\,\mathrm{sd}_\mu(r)$')
b.set_ylabel('comparisons reversed (%)')
b.set_ylim(0, 45)
b.legend(frameon=False, loc='upper center', bbox_to_anchor=(0.42, 1.0))
b.set_title('(b) the ratio decides whether a rank holds', loc='left')

# c: what the questions buy, in the tail
c = fig.add_subplot(gs[0, 1])
w = 0.38; q = np.arange(len(stats))
c.bar(q - w / 2, crit, w, color=INK, label=r'choose by $R^{2}$')
c.bar(q + w / 2, six, w, color=RED, label='with six answers')
for i in range(len(stats)):
    c.text(q[i] - w / 2, crit[i] + 1.2, f'{crit[i]:.1f}', ha='center', fontsize=7.5, color=INK)
    c.text(q[i] + w / 2, six[i] + 1.2, f'{six[i]:.1f}', ha='center', fontsize=7.5, color=RED)
c.set_xticks(q); c.set_xticklabels([s for s, _ in stats])
c.set_xlabel('over the selection problems')
c.set_ylabel('regret ratio (% of benchmark risk)')
c.legend(frameon=False, loc='upper left')
c.set_title('(c) on the suite, the answers cut the costly tail', loc='left')

fig.savefig('fig_summary.pdf', bbox_inches='tight')
fig.savefig('fig_summary.png', dpi=220, bbox_inches='tight')
print(f'certified {rev_cert} of {n_cert}; unseen half {ho_rev} of {ho_cert}')
for name, rate, n, *_ in series:
    print(f'{name:20s}', ' '.join(f'{r:5.1f}({k})' for r, k in zip(rate, n)))
print('criterion', np.round(crit, 1), 'six', np.round(six, 1))
