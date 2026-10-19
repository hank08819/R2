"""Supplementary Figure S8: the deployment regret test (DRT), step by step.

The flow of Methods and Suppl. Note 8, with the suite's numbers read from the stored result files:
coverage of the chosen-row band and of the joint band (half size, positive regret), and the share
of choices in each verdict at a tolerance of 10 percent.
"""
import pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon
from pathlib import Path

_H = Path(__file__).resolve().parents[1]
R = next(p for p in (_H / 'optionJF' / 'results' / 'universal', _H / 'results' / 'universal') if p.exists())
plt.rcParams.update({'font.size': 9})
BLUE, DARK, GREEN, ORANGE, RED, GREY = '#2B6CB0', '#1A365D', '#2E7D32', '#C05621', '#C53030', '#718096'
FILL = {BLUE: '#EBF4FF', GREEN: '#E8F5E9', ORANGE: '#FFF3E0', RED: '#FDECEC', GREY: '#F4F5F7'}

# ---------------------------------------------------------------- numbers from the result files
old = pd.read_csv(R / 'regret_test_sel.csv'); oh = old[old.frac == 0.5]
cov_row = 100 * (oh.rr_pop <= oh.rr_ucb_old + 1e-12)[oh.rr_pop > 0].mean()
jt = pd.read_csv(R / 'regret_test_joint.csv'); jh = jt[jt.frac == 0.5]
cov_joint = 100 * (jh.rr_pop <= jh.rr_ucb_joint + 1e-12)[jh.rr_pop > 0].mean()
cert = 100 * jh['cert_joint_0.1'].mean(); exc = 100 * jh['exceed_joint_0.1'].mean(); und = 100 - cert - exc

fig, ax = plt.subplots(figsize=(7.4, 8.2))
ax.set_xlim(0, 10); ax.set_ylim(0, 12); ax.axis('off')


def box(x, y, w, h, col, lw=1.3, dashed=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.02,rounding_size=0.12',
                                fc=FILL[col], ec=col, lw=lw, ls='--' if dashed else '-'))


def step(n, x, y, col=DARK):
    ax.add_patch(plt.Circle((x, y), 0.21, color=col, zorder=5))
    ax.text(x, y, n, color='white', ha='center', va='center', fontsize=9, fontweight='bold', zorder=6)


def arrow(x0, y0, x1, y1, col=DARK, ls='-'):
    ax.annotate('', (x1, y1), (x0, y0), arrowprops=dict(arrowstyle='-|>', color=col, lw=1.2, ls=ls))


ax.text(5, 11.75, 'The deployment regret test (DRT)', ha='center', fontsize=12, fontweight='bold', color=DARK)
ax.text(5, 11.38, 'Can the model the benchmark selects lose more than a tolerance fixed in advance?',
        ha='center', fontsize=8.5, color=GREY)

X, W = 0.3, 6.6
rows = [('1', 'Inputs', ['per-instance benchmark losses $\\ell_{f,i}$',
                         'stated deployment weight $w$, with $\\hat r=w/\\bar w$',
                         'tolerance $\\varepsilon$, fixed in advance'], 9.85, 1.2),
        ('2', 'Select', ['benchmark choice $f^\\star=\\arg\\min_f \\hat m_f$'], 8.95, 0.55),
        ('3', 'Estimate', ['deployment gaps $\\hat\\theta_{fg}$, every ordered pair',
                           'benchmark risks $\\hat m_f$, every candidate',
                           'their influence functions $\\chi_{fg}$ and $\\phi_f$'], 7.4, 1.2)]
for n, title, items, y, h in rows:
    box(X, y, W, h, BLUE)
    step(n, X + 0.35, y + h / 2)
    ax.text(X + 0.7, y + h / 2, title, va='center', fontsize=10, fontweight='bold', color=DARK)
    top = y + h / 2 + 0.3 * (len(items) - 1) / 2
    for k, it in enumerate(items):
        ax.text(X + 2.3, top - 0.3 * k, '• ' + it, va='center', fontsize=8.3)
y4, h4 = 5.15, 1.85
box(X, y4, W, h4, BLUE)
step('4', X + 0.35, y4 + h4 - 0.3)
ax.text(X + 0.7, y4 + h4 - 0.3, 'Selection-adjusted band', va='center', fontsize=10, fontweight='bold', color=DARK)
for k, it in enumerate(['one critical value $\\lambda$ from the multiplier bootstrap',
                        'covers all $K(K-1)+K$ quantities at once',
                        'so it holds for whichever $f^\\star$ the sample selected']):
    ax.text(X + 0.7, y4 + h4 - 0.68 - 0.27 * k, '• ' + it, va='center', fontsize=8.3)
box(X + 0.7, y4 + 0.12, W - 1.4, 0.36, BLUE, lw=1.0)
ax.text(X + W / 2, y4 + 0.3, 'upper bound $\\mathrm{RR}^{+}(f^\\star)$ on the regret ratio', ha='center', va='center',
        fontsize=9, fontweight='bold', color=DARK)
for y0, y1 in ((9.85, 9.52), (8.95, 8.62), (7.4, 7.02)):
    arrow(X + W / 2, y0, X + W / 2, y1)

# side notes
box(7.25, 8.85, 2.5, 0.75, RED, lw=1.0)
ax.text(8.5, 9.22, 'chosen on the same\nsample the test uses', ha='center', va='center', fontsize=7.8, color=RED)
arrow(7.25, 9.22, X + W + 0.04, 9.22, RED, '--')
box(7.25, 5.6, 2.5, 1.2, GREY, lw=1.0, dashed=True)
ax.text(8.5, 6.2, f'coverage of a positive regret,\nsuite, half size\n\njoint band: {cov_joint:.1f}%\nchosen row only: {cov_row:.1f}%',
        ha='center', va='center', fontsize=7.8, color='#2D3748')
arrow(7.25, 6.2, X + W + 0.04, 6.2, GREY, '--')

# decision
cx = X + W / 2
arrow(cx, y4, cx, 4.75)
ax.add_patch(Polygon([[cx, 4.75], [cx + 2.0, 4.2], [cx, 3.65], [cx - 2.0, 4.2]],
                     closed=True, fc=FILL[BLUE], ec=DARK, lw=1.3))
step('5', cx - 1.65, 4.65)
ax.text(cx, 4.2, 'compare $\\mathrm{RR}^{+}(f^\\star)$ with $\\varepsilon$', ha='center', va='center',
        fontsize=9, fontweight='bold', color=DARK)

cols = [(GREEN, 'Certified', '$\\mathrm{RR}^{+}\\leq\\varepsilon$', cert,
         'keep the choice;\nfor a new place or time,\nconfirm at step 4'),
        (ORANGE, 'Undetermined', 'neither verdict holds', und, 'labeled spot-check\n(step 4)'),
        (RED, 'Exceeds', 'a rival ahead by more than $\\varepsilon$', exc, 'choose again')]
xs = [0.3, 3.55, 6.8]
for (col, name, rule, pct, act), x in zip(cols, xs):
    box(x, 2.15, 2.95, 1.05, col)
    ax.text(x + 1.475, 2.92, name, ha='center', va='center', fontsize=10.5, fontweight='bold', color=col)
    ax.text(x + 1.475, 2.58, rule, ha='center', va='center', fontsize=8.2)
    ax.text(x + 1.475, 2.3, f'suite, $\\varepsilon=10\\%$: {pct:.1f}% of choices', ha='center', va='center', fontsize=7.2, color=GREY)
    box(x, 0.95, 2.95, 0.85, col)
    ax.text(x + 1.475, 1.375, act, ha='center', va='center', fontsize=8.6, fontweight='bold', color=col)
    arrow(x + 1.475, 2.15, x + 1.475, 1.82, col)
arrow(cx - 2.0, 4.2, xs[0] + 1.475, 3.22, GREEN)
arrow(cx, 3.65, xs[1] + 1.475, 3.22, ORANGE)
arrow(cx + 2.0, 4.2, xs[2] + 1.475, 3.22, RED)
step('6', 0.12, 1.98)

box(0.3, 0.12, 9.45, 0.58, BLUE, lw=1.0)
ax.text(5.02, 0.41, 'Output: a verdict against a tolerance fixed in advance, wrong with asymptotic probability at most $\\gamma$.\n'
        'It is read from the benchmark alone; no deployment labels enter.', ha='center', va='center', fontsize=7.8, color=DARK)

fig.savefig(_H / 'universal_metric' / 'fig_drt.pdf', bbox_inches='tight')
fig.savefig(_H / 'universal_metric' / 'fig_drt.png', dpi=600, bbox_inches='tight')
print(f'wrote fig_drt: coverage joint {cov_joint:.1f}, chosen row {cov_row:.1f}; verdicts {cert:.1f}/{und:.1f}/{exc:.1f}')
