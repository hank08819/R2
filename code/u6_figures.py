"""u6_figures.py -- the four figures, each drawn from a saved CSV.

Fig 1  R-squared reverses orderings, across a whole benchmark suite.
Fig 2  Why: the identity, drawn as the distance an ordering travels.
Fig 3  Both factors are necessary, and the regret bound holds.
Fig 4  Where the first-order extension to NMI stops being usable.
"""
import warnings, pickle; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, itertools
from pathlib import Path
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / 'results' / 'universal'
FIG = ROOT.parent / 'universal_metric' if (ROOT.parent / 'universal_metric').exists() else ROOT / 'figures'
FIG.mkdir(exist_ok=True)

# Figure style: default colour cycle, no top or
# right spine, a titled panel rather than a lettered one, and nothing else.
plt.rcParams.update({
    'font.size': 9, 'axes.titlesize': 9, 'axes.labelsize': 9,
    'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 7.5,
    'axes.spines.top': False, 'axes.spines.right': False,
    'figure.dpi': 600, 'savefig.dpi': 600,
    'savefig.bbox': 'tight', 'savefig.pad_inches': .02,
    'pdf.fonttype': 42, 'ps.fonttype': 42,   # embed real fonts, not outlines
})

_LET = {}


def panel(ax, letter, title=''):
    """Title each panel in one style: "(a) what it shows"."""
    ax.set_title(f'({letter}) {title}'.rstrip(), fontsize=9)


INK, HOT, COOL, GREY = 'C0', 'C3', 'C2', 'C7'
WARM, DEEP = 'C1', 'C0'


def save(fig, name):
    """Vector for the manuscript, 600 dpi raster for systems that want one."""
    fig.savefig(FIG / f'{name}.pdf')
    fig.savefig(FIG / f'{name}.png', dpi=600)
    plt.close(fig)
    print('wrote', name + '.pdf', 'and .png at 600 dpi')


# ============================================================ Figure 1 ===
def fig1(ax=None, letters='abc', only='abc'):
    """The suite. With `ax` given, draws the panels named in `only` into it."""
    sc = pd.read_csv(R / 'scale_r2.csv')
    sel = pd.read_csv(R / 'scale_selection.csv')
    sc['left'] = sc.pair.str.split('-').str[0]
    sc['right'] = sc.pair.str.split('-').str[1]
    own = ax is None
    if own:
        fig, ax = plt.subplots(1, len(only), figsize=(3.5 * len(only), 3.3),
                               gridspec_kw=dict(wspace=.44))
        ax = np.atleast_1d(ax)
    slot = {c: ax[i] for i, c in enumerate(only)}

    # (a) the selection whose excess loss is largest, drawn as a slopegraph
    t = sel.sort_values('rel').iloc[-1]
    g = sc[(sc.dataset == t.dataset) & (sc.seed == t.seed) &
           (sc.deployment == t.deployment)]
    ms = sorted(set(g.left) | set(g.right)); ix = {m: i for i, m in enumerate(ms)}
    K = len(ms); Amu = np.zeros((K, K)); Anu = np.zeros((K, K))
    for _, x in g.iterrows():
        i, j = ix[x.left], ix[x.right]
        Amu[i, j], Amu[j, i] = x.e_mu, -x.e_mu
        Anu[i, j], Anu[j, i] = x.e_nu, -x.e_nu
    ob = [ms[i] for i in np.argsort(Amu.sum(1))]
    od = [ms[i] for i in np.argsort(Anu.sum(1))]
    a = slot.get('a')
    for m in (ms if a is not None else []):
        y0, y1 = ob.index(m), od.index(m)
        cross = (y0 == 0) != (y1 == 0)
        a.plot([0, 1], [y0, y1], color=HOT if cross else GREY,
               lw=1.7 if cross else .9, zorder=3 if cross else 2)
        a.text(-.07, y0, m, ha='right', va='center', fontsize=7,
               color=HOT if cross else INK)
        a.text(1.07, y1, m, ha='left', va='center', fontsize=7,
               color=HOT if cross else INK)
    if a is not None:
        a.set_xlim(-.62, 1.62); a.set_ylim(K - .4, -.6)
        a.set_xticks([0, 1])
        a.set_xticklabels(['best by\n$R^{2}$', 'best in\ndeployment'])
        a.set_yticks([])
        for sp in ('left', 'bottom'): a.spines[sp].set_visible(False)
        panel(a, letters[only.index('a')], 'the costliest choice')

    # (b) every dataset in the suite
    per = sc.groupby('dataset').inverted.mean().mul(100).sort_values()
    b = slot.get('b')
    if b is not None:
        b.barh(np.arange(len(per)), per.values, color=INK, height=.74)
        pooled = 100 * sc.inverted.mean()
        b.axvline(pooled, color=HOT, lw=1.1, ls='--')
        b.text(pooled + .8, .4, f'all comparisons {pooled:.1f}%', color=HOT,
               fontsize=7.5, va='bottom')
        b.set_yticks([]); b.set_ylim(-.8, len(per) - .2)
        b.set_xlabel('pairs whose ordering reverses (%)')
        b.set_ylabel(f'{len(per)} datasets', labelpad=2)
        panel(b, letters[only.index('b')], 'every dataset affected')

    # (c) the rate against the mismatch, over every pair
    c = slot.get('c')
    if c is None:
        if own:
            save(fig, 'fig1_scale')
        return
    q = pd.qcut(sc.sd_r, 10, duplicates='drop')
    bn = sc.groupby(q, observed=True).agg(x=('sd_r', 'median'), y=('inverted', 'mean'))
    c.plot(bn.x, 100 * bn.y, 'o-', color=INK, ms=3.6, mfc='white', mew=.9)
    c.set_xscale('log')
    c.set_xticks([.3, 1, 3])
    c.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%g'))
    c.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    c.set_xticks([.3, .5, 1, 2, 3])
    c.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%g'))
    c.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    c.set_xlabel('$\\mathrm{sd}_\\mu(r)$')
    c.set_ylabel('reversal rate (%)')
    c.set_ylim(0, None)
    panel(c, letters[only.index('c')], 'the rate rises with mismatch')
    if own:
        save(fig, 'fig1_scale')


# ============================================================ Figure 2 ===
def fig2():
    st = pickle.load(open(R / 'per_pair.pkl', 'rb'))
    keys = [[x for x in st if x.startswith('mammography') and 'accuracy|cost-weighted 50' in x][0],
            [x for x in st if x.startswith('mammography') and 'balanced' in x][0]]
    titles = ['(a) accuracy, deployed at a 50:1 cost',
              '(b) balanced accuracy, same deployment']
    fig, ax = plt.subplots(1, 2, figsize=(7.4, 3.4), sharex=True, sharey=True,
                           gridspec_kw=dict(wspace=.12))
    for j, (k, t) in enumerate(zip(keys, titles)):
        recs = sorted(st[k]['recs'], key=lambda x: x['e_mu'])
        a = ax[j]
        nrev = 0
        for i, x in enumerate(recs):
            rev = x['inverted']; nrev += rev
            a.annotate('', xy=(x['e_nu'], i), xytext=(x['e_mu'], i),
                       arrowprops=dict(arrowstyle='-|>,head_width=.10,head_length=.24',
                                       color=HOT if rev else GREY,
                                       lw=1.5 if rev else .9, shrinkA=0, shrinkB=0))
            a.plot(x['e_mu'], i, 'o', ms=2.8, color=INK, zorder=4)
        a.axvline(0, color=INK, lw=.8)
        a.set_yticks([]); a.set_ylim(-1, len(recs))
        a.set_xlabel('$\\mathbb{E}[\\ell_f-\\ell_g]$')
        a.set_xticks([-.05, 0, .05, .10])
        panel(a, 'ab'[j], ['accuracy, 50:1 cost', 'balanced accuracy, 20:1 cost'][j])
        a.text(.02, .97, f'{nrev} of {len(recs)} reverse', transform=a.transAxes,
               fontsize=8, va='top', color=INK)
        a.spines['left'].set_visible(False)
    ax[0].set_ylabel('21 model pairs, ordered by benchmark gap', labelpad=4)
    fig.legend(handles=[
        Line2D([], [], marker='o', ls='none', color=INK, ms=3.4,
               label='benchmark, $\\mathbb{E}_\\mu[d]$'),
        Line2D([], [], color=GREY, lw=1.2,
               label='arrow $=\\mathrm{Cov}_\\mu(d,r)$; ordering survives'),
        Line2D([], [], color=HOT, lw=1.6,
               label='arrow crosses zero; ordering reverses')],
        loc='lower center', ncol=3, frameon=False, handlelength=1.9,
        bbox_to_anchor=(.5, -.10))
    save(fig, 'fig2_mechanism')


# ============================================================ Figure 3 ===
def fig3():
    grid = pd.read_csv(R / 'controlled_grid.csv')
    reg = pd.read_csv(R / 'regret.csv')
    fig, ax = plt.subplots(1, 2, figsize=(7.4, 3.4), gridspec_kw=dict(wspace=.32))
    a = ax[0]
    cols = [INK, COOL, '#6a51a3', HOT]
    for c, (rho, g) in zip(cols, grid.groupby('rho')):
        a.plot(g.sd_r, g.inv, 'o-', color=c, ms=3.4, mfc='white', mew=.9,
               label=f'$\\rho={rho:.1f}$')
    a.set_xlabel('$\\mathrm{sd}_\\mu(r)$'); a.set_ylabel('reversal rate (%)')
    a.legend(frameon=False, title='correlation of $d$ with $r$',
             title_fontsize=7.5, handlelength=1.6)
    panel(a, 'a', 'both factors are needed')

    b = ax[1]
    p = reg[reg.regret > 1e-12]
    b.scatter(p.bound, p.regret, s=4, color=INK, alpha=.28, lw=0)
    lim = [min(np.quantile(p.regret, .01), np.quantile(p.bound, .01)) * .5,
           np.quantile(p.bound, .995) * 2.0]
    b.plot(lim, lim, color=HOT, lw=1.1)
    b.text(lim[0] * 40, lim[0] * 95, 'bound $=$ regret', color=HOT,
           fontsize=7.5, rotation=41, ha='center')
    b.set_xscale('log'); b.set_yscale('log'); b.set_xlim(lim); b.set_ylim(lim)
    med = float(np.median(p.bound / p.regret))
    b.text(.97, .04, f'no violation in {len(p):,} draws\nmedian slack $\\times${med:.0f}',
           transform=b.transAxes, ha='right', va='bottom', fontsize=7.5, color=INK)
    b.set_xlabel('$\\Delta(\\mathcal{P})\\,\\mathrm{sd}_\\mu(r)$, selection bound')
    b.set_ylabel('realized regret')
    panel(b, 'b', 'the selection bound holds')
    save(fig, 'fig3_theory')


# ============================================================ Figure 4 ===
def fig4():
    sw = pd.read_csv(R / 'nmi_sweep.csv')
    cd = pd.read_csv(R / 'cross_domain_tabular.csv')
    fig, ax = plt.subplots(1, 2, figsize=(7.4, 3.4), gridspec_kw=dict(wspace=.32))
    a = ax[0]
    for c, (ds, g) in zip([INK, COOL], sw.groupby('dataset')):
        a.plot(g.sd_r, g.rel_med, 'o-', color=c, ms=3.4, mfc='white', mew=.9,
               label={'sick': 'thyroid'}.get(ds, ds))
    a.axhline(.10, color=HOT, lw=1.0, ls='--')
    a.axvspan(cd.sd_r.min(), cd.sd_r.max(), color=GREY, alpha=.16, lw=0)
    a.set_ylim(.05, 1.6)
    a.text(sw.sd_r.min() * 1.15, .062, 'ten percent of the first-order term',
           color=HOT, fontsize=7.5)
    a.text(np.sqrt(cd.sd_r.min() * cd.sd_r.max()), 1.25,
           'every setting in Table 1', color='#4a4a4a', fontsize=7.5, ha='center')
    a.set_xscale('log'); a.set_yscale('log')
    a.set_xlabel('$\\mathrm{sd}_\\mu(r)$'); a.set_ylabel('$|\\omega_2|\\,/\\,|$first order$|$')
    a.legend(frameon=False, handlelength=1.6)
    panel(a, 'a', 'the remainder grows with the mismatch')

    b = ax[1]
    sc = b.scatter(sw.rev_actual, sw.rev_pred, c=sw.sd_r, cmap='cividis_r',
                   norm=matplotlib.colors.LogNorm(), s=26, lw=.5, edgecolor='white',
                   zorder=3)
    hi = max(sw.rev_pred.max(), sw.rev_actual.max()) + 1.5
    b.plot([0, hi], [0, hi], color=HOT, lw=1.0)
    b.text(hi * .62, hi * .70, 'exact', color=HOT, fontsize=7.5, rotation=41)
    b.set_xlim(-.8, hi); b.set_ylim(-.8, hi)
    b.set_xlabel('pairs that actually reverse')
    b.set_ylabel('pairs the first-order term predicts')
    cb = fig.colorbar(sc, ax=b, pad=.02); cb.set_label('$\\mathrm{sd}_\\mu(r)$', fontsize=8)
    cb.ax.tick_params(labelsize=7); cb.outline.set_linewidth(.7)
    panel(b, 'b', 'predicted against actual reversals')
    save(fig, 'fig4_nmi')




# ============================================================ Figure 5 ===
def fig5(ax=None):
    """Panels a and b of the merged cost-and-crowding figure."""
    s = pd.read_csv(R / 'regret_ratio.csv')
    own = ax is None
    if own:
        fig, ax = plt.subplots(1, 2, figsize=(7.4, 3.4), gridspec_kw=dict(wspace=.32))
    a, b = ax[0], ax[1]
    v = np.sort(s.rr.values)
    surv = 1 - np.arange(len(v)) / len(v)
    a.step(100 * np.maximum(v, 1e-4), 100 * surv, color=INK, where='post')
    for q, lab in [(.90, '90th'), (.95, '95th')]:
        x = 100 * s.rr.quantile(q)
        a.plot([x, x], [0, 100 * (1 - q)], color=HOT, lw=.9, ls='--')
        a.text(x * 1.15, 100 * (1 - q) + 2, f'{lab}: {x:.0f}%', color=HOT, fontsize=8)
    a.set_xscale('log'); a.set_xlim(1e-2, None); a.set_ylim(0, 55)
    a.set_xlabel('regret ratio (%), log scale')
    a.set_ylabel('problems at least\nthis costly (%)')
    panel(a, 'a', 'the cost is heavy in the tail')

    # u20_visible.py keys on the dataset name, as scale_r2.csv does, so the join
    # needs no id-to-name translation.
    vm = pd.read_csv(R / 'visible_mismatch.csv')
    s = s.merge(vm[['dataset', 'seed', 'deployment', 'visible']],
                on=['dataset', 'seed', 'deployment'], how='inner')
    q = pd.qcut(s.visible, 4, duplicates='drop')
    g = s.groupby(q, observed=True).agg(x=('visible', 'median'),
                                        m=('rr', 'mean'), p=('rr', lambda z: z.quantile(.9)))
    w = .35; xs = np.arange(len(g))
    b.bar(xs - w / 2, 100 * g.m, w, color=INK, label='mean')
    b.bar(xs + w / 2, 100 * g.p, w, color=HOT, label='90th percentile')
    b.set_xticks(xs); b.set_xticklabels([f'{v:.2f}' for v in g.x])
    b.set_xlabel('visible mismatch $\\|P(r-1)\\|$, quartile median')
    b.set_ylabel('regret ratio (%)')
    b.legend(frameon=False, handlelength=1.2)
    panel(b, 'b', 'and tracks the visible mismatch')
    if own:
        save(fig, 'fig5_regret')


def fig11():
    """Figure 4 of the paper up to V7; make_fig4.py draws it now, from the same files."""
    fig, ax = plt.subplots(2, 2, figsize=(7.4, 6.0),
                           gridspec_kw=dict(wspace=.34, hspace=.52))
    fig5(ax[0])
    fig9(ax[1], letters='cd')
    save(fig, 'fig11_cost_crowding_v7')


# ============================================================ Figure 6 ===
def fig6():
    c = pd.read_csv(R / 'collapse.csv')
    K = int(c.K.iloc[0])
    ks = sorted(float(x[2:]) for x in c.columns if x.startswith('n_'))
    med = [c[f'n_{k}'].median() for k in ks]
    q1 = [c[f'n_{k}'].quantile(.25) for k in ks]
    q3 = [c[f'n_{k}'].quantile(.75) for k in ks]
    one = [100 * (c[f'n_{k}'] == 1).mean() for k in ks]
    fig, ax = plt.subplots(1, 2, figsize=(7.4, 3.4), gridspec_kw=dict(wspace=.34))
    a = ax[0]
    a.fill_between(ks, q1, q3, color=INK, alpha=.13, lw=0)
    a.plot(ks, med, 'o-', color=INK, ms=3.6, mfc='white', mew=.9)
    a.axhline(K, color=HOT, lw=1.1, ls='--')
    a.text(ks[0] * 1.2, K - .55, f'no budget: all {K} survive', color=HOT, fontsize=7.5)
    a.annotate('', xy=(ks[-1], med[-1] + .15), xytext=(ks[-1], K - .1),
               arrowprops=dict(arrowstyle='<->', color=GREY, lw=.9))
    a.text(ks[-1] * .72, (K + med[-1]) / 2, 'bought by naming\nany budget at all',
           color='#4a4a4a', fontsize=7.5, ha='right', va='center')
    a.set_xscale('log'); a.set_ylim(0, K + .6)
    a.set_xlabel('budget $\\kappa$ on $\\mathrm{sd}_\\mu(r)$')
    a.set_ylabel(f'models left in contention, of {K}')
    panel(a, 'a', 'candidates left by a budget')

    b = ax[1]
    b.plot(ks, one, 'o-', color=INK, ms=3.6, mfc='white', mew=.9)
    b.axhline(0, color=HOT, lw=1.1, ls='--')
    b.text(ks[0] * 1.2, 6, 'no budget: never decided', color=HOT, fontsize=7.5)
    b.set_xscale('log'); b.set_ylim(-4, 104)
    b.set_xlabel('budget $\\kappa$ on $\\mathrm{sd}_\\mu(r)$')
    b.set_ylabel('problems the benchmark decides (%)')
    panel(b, 'b', 'problems the benchmark settles alone')
    save(fig, 'fig6_collapse')


# ============================================================ Figure 7 ===
def fig7():
    d = pd.read_csv(R / 'scale_r2.csv')
    d['left'] = d.pair.str.split('-').str[0]; d['right'] = d.pair.str.split('-').str[1]
    fr = pd.read_csv(R / 'frontier.csv')
    t = fr.sort_values('n_frontier').iloc[-1]                 # the richest frontier
    g = d[(d.dataset == t.dataset) & (d.seed == t.seed)]
    h = g[g.deployment == t.deployment]
    ms = sorted(set(g.left) | set(g.right)); ix = {m: i for i, m in enumerate(ms)}
    K = len(ms); E = np.zeros((K, K)); S = np.zeros((K, K)); N = np.zeros((K, K))
    for _, x in g.iterrows():
        i, j = ix[x.left], ix[x.right]
        E[i, j], E[j, i] = x.e_mu, -x.e_mu
        S[i, j] = S[j, i] = x.sd_d
    for _, x in h.iterrows():
        i, j = ix[x.left], ix[x.right]
        N[i, j], N[j, i] = x.e_nu, -x.e_nu
    rmu = E.sum(1) / K; rnu = N.sum(1) / K
    W = (E + float(h.sd_r.iloc[0]) * S).max(1)
    front = [f for f in range(K) if not any(
        (rmu[j] <= rmu[f]) and (W[j] <= W[f]) and ((rmu[j] < rmu[f]) or (W[j] < W[f]))
        for j in range(K))]
    f_r2, g_star = int(np.argmin(rmu)), int(np.argmin(rnu))

    fig, ax = plt.subplots(1, 2, figsize=(7.4, 3.4), gridspec_kw=dict(wspace=.34))
    a = ax[0]
    a.scatter(rmu, W, s=40, color=GREY, edgecolor='k', lw=.4, zorder=2)
    o = sorted(front, key=lambda f: rmu[f])
    a.plot([rmu[f] for f in o], [W[f] for f in o], '-', color=INK, lw=1.1, zorder=3)
    a.scatter([rmu[f] for f in front], [W[f] for f in front], s=55, color=INK, edgecolor='k', lw=.4,
              zorder=4, label=f'frontier ({len(front)} of {K})')
    for f, lab, c, dx in [(f_r2, 'chosen by $R^{2}$', HOT, (8, 6)),
                          (g_star, 'best in deployment', COOL, (-6, 10))]:
        a.scatter([rmu[f]], [W[f]], s=64, facecolor='none', edgecolor=c, lw=1.5, zorder=5)
        a.annotate(lab, (rmu[f], W[f]), textcoords='offset points', xytext=dx,
                   fontsize=7.5, color=c, ha='left' if dx[0] > 0 else 'right')
    a.set_xlabel('benchmark risk $\\mathbb{E}_\\mu[\\ell_f]$ (shifted)')
    a.set_ylabel('worst-case regret $\\mathcal{W}_\\kappa(f)$')
    a.legend(frameon=False, handlelength=1.2, loc='center right')
    panel(a, 'a', 'one problem in two objectives')

    b = ax[1]
    eps = sorted(float(c[4:]) for c in fr.columns if c.startswith('exc_'))
    mean = [100 * fr[f'exc_{e}'].mean() for e in eps]
    wrong = [100 * fr[f'wrong_{e}'].mean() for e in eps]
    b.plot(eps, mean, 'o-', color=INK, ms=3.6, mfc='white', mew=.9, label='mean excess')
    b2 = b.twinx()
    b2.plot(eps, wrong, 's--', color=HOT, ms=3.4, mfc='white', mew=.9, label='picks wrong')
    b2.set_ylabel('picks a model deployment would not (%)', color=HOT, fontsize=8)
    b2.tick_params(axis='y', colors=HOT, labelsize=7.5)
    b2.spines['right'].set_visible(True); b2.spines['right'].set_color(HOT)
    b.set_xlabel('accuracy given up, as a share of the benchmark span')
    b.set_ylabel('mean excess deployment loss (%)')
    panel(b, 'b', 'what the trade buys')
    save(fig, 'fig7_frontier')


# ============================================================ Figure 8 ===
def fig8():
    d = pd.read_csv(R / 'mape_rr.csv')
    t = d.groupby(['criterion', 'deployment']).agg(
        rr=('rr', lambda x: 100 * x.mean()), sd_r=('sd_r', 'median')).reset_index()
    NAME = {'R2': '$R^{2}$', 'MAE': 'MAE', 'MAPE': 'MAPE', 'RSE': 'rel. sq. error'}
    SHORT = {'relative-error weighted': 'relative\nerror',
             'cost rising in outcome': 'cost rising\nin outcome',
             'top quartile of outcome': 'upper\nquartile',
             'top decile of outcome': 'upper\ndecile',
             'bottom decile of outcome': 'lower\ndecile'}
    ORDER = ['relative-error weighted', 'cost rising in outcome',
             'top quartile of outcome', 'top decile of outcome',
             'bottom decile of outcome']
    COL = {'R2': INK, 'MAE': GREY, 'MAPE': HOT, 'RSE': COOL}
    MK = {'R2': 'o', 'MAE': '^', 'MAPE': 's', 'RSE': 'D'}
    fig, ax = plt.subplots(1, 2, figsize=(7.4, 3.4), gridspec_kw=dict(wspace=.34))

    a = ax[0]
    x = np.arange(len(ORDER))
    for c in ['R2', 'MAE', 'MAPE', 'RSE']:
        g = t[t.criterion == c].set_index('deployment').reindex(ORDER)
        a.plot(x, g.rr, MK[c] + '-', color=COL[c], ms=4, mfc='white', mew=1,
               lw=1.4 if c in ('R2', 'MAPE') else 1.0, label=NAME[c],
               zorder=4 if c in ('R2', 'MAPE') else 3)
    a.set_xticks(x); a.set_xticklabels([SHORT[o] for o in ORDER], fontsize=7.5)
    a.set_xlabel('deployment measure')
    a.set_ylabel('regret ratio (%)')
    a.legend(frameon=False, handlelength=1.6, fontsize=7.5)
    panel(a, 'a', 'regret by criterion and deployment')

    b = ax[1]
    for c in ['R2', 'MAE', 'MAPE', 'RSE']:
        g = t[t.criterion == c]
        b.scatter(g.sd_r.clip(lower=.02), g.rr, s=55, marker=MK[c], color=COL[c],
                  edgecolor='k', lw=.4, label=NAME[c], zorder=3)
    q = t[t.sd_r > 0]
    z = np.polyfit(np.log(q.sd_r), q.rr, 1)
    xs = np.geomspace(q.sd_r.min(), q.sd_r.max(), 50)
    b.plot(xs, np.polyval(z, np.log(xs)), color=GREY, lw=1, ls='--', zorder=2)
    b.set_xscale('log')
    b.set_xlabel("mismatch of the criterion's own measure")
    b.set_ylabel('regret ratio (%)')
    panel(b, 'b', 'each criterion is right where its measure is')
    save(fig, 'fig8_criteria')



# ============================================================ Figure 9 ===
def fig9(ax=None, letters='ab'):
    c = pd.read_csv(R / 'crowding.csv')
    own = ax is None
    if own:
        fig, ax = plt.subplots(1, 2, figsize=(7.4, 3.4), gridspec_kw=dict(wspace=.34))
    COL = {'regression suite': INK, 'language models': HOT}
    a = ax[0]
    for s, g in c.groupby('setting'):
        g = g.sort_values('K')
        sl = np.polyfit(np.log(g.K), np.log(g.kstar), 1)
        a.plot(g.K, g.kstar, 'o-', color=COL[s], ms=4, mfc='white', mew=1,
               label=f"{ {'language models': 'Open LLM Leaderboard'}.get(s, s)} (slope ${sl[0]:.2f}$)")
        xs = np.geomspace(g.K.min(), g.K.max(), 30)
        a.plot(xs, np.exp(np.polyval(sl, np.log(xs))), color=COL[s], lw=.8, ls=':')
    a.set_xscale('log'); a.set_yscale('log')
    a.set_xticks([2, 5, 10, 20, 50, 77])
    a.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%g'))
    a.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    a.set_xlabel('$K$, candidates in the set')
    a.set_ylabel('ranking-stability radius $\\kappa^{\\star}$, median')
    a.legend(frameon=False, handlelength=1.6, fontsize=7.5)
    panel(a, letters[0], 'the radius falls as candidates are added')

    b = ax[1]
    pred = {'regression suite': -0.88, 'language models': -0.83}
    obs = {s: np.polyfit(np.log(g.K), np.log(g.kstar), 1)[0]
           for s, g in c.groupby('setting')}
    iv = pd.read_csv(R / 'intervals.csv').set_index('quantity')
    err = [[obs[s] - iv.loc[f'crowding slope, {s}', 'lo'] for s in pred],
           [iv.loc[f'crowding slope, {s}', 'hi'] - obs[s] for s in pred]]
    x = np.arange(len(pred)); w = .34
    b.bar(x - w / 2, [pred[s] for s in pred], w, color=GREY,
          label='predicted from\nthe radius tail')
    b.bar(x + w / 2, [obs[s] for s in pred], w, color=INK, label='observed in $K$',
          yerr=err, capsize=2.5, error_kw=dict(lw=.8, ecolor='#4a4a4a'))
    b.axhline(0, color=INK, lw=.7)
    b.set_xticks(x); b.set_xticklabels(['regression\nsuite', 'Open LLM\nLeaderboard'], fontsize=8)
    b.set_ylabel('slope of $\\log\\kappa^{\\star}$ on $\\log K$')
    lo_ = min(iv.loc[f'crowding slope, {s}', 'lo'] for s in pred)
    b.set_ylim(lo_ - .1, 0.24)
    b.legend(frameon=False, handlelength=1.2, fontsize=7.5, loc='upper center',
             ncol=2, columnspacing=1.1, borderaxespad=.1)
    panel(b, letters[1], 'measured slopes against predicted')
    if own:
        save(fig, 'fig9_crowding')


# =========================================================== Figure 10 ===
def fig10(ax=None, letters='ab'):
    """The leaderboard, and the same numbers averaged the other obvious way."""
    d = pd.read_csv(R / 'llm_raw.csv')
    AGG = {'leaderboard_bbh', 'leaderboard_gpqa', 'leaderboard_math_hard',
           'leaderboard_musr', 'leaderboard'}
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
    rl = lead.rank(ascending=False); rq = byq.rank(ascending=False)

    own = ax is None
    if own:
        fig, ax = plt.subplots(1, 2, figsize=(7.4, 3.4),
                               gridspec_kw=dict(wspace=.30, width_ratios=[1.15, 1]))
    a = ax[0]
    top = rl.sort_values().index[:12]
    for m in top:
        moved = abs(rl[m] - rq[m]) >= 3
        a.plot([0, 1], [rl[m], rq[m]], color=HOT if moved else GREY,
               lw=1.6 if moved else .8, zorder=3 if moved else 2)
        c = HOT if moved else INK
        a.text(-.07, rl[m], f'{int(rl[m])}  {m.split("/")[-1][:32]}', ha='right',
               va='center', fontsize=6.2, color=c)
        a.text(1.06, rq[m], f'{int(rq[m])}', ha='left', va='center', fontsize=6.2, color=c)
    ylo = max(12, max(rq[m] for m in top))          # keep every endpoint on the axes
    a.set_xlim(-2.35, 1.30); a.set_ylim(ylo + .8, .2)
    a.set_xticks([0, 1]); a.set_xticklabels(['leaderboard\n(groups equal)',
                                             'questions\nequal'], fontsize=7.5)
    a.set_yticks([]); a.set_ylabel('rank among 77 models', labelpad=2)
    for sp in ('left', 'bottom', 'right'):
        a.spines[sp].set_visible(False)
    panel(a, letters[0], 'groups equal against questions equal')

    b = ax[1]
    t = pd.read_csv(R / 'llm_pairs.csv')
    g = t.groupby('deployment').agg(sd=('sd_r', 'first'),
                                    inv=('inverted', lambda x: 100 * x.mean()))
    g = g.sort_values('inv')
    NICE = {'musr': 'MuSR', 'gpqa': 'GPQA', 'ifeval': 'IFEval', 'bbh': 'BBH',
            'math': 'MATH', 'mmlu_pro': 'MMLU-Pro'}
    lab = [NICE.get(x.replace('the ', '').replace(' group alone', ''),
                    x.replace('the ', '').replace(' of tasks', ''))
           for x in g.index]
    b.barh(np.arange(len(g)), g.inv, color=INK, height=.7)
    b.set_yticks(np.arange(len(g))); b.set_yticklabels(lab, fontsize=7.5)
    b.set_xlabel('pairs the leaderboard orders opposite to the deployment (%)')
    panel(b, letters[1], 'pairs each deployment reverses')
    if own:
        save(fig, 'fig10_llm')


def fig12():
    """Figure 1 of the paper up to V6; make_fig1.py draws it now, from the same files."""
    fig, ax = plt.subplots(2, 2, figsize=(7.6, 6.4),
                           gridspec_kw=dict(wspace=.34, hspace=.52,
                                            width_ratios=[1.12, 1]))
    fig10(ax[0], letters='ab')
    fig1(ax[1], letters='cd', only='bc')
    save(fig, 'fig12_two_benchmarks_v6')


def fig13():
    """The costliest wrong choice, on its own, for the Supplementary Information."""
    fig, ax = plt.subplots(1, 1, figsize=(3.6, 3.4))
    fig1(np.array([ax]), letters='a', only='a')
    save(fig, 'fig_costliest')


if __name__ == '__main__':
    import sys
    FIGS = {'1': fig1, '2': fig2, '3': fig3, '4': fig4,
            '5': fig5, '6': fig6, '7': fig7, '8': fig8, '9': fig9, '10': fig10,
            '11': fig11, '12': fig12, '13': fig13}
    for w in (sys.argv[1:] or sorted(FIGS)):
        try:
            FIGS[w]()
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f'fig{w} failed: {type(e).__name__}: {e}')
