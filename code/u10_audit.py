"""u10_audit.py -- every number in the paper, checked against the CSV it came from.

Fails loudly if a figure in the text has drifted from the script that produced it.
Run after any rerun of u1, u5, u7, or the NMI scripts.
"""
import warnings; warnings.filterwarnings('ignore')
import pandas as pd, numpy as np
from pathlib import Path
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / 'results' / 'universal'
import re
# The manuscript sits beside this tree in the working copy and one level up in the
# released package, so take the first directory that holds it.
P = next((d for d in (ROOT.parent / 'universal_metric', ROOT / 'paper', ROOT.parent, ROOT)
          if (d / 'NMI_V3.tex').exists()), ROOT.parent / 'universal_metric')
_src = [f for f in ('NMI_V3.tex', 'NMI_V3_SI.tex', 'main.tex', 'fig1.tex', 'table_cross_domain.tex')
        if (P / f).exists()]
_raw = ''.join((P / f).read_text() for f in _src)
# Compare on whitespace-normalised text, so a claim is checked for what it says
# rather than for where LaTeX happened to wrap the line.
norm = lambda x: re.sub(r'\s+', ' ', x).strip()
tex = norm(_raw)

sc = pd.read_csv(R / 'scale_r2.csv')
sel = pd.read_csv(R / 'scale_selection.csv')
cd = pd.read_csv(R / 'cross_domain.csv')
reg = pd.read_csv(R / 'regret.csv')
grid = pd.read_csv(R / 'controlled_grid.csv')
sw = pd.read_csv(R / 'nmi_sweep.csv')
opt = pd.read_csv(R / 'option_sdr.csv')
rr = pd.read_csv(R / 'regret_ratio.csv')
inf = pd.read_csv(R / 'inference.csv')
inf['rev'] = (np.sign(inf.e_mu) != np.sign(inf.e_nu)) & (inf.e_mu != 0)
inf['sm'] = inf.e_mu.abs() > 1.96 * inf.se_mu
inf['sn'] = inf.e_nu.abs() > 1.96 * inf.se_nu
bo = inf[inf.sm & inf.sn]
bq = bo.assign(q=pd.qcut(bo.n, 2, duplicates='drop')).groupby('q', observed=True)\
       .agg(n=('n','median'), rev=('rev', lambda x: 100*x.mean()))
li = pd.read_csv(R / 'llm_inference.csv')
li['rev'] = (np.sign(li.e_mu) != np.sign(li.e_nu)) & (li.e_mu != 0)
li['sm'] = li.e_mu.abs() > 1.96 * li.se_mu
li['sn'] = li.e_nu.abs() > 1.96 * li.se_nu
lb = li[li.sm & li.sn]
ce = pd.read_csv(R / 'crowding_exponent.csv').set_index('setting')
nr = pd.read_csv(R / 'neural.csv')
ii = pd.read_csv(R / 'iia.csv')
cw = pd.read_csv(R / 'crowding.csv')
lp = pd.read_csv(R / 'llm_pairs.csv')
ls_ = pd.read_csv(R / 'llm_selection.csv')
lg = lp.groupby('deployment').inverted.mean().mul(100)
nmod = lp.pair.str.split('|').explode().nunique()
ms = pd.read_csv(R / 'method_summary.csv', index_col=0)['value']
rq = rr.groupby(pd.qcut(rr.sd_r, 4, duplicates='drop'), observed=True).rr
qm = [100*v for v in rq.mean()]
qp = [100*v for v in rq.quantile(.9)]
cv = pd.read_csv(R / 'nmi_converge.csv')
ci = pd.read_csv(R / 'cluster_intervals.csv').set_index('rate')              # u57
rs_ap = pd.read_csv(R / 'acs_pairs.csv'); rs_bp = pd.read_csv(R / 'beijing_pairs.csv')          # u59, u61
rs_bq = pd.read_csv(R / 'beijing_problems.csv'); rs_bq10 = pd.read_csv(R / 'real_10models' / 'beijing_problems.csv')
rs_bq, rs_bq10 = (q.drop_duplicates(['year', 'month', 'seed']) for q in (rs_bq, rs_bq10))
rs_as = pd.read_csv(R / 'acs_spot_check.csv'); rs_bs = pd.read_csv(R / 'beijing_spot_check.csv')  # u60, u62
rs_c = rs_ap.certified == 1
rs_size = lambda t, col: max(100 * u[col][u.opt_true == 1].mean() for _, u in t.groupby('n'))
rs_cov = min(100 * (u.rr_true[u.rr_true > 0] <= u.rr_ucb[u.rr_true > 0] + 1e-12).mean() for _, u in rs_bs.groupby('n'))
rl = pd.concat([pd.read_csv(R / f'rule_real_{w}.csv') for w in ('census', 'beijing')])          # u70
rl['fam'] = rl.case.str.split('-').str[0]
rlc = pd.concat([pd.read_csv(R / f'rule_real_corr_{w}.csv') for w in ('census', 'beijing')])
rl_m = lambda fam, col: 100 * rl[rl.fam == fam][col].mean()
LB = R / 'leaderboards'                                                                    # u72, u73
lbs = pd.read_csv(LB / 'leaderboards_summary.csv').set_index('board')
lbk = {b: pd.read_csv(LB / f'{b}_selection.csv') for b in ['openllm', 'medical', 'finance', 'livebench']}
lbr = pd.read_csv(LB / 'leaderboards_regret_test.csv')
lb_none = sum(int(lbs.loc[b, 'deployments']) for b in ['openllm', 'finance', 'livebench'] if lbs.loc[b, 'leader_certified_in'] == 0)
st = pd.read_csv(R / 'signed_top.csv'); sn = pd.read_csv(R / 'signed_noise.csv').set_index('p_wrong')   # u44
rsg = pd.read_csv(R / 'rr_signed.csv'); fr = pd.read_csv(R / 'frontier.csv')                         # u46, u42
rsq = rsg.groupby(pd.qcut(rsg.sd_r, 4, duplicates='drop'), observed=True)[['rr_r2', 'rr_rule']].mean() * 100
ho = pd.read_csv(R / 'holdout_certificate.csv'); ho_c = ho.ratio >= 1                     # u56
drt = pd.read_csv(R / 'regret_test_sel.csv'); drt_h = drt[drt.frac == 0.5]   # u68
drt_pos = drt_h.rr_pop > 0
drt_wrong = max(u[f'cert_{e}'][u.rr_pop > e].mean() for _, u in drt.groupby('frac') for e in (0.01, 0.02, 0.05, 0.1, 0.2))
mx = pd.read_csv(R / 'minimax_check.csv')                                   # u55
mx['cond'] = mx.kappa <= mx.inv_zinf
mx['ok'] = mx.reg_bench <= mx.reg_other + 1e-12
mxe = mx[mx.kfg < mx.kappa].groupby('setting').ok.mean().mul(100)
mxc = mx.groupby('setting').cond.mean().mul(100)
vm = pd.read_csv(R / 'visible_mismatch.csv')
vj = vm.merge(rr[['dataset', 'seed', 'deployment', 'rr']],
              on=['dataset', 'seed', 'deployment'], how='inner')
vpos = vj[vj.rr > 1e-12]
vq = vj.groupby(pd.qcut(vj.visible, 4, duplicates='drop'), observed=True).rr
# Rank correlation of mismatch against reversal rate, by dataset, within each of the
# two deployments whose weight varies continuously.
cont = [spearmanr(*zip(*[(v.sd_r.mean(), v.inverted.mean())
                         for _, v in sc[sc.deployment == d].groupby('dataset')]))[0]
        for d in ('cost rising in outcome', 'relative-error weighted')]
le = pd.read_csv(R / 'llm_entropy.csv').set_index('quantity')['value']   # u50
ra = pd.read_csv(R / 'llm_rank_agreement.csv').iloc[0]                    # u51
tr = pd.read_csv(R / 'llm_two_readings.csv').iloc[0]                      # u52
lm = pd.read_csv(R / 'llm_models.csv')                                    # u53
dp = sc.drop_duplicates(['dataset', 'seed', 'pair'])      # 4,725 distinct pairs
kd = dp.e_mu.abs() / dp.sd_d; enc = kd**2 / (1 + kd**2)     # pair radius, encoded share

per = sc.groupby('dataset').inverted.mean().mul(100)
bn = sc.groupby(pd.qcut(sc.sd_r, 5, duplicates='drop'), observed=True).inverted.mean().mul(100)
w = sel[sel.wrong == 1]
pos = reg[reg.regret > 1e-12]
r2 = cd[cd.criterion.str.startswith('R2')]
G = {(r.sd_r, r.rho): r.inv for r in grid.itertuples()}
mam = sw[sw.dataset == 'mammography'].sort_values('sd_r')
at199 = mam.iloc[(mam.sd_r - 1.99).abs().argmin()]
lim = cv.iloc[-1].ratio                       # limiting |R2|/|first| per unit t
slope = lim / cv.iloc[0].sd_rt                # ... expressed per unit sd(r)

C = [
    # ---------------------------------------------- suite scale (section 4.1)
    ('suite pairs',        '22{,}545',               f'{len(sc):,}'.replace(',', '{,}')),
    ('suite reversal',     '$14.2$ percent',         f'${100*sc.inverted.mean():.1f}$ percent'),
    ('per-dataset median', '$14.1$ percent',         f'${per.median():.1f}$ percent'),
    ('per-dataset range',  '$1.9$ to $31.3$',        f'${per.min():.1f}$ to ${per.max():.1f}$'),
    ('quintile low',       '$6.8$ percent',          f'${bn.iloc[0]:.1f}$ percent'),
    ('quintile low sd',    'mismatch is $0.35$',     f'mismatch is ${sc.sd_r.quantile(.1):.2f}$'),
    ('quintile high',      '$18.3$ percent',         f'${bn.iloc[-1]:.1f}$ percent'),
    ('spearman scale',     '$0.108$',                f'${spearmanr(sc.sd_r, sc.inverted)[0]:.3f}$'),
    ('selection wrong',    '$38.9$ percent',         f'${100*sel.wrong.mean():.1f}$ percent'),
    ('selection count',    '$501$ selection',        f'${len(sel)}$ selection'),
    ('datasets affected',  '$31$ of the $35$',       f'${int(sel.groupby("dataset").wrong.max().sum())}$ of the ${sel.dataset.nunique()}$'),
    # ------------------------------------------------- regret ratio (eq. 5)
    ('RR q3',              f'${100*rr.rr.quantile(.75):.1f}$ percent at the third quartile',
                           f'${100*rr.rr.quantile(.75):.1f}$ percent at the third quartile'),
    ('RR p90',             f'${100*rr.rr.quantile(.90):.1f}$ percent at the ninetieth percentile',
                           f'${100*rr.rr.quantile(.90):.1f}$ percent at the ninetieth percentile'),
    ('RR p95',             f'${100*rr.rr.quantile(.95):.1f}$ percent at the ninety-fifth',
                           f'${100*rr.rr.quantile(.95):.1f}$ percent at the ninety-fifth'),
    ('RR quartile p90',    f'runs ${qp[0]:.1f}$, ${qp[1]:.1f}$, ${qp[2]:.1f}$\nand ${qp[3]:.1f}$ percent',
                           f'runs ${qp[0]:.1f}$, ${qp[1]:.1f}$, ${qp[2]:.1f}$\nand ${qp[3]:.1f}$ percent'),
    # ------------------------------------------------------ inference (sec 4.1)
    ('resolved share',     f'against\n${100*inf.sm.mean():.1f}$ percent, because its subtasks',
                           f'against\n${100*inf.sm.mean():.1f}$ percent, because its subtasks'),
    ('both decisive',      f'leave ${100*(inf.sm&inf.sn).mean():.1f}$ percent on which both are',
                           f'leave ${100*(inf.sm&inf.sn).mean():.1f}$ percent on which both are'),
    ('conditional rev',    f'reversal rate is ${100*bo.rev.mean():.1f}$ percent',
                           f'reversal rate is ${100*bo.rev.mean():.1f}$ percent'),
    ('unresolved',         f'on ${100*(~inf.sm).mean():.1f}$ percent of pairs the',
                           f'on ${100*(~inf.sm).mean():.1f}$ percent of pairs the'),
    ('rev by n low',       f'it is ${bq.rev.iloc[0]:.1f}$ percent where the median test set holds ${int(bq.n.iloc[0])}$',
                           f'it is ${bq.rev.iloc[0]:.1f}$ percent where the median test set holds ${int(bq.n.iloc[0])}$'),
    ('rev by n high',      f'${bq.rev.iloc[-1]:.1f}$ percent where it holds ${int(bq.n.iloc[-1])}$',
                           f'${bq.rev.iloc[-1]:.1f}$ percent where it holds ${int(bq.n.iloc[-1])}$'),
    # -------------------------------------------------- language models (sec 4.2)
    ('LLM models',         f'the {nmod} models evaluated on all',
                           f'the {nmod} models evaluated on all'),
    ('LLM reversal',       f'${100*lp.inverted.mean():.1f}$ percent are ordered by the leaderboard',
                           f'${100*lp.inverted.mean():.1f}$ percent are ordered by the leaderboard'),
    ('LLM range',          f'from ${lg.min():.1f}$ percent',
                           f'from ${lg.min():.1f}$ percent'),
    ('LLM range hi',       f'to ${lg.max():.1f}$ percent',
                           f'to ${lg.max():.1f}$ percent'),
    ('LLM zero of eight',  'the best available is zero',
                           'the best available is zero' if ls_.same.sum() == 0 else 'NOT zero'),
    ('LLM regret lo',      f'from ${100*ls_.regret_ratio.min():.1f}$ to',
                           f'from ${100*ls_.regret_ratio.min():.1f}$ to'),
    ('LLM regret hi',      f'to ${100*ls_.regret_ratio.max():.1f}$ percent depending',
                           f'to ${100*ls_.regret_ratio.max():.1f}$ percent depending'),
    ('LLM question count', f'the top-ranked model is ${100*ls_[ls_.deployment=="by question count"].regret_ratio.iloc[0]:.1f}$ percent worse',
                           f'the top-ranked model is ${100*ls_[ls_.deployment=="by question count"].regret_ratio.iloc[0]:.1f}$ percent worse'),
    ('LLM resolves',       f'resolves\n${100*li.sm.mean():.1f}$ percent of pairs at the five percent level',
                           f'resolves\n${100*li.sm.mean():.1f}$ percent of pairs at the five percent level'),
    ('LLM cond rev',       f'within those the reversal rate is ${100*lb.rev.mean():.1f}$ percent',
                           f'within those the reversal rate is ${100*lb.rev.mean():.1f}$ percent'),
    ('crowd pred reg',     f'predicts a slope of ${ce.loc["regression suite","predicted"]:.2f}$',
                           f'predicts a slope of ${ce.loc["regression suite","predicted"]:.2f}$'),
    ('crowd obs llm',      f'against a measured\n${ce.loc["language models","observed"]:.2f}$',
                           f'against a measured\n${ce.loc["language models","observed"]:.2f}$'),
    # ------------------------------------------------ method section (sec 8)
    ('naive p90',          f'from ${ms.naive_p90_r2:.1f}$ to ${ms.naive_p90_stated:.1f}$ percent',
                           f'from ${ms.naive_p90_r2:.1f}$ to ${ms.naive_p90_stated:.1f}$ percent'),
    ('ball mean',          f'raises the mean excess from ${ms.ball_mean_r2:.1f}$ to ${ms.ball_mean_stated:.1f}$',
                           f'raises the mean excess from ${ms.ball_mean_r2:.1f}$ to ${ms.ball_mean_stated:.1f}$'),
    ('ball counts',        f'it improves ${int(ms.ball_better)}$ and damages\n${int(ms.ball_worse)}$',
                           f'it improves ${int(ms.ball_better)}$ and damages\n${int(ms.ball_worse)}$'),
    ('scenario',           f'gives ${ms.scen_mean:.1f}$\npercent against ${ms.scen_mean_r2:.1f}$',
                           f'gives ${ms.scen_mean:.1f}$\npercent against ${ms.scen_mean_r2:.1f}$'),
    ('matched criterion',  f'excess of ${ms.rel_rel_exc:.1f}$ percent',
                           f'excess of ${ms.rel_rel_exc:.1f}$ percent'),
    ('matched mismatch',   f'is\n${ms.rel_rel_sdr:.2f}$', f'is\n${ms.rel_rel_sdr:.2f}$'),
    ('MAPE relative',      f'mismatch is ${ms.rel_mape_sdr:.2f}$ and\nits excess ${ms.rel_mape_exc:.1f}$ percent',
                           f'mismatch is ${ms.rel_mape_sdr:.2f}$ and\nits excess ${ms.rel_mape_exc:.1f}$ percent'),
    ('MAPE top decile',    f'rises to\n${ms.top_mape_sdr:.2f}$', f'rises to\n${ms.top_mape_sdr:.2f}$'),
    ('RR one in ten',      'exceeds a quarter in one problem in\nten',
                           'exceeds a quarter in one problem in\nten' if abs(100*(rr.rr>.25).mean()-10) < 2 else f'{100*(rr.rr>.25).mean():.1f}%'),
    # -------------------------------------------- ten settings (section 4.2)
    ('table pairs',        '246 distinct',           f'{int(r2.n_pairs.sum())} distinct'),
    ('R2 mean',            '$25.3$ percent',         f'${r2.inv_pct.mean():.1f}$ percent'),
    ('R2 tight',           '$11.9$ percent',         f'${r2[r2.sd_r<1].inv_pct.mean():.1f}$ percent'),
    ('R2 wide',            '$34.2$ percent',         f'${r2[r2.sd_r>1.7].inv_pct.mean():.1f}$ percent'),
    ('continuous spearman', '$0.55$ under a cost rising in the outcome and $0.80$ under relative-error weighting',
                           '${:.2f}$ under a cost rising in the outcome and ${:.2f}$ under relative-error weighting'.format(*cont)),
    ('pooled certified',  f'none of\n${(sc.e_mu.abs()/sc.sd_d >= sc.sd_r).sum() + int(le.certified_pairs):,}$ comparisons reverses'.replace(',', '{,}'),
                          f'none of\n${(sc.e_mu.abs()/sc.sd_d >= sc.sd_r).sum() + int(le.certified_pairs):,}$ comparisons reverses'.replace(',', '{,}')),
    ('held-out abstract', '1 of 1{,}180 comparisons it certifies reverses, against 24.2\npercent of others',
     f'{ho.rev_out[ho_c].sum()} of {ho_c.sum():,} comparisons it certifies reverses, against {100*ho.rev_out[~ho_c].mean():.1f}\npercent of others'.replace(',', '{,}', 1)),
    ('encoded share abstract', 'that gap holds 2.1 percent of how they differ. Deployments act on the other 97.9 percent',
     f'that gap holds {100*enc.median():.1f} percent of how they differ. Deployments act on the other {100-round(100*enc.median(),1):.1f} percent'),
    ('suite resolved',    f'${100*inf.sm.mean():.1f}$ percent', f'${100*inf.sm.mean():.1f}$ percent'),
    ('LLM resolved pair',  f'${100*li.sm.mean():.1f}$\nagainst ${100*inf.sm.mean():.1f}$ percent',
                           f'${100*li.sm.mean():.1f}$ against ${100*inf.sm.mean():.1f}$ percent'),
    ('neural classes',     f'classical pairs reverse at ${100*nr[nr.neural==0].inverted.mean():.1f}$ percent, mixed pairs at ${100*nr[nr.neural==1].inverted.mean():.1f}$ and pairs of networks at ${100*nr[nr.neural==2].inverted.mean():.1f}$',
                           f'classical pairs reverse at ${100*nr[nr.neural==0].inverted.mean():.1f}$ percent, mixed pairs at ${100*nr[nr.neural==1].inverted.mean():.1f}$ and pairs of networks at ${100*nr[nr.neural==2].inverted.mean():.1f}$'),
    ('iia drops',         f'moves its answer in ${100*ii.flips_w.sum()/ii.drops_w.sum():.1f}$ percent of drops',
                          f'moves its answer in ${100*ii.flips_w.sum()/ii.drops_w.sum():.1f}$ percent of drops'),
    # ------------------------- winners and certificate on the leaderboard (u50)
    ('LLM winners',       'select three different best models' if le.distinct_winners == 3 else 'WRONG COUNT',
                          'select three different best models' if le.distinct_winners == 3 else 'WRONG COUNT'),
    ('LLM certified',     f'of the ${int(le.certified_pairs)}$ certified', f'of the ${int(le.certified_pairs)}$ certified'),
    ('LLM cert reverse',  'certified none reverses', 'certified none reverses' if le.certified_reversals == 0 else 'SOME REVERSE'),
    ('LLM exposed rate',  f'${le.exposed_reversal_pct:.1f}$ percent of the rest do',
                          f'${le.exposed_reversal_pct:.1f}$ percent of the rest do'),
    # ------------------------------------------------ the model list (u53)
    ('model count',       f'names all ${len(lm)}$ models', f'names all ${len(lm)}$ models'),
    ('model range',       f'the widest range is ${int((lm.worst-lm.best).max())}$ places',
                          f'the widest range is ${int((lm.worst-lm.best).max())}$ places'),
    ('model none fixed',  'Every model changes rank across them',
                          'Every model changes rank across them' if (lm.worst != lm.best).all() else 'SOME FIXED'),
    # --------------------------------------- two readings of one table (u52)
    ('two readings rate',  f'order ${tr.reversed_pct:.1f}$ percent of the comparisons', f'order ${tr.reversed_pct:.1f}$ percent of the comparisons'),
    ('two readings models', f'among its ${tr.models:,}$ archived models'.replace(',', '{,}'), f'among its ${tr.models:,}$ archived models'.replace(',', '{,}')),
    ('two readings sd_r',  f'$\\sd_\\mu(r)={tr.sd_r:.2f}$ is the smallest in this paper', f'$\\sd_\\mu(r)={tr.sd_r:.2f}$ is the smallest in this paper'),
    ('two readings pairs', f'Over ${tr.pairs:,}$ comparisons'.replace(',', '{,}'), f'Over ${tr.pairs:,}$ comparisons'.replace(',', '{,}')),
    ('two readings spearman', f'Spearman correlation of ${tr.spearman:.4f}$', f'Spearman correlation of ${tr.spearman:.4f}$'),
    ('crowd saturation',  f'falls from ${tr.kstar_K2:.2f}$ at $K=2$ to ${tr.kstar_K34:.2f}$ at $K=34$',
                          f'falls from ${tr.kstar_K2:.2f}$ at $K=2$ to ${tr.kstar_K34:.2f}$ at $K=34$'),
    ('crowd settles',     f'settles near ${tr.kstar_Kmax:.2f}$', f'settles near ${tr.kstar_Kmax:.2f}$'),
    ('crowd slope full',  f'slope of ${tr.crowd_slope_full:.2f}$ against ${tr.crowd_slope_pred:.2f}$ predicted',
                          f'slope of ${tr.crowd_slope_full:.2f}$ against ${tr.crowd_slope_pred:.2f}$ predicted'),
    ('two readings open',  f'${tr.open_top100_pct:.1f}$ percent of the pairs among the top hundred',
                           f'${tr.open_top100_pct:.1f}$ percent of the pairs among the top hundred'),
    ('two readings rank19', 'could stand anywhere from ninth to twenty-ninth',
                           'could stand anywhere from ninth to twenty-ninth' if (tr.rank19_best, tr.rank19_worst) == (9, 29) else f'{tr.rank19_best}-{tr.rank19_worst}'),
    ('two readings fit',   f'at a correlation of ${tr.reweight_fit:.4f}$', f'at a correlation of ${tr.reweight_fit:.4f}$'),
    # ---------------------------------- rebuilt against published ranking (u51)
    ('LLM spearman',       f'Spearman correlation of ${ra.spearman:.3f}$', f'Spearman correlation of ${ra.spearman:.3f}$'),
    ('LLM kendall',        f'Kendall rank correlation of ${ra.kendall:.3f}$', f'Kendall rank correlation of ${ra.kendall:.3f}$'),
    ('LLM same first',     'variants change places',
                           'variants change places' if not ra.same_first else 'SAME MODEL FIRST'),
    # ------------------------------------------- leaderboard spectrum (u50)
    ('LLM entropy bits',   f'$h(D)={le.h_bits:.2f}$ bits', f'$h(D)={le.h_bits:.2f}$ bits'),
    ('LLM eff dirs',       f'$2^{{h(D)}}={le.effective_directions:.1f}$ effective directions',
                           f'$2^{{h(D)}}={le.effective_directions:.1f}$ effective directions'),
    ('LLM leading share',  f'the leading one carrying ${le.leading_energy_pct:.1f}$ percent and the first five ${le.first_five_energy_pct:.1f}$',
                           f'the leading one carrying ${le.leading_energy_pct:.1f}$ percent and the first five ${le.first_five_energy_pct:.1f}$'),
    ('LLM h ratio',        f'${le.h_over_log2q:.2f}$ on the leaderboard', f'${le.h_over_log2q:.2f}$ on the leaderboard'),
    ('LLM loadings',       f'are ${le.loading_bbh:.2f}$ on the', f'are ${le.loading_bbh:.2f}$ on the'),
    ('LLM loadings math',  f'${le.loading_math:.2f}$ on the seven MATH subtasks and ${le.loading_gpqa:.2f}$',
                           f'${le.loading_math:.2f}$ on the seven MATH subtasks and ${le.loading_gpqa:.2f}$'),
    ('ten-row spearman',   'rate is $0.75$',         f'rate is ${cd.sd_r.corr(cd.inv_pct, method="spearman"):.2f}$'),
    ('diabetes decile',    'reverses $47.6$ percent', f'reverses ${cd[(cd.dataset=="diabetes")&(cd.deployment=="top decile of outcome")].inv_pct.iloc[0]:.1f}$ percent'),
    ('option row',         '$26.5$ percent of $162$', f'${cd[cd.domain=="Option pricing"].inv_pct.iloc[0]:.1f}$ percent of ${int(cd[cd.domain=="Option pricing"].n_pairs.iloc[0])}$'),
    ('option sd_r',        f'${opt.sd_r.median():.2f}$', f'${cd[cd.domain=="Option pricing"].sd_r.iloc[0]:.2f}$'),
    ('option range',       f'run from ${opt.sd_r.min():.2f}$ to ${opt.sd_r.max():.2f}$',
                           f'run from ${opt.sd_r.min():.2f}$ to ${opt.sd_r.max():.2f}$'),
    ('panel widest',       f'$\\sd_\\mu(r)={opt.sd_r.max():.1f}$', f'$\\sd_\\mu(r)={opt.sd_r.max():.1f}$'),
    ('panel narrowest',    f'narrowest, ${opt.sd_r.min():.1f}$', f'narrowest, ${opt.sd_r.min():.1f}$'),
    ('panel pairs',        f'all ${len(pd.read_csv(R/"option_rho.csv"))}$ pairs', f'all ${len(pd.read_csv(R/"option_rho.csv"))}$ pairs'),
    ('option widest',      'the widest in the table',
                           'the widest in the table' if cd.sd_r.idxmax() == cd[cd.domain=="Option pricing"].index[0] else 'NOT widest'),
    # ------------------------------------------- controlled grid (section 7)
    ('grid rho .6 tight',  'only $22.2$',            f'only ${G[(0.3,0.6)]:.1f}$'),
    ('grid rho 0 wide',    '$12.7$ percent',         f'${G[(3.0,0.0)]:.1f}$ percent'),
    # ------------------------------------------------ regret bound (Thm 2)
    ('regret slack',       'factor of $47$',         f'factor of ${np.median(pos.bound/pos.regret):.0f}$'),
    ('regret draws',       'the $419$ draws',        f'the ${len(pos)}$ draws'),
    ('regret violations',  'holds over all $3{,}000$', 'holds over all $3{,}000$' if (pos.regret <= pos.bound + 1e-12).all() else 'VIOLATED'),
    # ------------------------------------------------------- NMI (section 6)
    ('NMI limit ratio',    'settling at $13.7$',     f'settling at ${lim:.1f}$'),
    ('NMI slope',          r'\approx 4\,\sd_\mu(r)', r'\approx 4\,\sd_\mu(r)' if 3.5 <= slope < 4.5 else f'slope {slope:.2f}'),
    ('NMI threshold',      r'\lesssim 0.025',        r'\lesssim 0.025' if abs(.10/slope - .025) < .004 else f'{.10/slope:.3f}'),
    ('NMI at 1.99',        'remainder is $86$',      f'remainder is ${100*at199.rel_med:.0f}$'),
    ('NMI reversals',      'eighteen reversals where six', f'eighteen reversals where six' if (at199.rev_pred, at199.rev_actual) == (18, 6) else f'{int(at199.rev_pred)}/{int(at199.rev_actual)}'),
    ('NMI factor five',    'by a factor of five',    'by a factor of five' if abs(cd.sd_r.min()/.025 - 5) < 1 else f'{cd.sd_r.min()/.025:.1f}'),
    # ---------------------------------------------- where it happens, by deployment
    ('top decile turns',   '$16.7$ percent of comparisons', f"${100*sc[sc.deployment=='top decile of outcome'].inverted.mean():.1f}$ percent of comparisons"),
    ('relative turns',     '$15.4$ percent of comparisons', f"${100*sc[sc.deployment=='relative-error weighted'].inverted.mean():.1f}$ percent of comparisons"),
    ('smooth cost turns',  '$10.2$ for a cost rising smoothly', f"${100*sc[sc.deployment=='cost rising in outcome'].inverted.mean():.1f}$ for a cost rising smoothly"),
    # ---------------------------------------------- when the gap bites
    ('certified pairs',    'the $1{,}229$ comparisons', f'the ${(sc.e_mu.abs()/sc.sd_d >= sc.sd_r).sum():,}$ comparisons'.replace(',', '{,}')),
    ('certified reverse',  'that meet it, none reverses', 'that meet it, none reverses' if sc[sc.e_mu.abs()/sc.sd_d >= sc.sd_r].inverted.sum()==0 else 'SOME REVERSE'),
    ('exposed reverse',    '$15.1$ percent reverse',     f'${100*sc[sc.e_mu.abs()/sc.sd_d < sc.sd_r].inverted.mean():.1f}$ percent reverse'),
    # -------------------------------------------- what a benchmark sees (u20)
    ('visible share',      '$0.37$',                 f'${(vm.visible/vm.sd_r).median():.2f}$'),
    ('effective dirs',     '$6.1$ of nine',          f'${vm.eff_dim.median():.1f}$ of nine'),
    ('encoded share',      '$2.1$ percent',          f'${100*enc.median():.1f}$ percent'),
    ('encoded below one',  'On $36$ percent of pairs the encoded share', f'On ${100*(enc<0.01).mean():.0f}$ percent of pairs the encoded share'),
    ('median pair radius', 'the median $\\kappa_{fg}$ is $0.145$', f'the median $\\kappa_{{fg}}$ is ${kd.median():.3f}$'),
    ('discarded share',    '$97.9$ percent in directions', f'${100-100*enc.median():.1f}$ percent in directions'),
    ('visible vs cost',    '$0.55$ against $0.27$',  f'${spearmanr(vpos.visible, vpos.rr)[0]:.2f}$ against ${spearmanr(vpos.sd_r, vpos.rr)[0]:.2f}$'),
    ('costly problems',    'Over the $194$ selection problems that incur a cost', f'Over the ${len(vpos)}$ selection problems that incur a cost'),
    ('exposed pairs',      '$96.7$ percent of pairs have a radius below the median',
                           f'${100*(kd < vm.visible.median()).mean():.1f}$ percent of pairs have a radius below the median'),
    ('median visible',     'visible mismatch of a deployment, which is $0.62$',
                           f'visible mismatch of a deployment, which is ${vm.visible.median():.2f}$'),
    ('visible coverage',   'covers $497$ of the $498$', f'covers ${len(vm)}$ of the ${len(rr)}$'),
    ('visible quartiles',  'from $1.1$ to $29.4$',   f'from ${100*vq.mean().iloc[0]:.1f}$ to ${100*vq.mean().iloc[-1]:.1f}$'),
    ('visible ninetieth',  'from $2.9$ to $92$',     f'from ${100*vq.quantile(.9).iloc[0]:.1f}$ to ${100*vq.quantile(.9).iloc[-1]:.0f}$'),
    # ------------------------------------- the minimax theorem beyond its condition (u55)
    ('minimax cond suite', "it holds for $1.7$ percent of the suite's comparisons",
                           f"it holds for ${mxc['suite']:.1f}$ percent of the suite's comparisons"),
    ('minimax cond llm',   'and for none on the leaderboard',
                           'and for none on the leaderboard' if mxc['leaderboard'] == 0 else f"{mxc['leaderboard']:.1f}"),
    ('minimax suite',      'still the minimax choice in $90.1$ percent on the suite',
                           f"still the minimax choice in ${mxe['suite']:.1f}$ percent on the suite"),
    ('minimax llm',        'and $95.7$ percent on the leaderboard',
                           f"and ${mxe['leaderboard']:.1f}$ percent on the leaderboard"),
    # ------------------------------------------- headline rates, resampled by dataset (u57)
    ('cluster reversal',   'the reversal rate of $14.3$ percent has interval $[11.7,17.0]$',
     'the reversal rate of ${:.1f}$ percent has interval $[{:.1f},{:.1f}]$'.format(*ci.loc['pair reversal', ['point', 'lo', 'hi']])),
    ('cluster selection',  'the selection error of $39.2$ percent has interval $[30.6,47.9]$',
     'the selection error of ${:.1f}$ percent has interval $[{:.1f},{:.1f}]$'.format(*ci.loc['selection error', ['point', 'lo', 'hi']])),
    # -------------------------------- the deployment regret test, corrected for selection (u68)
    ('drt draws',          'drew $9{,}960$ benchmarks of half its size',
                           f'drew ${len(drt_h):,}$ benchmarks of half its size'.replace(',', '{,}')),
    ('drt chosen row',     'covered a positive regret in only $92.7$ percent of tests',
                           f'covered a positive regret in only ${100*(drt_h.rr_pop[drt_pos] <= drt_h.rr_ucb_old[drt_pos] + 1e-12).mean():.1f}$ percent of tests'),
    ('drt coverage',       'covered a positive regret ratio in $97.5$ percent of them',
                           f'covered a positive regret ratio in ${100*(drt_h.rr_pop[drt_pos] <= drt_h.rr_ucb_sel[drt_pos] + 1e-12).mean():.1f}$ percent of them'),
    ('drt verdicts',       'the test certified $18.9$ percent of choices and found $2.5$',
                           f"the test certified ${100*drt_h['cert_0.1'].mean():.1f}$ percent of choices and found ${100*drt_h['exceed_0.1'].mean():.1f}$"),
    ('drt wrong cert',     'at most $0.6$ percent were certified',
                           f'at most ${100*drt_wrong:.1f}$ percent were certified'),
    # ------------------------------------------------------- two real shifts (u59 to u62)
    ('real certified',     'Of the $730$ census comparisons certified, none reversed, against $11.0$ percent',
     f'Of the ${rs_c.sum()}$ census comparisons certified, none reversed, against ${100*rs_ap.reversed_true[~rs_c].mean():.1f}$ percent'
     if rs_ap.reversed_true[rs_c].sum() == 0 else 'some certified comparison reversed'),
    ('real beijing wrong', 'not the best model in $70.8$ percent of', f'not the best model in ${100*(1-rs_bq.opt_true.mean()):.1f}$ percent of'),
    ('real beijing max',   'its excess reached $61.4$ percent', f'its excess reached ${100*rs_bq.rr_true.max():.1f}$ percent'),
    ('real beijing ten',   'learners alone, the share is $30.6$ percent', f'learners alone, the share is ${100*(1-rs_bq10.opt_true.mean()):.1f}$ percent'),
    ('real beijing bins',  '$3.0$ percent of comparisons at one tenth or more reversed,\nagainst $21.6$ percent',
     f'${100*rs_bp.reversed_true[rs_bp.ratio >= .1].mean():.1f}$ percent of comparisons at one tenth or more reversed,\nagainst ${100*rs_bp.reversed_true[rs_bp.ratio < .1].mean():.1f}$ percent'),
    ('real acs spot',      'the same test flagged at most $0.5$ percent', f"the same test flagged at most ${rs_size(rs_as, 'reject'):.1f}$ percent"),
    ('real bj spot',       'it flagged at most\n$1.7$ percent', f"it flagged at most\n${rs_size(rs_bs, 'reject'):.1f}$ percent"),
    ('real bj cover',      'held in at least $96.3$ percent', f'held in at least ${rs_cov:.1f}$ percent'),
    ('rule real beijing',  'mean regret ratio from $8.4$ to $12.4$ percent in Beijing',
     f"mean regret ratio from ${rl_m('beijing', 'rr_criterion'):.1f}$ to ${rl_m('beijing', 'rr_rule_0.05'):.1f}$ percent in Beijing"),
    ('rule real census',   'from $0.6$ to\n$2.1$ percent in the census',
     f"from ${rl_m('census', 'rr_criterion'):.1f}$ to\n${rl_m('census', 'rr_rule_0.05'):.1f}$ percent in the census"),
    ('rule real reweight', 'at $8.8$ and $0.6$ percent',
     f"at ${rl_m('beijing', 'rr_reweighted'):.1f}$ and ${rl_m('census', 'rr_reweighted'):.1f}$ percent"),
    ('rule real corr',     'median correlation of $0.012$', f'median correlation of ${rlc.abs_corr.median():.3f}$'),
    ('rule real corr SI',  'median correlation of $0.011$\nin the census and $0.017$ in Beijing',
     f"median correlation of ${rlc[rlc.case.str.startswith('census')].abs_corr.median():.3f}$\nin the census and ${rlc[rlc.case.str.startswith('beijing')].abs_corr.median():.3f}$ in Beijing"),
    ('rule real ends',     'recovers it in $46$ to $52$ percent of\npairs, and the mean weight of each end\'s tenth in $76$ to $88$ percent',
     f"recovers it in ${100*rl.groupby('case').one_record.mean().min():.0f}$ to ${100*rl.groupby('case').one_record.mean().max():.0f}$ percent of\npairs, and the mean weight of each end's tenth in ${100*rl.groupby('case').group.mean().min():.0f}$ to ${100*rl.groupby('case').group.mean().max():.0f}$ percent"),
    ('boards abstract',    'guaranteed under none of 28 uses, not even the mildest',
     f'guaranteed under none of {lb_none} uses, not even the mildest'
     if all(lbk[x].kappa_star[0] < lbk[x].sd_r.min() for x in ('openllm', 'finance', 'livebench')) else 'some board certifies its leader under the mildest use'),
    ('boards radius',      'The leader\'s stability radius is $0.024$ on\nLiveBench and $0.055$ in finance, against $0.027$ on the Open LLM Leaderboard',
     f"The leader's stability radius is ${lbk['livebench'].kappa_star[0]:.3f}$ on\nLiveBench and ${lbk['finance'].kappa_star[0]:.3f}$ in finance, against ${lbk['openllm'].kappa_star[0]:.3f}$ on the Open LLM Leaderboard"),
    ('boards best',        'only $2$ of $9$ deployments on LiveBench and $2$ of\n$11$ in finance',
     f"only ${int(lbs.loc['livebench','leader_best_in'])}$ of ${int(lbs.loc['livebench','deployments'])}$ deployments on LiveBench and ${int(lbs.loc['finance','leader_best_in'])}$ of\n${int(lbs.loc['finance','deployments'])}$ in finance"),
    ('boards medical',     'radius of $1.10$ and\nstays best under $5$ of $6$ deployments',
     f"radius of ${lbk['medical'].kappa_star[0]:.2f}$ and\nstays best under ${int(lbs.loc['medical','leader_best_in'])}$ of ${int(lbs.loc['medical','deployments'])}$ deployments"),
    ('boards models',      '($185$ models, $9$ tasks)', f"(${int(lbs.loc['medical','models'])}$ models, ${int(lbs.loc['medical','tasks'])}$ tasks)"),
    ('boards regret',      'gives no wrong verdict in $34$\ndeployments. With $9$ to $39$ tasks it leaves $27$ of them undetermined',
     (f"gives no wrong verdict in ${len(lbr)}$\ndeployments. With ${lbr.tasks.min()}$ to ${lbr.tasks.max()}$ tasks it leaves ${(lbr.verdict_10=='undetermined').sum()}$ of them undetermined"
      if (((lbr.verdict_10=='certified')&(lbr.rr_exact>.10))|((lbr.verdict_10=='exceeds')&(lbr.rr_exact<=.10))|((lbr.verdict_05=='certified')&(lbr.rr_exact>.05))|((lbr.verdict_05=='exceeds')&(lbr.rr_exact<=.05))).sum()==0 else 'a wrong verdict')),
    # ------------------------------------------- research question 4, the six answers (u44, u46)
    ('six mean',           'lower the mean excess from $4.7$ to $2.9$ percent of the candidate range, and the ninetieth percentile from $17.4$ to $8.7$',
     f'lower the mean excess from ${100*st.r2.mean():.1f}$ to ${100*st.m4.mean():.1f}$ percent of the candidate range, and the ninetieth percentile from ${100*st.r2.quantile(.9):.1f}$ to ${100*st.m4.quantile(.9):.1f}$'),
    ('six counts',         'They improve $85$ of the $498$ selection problems and damage $22$',
     f'They improve ${int((st.m4 < st.r2-1e-12).sum())}$ of the ${len(st)}$ selection problems and damage ${int((st.m4 > st.r2+1e-12).sum())}$'),
    ('six beat 45',        'Six answers also beat forty-five', 'Six answers also beat forty-five' if st.m4.mean() < st.m10.mean() else 'forty-five better'),
    ('six rr',             'from $11.2$ to $7.1$ percent of the benchmark choice',
     f'from ${100*rsg.rr_r2.mean():.1f}$ to ${100*rsg.rr_rule.mean():.1f}$ percent of the benchmark choice'),
    ('six quartiles',      'In the widest quartile the cost falls from $22.7$ to $10.4$ percent, more than half; in the narrowest, where there is least to recover, from $1.2$ to $1.0$',
     f'In the widest quartile the cost falls from ${rsq.rr_r2.iloc[3]:.1f}$ to ${rsq.rr_rule.iloc[3]:.1f}$ percent, ' + ('more than half' if rsq.rr_rule.iloc[3] < rsq.rr_r2.iloc[3] / 2 else 'NOT more than half')
     + f'; in the narrowest, where there is least to recover, from ${rsq.rr_r2.iloc[0]:.1f}$ to ${rsq.rr_rule.iloc[0]:.1f}$'),
    ('six noise',          "flipped at random with probability $0.4$, the mean excess rises from $2.9$ to $4.6$ percent, still below the criterion's $4.7$. Only at $0.5$",
     (f"flipped at random with probability $0.4$, the mean excess rises from ${sn.loc[0.0].iloc[0]:.1f}$ to ${sn.loc[0.4].iloc[0]:.1f}$ percent, still below the criterion's ${100*st.r2.mean():.1f}$. Only at $0.5$"
      if sn.loc[0.4].iloc[0] < 100 * st.r2.mean() < sn.loc[0.5].iloc[0] else 'the crossing is not at 0.5')),
    ('tolerance rule',     'The mean excess runs from $5.0$ percent at two percent of the benchmark span to $12.7$ at thirty-five',
     f"The mean excess runs from ${100*fr['exc_0.02'].mean():.1f}$ percent at two percent of the benchmark span to ${100*fr['exc_0.35'].mean():.1f}$ at thirty-five"),
    # ---------------------------- headline numbers, in the wording of the NMI manuscript
    ('nmi suite comparisons', 'problems and $22{,}410$ model-pair comparisons', f'problems and ${len(sc):,}$ model-pair comparisons'.replace(',', '{,}')),
    ('nmi suite reversal', 'puts $14.3$ percent of the comparisons opposite', f'puts ${100*sc.inverted.mean():.1f}$ percent of the comparisons opposite'),
    ('nmi selection error', 'in $39.2$ percent of the $498$ selection problems', f'in ${100*sel.wrong.mean():.1f}$ percent of the ${len(sel)}$ selection problems'),
    ('nmi resolved rate',  'the rate is $3.9$ percent, a lower bound', f'the rate is ${100*bo.rev.mean():.1f}$ percent, a lower bound'),
    ('nmi resolved pair',  'Restricted to resolved pairs the figures are $6.6$ and $3.9$ percent', f'Restricted to resolved pairs the figures are ${100*lb.rev.mean():.1f}$ and ${100*bo.rev.mean():.1f}$ percent'),
    ('nmi resolves more',  'resolves more of its pairs, $90.4$ against $74.5$ percent', f'resolves more of its pairs, ${100*li.sm.mean():.1f}$ against ${100*inf.sm.mean():.1f}$ percent'),
    ('nmi rr p90',         'reaching $25.5$ percent at the ninetieth percentile', f'reaching ${100*rr.rr.quantile(.9):.1f}$ percent at the ninetieth percentile'),
    ('nmi crowding',       'The predicted slopes, $-0.88$ for the suite and $-0.83$ for the leaderboard, sit close to the measured $-0.75$ and $-0.78$',
     f'The predicted slopes, ${ce.loc["regression suite","predicted"]:.2f}$ for the suite and ${ce.loc["language models","predicted"]:.2f}$ for the leaderboard, sit close to the measured ${ce.loc["regression suite","observed"]:.2f}$ and ${ce.loc["language models","observed"]:.2f}$'),
    ('nmi ball counts',    'It improves $33$ of the $498$ problems and damages $347$', f'It improves ${int(ms.ball_better)}$ of the ${len(sel)}$ problems and damages ${int(ms.ball_worse)}$'),
    ('nmi grid tight',     'a tight ratio reverses $22.2$ percent of pairs', f'a tight ratio reverses ${G[(0.3,0.6)]:.1f}$ percent of pairs'),
    ('nmi option panels',  'from $3.3$ to $26.3$, and their within-domain', f'from ${opt.sd_r.min():.1f}$ to ${opt.sd_r.max():.1f}$, and their within-domain'),
    ('iia main',           'changes its answer in $5.6$ percent of drops, and in $46.6$ percent of problems',
     f'changes its answer in ${100*ii.flips_w.sum()/ii.drops_w.sum():.1f}$ percent of drops, and in ${100*ii.any_w.mean():.1f}$ percent of problems'),
    ('iia SI',             'moves its answer in $5.6$ percent of drops, and in $46.6$ percent of problems',
     f'moves its answer in ${100*ii.flips_w.sum()/ii.drops_w.sum():.1f}$ percent of drops, and in ${100*ii.any_w.mean():.1f}$ percent of problems'),
    ('rq3 r2 relative',    'against $14.4$ percent for $R^{2}$', f"against ${100*rr[rr.deployment=='relative-error weighted'].rr.mean():.1f}$ percent for $R^{{2}}$"),
    ('suite median corr',  'below the suite median of $0.076$', f'below the suite median of ${((sc.e_nu-sc.e_mu).abs()/(sc.sd_r*sc.sd_d)).median():.3f}$'),
    ('holdout main',       '$1$ of $1{,}180$ comparisons reverses on the other half, against $24.2$ percent',
     f'${ho.rev_out[ho_c].sum()}$ of ${ho_c.sum():,}$ comparisons reverses on the other half, against ${100*ho.rev_out[~ho_c].mean():.1f}$ percent'.replace(',', '{,}', 1)),
    ('distinct pairs',     'Over the $4{,}725$ distinct pairs', f'Over the ${sc.drop_duplicates(["dataset", "seed", "pair"]).shape[0]:,}$ distinct pairs'.replace(',', '{,}')),
    ('llm certified',      'on its own $23{,}408$ comparisons: of the $582$ certified none reverses, and $14.9$ percent of the rest do',
     (lambda q: f'on its own ${len(lp):,}$ comparisons: of the ${int(q.sum())}$ certified none reverses, and ${100*lp.inverted[~q].mean():.1f}$ percent of the rest do'.replace(',', '{,}', 1)
      if lp.inverted[q].sum() == 0 else 'a certified pair reversed')(lp.e_mu.abs() / lp.sd_d >= lp.sd_r)),
    ('real bj iid',        'flagged up to $12.4$ percent', f"flagged up to ${rs_size(rs_bs, 'reject_iid'):.1f}$ percent"),
]

# Items written for an earlier manuscript (main.tex), whose wording the NMI manuscript does not
# contain.  They are kept for the record and neither run nor counted.
LEGACY = ['LLM cond rev', 'LLM range', 'LLM range hi', 'LLM regret hi', 'LLM regret lo', 'LLM resolves', 'LLM zero of eight', 'MAPE relative', 'MAPE top decile', 'NMI at 1.99', 'NMI factor five', 'NMI limit ratio', 'NMI reversals', 'NMI slope', 'NMI threshold', 'R2 tight', 'R2 wide', 'RR one in ten', 'RR p95', 'RR q3', 'RR quartile p90', 'ball counts', 'both decisive', 'conditional rev', 'crowd obs llm', 'crowd pred reg', 'datasets affected', 'diabetes decile', 'grid rho .6 tight', 'matched criterion', 'matched mismatch', 'option range', 'option row', 'option widest', 'panel narrowest', 'panel pairs', 'panel widest', 'per-dataset median', 'per-dataset range', 'quintile high', 'quintile low', 'quintile low sd', 'regret draws', 'rev by n high', 'rev by n low', 'selection count', 'selection wrong', 'spearman scale', 'suite pairs', 'suite reversal', 'table pairs', 'unresolved']
C = [c for c in C if c[0] not in LEGACY]
# the paper and its Supplementary Information are the only text audited
tex = norm(''.join((P / f).read_text() for f in ('NMI_V3.tex', 'NMI_V3_SI.tex') if (P / f).exists()))

# the audit's own size, as the paper states it in digits (Code availability) and in words
_ones = 'zero one two three four five six seven eight nine'.split()
_teens = 'ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen'.split()
_tens = 'zero ten twenty thirty forty fifty sixty seventy eighty ninety'.split()
def _words(k):
    h, r = divmod(k, 100); t, o = divmod(r, 10)
    rest = _teens[o] if t == 1 else (_tens[t] + ('-' + _ones[o] if o else '') if t else _ones[o])
    return (_ones[h] + ' hundred' + (f'\nand {rest}' if r else '')) if h else rest
_n = len(C) + 2
C.append(('audit count digits', f'each of the ${_n}$ numbers', f'each of the ${_n}$ numbers'))
C.append(('audit count words', f'{_words(_n).capitalize()} such checks', f'{_words(_n).capitalize()} such checks'))

bad = []
for name, intext, truth in C:
    intext, truth = norm(intext), norm(truth)
    if intext not in tex:
        bad.append(f'{name}: text does not contain "{intext}"')
    elif intext != truth:
        bad.append(f'{name}: text "{intext}" != data "{truth}"')
print(f'audit {len(C)} items: {len(C)-len(bad)} consistent, {len(bad)} stale')
for b in bad:
    print('  FAIL', b)
raise SystemExit(1 if bad else 0)
