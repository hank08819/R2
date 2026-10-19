"""u10_audit.py -- every number in the paper, checked against the CSV it came from.

Fails loudly if a figure in the text has drifted from the script that produced it.
Run after any rerun of u1, u5, u7, or the NMI scripts.
"""
import warnings; warnings.filterwarnings('ignore')
import json
import pandas as pd, numpy as np
from pathlib import Path
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / 'results' / 'universal'
import re
# The manuscript sits beside this tree in the working copy and one level up in the
# released package, so take the first directory that holds it.
P = next((d for d in (ROOT.parent / 'universal_metric', ROOT / 'paper', ROOT.parent, ROOT)
          if (d / 'NMI_V9.tex').exists()), ROOT.parent / 'universal_metric')
_src = [f for f in ('NMI_V9.tex', 'NMI_V9_SI.tex', 'main.tex', 'fig1.tex', 'table_cross_domain.tex')
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
rs_ap = pd.read_csv(R / 'acs_pairs.csv')                                                       # u59
rs_as = pd.read_csv(R / 'acs_spot_check.csv')                                                  # u60
cm_p = pd.read_csv(R / 'commute_pairs.csv'); cm_q = pd.read_csv(R / 'commute_problems.csv'); cm_s = pd.read_csv(R / 'commute_spot.csv')  # u80
cm_sum = pd.read_csv(R / 'commute_summary.csv'); cs_sum = pd.read_csv(R / 'conditional_shift_summary.csv')  # u81, u83
sh_full = pd.read_csv(R / 'sharp_radius.csv'); sh_ho = pd.read_csv(R / 'holdout_sharp.csv')  # u82, u84
wx_p = pd.read_csv(R / 'weather_pairs.csv'); wx_q = pd.read_csv(R / 'weather_problems.csv'); wx_s = pd.read_csv(R / 'weather_spot.csv'); wx_sum = pd.read_csv(R / 'weather_summary.csv')  # u86, u87
rt_acs = pd.read_csv(R / 'acs_problems_u78.csv')  # u78
rs_c = rs_ap.certified == 1
rs_size = lambda t, col: max(100 * u[col][u.opt_true == 1].mean() for _, u in t.groupby('n'))
rs_cov = min(100 * (u.rr_true[u.rr_true > 0] <= u.rr_ucb[u.rr_true > 0] + 1e-12).mean() for _, u in rs_as.groupby('n'))
rl = pd.concat([pd.read_csv(R / f'rule_real_{w}.csv') for w in ('census',)])          # u70
rl['fam'] = rl.case.str.split('-').str[0]
rlc = pd.concat([pd.read_csv(R / f'rule_real_corr_{w}.csv') for w in ('census',)])
rl_m = lambda fam, col: 100 * rl[rl.fam == fam][col].mean()
LB = R / 'leaderboards'                                                                    # u72, u73
lbs = pd.read_csv(LB / 'leaderboards_summary.csv').set_index('board')
lbk = {b: pd.read_csv(LB / f'{b}_selection.csv') for b in ['openllm', 'medical', 'finance', 'livebench']}
lbr = pd.read_csv(LB / 'leaderboards_regret_test.csv')
lb_none = sum(int(lbs.loc[b, 'deployments']) for b in ['openllm', 'finance', 'livebench'] if lbs.loc[b, 'leader_certified_in'] == 0)
_rsz = pd.read_csv(R / 'rule_size.csv')   # u75
_rts = pd.read_csv(R / 'regret_test_sel.csv'); _rt5 = _rts[_rts.frac == 0.5]; _und5 = _rt5[(_rt5['cert_0.1'] == 0) & (_rt5['exceed_0.1'] == 0)]
_spl = pd.read_csv(R / 'regret_test_split.csv'); _sp5 = _spl[_spl.frac == 0.5]; _EPS = (0.01, 0.02, 0.05, 0.1, 0.2)   # u74
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
    ('RR p90',             f'${100*rr.rr.quantile(.90):.1f}$ percent at the 90th percentile',
                           f'${100*rr.rr.quantile(.90):.1f}$ percent at the 90th percentile'),
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
    ('LLM models',         f'for ${nmod}$ models evaluated on all',
                           f'for ${nmod}$ models evaluated on all'),
    ('LLM reversal',       f'${100*lp.inverted.mean():.1f}$ percent reverse, almost the same rate',
                           f'${100*lp.inverted.mean():.1f}$ percent reverse, almost the same rate'),
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
    ('LLM question count', f'leaves the original leader ${100*ls_[ls_.deployment=="by question count"].regret_ratio.iloc[0]:.1f}$ percent behind the new winner',
                           f'leaves the original leader ${100*ls_[ls_.deployment=="by question count"].regret_ratio.iloc[0]:.1f}$ percent behind the new winner'),
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
    ('pooled certified',  f'${(sc.e_mu.abs()/sc.sd_d >= sc.sd_r).sum() + int(le.certified_pairs):,}$'.replace(',', '{,}') + ' certified comparisons reverses',
                          f'${(sc.e_mu.abs()/sc.sd_d >= sc.sd_r).sum() + int(le.certified_pairs):,}$'.replace(',', '{,}') + ' certified comparisons reverses'),
    ('held-out abstract', 'On held-out data, 1 of 1{,}180 certified comparisons reverses, versus 24.2 percent of uncertified comparisons',
     f'On held-out data, {ho.rev_out[ho_c].sum()} of ' + f'{ho_c.sum():,}'.replace(',', '{,}') + f' certified comparisons reverses, versus {100*ho.rev_out[~ho_c].mean():.1f} percent of uncertified comparisons'),
    ('encoded share results', 'The median pair has just $2.1$ percent of its second moment',
     'The median pair has just $' + f'{100*enc.median():.1f}' + '$ percent of its second moment'),
    ('suite resolved',    f'${100*inf.sm.mean():.1f}$ percent', f'${100*inf.sm.mean():.1f}$ percent'),
    ('LLM resolved pair',  f'${100*li.sm.mean():.1f}$\nagainst ${100*inf.sm.mean():.1f}$ percent',
                           f'${100*li.sm.mean():.1f}$ against ${100*inf.sm.mean():.1f}$ percent'),
    ('neural classes',     '$14.3$ percent among pairs of classical learners, $13.6$ percent for mixed pairs and $17.1$ percent for pairs of networks',
                           f'${100*nr[nr.neural==0].inverted.mean():.1f}$ percent among pairs of classical learners, ${100*nr[nr.neural==1].inverted.mean():.1f}$ percent for mixed pairs and ${100*nr[nr.neural==2].inverted.mean():.1f}$ percent for pairs of networks'),
    ('iia drops',         f'moves its answer in ${100*ii.flips_w.sum()/ii.drops_w.sum():.1f}$ percent of drops',
                          f'moves its answer in ${100*ii.flips_w.sum()/ii.drops_w.sum():.1f}$ percent of drops'),
    # ------------------------- winners and certificate on the leaderboard (u50)
    ('LLM certified',     f'${int(le.certified_pairs)}$ comparisons are certified', f'${int(le.certified_pairs)}$ comparisons are certified'),
    ('LLM cert reverse',  'its ordering cannot reverse', 'its ordering cannot reverse' if le.certified_reversals == 0 else 'SOME REVERSE'),
    ('LLM exposed rate',  f'${le.exposed_reversal_pct:.1f}$ percent of the exposed comparisons reverse',
                          f'${le.exposed_reversal_pct:.1f}$ percent of the exposed comparisons reverse'),
    # ------------------------------------------------ the model list (u53)
    ('model count',       f'names all ${len(lm)}$ models', f'names all ${len(lm)}$ models'),
    ('model range',       f'the widest range is ${int((lm.worst-lm.best).max())}$ places',
                          f'the widest range is ${int((lm.worst-lm.best).max())}$ places'),
    ('model none fixed',  'Every model changes rank across them',
                          'Every model changes rank across them' if (lm.worst != lm.best).all() else 'SOME FIXED'),
    # --------------------------------------- two readings of one table (u52)
    ('two readings rate',  f'ordered oppositely ${tr.reversed_pct:.1f}$ percent of the time', f'ordered oppositely ${tr.reversed_pct:.1f}$ percent of the time'),
    ('two readings models', f'Fitted across the ${tr.models:,}$ archived models'.replace(',', '{,}'), f'Fitted across the ${tr.models:,}$ archived models'.replace(',', '{,}')),
    ('two readings sd_r',  f'a density ratio of dispersion $\\sd_\\mu(r)={tr.sd_r:.2f}$', f'a density ratio of dispersion $\\sd_\\mu(r)={tr.sd_r:.2f}$'),
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
    ('LLM loadings',       f"${le.loading_ifeval:.2f}$ on IFEval, ${le.loading_musr:.2f}$ on the three MuSR subtasks, ${le.loading_gpqa:.2f}$ on the three GPQA ones and ${le.loading_bbh:.2f}$ on the twenty-four BBH subtasks",
                           f"${le.loading_ifeval:.2f}$ on IFEval, ${le.loading_musr:.2f}$ on the three MuSR subtasks, ${le.loading_gpqa:.2f}$ on the three GPQA ones and ${le.loading_bbh:.2f}$ on the twenty-four BBH subtasks"),
    ('LLM dims discussion', f"${le.effective_directions:.1f}$ of $38$ for the $77$ leaderboard models", f"${le.effective_directions:.1f}$ of $38$ for the $77$ leaderboard models"),
    ('LLM h ratio',        f'${le.h_over_log2q:.2f}$ on the leaderboard', f'${le.h_over_log2q:.2f}$ on the leaderboard'),
    ('ten-row spearman',   'rank correlation $0.75$',         f'rank correlation ${cd.sd_r.corr(cd.inv_pct, method="spearman"):.2f}$'),
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
    ('certified pairs',    'none of the $1{,}229$ certified comparisons reverses', f'none of the ${(sc.e_mu.abs()/sc.sd_d >= sc.sd_r).sum():,}$ certified comparisons reverses'.replace(',', '{,}')),
    ('second split',       'five reversals among $1{,}282$ certified comparisons, against $19.7$ percent',
     '{} reversals among ${}$ certified comparisons, against ${:.1f}$ percent'.format(['zero','one','two','three','four','five','six','seven','eight','nine','ten'][sh_full.rev_out[sh_full.kappa_A >= sh_full.sd_rA].sum()], f'{(sh_full.kappa_A >= sh_full.sd_rA).sum():,}'.replace(',', '{,}'), 100*sh_full.rev_out[sh_full.kappa_A < sh_full.sd_rA].mean())),
    ('certified reverse',  'its ordering cannot reverse', 'its ordering cannot reverse' if sc[sc.e_mu.abs()/sc.sd_d >= sc.sd_r].inverted.sum()==0 else 'SOME REVERSE'),
    ('exposed reverse',    '$15.1$ percent of the remaining',     f'${100*sc[sc.e_mu.abs()/sc.sd_d < sc.sd_r].inverted.mean():.1f}$ percent of the remaining'),
    # -------------------------------------------- what a benchmark sees (u20)
    ('visible share',      '$0.39$',                 f'${(vm.visible/vm.sd_r).median():.2f}$'),
    ('effective dirs',     '$6.1$ of nine',          f'${vm.eff_dim.median():.1f}$ of nine'),
    ('encoded share',      '$2.1$ percent',          f'${100*enc.median():.1f}$ percent'),
    ('encoded below one',  'On $36$ percent of pairs the encoded share', f'On ${100*(enc<0.01).mean():.0f}$ percent of pairs the encoded share'),
    ('median pair radius', 'the median $\\kappa_{fg}$ is $0.145$', f'the median $\\kappa_{{fg}}$ is ${kd.median():.3f}$'),
    ('discarded share',    '$97.9$ percent in directions', f'${100-100*enc.median():.1f}$ percent in directions'),
    ('visible vs cost',    '$0.54$ and $0.27$',  f'${spearmanr(vpos.visible, vpos.rr)[0]:.2f}$ and ${spearmanr(vpos.sd_r, vpos.rr)[0]:.2f}$'),
    ('costly problems',    'Over the $194$ selection problems that incur a cost', f'Over the ${len(vpos)}$ selection problems that incur a cost'),
    ('exposed pairs',      '$97.3$ percent of pairs have a radius below the median',
                           f'${100*(kd < vm.visible.median()).mean():.1f}$ percent of pairs have a radius below the median'),
    ('median visible',     'visible mismatch of a deployment, which is $0.65$',
                           f'visible mismatch of a deployment, which is ${vm.visible.median():.2f}$'),
    ('visible coverage',   'covers $497$ of the $498$', f'covers ${len(vm)}$ of the ${len(rr)}$'),
    ('visible quartiles',  'from $1.1$ to $29.2$',   f'from ${100*vq.mean().iloc[0]:.1f}$ to ${100*vq.mean().iloc[-1]:.1f}$'),
    ('visible ninetieth',  'from $3.1$ to $92$',     f'from ${100*vq.quantile(.9).iloc[0]:.1f}$ to ${100*vq.quantile(.9).iloc[-1]:.0f}$'),
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
    ('drt draws',          'drew $9{,}960$ half-sized benchmark samples',
                           f'drew ${len(drt_h):,}$ half-sized benchmark samples'.replace(',', '{,}')),
    ('drt chosen row',     'compared with $92.7$ percent for inference restricted to the benchmark winner',
                           f'compared with ${100*(drt_h.rr_pop[drt_pos] <= drt_h.rr_ucb_old[drt_pos] + 1e-12).mean():.1f}$ percent for inference restricted to the benchmark winner'),
    ('drt coverage',       'covered positive population regret in $97.6$ percent of draws',
                           f'covered positive population regret in ${100*(drt_h.rr_pop[drt_pos] <= drt_h.rr_ucb_sel[drt_pos] + 1e-12).mean():.1f}$ percent of draws'),
    ('drt verdicts',       'the DRT certified $18.5$ percent of choices, found $2.5$',
                           f"the DRT certified ${100*drt_h['cert_0.1'].mean():.1f}$ percent of choices, found ${100*drt_h['exceed_0.1'].mean():.1f}$"),
    ('drt wrong cert',     'at most $0.6$ percent were certified',
                           f'at most ${100*drt_wrong:.1f}$ percent were certified'),
    # ------------------------------------------------------- two real shifts (u59 to u62)
    # ------------------------------------------------ two real census shifts (u59, u60, u80 to u84)
    ('real certified',     'None of $730$ certified income comparisons',
     f'None of ${rs_c.sum()}$ certified income comparisons' if rs_ap.reversed_true[rs_c].sum() == 0 else 'some certified comparison reversed'),
    ('real exposed rates', 'reversal rates among exposed comparisons are $11.0$ and $12.6$ percent',
     f'reversal rates among exposed comparisons are ${100*rs_ap.reversed_true[~rs_c].mean():.1f}$ and ${100*cm_p.reversed_true[cm_p.certified != 1].mean():.1f}$ percent'),
    ('real both certified', 'none of $881$ comparisons certified with weights fixed before deployment labels were read reverses',
     f'none of ${rs_c.sum() + (cm_p.certified == 1).sum()}$ comparisons certified with weights fixed before deployment labels were read reverses'
     if rs_ap.reversed_true[rs_c].sum() + cm_p.reversed_true[cm_p.certified == 1].sum() == 0 else 'some certified comparison reversed'),
    ('commute certified',  'or $151$ certified commute comparisons reverses',
     (lambda c: f'or ${c.sum()}$ certified commute comparisons reverses' if cm_p.reversed_true[c].sum() == 0 else 'a certified commute comparison reversed')(cm_p.certified == 1)),
    ('commute selection',  'suboptimal in $63.5$ percent of commute choices',
     (lambda q: f"suboptimal in ${100*(q.opt_true == 0).mean():.1f}$ percent of commute choices")(cm_q.drop_duplicates(['state', 'year', 'seed']))),
    ('real max regret',    'with regret ratios reaching $29.2$ and $215$ percent',
     f"with regret ratios reaching ${100*cm_q.rr_true.max():.1f}$ and ${100*wx_q.rr_true.max():.0f}$ percent"),
    ('real acs cover',     'Its bound holds in at\nleast $99.4$ percent of tests with positive regret', f'Its bound holds in at\nleast ${rs_cov:.1f}$ percent of tests with positive regret'),
    ('commute flag',       'Only $0.4$ percent of right choices were flagged',
     (lambda a: f"Only ${100*a.false_alarm_rule:.1f}$ percent of right choices were flagged" if (a.wrong_pass_rule < a.wrong_pass_no_rule and a.false_alarm_rule <= .05) else 'CRITERION NOT MET')
     (cm_sum[(cm_sum.eps == .05) & (cm_sum.n == 1000) & (cm_sum.weight == 'all')].iloc[0])),
    ('commute ten',        'At a $10$ percent tolerance, none of the costly choices passed',
     'At a $10$ percent tolerance, none of the costly choices passed' if cm_sum[(cm_sum.eps == .1) & (cm_sum.n == 1000) & (cm_sum.weight == 'all')].wrong_pass_rule.iloc[0] == 0 else 'SOME PASS'),
    ('cond shift D',       "about $5$ percent of benchmark risk for commute. It is $23$ to $33$ percent for income and $17$ to $38$ percent for weather",
     (lambda c, i, w: f"about ${100*c.D_median.median():.0f}$ percent of benchmark risk for commute. It is ${100*i.D_median.min():.0f}$ to ${100*i.D_median.max():.0f}$ percent for income and ${100*w.min():.0f}$ to ${100*w.max():.0f}$ percent for weather")
     (cs_sum[(cs_sum.case == 'census commute') & (cs_sum.eps == .05)], cs_sum[(cs_sum.case == 'census income') & (cs_sum.eps == .05)],
      (lambda p: p.assign(D=[(np.array(json.loads(a)) - np.array(json.loads(b))).max() / m for a, b, m in zip(p.th_true, p.th_hat, p.m_f)]).groupby('weight').D.median())(wx_q))),
    ('sharp full',         '$1{,}730$ comparisons, against $1{,}229$ for $\\kappa_{fg}$, with none reversing',
     (lambda c0, c1: '${}$ comparisons, against ${}$ for $\\kappa_{{fg}}$, with none reversing'.format(*[f'{x:,}'.replace(',', '{,}') for x in (c1.sum(), c0.sum())])
      if sh_full.rev_full[c1].sum() == 0 else 'a sharp certificate reversed')(sh_full.kappa >= sh_full.sd_r, sh_full.sharp >= sh_full.sd_r)),
    ('sharp holdout SI',   'it certifies $1{,}648$ comparisons, and $99.3$ percent keep their order, against $24.7$ percent reversing among the rest',
     'it certifies ${}$ comparisons, and ${:.1f}$ percent keep their order, against ${:.1f}$ percent reversing among the rest'.format(
         f'{(sh_ho.sharp_ratio >= 1).sum():,}'.replace(',', '{,}'), 100 - 100 * sh_ho.rev_out[sh_ho.sharp_ratio >= 1].mean(), 100 * sh_ho.rev_out[sh_ho.sharp_ratio < 1].mean())),
    ('rule real corr',     'median correlation of $0.011$ with a loss difference, against $0.076$ on the suite',
     f"median correlation of ${rlc.abs_corr.median():.3f}$ with a loss difference, against $0.076$ on the suite"),
    ('rule real ends',     'The mean weight of each end\'s tenth\nrecovers it in $76$ to $82$ percent of pairs, against $48$ to $52$ percent from one record per end',
     f"The mean weight of each end's tenth\nrecovers it in ${100*rl.groupby('case').group.mean().min():.0f}$ to ${100*rl.groupby('case').group.mean().max():.0f}$ percent of pairs, against ${100*rl.groupby('case').one_record.mean().min():.0f}$ to ${100*rl.groupby('case').one_record.mean().max():.0f}$ percent from one record per end"),
    # ------------------------------------------------ the pre-registered weather case (u86, u87)
    # ------------------------------------------------ rho_bar sweep (u44) and the summary card (u89)
    ('rho sweep',          'each $\\bar\\rho$ tested from $0.025$ to $0.077$ gives a mean excess of $3.8$ to $4.5$ percent',
     (lambda sg: f"each $\\bar\\rho$ tested from $0.025$ to $0.077$ gives a mean excess of ${100*min(sg['e0.025'].mean(), sg['e0.05'].mean(), sg['e0.077'].mean()):.1f}$ to ${100*max(sg['e0.025'].mean(), sg['e0.05'].mean(), sg['e0.077'].mean()):.1f}$ percent")(pd.read_csv(R / 'signed.csv'))),
    ('card share',         'In the strongest tenth (\\%) & $22.0$ & $17.1$',
     (lambda c: f"In the strongest tenth (\\%) & ${c['in top decile (%)'][0]:.1f}$ & ${c['in top decile (%)'][1]:.1f}$")(pd.read_csv(R / 'card_suite.csv'))),
    ('card strength',      'Strength (MPa), mean & $44.3$ & $42.7$',
     (lambda c: f"Strength (MPa), mean & ${c['strength (MPa)'][0]:.1f}$ & ${c['strength (MPa)'][1]:.1f}$")(pd.read_csv(R / 'card_suite.csv'))),
    ('sharp solar',        'Ten of the eleven reversals on the split of the main text are solar flare comparisons',
     (lambda h: (lambda r: f"{['No','One','Two','Three','Four','Five','Six','Seven','Eight','Nine','Ten'][int((r.dataset=='solar_flare').sum())]} of the {['zero','one','two','three','four','five','six','seven','eight','nine','ten','eleven'][len(r)]} reversals on the split of the main text are solar flare comparisons")(h[(h.sharp_ratio>=1)&(h.rev_out==1)]))(pd.read_csv(R / 'holdout_sharp.csv'))),
    ('sharp solar kfg',    'while $\\kappa_{fg}$ stayed between $0.004$ and $0.012$',
     (lambda r: f"while $\\kappa_{{fg}}$ stayed between ${r.ratio.min():.3f}$ and ${r.ratio.max():.3f}$")((lambda h: h[(h.sharp_ratio>=1)&(h.rev_out==1)&(h.dataset=='solar_flare')])(pd.read_csv(R / 'holdout_sharp.csv')))),
    ('weather loses',      'and $66.7$ percent of weather deployments',
     (lambda q: f"and ${100*(q.opt_true == 0).mean():.1f}$ percent of weather deployments"
      if (q[q.cell.str.startswith(('dry', 'polar'))].opt_true == 0).all() else 'NOT EVERY DRY AND POLAR')(wx_q.drop_duplicates(['cell', 'seed']))),
    ('weather bins',       '$3.0$ percent of comparisons at one tenth or more reverse, against $20.8$ percent below it',
     f'${100*wx_p.reversed_true[wx_p.ratio >= .1].mean():.1f}$ percent of comparisons at one tenth or more reverse, against ${100*wx_p.reversed_true[wx_p.ratio < .1].mean():.1f}$ percent below it'
     if (wx_p.certified == 1).sum() == 0 else 'SOME CERTIFIED'),
    ('weather criterion',  'with $0.9$ percent of right choices flagged',
     (lambda a: f"with ${100*wx_sum[(wx_sum.weight == 'all')].false_alarm_rule.max():.1f}$ percent of right choices flagged"
      if (a.wrong_pass_rule < a.wrong_pass_no_rule and a.false_alarm_rule <= .05) else 'CRITERION NOT MET')
     (wx_sum[(wx_sum.eps == .05) & (wx_sum.n == 1000) & (wx_sum.weight == 'all')].iloc[0])),
    ('weather SI pass',    'falls from $58.7$ to $44.6$ percent',
     (lambda a: f'falls from ${100*a.wrong_pass_no_rule:.1f}$ to ${100*a.wrong_pass_rule:.1f}$ percent')(wx_sum[(wx_sum.eps == .05) & (wx_sum.n == 1000) & (wx_sum.weight == 'all')].iloc[0])),
    ('real acs spot',      'it flags at most $0.5$ percent', f"it flags at most ${rs_size(rs_as, 'reject'):.1f}$ percent"),
    ('boards abstract',    'winners lose their lead under 24 of 28 reweightings',
     f"winners lose their lead under {int(sum(lbs.loc[x,'deployments'] - lbs.loc[x,'leader_best_in'] for x in ('openllm','finance','livebench')))} of {int(sum(lbs.loc[x,'deployments'] for x in ('openllm','finance','livebench')))} reweightings"),
    ('boards none cert',   'In none of those $28$ does the leader\'s stability radius cover the mismatch',
     f"In none of those ${lb_none}$ does the leader's stability radius cover the mismatch"
     if all(lbk[x].kappa_star[0] < lbk[x].sd_r.min() for x in ('openllm', 'finance', 'livebench')) else 'some board certifies its leader under the mildest use'),
    ('boards medical',     'larger stability radius, $1.10$, and stays first under five of six weightings',
     f"larger stability radius, ${lbk['medical'].kappa_star[0]:.2f}$, and stays first under {['zero','one','two','three','four','five','six','seven','eight','nine','ten','eleven','twelve'][int(lbs.loc['medical','leader_best_in'])]} of {['zero','one','two','three','four','five','six','seven','eight','nine','ten','eleven','twelve'][int(lbs.loc['medical','deployments'])]} weightings"),
    ('boards models',      '($185$ models, nine tasks)', f"(${int(lbs.loc['medical','models'])}$ models, {['zero','one','two','three','four','five','six','seven','eight','nine','ten','eleven','twelve'][int(lbs.loc['medical','tasks'])]} tasks)"),
    ('boards regret',      'the test returned three certified, four exceeds and $27$ undetermined verdicts across $34$ weightings; none was incorrect',
     f"the test returned {['zero','one','two','three','four','five','six','seven','eight','nine','ten','eleven','twelve'][(lbr.verdict_10=='certified').sum()]} certified, {['zero','one','two','three','four','five','six','seven','eight','nine','ten','eleven','twelve'][(lbr.verdict_10=='exceeds').sum()]} exceeds and ${(lbr.verdict_10=='undetermined').sum()}$ undetermined verdicts across ${len(lbr)}$ weightings; none was incorrect"
     if (((lbr.verdict_10=='certified')&(lbr.rr_exact>.10))|((lbr.verdict_10=='exceeds')&(lbr.rr_exact<=.10))|((lbr.verdict_05=='certified')&(lbr.rr_exact>.05))|((lbr.verdict_05=='exceeds')&(lbr.rr_exact<=.05))).sum()==0
        and (lbr.verdict_10=='undetermined').mean() > .5 else 'a wrong verdict or not most'),
    # ------------------------------------------- research question 4, the six answers (u44, u46)
    ('six mean',           'mean excess falls from $4.7$ to $2.9$ percent of the candidate range, and its 90th percentile from $17.4$ to $8.7$',
     f'mean excess falls from ${100*st.r2.mean():.1f}$ to ${100*st.m4.mean():.1f}$ percent of the candidate range, and its 90th percentile from ${100*st.r2.quantile(.9):.1f}$ to ${100*st.m4.quantile(.9):.1f}$'),
    ('six rr',             'from $11.2$ to $7.1$ percent, with the largest gains',
     f'from ${100*rsg.rr_r2.mean():.1f}$ to ${100*rsg.rr_rule.mean():.1f}$ percent, with the largest gains'),
    ('six noise',          "Flipping each answer at random with probability $0.4$ moves the mean excess from $2.9$ to $4.6$ percent, still below the criterion's $4.7$.",
     (f"Flipping each answer at random with probability $0.4$ moves the mean excess from ${sn.loc[0.0].iloc[0]:.1f}$ to ${sn.loc[0.4].iloc[0]:.1f}$ percent, still below the criterion's ${100*st.r2.mean():.1f}$."
      if sn.loc[0.4].iloc[0] < 100 * st.r2.mean() < sn.loc[0.5].iloc[0] else 'the crossing is not at 0.5')),
    ('tolerance rule',     'The mean excess runs from $5.0$ percent at two percent of the benchmark span to $12.7$ at thirty-five',
     f"The mean excess runs from ${100*fr['exc_0.02'].mean():.1f}$ percent at two percent of the benchmark span to ${100*fr['exc_0.35'].mean():.1f}$ at thirty-five"),
    # ---------------------------- headline numbers, in the wording of the NMI manuscript
    ('nmi suite comparisons', 'problems and $22{,}410$ pairwise comparisons', f'problems and ${len(sc):,}$ pairwise comparisons'.replace(',', '{,}')),
    ('nmi suite reversal', '$14.3$ percent of model pairs reverse their ordering', f'${100*sc.inverted.mean():.1f}$ percent of model pairs reverse their ordering'),
    ('nmi selection error', 'in $39.2$ percent of the $498$ problems', f'in ${100*sel.wrong.mean():.1f}$ percent of the ${len(sel)}$ problems'),
    ('nmi resolved rate',  'sampling error, $3.9$ percent reverse', f'sampling error, ${100*bo.rev.mean():.1f}$ percent reverse'),
    ('nmi resolved pair',  'Restricted to resolved pairs the reversal rates are $6.6$ percent on the leaderboard and $3.9$ on the suite', f'Restricted to resolved pairs the reversal rates are ${100*lb.rev.mean():.1f}$ percent on the leaderboard and ${100*bo.rev.mean():.1f}$ on the suite'),
    ('nmi resolves more',  'resolves more of its pairs, $90.4$ against $74.5$ percent', f'resolves more of its pairs, ${100*li.sm.mean():.1f}$ against ${100*inf.sm.mean():.1f}$ percent'),
    ('zero regret share',  'zero in $60.8$ percent of selection problems',
     f"zero in ${100*(pd.read_csv(R / 'cases.csv').exc_r2 <= 1e-12).mean():.1f}$ percent of selection problems"),
    ('split certified',    'the split test certified $12.7$ percent of choices against $18.5$',
     f"the split test certified ${100*_sp5['cert_split_0.1'].mean():.1f}$ percent of choices against ${100*_sp5['cert_0.1'].mean():.1f}$"),
    ('split fewer',        'it certified fewer choices and selected less accurate models',
     'it certified fewer choices and selected less accurate models'
     if _sp5['cert_split_0.1'].mean() < _sp5['cert_0.1'].mean() and _sp5.rr_pop_split.mean() > _sp5.rr_pop_sel.mean() else 'NOT FEWER OR NOT WORSE'),
    ('split fewer all',    'It certified fewer at every tolerance',
     'It certified fewer at every tolerance' if all(_sp[f'cert_split_{e}'].mean() < _sp[f'cert_{e}'].mean() for _, _sp in _spl.groupby('frac') for e in _EPS) else 'NOT FEWER'),
    ('split wrong max',    '$0.84$ for the split test and $0.55$ for ours',
     f"${max(100*_sp[f'cert_split_{e}'][_sp.rr_pop_split > e].mean() for _, _sp in _spl.groupby('frac') for e in _EPS):.2f}$ for the split test and ${max(100*_sp[f'cert_{e}'][_sp.rr_pop_sel > e].mean() for _, _sp in _spl.groupby('frac') for e in _EPS):.2f}$ for ours"),
    ('split worse',        'their mean true regret ratio was $22.6$ percent against $18.9$',
     f"their mean true regret ratio was ${100*_sp5.rr_pop_split.mean():.1f}$ percent against ${100*_sp5.rr_pop_sel.mean():.1f}$"),
    ('split coverage',     'covered a positive regret in $98.2$ and $97.6$ percent',
     f"covered a positive regret in ${100*(_sp5.rr_pop_split[_sp5.rr_pop_split > 0] <= _sp5.rr_ucb_split[_sp5.rr_pop_split > 0] + 1e-12).mean():.1f}$ and ${100*(_sp5.rr_pop_sel[_sp5.rr_pop_sel > 0] <= _sp5.rr_ucb_sel[_sp5.rr_pop_sel > 0] + 1e-12).mean():.1f}$ percent"),
    ('split same draws',   'on the draws of the validation above',
     'on the draws of the validation above' if (pd.read_csv(R / 'regret_test_sel.csv').merge(_spl, on=['dataset','seed','deployment','frac','rep']).pipe(lambda m: (m.choice_x == m.choice_y).all() and len(m) == len(_spl))) else 'DIFFERENT DRAWS'),
    ('undetermined within', 'Of these, $73.3$ percent have a true regret ratio within the tolerance',
     f"Of these, ${100*_und5[_und5.rr_pop <= 0.1].shape[0]/len(_und5):.1f}$ percent have a true regret ratio within the tolerance"),
    ('undetermined SI',    'Of these, $73.3$ percent have a true regret ratio within the tolerance and $49.6$ percent a true regret of zero',
     f"Of these, ${100*(_und5.rr_pop <= 0.1).mean():.1f}$ percent have a true regret ratio within the tolerance and ${100*(_und5.rr_pop <= 1e-12).mean():.1f}$ percent a true regret of zero"),
    ('rule size median',   'Fixing the mismatch size in advance still gives $3.2$ percent mean excess',
     f"Fixing the mismatch size in advance still gives ${100*_rsz.exc_median.mean():.1f}$ percent mean excess"),
    ('rule size methods',  'by the suite median, $1.73$, yields a mean excess of $3.2$ percent, against $2.9$ percent when the actual mismatch size is used. Fixing the size at one yields $4.1$ percent; half and twice the true mismatch yield $3.4$ and $2.6$ percent',
     f"by the suite median, ${_rsz.size_median.iloc[0]:.2f}$, yields a mean excess of ${100*_rsz.exc_median.mean():.1f}$ percent, against ${100*_rsz.exc_true.mean():.1f}$ percent when the actual mismatch size is used. Fixing the size at one yields ${100*_rsz.exc_one.mean():.1f}$ percent; half and twice the true mismatch yield ${100*_rsz.exc_half.mean():.1f}$ and ${100*_rsz.exc_double.mean():.1f}$ percent"),
    ('nmi rr p90',         'reaches $25.5$ percent at the 90th percentile', f'reaches ${100*rr.rr.quantile(.9):.1f}$ percent at the 90th percentile'),
    ('nmi crowding',       'Its measured exponents are $-0.75$ in the regression suite and $-0.78$ on the leaderboard, against predictions of $-0.88$ and $-0.83$',
     f'Its measured exponents are ${ce.loc["regression suite","observed"]:.2f}$ in the regression suite and ${ce.loc["language models","observed"]:.2f}$ on the leaderboard, against predictions of ${ce.loc["regression suite","predicted"]:.2f}$ and ${ce.loc["language models","predicted"]:.2f}$'),
    ('nmi ball counts',    'improving $33$ selection problems while worsening $347$', f'improving ${int(ms.ball_better)}$ selection problems while worsening ${int(ms.ball_worse)}$'),
    ('nmi grid tight',     'reverses $22.2$ percent\nof pairs', f'reverses ${G[(0.3,0.6)]:.1f}$ percent\nof pairs'),
    ('nmi option panels',  'from $3.3$ to $26.3$, and their within-domain', f'from ${opt.sd_r.min():.1f}$ to ${opt.sd_r.max():.1f}$, and their within-domain'),
    ('iia SI',             'moves its answer in $5.6$ percent of drops, and in $46.6$ percent of problems',
     f'moves its answer in ${100*ii.flips_w.sum()/ii.drops_w.sum():.1f}$ percent of drops, and in ${100*ii.any_w.mean():.1f}$ percent of problems'),
    ('rq3 r2 relative',    'against $14.4$ percent for $R^{2}$', f"against ${100*rr[rr.deployment=='relative-error weighted'].rr.mean():.1f}$ percent for $R^{{2}}$"),
    ('suite median corr',  'below the median absolute pairwise correlation of $0.076$', f'below the median absolute pairwise correlation of ${((sc.e_nu-sc.e_mu).abs()/(sc.sd_r*sc.sd_d)).median():.3f}$'),
    ('holdout main',       'only one of $1{,}180$ certified comparisons reverses on the held-out half, versus $24.2$ percent',
     f'only {['zero','one','two','three','four','five','six','seven','eight','nine','ten','eleven','twelve'][ho.rev_out[ho_c].sum()]} of ' + f'${ho_c.sum():,}$'.replace(',', '{,}') + f' certified comparisons reverses on the held-out half, versus ${100*ho.rev_out[~ho_c].mean():.1f}$ percent'),
    ('distinct pairs',     'Over the $4{,}725$ distinct pairs', f'Over the ${sc.drop_duplicates(["dataset", "seed", "pair"]).shape[0]:,}$ distinct pairs'.replace(',', '{,}')),
    ('llm certified',      '$582$ comparisons are certified, while $14.9$ percent of the exposed comparisons reverse',
     (lambda q: f'${int(q.sum())}$ comparisons are certified, while ${100*lp.inverted[~q].mean():.1f}$ percent of the exposed comparisons reverse'
      if lp.inverted[q].sum() == 0 else 'a certified pair reversed')(lp.e_mu.abs() / lp.sd_d >= lp.sd_r)),
    ('alabama example',    "whose realized regret ratio was $21.2$ percent; a spot-check with $500$ labeled cases flagged it in $95$ percent of draws",
     (lambda q, v: f"whose realized regret ratio was ${100*q.rr_true:.1f}$ percent; a spot-check with $500$ labeled cases flagged it in ${100*v.reject.mean():.0f}$ percent of draws"
      if q['cert_j_0.05'] == 1 else 'NOT CERTIFIED')(rt_acs[(rt_acs.state == 'AL') & (rt_acs.seed == 1) & (rt_acs.weight == 'marginals')].iloc[0],
                                                     rs_as[(rs_as.state == 'AL') & (rs_as.seed == 1) & (rs_as.n == 500)])),
]

# Items written for an earlier manuscript (main.tex), whose wording the NMI manuscript does not
# contain.  They are kept for the record and neither run nor counted.
LEGACY = ['LLM cond rev', 'LLM range', 'LLM range hi', 'LLM regret hi', 'LLM regret lo', 'LLM resolves', 'LLM zero of eight', 'MAPE relative', 'MAPE top decile', 'NMI at 1.99', 'NMI factor five', 'NMI limit ratio', 'NMI reversals', 'NMI slope', 'NMI threshold', 'R2 tight', 'R2 wide', 'RR one in ten', 'RR p95', 'RR q3', 'RR quartile p90', 'ball counts', 'both decisive', 'conditional rev', 'crowd obs llm', 'crowd pred reg', 'datasets affected', 'diabetes decile', 'grid rho .6 tight', 'matched criterion', 'matched mismatch', 'option range', 'option row', 'option widest', 'panel narrowest', 'panel pairs', 'panel widest', 'per-dataset median', 'per-dataset range', 'quintile high', 'quintile low', 'quintile low sd', 'regret draws', 'rev by n high', 'rev by n low', 'selection count', 'selection wrong', 'spearman scale', 'suite pairs', 'suite reversal', 'table pairs', 'unresolved']
C = [c for c in C if c[0] not in LEGACY]
# the paper and its Supplementary Information are the only text audited
tex = norm(''.join((P / f).read_text() for f in ('NMI_V9.tex', 'NMI_V9_SI.tex') if (P / f).exists()))

# the audit's own size, as the paper states it in digits (Code availability) and in words
_ones = 'zero one two three four five six seven eight nine'.split()
_teens = 'ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen'.split()
_tens = 'zero ten twenty thirty forty fifty sixty seventy eighty ninety'.split()
def _words(k):
    h, r = divmod(k, 100); t, o = divmod(r, 10)
    rest = _teens[o] if t == 1 else (_tens[t] + ('-' + _ones[o] if o else '') if t else _ones[o])
    return (_ones[h] + ' hundred' + (f'\nand {rest}' if r else '')) if h else rest
_rtq = _rts[_rts.frac == 0.25]
C.append(('regret test by size', "a quarter of each test set certifies $10.1$ percent of choices at a tolerance of $10$ percent, and half certifies $18.5$ percent",
                                 f"a quarter of each test set certifies ${100*_rtq['cert_0.1'].mean():.1f}$ percent of choices at a tolerance of $10$ percent, and half certifies ${100*_rt5['cert_0.1'].mean():.1f}$ percent"))

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
