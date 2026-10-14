"""Figure 4, the direction rule, from the stored result files (u42, u44, u46)."""
import numpy as np, pandas as pd, matplotlib
from pathlib import Path
matplotlib.use('Agg')
_H = Path(__file__).resolve().parents[1]
R = next(p for p in (_H / 'optionJF' / 'results' / 'universal', _H / 'results' / 'universal') if p.exists())
ms = pd.read_csv(R / 'method_summary.csv', index_col=0)['value']
st = pd.read_csv(R / 'signed_top.csv'); fr = pd.read_csv(R / 'frontier.csv'); rsg = pd.read_csv(R / 'rr_signed.csv')
rsq = rsg.groupby(pd.qcut(rsg.sd_r, 4, duplicates='drop'), observed=True)[['rr_r2', 'rr_rule']].mean() * 100
import matplotlib.pyplot as plt
# figure style
plt.rcParams.update({'font.size':9,'axes.titlesize':9,'axes.labelsize':9,
                     'xtick.labelsize':8,'ytick.labelsize':8,'legend.fontsize':7.5,
                     'axes.spines.top':False,'axes.spines.right':False,
                     'pdf.fonttype':42,'ps.fonttype':42})
INK='C0'; RED='C3'; BLUE='C0'; GREY='C7'
fig,(a,b)=plt.subplots(1,2,figsize=(7.4,3.4),gridspec_kw={'width_ratios':[1.15,1]})

# (a) mean excess, share of the candidate range, over the selection problems
rules=['worst-case\nregret','stated\nscenarios','stated\ntolerance',
       'the criterion','six binary\nquestions']
vals =[round(v, 1) for v in (ms.ball_mean_stated, ms.scen_mean, 100 * fr['exc_0.02'].mean(),
                               100 * st.r2.mean(), 100 * st.m4.mean())]
cols =[GREY, GREY, GREY, INK, RED]
x=np.arange(len(vals))
a.bar(x,vals,color=cols,width=0.68)
for i,v in enumerate(vals):
    a.text(i,v+0.35,f'{v}',ha='center',fontsize=8.6,color=cols[i],fontweight='bold')
a.axhline(vals[3],color=INK,lw=0.8,ls=(0,(4,3)))
a.set_xticks(x); a.set_xticklabels(rules,fontsize=7.9)
a.set_ylabel('mean excess (% of candidate range)',fontsize=8.6)
a.set_ylim(0,19.5)
a.text(0.97,0.95,'budget rules',transform=a.transAxes,fontsize=8.2,va='top',ha='right',color='C7')
a.set_title('(a) what a budget buys, and what a direction buys',fontsize=9)

# (b) regret ratio by quartile of the mismatch
q=np.arange(4); w=0.38
crit=[round(v, 1) for v in rsq.rr_r2]; six=[round(v, 1) for v in rsq.rr_rule]
b.bar(q-w/2,crit,w,color=INK,label='criterion')
b.bar(q+w/2,six ,w,color=RED ,label='with six answers')
for i in range(4):
    b.text(q[i]+w/2,six[i]+0.5,f'{six[i]}',ha='center',fontsize=8.2,color=RED)
    b.text(q[i]-w/2,crit[i]+0.5,f'{crit[i]}',ha='center',fontsize=8.2,color=INK)
b.set_xticks(q); b.set_xticklabels(['narrowest','2nd','3rd','widest'],fontsize=8.4)
b.set_xlabel(r'quartile of the mismatch $\mathrm{sd}_\mu(r)$',fontsize=8.6)
b.set_ylabel('regret ratio (% of benchmark risk)',fontsize=8.6)
b.set_ylim(0,26); b.legend(fontsize=8.2,frameon=False,loc='upper left')
b.set_title('(b) the correction acts where the mismatch is',fontsize=9)
fig.tight_layout()
fig.savefig('fig_direction.pdf',bbox_inches='tight')
fig.savefig('fig_direction.png',dpi=220,bbox_inches='tight')
