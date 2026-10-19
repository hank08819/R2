import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from matplotlib.colors import ListedColormap
# figure style
plt.rcParams.update({'font.size':9,'axes.titlesize':9,'axes.labelsize':9,
                     'xtick.labelsize':8,'ytick.labelsize':8,'legend.fontsize':7.5,
                     'axes.spines.top':False,'axes.spines.right':False,
                     'pdf.fonttype':42,'ps.fonttype':42})
INK='k'; RED='C3'; BLUE='C0'; GREY='C7'
fig,(a,b,cc_ax)=plt.subplots(1,3,figsize=(11.1,3.4),gridspec_kw={'wspace':.30})

# ---- (a) same projection, different size: same ordering
a.set_xlim(-0.2,1.35); a.set_ylim(-0.32,1.15); a.axis('off')
a.plot([-0.05,1.2],[0,0],color=INK,lw=1.1)
a.text(1.22,0,r'$\mathcal{H}_{\mathcal{P}}$',color=INK,va='center',fontsize=10)
a.plot([0,0],[-0.05,1.05],color=GREY,lw=1.0,ls=(0,(4,3)))
a.text(0.03,1.07,r'$\mathcal{H}_{\mathcal{P}}^{\perp}$',color=GREY,fontsize=10)
a.plot([0,0.07,0.07],[0.07,0.07,0],color=INK,lw=0.7)
for (y,c,lab,dy) in [(0.92,RED,r'$r-1$',0.04),(0.34,BLUE,r"$r'-1$",0.04)]:
    a.add_patch(FancyArrowPatch((0,0),(0.72,y),arrowstyle='-|>',mutation_scale=9,color=c,lw=1.5))
    a.text(0.75,y+dy,lab,color=c,fontsize=9.5)
    a.plot([0.72,0.72],[y,0],color=c,lw=0.8,ls=(0,(2,2)))
a.add_patch(FancyArrowPatch((0,0),(0.72,0),arrowstyle='-|>',mutation_scale=10,color=INK,lw=2.6))
a.text(0.36,-0.12,r'$P(r-1)=P(r\,\!^{\prime}-1)$',color=INK,ha='center',fontsize=9.5)
a.text(0.36,-0.26,'same ordering of every pair',color=INK,ha='center',fontsize=8.8,style='italic')
a.set_title('(a) the shift splits, and only the flat part acts',fontsize=9)

# ---- (b) selection regions: risks are linear in P(r-1), so boundaries are concurrent
v = {1:np.array([ 0.95, 0.35]), 2:np.array([-0.15,-1.00]), 3:np.array([-0.80, 0.65])}
c = {1:0.00, 2:0.62, 3:0.78}                      # benchmark risks, f1 best at the origin
g = np.linspace(-1.3,1.3,600); X,Y = np.meshgrid(g,g)
R = np.stack([c[k]+v[k][0]*X+v[k][1]*Y for k in (1,2,3)])
win = R.argmin(0)
b.imshow(win,origin='lower',extent=[-1.3,1.3,-1.3,1.3],
         cmap=ListedColormap(['#e9eff7','#f6ecec','#eef4ee']),interpolation='nearest',aspect='equal')
# boundaries: only the active part of each tie line, where the two tied models are both best
for (i1,i2) in [(0,1),(0,2),(1,2)]:
    other=[k for k in range(3) if k not in (i1,i2)][0]
    tie=R[i1]-R[i2]; active=np.where(R[[i1,i2]].min(0)<=R[other], tie, np.nan)
    b.contour(X,Y,active,levels=[0],colors=INK,linewidths=0.9)
b.scatter([0],[0],s=26,color=INK,zorder=6)
b.text(0.16,-0.10,'benchmark\n$P(r-1)=0$',fontsize=8.4,color=INK,va='top')
tip=np.array([-0.45,1.05])
b.add_patch(FancyArrowPatch((0,0),tuple(tip),arrowstyle='-|>',mutation_scale=9,color=RED,lw=1.6,zorder=7))
b.text(-1.24,1.16,'stated deployment',color=RED,fontsize=8.8,ha='left')
def risks(p): return [c[k]+v[k]@np.asarray(p) for k in (1,2,3)]
assert int(np.argmin(risks(tip)))==1, 'the stated deployment must select f2'
# kappa*: distance from the benchmark to the nearest selection face of f1, drawn to that face
feet={}
for jj in (2,3):
    d=v[jj]-v[1]; foot=-(c[jj]-c[1])*d/(d@d)
    r=risks(foot); assert abs(r[0]-r[jj-1])<1e-9 and r[0]<=min(r)+1e-9, 'face not active'
    feet[jj]=foot
jn=min(feet,key=lambda jj:np.linalg.norm(feet[jj])); foot=feet[jn]; k=float(np.linalg.norm(foot))
b.plot([0,foot[0]],[0,foot[1]],color=INK,lw=1.0,ls=(0,(3,2)),zorder=6)
u=foot/k; t=np.array([-u[1],u[0]]); h=0.07                      # right-angle mark at the foot
A=foot-u*h; B=A+t*h; C=foot+t*h; b.plot([A[0],B[0],C[0]],[A[1],B[1],C[1]],color=INK,lw=0.7,zorder=6)
b.text(0.0,0.21,r'$\kappa^{\star}$',color='k',fontsize=10,ha='center',va='center')
# region labels, each checked against the risks at its own position
for (x,y,kk,lab) in [(0.70,-1.15,3,r'$f_{3}$ selected'),(0.68,0.72,2,r'$f_{2}$ selected'),
                     (-1.22,-1.00,1,'$f_{1}$ selected,\nas on the benchmark')]:
    assert int(np.argmin(risks((x,y))))+1==kk, (lab,'is not in its region')
    b.text(x,y,lab,fontsize=8.6,color=INK)
b.set_xticks([]); b.set_yticks([])
for s in b.spines.values(): s.set_color(GREY)
b.set_title('(b) a benchmark partitions deployment space',fontsize=9)
# ---- (c) the same picture, measured on one dataset of the suite
from pathlib import Path
_H = Path(__file__).resolve().parents[1]
_R = next(p for p in (_H / 'optionJF' / 'results' / 'universal', _H / 'results' / 'universal') if p.exists())
z=np.load(_R / 'geometry_slice.npz',allow_pickle=True)
names=list(z['names']); best=int(z['best'])
lines=[r for r in z['lines'] if int(r[3])==best or int(r[4])==best]
xs=np.linspace(-.42,.42,2)
for r in lines:
    aa,bb,off=r[0],r[1],r[2]
    if abs(bb)>1e-9:
        cc_ax.plot(xs,(off-aa*xs)/bb,color='k',lw=.7,zorder=2)
    elif abs(aa)>1e-9:
        cc_ax.plot([off/aa,off/aa],[-.42,.42],color='k',lw=.7,zorder=2)
cc_ax.scatter([0],[0],s=28,color='k',zorder=6)
cc_ax.text(-.012,.012,'benchmark',fontsize=8,color='k',ha='right')
SW={'upper decile':True,'upper quartile':True,'relative error':True,
    'lower decile':False,'cost rising in the outcome':False}
for lab,pt in zip(z['labels'],z['pts']):
    sw=SW.get(str(lab),False); col=RED if sw else BLUE
    cc_ax.scatter([pt[0]],[pt[1]],s=46,color=col,edgecolor='k',lw=.4,zorder=7)
    OFF={'relative error':(8,-11),'lower decile':(8,-4),'cost rising in the outcome':(8,3)}
    cc_ax.annotate(str(lab),(pt[0],pt[1]),textcoords='offset points',
               xytext=OFF.get(str(lab),(8,4)),fontsize=7.4,color=col)
cc_ax.set_xlim(-.42,.42); cc_ax.set_ylim(-.42,.42)
cc_ax.set_xlabel('leading direction of $\\mathcal{H}_{\\mathcal{P}}$')
cc_ax.set_ylabel('second direction')
cc_ax.set_title('(c) the same picture, measured on one dataset',fontsize=9)
cc_ax.set_aspect('equal')

fig.tight_layout()
fig.savefig('fig_projection.pdf',bbox_inches='tight')
fig.savefig('fig_projection.png',dpi=220,bbox_inches='tight')
print('kappa* =',round(k,3))
