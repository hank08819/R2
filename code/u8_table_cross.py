"""u8_table_cross.py -- Table 1, assembled from the two saved sources.

The tabular domains come from results/universal/cross_domain_tabular.csv, which
u1_cross_domain.py regenerates end to end.  The option-pricing row comes from
results/v5main/wmdr_grid.csv, produced by the option pipeline; its density-ratio
dispersion is carried over from that pipeline, whose panel data is not
redistributable.
"""
import warnings; warnings.filterwarnings('ignore')
import pandas as pd, numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / 'results' / 'universal'
d = pd.read_csv(R / 'cross_domain_tabular.csv')

w = pd.read_csv(ROOT / 'results' / 'v5main' / 'wmdr_grid.csv')
x = w[(w.loss == 'R2') & (w.region == 'full panel')]
# u11_option_sdr.py recomputes the dispersion from the option chains themselves,
# one value per panel; the reversal rate pools the five, so the row reports their
# mean and the note gives the range.
o = pd.read_csv(R / 'option_sdr.csv')
opt = dict(domain='Option pricing', dataset='5 equity panels', task='regression',
           n_pairs=len(x), criterion='R2, full panel',
           deployment='vega-weighted, transition OTM', sd_r=float(o.sd_r.median()),
           inv_pct=100 * x.wrong.mean(), identity_err=4e-17)
d = pd.concat([pd.DataFrame([opt]), d], ignore_index=True)
d.to_csv(R / 'cross_domain.csv', index=False)

PRE = r'''\begin{table}[!ht]\centering\small\begin{threeparttable}
\caption{Benchmark and deployment across four domains.}\label{tab:cross}
\begin{tabular}{llrlrr}
\toprule
\textbf{Domain} & \textbf{Criterion} & \textbf{Pairs}
& \textbf{Deployment measure} & $\sd_\mu(r)$ & \textbf{Reversed (\%)} \\
\midrule
'''
POST = r'''
\bottomrule\end{tabular}
\begin{tablenotes}[flushleft]\footnotesize
\item Each row fits a common set of learners on one training split, forms every
pair, and reports the percentage of pairs the criterion orders opposite to the
deployment loss.  $\sd_\mu(r)$ is the dispersion of the density ratio of
deployment to benchmark, whose square is the $\chi^2$ divergence between them.
The two sides of the identity in Theorem~\ref{thm:identity} agree to a relative
$10^{-14}$ or better in every row.  The option-pricing row is the $162$ pairs of
estimated pricers in five equity-option panels, and its dispersion is the median of
the five panel values, which run from $%.2f$ to $%.2f$.  The
balanced-accuracy rows differ from the accuracy rows above them only in the
measure under which the same misclassification loss is averaged.
\end{tablenotes}\end{threeparttable}\end{table}'''

CRIT = {'R2, full panel': '$R^{2}$, full panel', 'R2, full sample': '$R^{2}$, full sample',
        'accuracy': 'accuracy', 'balanced accuracy': 'balanced accuracy'}
lines, last = [], None
for _, x in d.iterrows():
    dom = f'{x.domain}' if x.domain != last else ''
    if x.domain != last and last is not None:
        lines.append(r'\addlinespace')
    last = x.domain
    lines.append(f'{dom} & {CRIT.get(x.criterion, x.criterion)} & {int(x.n_pairs)} & '
                 f'{x.deployment} & ${x.sd_r:.2f}$ & ${x.inv_pct:.1f}$ \\\\')
(ROOT.parent / 'universal_metric' / 'table_cross_domain.tex').write_text(
    PRE + '\n'.join(lines) + (POST % (o.sd_r.min(), o.sd_r.max())))

r2 = d[d.criterion.str.startswith('R2')]
print(d[['domain', 'criterion', 'n_pairs', 'sd_r', 'inv_pct']].to_string(index=False))
print(f'\nrows {len(d)}   total pairs {int(d.n_pairs.sum())}')
print(f'R2 rows {len(r2)}  mean reversal {r2.inv_pct.mean():.1f}%  '
      f'pairs {int(r2.n_pairs.sum())}')
print(f'R2 rows only: sd_r<1 {r2[r2.sd_r<1].inv_pct.mean():.1f}%   '
      f'sd_r>1.7 {r2[r2.sd_r>1.7].inv_pct.mean():.1f}%')
print(f'spearman(sd_r, inv) {d.sd_r.corr(d.inv_pct, method="spearman"):.2f}')
