"""u2_nmi_remainder.py -- does the covariance identity extend to NMI?

NMI is not an average of a per-instance score, so Theorem 1 does not apply to it
directly. The von Mises expansion gives a first-order version with the same
covariance form, the per-instance score replaced by the influence function.  This
script measures the second-order remainder that expansion leaves behind, on real
data with real fitted models, and asks whether the first-order term is accurate
enough to predict which model deployment prefers.
"""
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, itertools, json
from pathlib import Path
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, \
    GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer

OUT = Path(__file__).resolve().parents[1] / 'results' / 'universal'
OUT.mkdir(parents=True, exist_ok=True)


# ----------------------------------------------------------- information ---
def joint(y, yhat, w):
    """Joint pmf of (Y, Yhat) under the measure with weights w summing to one."""
    ky, kh = int(y.max()) + 1, int(yhat.max()) + 1
    p = np.zeros((ky, kh))
    np.add.at(p, (y.astype(int), yhat.astype(int)), w)
    return p


def _H(q):
    q = q[q > 0]
    return float(-(q * np.log(q)).sum())


def MI(p):
    py, ph = p.sum(1, keepdims=True), p.sum(0, keepdims=True)
    m = p > 0
    return float((p[m] * np.log(p[m] / (py @ ph)[m])).sum())


def NMI(p):
    return 2 * MI(p) / (_H(p.sum(1)) + _H(p.sum(0)))


def IF_NMI(p):
    """Influence function of NMI as a table over cells (y, yhat), under p.

    IF_MI = pointwise mutual information minus MI; the normaliser follows by the
    quotient rule.  Verified against a central difference to O(t^2).
    """
    py, ph = p.sum(1), p.sum(0)
    I0, D = MI(p), _H(py) + _H(ph)
    with np.errstate(divide='ignore', invalid='ignore'):
        pmi = np.log(p / np.outer(py, ph))
    pmi[p <= 0] = 0.0
    if_mi = pmi - I0
    if_d = (-np.log(py) - _H(py))[:, None] + (-np.log(ph) - _H(ph))[None, :]
    return 2 * (if_mi * D - I0 * if_d) / D ** 2


# ------------------------------------------------------------------ data ---
def load(name):
    d = fetch_openml(name, version=1, as_frame=True, parser='auto')
    X = pd.get_dummies(d.data, drop_first=True).astype(float)
    X = pd.DataFrame(SimpleImputer(strategy='median').fit_transform(X))
    y = d.target
    pos = y.value_counts().idxmin()          # minority class is the positive one
    return X.values, (y == pos).astype(int).values


MODELS = lambda: {
    'rf': RandomForestClassifier(n_estimators=120, random_state=0),
    'et': ExtraTreesClassifier(n_estimators=120, random_state=0),
    'gb': GradientBoostingClassifier(random_state=0),
    'ada': AdaBoostClassifier(random_state=0),
    'dt': DecisionTreeClassifier(max_depth=6, random_state=0),
    'lr': make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
    'nb': make_pipeline(StandardScaler(), GaussianNB()),
}

rows, cells = [], []
for name, costs in [('mammography', [20, 50]), ('sick', [50])]:
    X, y = load(name)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.4, random_state=0, stratify=y)
    pred = {k: m.fit(Xtr, ytr).predict(Xte).astype(int) for k, m in MODELS().items()}
    n = len(yte)
    mu = np.full(n, 1 / n)

    for C in costs:
        w = np.where(yte == 1, float(C), 1.0)
        r = w / (mu @ w)                                  # mean one under mu
        sd_r = float(np.sqrt(mu @ (r - 1) ** 2))

        nmi_mu, nmi_nu, IFs = {}, {}, {}
        for k, yh in pred.items():
            p_mu = joint(yte, yh, mu)
            p_nu = joint(yte, yh, mu * r)
            nmi_mu[k], nmi_nu[k] = NMI(p_mu), NMI(p_nu)
            IFs[k] = IF_NMI(p_mu)[yte.astype(int), yh]     # per-instance
            cells.append(dict(dataset=name, model=k,
                              min_cell=int(round(p_mu.min() * n))))

        for f, g in itertools.combinations(pred, 2):
            gap_mu = nmi_mu[f] - nmi_mu[g]
            gap_nu = nmi_nu[f] - nmi_nu[g]
            d = IFs[f] - IFs[g]
            first = float(mu @ (d * r)) - float(mu @ d)     # Cov_mu(d, r)
            exact = gap_nu - gap_mu
            rows.append(dict(
                dataset=name, cost=C, sd_r=sd_r, pair=f'{f}-{g}',
                gap_mu=gap_mu, gap_nu=gap_nu, exact=exact, first=first,
                remainder=exact - first,
                reversed_actual=int(np.sign(gap_mu) != np.sign(gap_nu) and gap_mu != 0),
                reversed_pred=int(np.sign(gap_mu) != np.sign(gap_mu + first) and gap_mu != 0)))

df = pd.DataFrame(rows)
df.to_csv(OUT / 'nmi_remainder.csv', index=False)

print(f'pairs = {len(df)}   min cell count over all models = '
      f'{pd.DataFrame(cells).min_cell.min()}')
print('\n  data          cost  sd(r)  |R2|/|first| med   worst   sign(first) ok   '
      'reversals pred/actual')
for (ds, C), g in df.groupby(['dataset', 'cost']):
    rel = (g.remainder.abs() / g['first'].abs().clip(lower=1e-12))
    ok = float((np.sign(g['first']) == np.sign(g.exact)).mean())
    print(f'  {ds:<12} {C:>5}  {g.sd_r.iloc[0]:5.2f}   {rel.median():10.2f} '
          f'{rel.max():7.2f}   {ok:10.2f}      '
          f'{g.reversed_pred.sum():4d} / {g.reversed_actual.sum()}')

agree = float((df.reversed_pred == df.reversed_actual).mean())
print(f'\nfirst-order prediction of reversal agrees with truth on '
      f'{100*agree:.1f}% of {len(df)} pairs')
