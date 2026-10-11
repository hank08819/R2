"""u1_cross_domain.py -- regenerate every tabular row of Table 1 from one script.

Each setting fits the same seven learners on a common training split, forms all
pairs, and asks how often the benchmark criterion orders a pair opposite to the
deployment loss.  Writes results/universal/cross_domain_tabular.csv and the
per-instance quantities the figures need.
"""
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, itertools, pickle
from pathlib import Path
from sklearn.datasets import fetch_openml, load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
    GradientBoostingClassifier, AdaBoostClassifier, RandomForestRegressor,
    ExtraTreesRegressor, GradientBoostingRegressor, AdaBoostRegressor)
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer

OUT = Path(__file__).resolve().parents[1] / 'results' / 'universal'
OUT.mkdir(parents=True, exist_ok=True)
SEED = 0

CLF = lambda: {
    'RF': RandomForestClassifier(n_estimators=120, random_state=SEED),
    'ET': ExtraTreesClassifier(n_estimators=120, random_state=SEED),
    'GB': GradientBoostingClassifier(random_state=SEED),
    'AdaB': AdaBoostClassifier(random_state=SEED),
    'Tree': DecisionTreeClassifier(max_depth=6, random_state=SEED),
    'Logit': make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
    'NB': make_pipeline(StandardScaler(), GaussianNB()),
}
REG = lambda: {
    'RF': RandomForestRegressor(200, random_state=SEED, n_jobs=-1),
    'ET': ExtraTreesRegressor(200, random_state=SEED, n_jobs=-1),
    'GBM': GradientBoostingRegressor(random_state=SEED),
    'OLS': LinearRegression(),
    'Ridge': Ridge(1.0),
    'kNN': KNeighborsRegressor(10),
    'CART': DecisionTreeRegressor(max_depth=6, random_state=SEED),
}


def openml(name):
    d = fetch_openml(name, version=1, as_frame=True, parser='auto')
    X = pd.get_dummies(d.data, drop_first=True).astype(float)
    X = SimpleImputer(strategy='median').fit_transform(X)
    y = d.target
    pos = y.value_counts().idxmin()
    return X, (y == pos).astype(int).values


def evaluate(losses, r, mu, names):
    """Return per-pair records under benchmark mu and deployment mu*r."""
    recs = []
    for f, g in itertools.combinations(names, 2):
        d = losses[f] - losses[g]
        e_mu, e_nu = float(mu @ d), float(mu @ (d * r))
        sd_d = float(np.sqrt(mu @ (d - e_mu) ** 2))
        sd_r = float(np.sqrt(mu @ (r - 1) ** 2))
        rho = (e_nu - e_mu) / (sd_d * sd_r) if sd_d * sd_r > 0 else 0.0
        recs.append(dict(pair=f'{f}–{g}', e_mu=e_mu, e_nu=e_nu, rho=rho,
                         cov=e_nu - e_mu, sd_d=sd_d,
                         inverted=int(e_mu != 0 and np.sign(e_mu) != np.sign(e_nu)),
                         identity_err=abs((e_nu - e_mu) - float(mu @ ((d - e_mu) * (r - 1))))
                                      / max(abs(e_nu - e_mu), 1e-300)))
    return recs


rows, store = [], {}


def add(domain, dataset, task, criterion, deployment, losses, r, mu, names):
    recs = evaluate(losses, r, mu, names)
    sd_r = float(np.sqrt(mu @ (r - 1) ** 2))
    key = f'{dataset}|{criterion}|{deployment}'
    store[key] = dict(recs=recs, losses=losses, r=r, mu=mu, names=names)
    rows.append(dict(domain=domain, dataset=dataset, task=task, n_pairs=len(recs),
                     criterion=criterion, deployment=deployment, sd_r=sd_r,
                     inv_pct=100 * np.mean([x['inverted'] for x in recs]),
                     identity_err=max(x['identity_err'] for x in recs)))


# ------------------------------------------------- medical screening ---
for name, costs in [('mammography', [20, 50]), ('sick', [50])]:
    X, y = openml(name)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.4, random_state=SEED, stratify=y)
    pred = {k: m.fit(Xtr, ytr).predict(Xte).astype(int) for k, m in CLF().items()}
    n = len(yte); mu = np.full(n, 1 / n); names = list(pred)
    err = {k: (v != yte).astype(float) for k, v in pred.items()}          # 0-1 loss
    prev = yte.mean()
    for C in costs:
        w = np.where(yte == 1, float(C), 1.0)
        add('Medical screening', name, 'classification', 'accuracy',
            f'cost-weighted {C}:1', err, w / (mu @ w), mu, names)
    # Balanced accuracy is the same 0-1 loss averaged under a class-balanced
    # benchmark measure, so it is the benchmark that moves, not the loss.
    wbal = np.where(yte == 1, .5 / prev, .5 / (1 - prev))
    mubal = wbal / wbal.sum()
    wdep = np.where(yte == 1, 20.0, 1.0)
    nu = wdep / wdep.sum()
    rbal = nu / mubal
    add('Medical screening', name, 'classification', 'balanced accuracy',
        'natural prevalence + cost', err, rbal / (mubal @ rbal), mubal, names)

# ------------------------------------------------ clinical regression ---
db = load_diabetes()
Xtr, Xte, ytr, yte = train_test_split(db.data, db.target, test_size=.4, random_state=SEED)
pred = {k: m.fit(Xtr, ytr).predict(Xte) for k, m in REG().items()}
n = len(yte); mu = np.full(n, 1 / n); names = list(pred)
sq = {k: (v - yte) ** 2 for k, v in pred.items()}                        # R2 <-> MSE
# A tiny floor keeps the deployment measure absolutely continuous with respect
# to the benchmark, so the density ratio is finite everywhere.
for lab, w in [('top decile of outcome', (yte > np.percentile(yte, 90)).astype(float) + 1e-9),
               ('top quartile of outcome', (yte > np.percentile(yte, 75)).astype(float) + 1e-9),
               ('cost rising in outcome', yte / yte.mean())]:
    add('Clinical regression', 'diabetes', 'regression', 'R2, full sample',
        lab, sq, w / (mu @ w), mu, names)

# ------------------------------------------------------- real estate ---
try:
    h = fetch_openml('house_sales', version=3, as_frame=True, parser='auto')
    X = pd.get_dummies(h.data, dummy_na=True).fillna(0).values.astype(float)[:8000]
    y = h.target.values.astype(float)[:8000]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.4, random_state=SEED)
    pred = {k: m.fit(Xtr, ytr).predict(Xte) for k, m in REG().items()}
    n = len(yte); mu = np.full(n, 1 / n); names = list(pred)
    sq = {k: (v - yte) ** 2 for k, v in pred.items()}
    w = 1.0 / np.maximum(yte, 1) * yte.mean()               # relative-error weighting
    add('Real estate', 'house_sales', 'regression', 'R2, full sample',
        'relative-error weighted', sq, w / (mu @ w), mu, names)
except Exception as e:
    print('real estate unavailable:', type(e).__name__, e)

pd.DataFrame(rows).to_csv(OUT / 'cross_domain_tabular.csv', index=False)
pickle.dump(store, open(OUT / 'per_pair.pkl', 'wb'))
print(pd.DataFrame(rows)[['dataset', 'criterion', 'deployment', 'n_pairs',
                          'sd_r', 'inv_pct', 'identity_err']].to_string(index=False))
