"""u72_leaderboards.py -- the identity, the certificate and the leader on further leaderboards.

Same analysis as u34 for the Open LLM Leaderboard, on the boards fetched by u71, with the
deployments fixed in data/leaderboards/protocol.md before any analysis.  For each board:
  <board>_pairs.csv      one row per deployment and pair, u34's columns plus kappa and
                         whether the comparison is certified (kappa_fg >= sd_mu(r))
  <board>_selection.csv  one row per deployment: does the leader stay best, its regret
                         ratio, its kappa* (smallest radius against any rival) and sd_mu(r)
and one summary row per board in leaderboards_summary.csv.

    python u72_leaderboards.py [board ...]
"""
import itertools, sys, warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / 'data' / 'leaderboards'; R = ROOT / 'results' / 'universal' / 'leaderboards'
R.mkdir(parents=True, exist_ok=True)
ASR_OLD = ['AMI-Cleaned', 'Earnings22-Cleaned-AA-chunked', 'Gigaspeech-Cleaned', 'LS Clean',
           'LS Other', 'SPGISpeech', 'Voxpopuli-AA-Cleaned']


def onehot(mask):
    m = np.asarray(mask, float); return m / m.sum()


def measures(board, t, tasks, L):
    """The board's stated aggregation mu, and the deployments of the protocol."""
    G = t.drop_duplicates('task').set_index('task')['group'].loc[tasks]
    gsz = G.value_counts()
    eq_task = np.full(len(tasks), 1.0 / len(tasks))
    eq_group = np.array([1.0 / (len(gsz) * gsz[G[k]]) for k in tasks])
    if board == 'openllm':
        n = t.drop_duplicates('task').set_index('task')['n'].loc[tasks].astype(float).values
        mu, mild = eq_group, ('by question count', n / n.sum())
    elif board == 'livebench':
        mu, mild = eq_group, ('every task equally', eq_task)
    elif board == 'finance':
        mu, mild = eq_task, ('task types equally', eq_group)
    elif board == 'medical':
        n = t.drop_duplicates('task').set_index('task')['n'].loc[tasks].astype(float).values
        mu, mild = eq_task, ('every question equally', n / n.sum())
    elif board == 'open_asr':
        mu, mild = eq_task, ('the earlier seven sets', onehot([k in ASR_OLD for k in tasks]))
    else:
        raise ValueError(board)
    dep = {mild[0]: mild[1]}
    for g in sorted(gsz.index):
        dep[f'{g} alone'] = onehot([G[k] == g for k in tasks])
    if board == 'finance':
        lang = t.drop_duplicates('task').set_index('task')['lang'].loc[tasks]
        for l in ['EN', 'ZH', 'ES', 'EL']:
            dep[f'language {l} alone'] = onehot([lang[k] == l for k in tasks])
    hard = L.mean(0) >= np.quantile(L.mean(0), .75)
    dep['the hardest quartile of tasks'] = onehot(hard)
    return mu, dep


def load(board):
    if board != 'openllm':
        return pd.read_csv(D / board / 'raw.csv')
    # the board of the main text (u33/u34): same filtering as u34, groups by task prefix
    t = pd.read_csv(ROOT / 'results' / 'universal' / 'llm_raw.csv')
    t = t[~t.task.isin({'leaderboard_bbh', 'leaderboard_gpqa', 'leaderboard_math_hard',
                        'leaderboard_musr', 'leaderboard'})]
    grp = lambda k: next((g for g in ['bbh', 'math', 'gpqa', 'musr', 'mmlu_pro', 'ifeval']
                          if k.startswith('leaderboard_' + g)), 'other')
    t = t.assign(group=t.task.map(grp)); t = t[t.group != 'other']
    W = t.pivot(index='model', columns='task', values='score')
    keep = [k for k in W.columns if W[k].notna().sum() >= .8 * len(W)]
    models = W[keep].dropna(axis=0).index
    return t[t.task.isin(keep) & t.model.isin(models)]


def analyse(board):
    t = load(board)
    assert not t.duplicated(['model', 'task']).any(), 'a model has two scores for one task'
    assert t.score.between(0, 1).all(), 'scores must lie in [0, 1]'
    S = t.pivot_table(index='model', columns='task', values='score', aggfunc='first')
    S = S.dropna(axis=0)
    models, tasks = list(S.index), list(S.columns)
    L = 1.0 - S.values
    assert np.isfinite(L).all()
    mu, dep = measures(board, t, tasks, L)
    rows, sel = [], []
    for name, nu in dep.items():
        r = nu / mu
        sd_r = float(np.sqrt(mu @ (r - 1) ** 2))
        for i, j in itertools.combinations(range(len(models)), 2):
            dd = L[i] - L[j]
            e_mu, e_nu = float(mu @ dd), float(nu @ dd)
            sd_d = float(np.sqrt(mu @ (dd - e_mu) ** 2))
            kappa = abs(e_mu) / sd_d if sd_d > 0 else np.inf
            rows.append(dict(deployment=name, pair=f'{models[i]}|{models[j]}', sd_r=sd_r, sd_d=sd_d,
                             e_mu=e_mu, e_nu=e_nu, kappa=kappa, certified=int(kappa >= sd_r),
                             inverted=int(e_mu != 0 and np.sign(e_mu) != np.sign(e_nu)),
                             identity_err=abs((e_nu - e_mu) - float(mu @ ((dd - e_mu) * (r - 1))))
                                          / max(abs(e_nu - e_mu), 1e-300)))
        f = int(np.argmin(L @ mu)); g = int(np.argmin(L @ nu))
        kap = []
        for h in range(len(models)):
            if h == f: continue
            dd = L[f] - L[h]; e = float(mu @ dd); s = float(np.sqrt(mu @ (dd - e) ** 2))
            kap.append(abs(e) / s if s > 0 else np.inf)
        ks = float(min(kap))
        sel.append(dict(deployment=name, leader=models[f], best=models[g],
                        same=int(float(nu @ L[f]) <= float(nu @ L[g]) + 1e-12),   # a tie counts as best
                        regret_ratio=(float(nu @ L[f]) - float(nu @ L[g])) / float(mu @ L[f]),
                        leader_rank=int(1 + (L @ nu < float(nu @ L[f]) - 1e-15).sum()),
                        sd_r=sd_r, kappa_star=ks, leader_certified=int(ks >= sd_r)))
    p, s = pd.DataFrame(rows), pd.DataFrame(sel)
    p.to_csv(R / f'{board}_pairs.csv', index=False); s.to_csv(R / f'{board}_selection.csv', index=False)
    mild = list(dep)[0]
    out = dict(board=board, models=len(models), tasks=len(tasks), deployments=len(dep),
               comparisons=len(p), reversed_pct=100 * p.inverted.mean(),
               certified=int(p.certified.sum()), certified_reversed=int(p.inverted[p.certified == 1].sum()),
               exposed_reversed_pct=100 * p.inverted[p.certified == 0].mean(),
               leader=s.leader.iloc[0], leader_best_in=int(s.same.sum()),
               mild=mild, mild_same=int(s.same.iloc[0]), mild_rr_pct=100 * s.regret_ratio.iloc[0],
               mild_best=s.best.iloc[0], mild_leader_rank=int(s.leader_rank.iloc[0]),
               max_rr_pct=100 * s.regret_ratio.max(), leader_certified_in=int(s.leader_certified.sum()),
               identity_err=float(p.identity_err.max()))
    print(f"\n== {board}: {out['models']} models, {out['tasks']} tasks, {out['deployments']} deployments")
    print(f"   leader {out['leader']}; best under {out['leader_best_in']} of {out['deployments']} deployments; "
          f"certified against {out['leader_certified_in']}")
    print(f"   mildest ({mild}): best is {out['mild_best']}, leader ranks {out['mild_leader_rank']}, "
          f"regret ratio {out['mild_rr_pct']:.1f}%")
    print(f"   comparisons {out['comparisons']}, reversed {out['reversed_pct']:.1f}%, certified {out['certified']} "
          f"(reversed {out['certified_reversed']}), exposed reversed {out['exposed_reversed_pct']:.1f}%, "
          f"identity err {out['identity_err']:.1e}")
    print(s[['deployment', 'best', 'same', 'leader_rank', 'regret_ratio', 'sd_r', 'kappa_star']].round(3).to_string(index=False))
    return out


if __name__ == '__main__':
    boards = sys.argv[1:] or ['openllm', 'medical', 'finance', 'livebench']
    res = [analyse(b) for b in boards]
    f = R / 'leaderboards_summary.csv'
    old = pd.read_csv(f) if f.exists() else pd.DataFrame()
    new = pd.DataFrame(res)
    if len(old):
        old = old[~old.board.isin(new.board)]
    pd.concat([old, new]).to_csv(f, index=False)
