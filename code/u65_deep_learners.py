"""u65_deep_learners.py -- the three added learners for the two real-shift cases.

  MLP             : two hidden layers (64, 32), scikit-learn, early stopping.
  FT-Transformer  : the feature-tokenizer Transformer of Gorishniy et al. (NeurIPS 2021),
                    their default size (3 blocks, width 192, 8 heads, ReGLU), AdamW,
                    early stopping on a tenth of the training rows; CPU, seeded.
  LLM + ridge     : a ridge regression on frozen LLM embeddings of each row written as a
                    sentence (Qwen2.5-1.5B, 256 principal components; u64).  It sees only
                    the embedding columns; every other learner sees only the table.

learners13(seed, ntab) returns the ten of u58 followed by these three, each restricted to
its columns, for an input [table | embedding] with ntab table columns.
"""
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.neural_network import MLPRegressor

NAMES = ['random forest', 'extra trees', 'gradient boosting', 'AdaBoost', 'bagging', 'tree depth 5',
         'tree depth 12', 'linear', 'ridge', 'k nearest neighbours', 'MLP', 'FT-Transformer', 'LLM + ridge']


class Cols(BaseEstimator, RegressorMixin):
    def __init__(self, est, lo, hi):
        self.est, self.lo, self.hi = est, lo, hi
    def fit(self, X, y):
        self.est.fit(X[:, self.lo:self.hi], y); return self
    def predict(self, X):
        return self.est.predict(X[:, self.lo:self.hi])


class FTTransformer(BaseEstimator, RegressorMixin):
    def __init__(self, seed=0, d=192, blocks=3, heads=8, epochs=200, patience=16, batch=256, lr=1e-4):
        self.seed, self.d, self.blocks, self.heads = seed, d, blocks, heads
        self.epochs, self.patience, self.batch, self.lr = epochs, patience, batch, lr

    def _net(self, p):
        import torch, torch.nn as nn
        d, heads = self.d, self.heads
        class ReGLU(nn.Module):
            def forward(self, x):
                a, b = x.chunk(2, -1); return a * torch.relu(b)
        class Block(nn.Module):
            def __init__(self, first):
                super().__init__()
                self.n1 = nn.Identity() if first else nn.LayerNorm(d)
                self.att = nn.MultiheadAttention(d, heads, dropout=0.2, batch_first=True)
                self.n2 = nn.LayerNorm(d)
                h = int(d * 4 / 3)
                self.ff = nn.Sequential(nn.Linear(d, 2 * h), ReGLU(), nn.Dropout(0.1), nn.Linear(h, d))
            def forward(self, x):
                z = self.n1(x); x = x + self.att(z, z, z, need_weights=False)[0]
                return x + self.ff(self.n2(x))
        class Net(nn.Module):
            def __init__(s):
                super().__init__()
                s.W = nn.Parameter(torch.randn(p, d) / d ** 0.5); s.b = nn.Parameter(torch.zeros(p, d))
                s.cls = nn.Parameter(torch.randn(1, 1, d) / d ** 0.5)
                s.blocks = nn.ModuleList([Block(i == 0) for i in range(self.blocks)])
                s.head = nn.Sequential(nn.LayerNorm(d), nn.ReLU(), nn.Linear(d, 1))
            def forward(s, x):
                t = x.unsqueeze(-1) * s.W + s.b
                t = torch.cat([s.cls.expand(len(x), -1, -1), t], 1)
                for blk in s.blocks: t = blk(t)
                return s.head(t[:, 0]).squeeze(-1)
        return Net()

    def fit(self, X, y):
        import torch
        torch.manual_seed(self.seed); torch.set_num_threads(8)
        rng = np.random.default_rng(self.seed)
        self.sx = StandardScaler().fit(X); self.my, self.sy = float(y.mean()), float(y.std()) or 1.0
        Z = torch.tensor(self.sx.transform(X), dtype=torch.float32); t = torch.tensor((y - self.my) / self.sy, dtype=torch.float32)
        idx = rng.permutation(len(Z)); nv = len(Z) // 10; va, tr = idx[:nv], idx[nv:]
        self.net_ = self._net(Z.shape[1])
        opt = torch.optim.AdamW(self.net_.parameters(), lr=self.lr, weight_decay=1e-5)
        best, bad, state = np.inf, 0, None
        for ep in range(self.epochs):
            self.net_.train()
            for b in np.array_split(rng.permutation(tr), max(1, len(tr) // self.batch)):
                opt.zero_grad(); loss = ((self.net_(Z[b]) - t[b]) ** 2).mean(); loss.backward(); opt.step()
            self.net_.eval()
            with torch.no_grad():
                v = float(((self.net_(Z[va]) - t[va]) ** 2).mean())
            if v < best - 1e-6:
                best, bad = v, 0; state = {k: x.clone() for k, x in self.net_.state_dict().items()}
            else:
                bad += 1
                if bad >= self.patience: break
        self.net_.load_state_dict(state); self.epochs_run_ = ep + 1
        return self

    def predict(self, X):
        import torch
        Z = torch.tensor(self.sx.transform(X), dtype=torch.float32); self.net_.eval(); out = []
        with torch.no_grad():
            for i in range(0, len(Z), 8192): out.append(self.net_(Z[i:i + 8192]).numpy())
        return np.concatenate(out) * self.sy + self.my


def learners13(seed, ntab, ten):
    tab = [Cols(m, 0, ntab) for m in ten(seed)]
    tab.append(Cols(make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(64, 32), early_stopping=True,
                                                                 max_iter=300, random_state=seed)), 0, ntab))
    tab.append(Cols(FTTransformer(seed=seed), 0, ntab))
    tab.append(Cols(make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 4, 13))), ntab, None))
    return tab
