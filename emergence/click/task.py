"""A hidden random layered Boolean circuit, seen by the learner only as
input -> output pairs.

    inputs  x ∈ {0,1}^n_in
    hidden  g_j(x) = T_j(x[S_j])   random truth table on a random k-subset
    outputs y_k(x) = U_k(g[R_k])   random truth table on a random k-subset of gates

Each hidden gate feeds several outputs, so a shared intermediate is cheaper
than separate functions, but no gate, boundary or table is visible to the
learner, and many other circuits compute the same function.  The generator
is kept for evaluation only.

Draws are rejected (before anything is trained) when an output is nearly
constant, or coincides with a single gate or a single input: such outputs
would make the decomposition trivially readable from the task itself.
"""

from __future__ import annotations

from itertools import product

import numpy as np


class Circuit:
    def __init__(self, n_in=12, n_hid=6, n_out=6, k=3, seed=0, balance=(0.25, 0.75)):
        rng = np.random.default_rng(seed)
        self.n_in, self.n_hid, self.n_out, self.k = n_in, n_hid, n_out, k
        self.draws = 0
        while True:
            self.draws += 1
            self.S = [np.sort(rng.choice(n_in, k, replace=False)) for _ in range(n_hid)]
            self.T = [self._table(rng, k) for _ in range(n_hid)]
            self.R = [np.sort(rng.choice(n_hid, k, replace=False)) for _ in range(n_out)]
            self.U = [self._table(rng, k) for _ in range(n_out)]
            if self._acceptable(balance):
                break

    def _acceptable(self, balance):
        X = self.all_inputs()
        G, Y = self.hidden(X), self.outputs(X)
        m = Y.mean(0)
        if (m < balance[0]).any() or (m > balance[1]).any():
            return False
        for k in range(self.n_out):
            for src in (G.T, X.T):
                for s in src:
                    if (Y[:, k] == s).all() or (Y[:, k] == 1 - s).all():
                        return False
        return True

    @staticmethod
    def _table(rng, k):
        while True:
            t = rng.integers(0, 2, 2**k)
            if 0 < t.sum() < 2**k:  # non-constant
                return t

    @staticmethod
    def _idx(bits):  # (B, k) bits -> table index
        return (bits * (2 ** np.arange(bits.shape[1]))[None]).sum(1)

    def hidden(self, X):
        return np.stack([self.T[j][self._idx(X[:, self.S[j]])] for j in range(self.n_hid)], 1)

    def outputs(self, X):
        G = self.hidden(X)
        return np.stack([self.U[kk][self._idx(G[:, self.R[kk]])] for kk in range(self.n_out)], 1)

    def all_inputs(self):
        return np.array(list(product([0, 1], repeat=self.n_in)), dtype=np.int64)

    def reuse(self):
        """How many outputs each hidden gate feeds (evaluation only)."""
        return [int(sum(j in R for R in self.R)) for j in range(self.n_hid)]


def split(n, frac, seed):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    m = int(round(frac * n))
    return np.sort(perm[:m]), np.sort(perm[m:])
