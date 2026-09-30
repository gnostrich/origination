"""What discovery is allowed to touch.

A sequence substrate exposes pieces (configurations carried forward after a
prefix of actions), interactions (continue a piece with an action string),
and behaviour (the output distribution at every step).  Coordinates of the
pieces exist only so that they can be re-inserted; nothing in discovery
reads them.  `basis` optionally applies an invertible affine change of
basis to every exposed piece (Level 1 of the ladder): discovery must be
blind to it.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


class SeqSubstrate:
    def __init__(self, model, n_actions, n_obs, basis=None):
        self.model = model
        self.n_actions, self.n_obs = n_actions, n_obs
        self.basis = basis  # (Q, b): exposed = h @ Q.T + b
        self.calls = 0      # substrate queries made (observation budget)

    # -- coordinates: only for re-insertion
    def _expose(self, h):
        if self.basis is None:
            return h
        Q, b = self.basis
        return h @ Q.T + b

    def _internal(self, x):
        if self.basis is None:
            return x
        Q, b = self.basis
        return (x - b) @ Q

    # -- pieces
    def pieces(self, prefixes):
        """Exposed configurations after each prefix (tuple of actions; () = initial)."""
        by_len = {}
        for i, p in enumerate(prefixes):
            by_len.setdefault(len(p), []).append(i)
        H = torch.empty(len(prefixes), self.model.d)
        with torch.no_grad():
            for l, idx in by_len.items():
                if l == 0:
                    H[idx] = self.model.initial().expand(len(idx), -1)
                    continue
                A = torch.tensor([prefixes[i] for i in idx], dtype=torch.long)
                _, sites = self.model(A)
                H[idx] = sites["hidden"][:, -1]
        self.calls += len(prefixes)
        return self._expose(H)

    # -- interactions
    def behave(self, X, contexts):
        """Output distributions when each exposed piece is continued with each
        context (a list of action tuples, all the same length): (n, C, L, o)."""
        h = self._internal(X)
        n = h.shape[0]
        Lmax = max(len(c) for c in contexts)
        out = np.full((n, len(contexts), Lmax, self.n_obs), np.nan)   # NaN pads shorter contexts
        with torch.no_grad():
            for i, c in enumerate(contexts):
                A = torch.tensor(c, dtype=torch.long)[None].expand(n, -1)
                lg, _ = self.model(A, h0=h)
                out[:, i, : len(c)] = F.softmax(lg, -1).numpy()
        self.calls += n * len(contexts)
        return out

    def step(self, X, chunk):
        """The pieces obtained by continuing each exposed piece with `chunk`."""
        h = self._internal(X)
        with torch.no_grad():
            A = torch.tensor(chunk, dtype=torch.long)[None].expand(h.shape[0], -1)
            _, sites = self.model(A, h0=h)
        self.calls += h.shape[0]
        return self._expose(sites["hidden"][:, -1])

    def run(self, X, string):
        """Output distributions along one action string from each piece: (n, L, o)."""
        return self.behave(X, [tuple(string)])[:, 0]


def random_basis(d, seed):
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.normal(size=(d, d)))
    b = rng.normal(size=d) * 3.0
    return torch.as_tensor(Q, dtype=torch.float32), torch.as_tensor(b, dtype=torch.float32)


def unique_rows(S, decimals=4):
    """Indices of unique signatures (rounded) and the inverse map; exact
    deduplication so that identical behaviours are typed once."""
    flat = np.nan_to_num(np.round(S.reshape(S.shape[0], -1), decimals), nan=-1.0)
    _, first, inv = np.unique(flat, axis=0, return_index=True, return_inverse=True)
    return first, np.asarray(inv).ravel()


def beh_dist(S1, S2):
    """Behavioural distance between signatures (…, C, L, o): the LARGEST JS
    distance over contexts and steps — `x ~ y` requires closeness in every
    context, as INTERFACE_LANGUAGE.md states.  (A mean over contexts dilutes
    rare but decisive differences; Milestone 2 exposed this.)"""
    return np.nanmax(js_distance(S1, S2), axis=(-2, -1))


def js_distance(P, Q, eps=1e-12):
    """Jensen–Shannon distance (sqrt of JS divergence in bits) along the last axis."""
    M = 0.5 * (P + Q)
    def kl(A, B):
        return np.sum(np.where(A > 0, A * (np.log2(A + eps) - np.log2(B + eps)), 0.0), axis=-1)
    return np.sqrt(np.maximum(0.5 * kl(P, M) + 0.5 * kl(Q, M), 0.0))
