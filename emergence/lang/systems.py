"""Milestone 2 system: one external computation with several genuinely
different compositional decompositions, and no slot ontology exposed.

The reader
----------
Three input slots A, B, D over Z_3 are set by actions `set(slot, v)`
(9 actions), in ANY order; each slot is write-once (a set on an already
set slot is a no-op, so no continuation can re-read a slot separately).  A
tenth action `query` emits the result

    Y = (a·b + d) mod 3

when all three slots are set, and ⊥ otherwise; set actions emit ⊥.
Observations: {⊥, 0, 1, 2}.

Why the decompositions are genuinely inequivalent
-------------------------------------------------
The same Y factors through different intermediates depending on which pair
is combined first:

    D1:  (A, B) ⇝ C,  (C, D) ⇝ Y     with C = a·b mod 3            (3 classes)
    D2:  (B, D) ⇝ E,  (A, E) ⇝ Y     with E = the class of the map a ↦ a·b + d   (9 classes)
    D3:  (A, D) ⇝ F,  (F, B) ⇝ Y     with F = the class of the map b ↦ a·b + d   (9 classes)

C, E and F are quotients of different sets (A×B, B×D, A×D), of different
sizes, and none is a refinement, coarsening, permutation or relabelling of
another; each supports a compact executable explanation of Y for its own
reading order.  Because the reader accepts any order, the substrate
privileges none of them.  All three are sealed: discovery never sees slot
identities, values, the table, or the decompositions.  They are used only
in `unseal_m2` after evaluation is frozen.

Substrate exposure
------------------
Internal configurations are random 16-dimensional vectors, one per
partial assignment (64 of them; 34 behaviourally distinct); the map from
configuration to vector is a fixed random embedding.  Discovery receives pieces (these vectors),
interactions (continuation with action strings) and behaviour (one-hot
observation distributions), nothing else.
"""

from __future__ import annotations

from itertools import product

import numpy as np

N_VAL = 3
SLOTS = ("A", "B", "D")
UNSET = N_VAL          # slot value meaning "unset"
N_ACTIONS = 3 * N_VAL + 1
QUERY = 3 * N_VAL
N_OBS = N_VAL + 1      # ⊥ plus the three values
BOT = 0


def f_table(a, b, d):
    return (a * b + d) % N_VAL


class ReaderMachine:
    """Table-driven transducer with a distributed (random-vector) state encoding."""

    n_actions = N_ACTIONS
    n_obs = N_OBS

    def __init__(self, dim=16, seed=0):
        rng = np.random.default_rng(seed)
        self.states = list(product(range(N_VAL + 1), repeat=3))       # (a, b, d) with UNSET allowed
        self.index = {s: i for i, s in enumerate(self.states)}
        self.E = rng.normal(size=(len(self.states), dim))               # sealed embedding
        self.E /= np.linalg.norm(self.E, axis=1, keepdims=True)
        self.d = dim
        self.calls = 0
        self.T = np.zeros((len(self.states), N_ACTIONS), dtype=int)     # transition
        self.O = np.zeros((len(self.states), N_ACTIONS), dtype=int)     # observation index
        for i, s in enumerate(self.states):
            for act in range(N_ACTIONS):
                if act == QUERY:
                    self.T[i, act] = i
                    self.O[i, act] = BOT if UNSET in s else 1 + f_table(*s)
                else:
                    slot, v = divmod(act, N_VAL)
                    t = list(s)
                    if t[slot] == UNSET:          # write-once: a set slot is never overwritten
                        t[slot] = v
                    self.T[i, act] = self.index[tuple(t)]
                    self.O[i, act] = BOT
        self.init = self.index[(UNSET, UNSET, UNSET)]

    # ---- coordinates are only for re-insertion
    def _expose(self, idx):
        return self.E[idx]

    def _decode(self, X):
        return np.argmax(X @ self.E.T, axis=1)

    # ---- pieces
    def pieces(self, prefixes):
        idx = np.empty(len(prefixes), dtype=int)
        for i, p in enumerate(prefixes):
            s = self.init
            for a in p:
                s = self.T[s, a]
            idx[i] = s
        self.calls += len(prefixes)
        return self._expose(idx)

    # ---- interactions
    def _run_idx(self, idx, string):
        obs = np.empty((len(idx), len(string)), dtype=int)
        s = idx.copy()
        for t, a in enumerate(string):
            obs[:, t] = self.O[s, a]
            s = self.T[s, a]
        return obs, s

    def behave(self, X, contexts):
        idx = self._decode(X)
        Lmax = max(len(c) for c in contexts)
        out = np.full((len(idx), len(contexts), Lmax, N_OBS), np.nan)
        for i, c in enumerate(contexts):
            obs, _ = self._run_idx(idx, c)
            out[:, i, : len(c)] = 0.0
            out[np.arange(len(idx))[:, None], i, np.arange(len(c))[None, :], obs] = 1.0
        self.calls += len(idx) * len(contexts)
        return out

    def step(self, X, chunk):
        idx = self._decode(X)
        _, s = self._run_idx(idx, chunk)
        self.calls += len(idx)
        return self._expose(s)

    def run(self, X, string):
        return self.behave(X, [tuple(string)])[:, 0]

    # ---- sealed: for unsealing only
    def decode_assignment(self, X):
        return [self.states[i] for i in self._decode(X)]

    def substrate_bits(self):
        """Description of the substrate itself: transition + observation tables."""
        n = len(self.states)
        return float(n * N_ACTIONS * (np.log2(n) + np.log2(N_OBS)))


# ------------------------------------------------------------ sealed decompositions
def sealed_decompositions():
    """The three factorisations, as functions from (a, b, d)-partial states to
    intermediate class ids (None where the intermediate is not defined)."""
    vals = range(N_VAL)
    C = {(a, b): (a * b) % N_VAL for a in vals for b in vals}
    E_fn = {(b, d): tuple(f_table(a, b, d) for a in vals) for b in vals for d in vals}
    E_ids = {v: i for i, v in enumerate(sorted(set(E_fn.values())))}
    E = {k: E_ids[v] for k, v in E_fn.items()}
    F_fn = {(a, d): tuple(f_table(a, b, d) for b in vals) for a in vals for d in vals}
    F_ids = {v: i for i, v in enumerate(sorted(set(F_fn.values())))}
    F = {k: F_ids[v] for k, v in F_fn.items()}
    return dict(C=C, E=E, F=F, sizes=dict(C=len(set(C.values())), E=len(E_ids), F=len(F_ids)))


def order_policy_prefixes(policy, rng, n, max_len=3):
    """Discovery pools under a reading-order policy.  'free': random actions.
    'ABD' etc.: slots set in that order (values random), with queries and
    partial prefixes; all realised as ordinary action strings."""
    slot_of = {"A": 0, "B": 1, "D": 2}
    pre = [()]
    if policy == "free":
        from emergence.lang.discover import all_chunks
        pre += all_chunks(N_ACTIONS, max_len)
        pre += [tuple(int(a) for a in rng.integers(0, N_ACTIONS, 8)) for _ in range(n)]
        return pre
    order = [slot_of[c] for c in policy]
    seen = set([()])
    for _ in range(n * 4):
        k = int(rng.integers(0, 4))                      # how many slots already set, in policy order
        p = []
        for j in range(k):
            p.append(order[j] * N_VAL + int(rng.integers(0, N_VAL)))
        if rng.random() < 0.3:
            p.append(QUERY)
        p = tuple(p)
        if p not in seen:
            seen.add(p); pre.append(p)
    # every complete assignment in the policy order, for coverage
    for vals in product(range(N_VAL), repeat=3):
        p = tuple(order[j] * N_VAL + vals[j] for j in range(3))
        if p not in seen:
            seen.add(p); pre.append(p)
        for j in (1, 2):
            q = p[:j]
            if q not in seen:
                seen.add(q); pre.append(q)
    return pre
