"""Small dynamical worlds whose objective is *prediction*, not a table.

A world has a hidden state, a finite action alphabet and a lossy
observation.  The learner sees only action tokens and must predict the
observation after each action.  Nothing algebraic is presented: no
operation table, no state labels.  The algebra that the extractor may or
may not find (a transformation monoid / group acting on the induced
states) lives in the dynamics, not in the objective.

Worlds (``make_world``):

    perm4        4 items in 4 slots; actions: swap01, swap12, swap23, rotate;
                 observe the item in slot 0.  Reversible: the induced
                 structure is S_4 (order 24, non-abelian).
    perm4reset   same plus a `reset` action that puts the items back in
                 order.  Not reversible: the induced structure is a
                 transformation monoid, not a group.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import permutations

import numpy as np
import torch


@dataclass
class World:
    name: str
    n_actions: int
    n_obs: int
    states: list  # canonical hidden states (tuples)
    step_table: np.ndarray  # (n_states, n_actions) -> state index
    obs_table: np.ndarray  # (n_states,) -> observation
    init_state: int
    meta: dict = field(default_factory=dict)

    def run(self, actions: np.ndarray) -> np.ndarray:
        """actions (B, L) -> observations (B, L)."""
        B, L = actions.shape
        s = np.full(B, self.init_state)
        out = np.empty((B, L), dtype=int)
        for t in range(L):
            s = self.step_table[s, actions[:, t]]
            out[:, t] = self.obs_table[s]
        return out

    def minimal_automaton(self) -> tuple[int, np.ndarray]:
        """Nerode-minimise the reachable part w.r.t. observations (the
        external reference for the number of distinguishable states)."""
        n, k = self.step_table.shape
        reach = {self.init_state}
        frontier = [self.init_state]
        while frontier:
            s = frontier.pop()
            for a in range(k):
                t = int(self.step_table[s, a])
                if t not in reach:
                    reach.add(t); frontier.append(t)
        reach = sorted(reach)
        part = {s: int(self.obs_table[s]) for s in reach}
        while True:
            sig = {s: (part[s],) + tuple(part[int(self.step_table[s, a])] for a in range(k)) for s in reach}
            ids = {v: i for i, v in enumerate(sorted(set(sig.values())))}
            new = {s: ids[sig[s]] for s in reach}
            if len(set(new.values())) == len(set(part.values())):
                part = new
                break
            part = new
        m = len(set(part.values()))
        table = np.zeros((m, k), dtype=int)
        for s in reach:
            for a in range(k):
                table[part[s], a] = part[int(self.step_table[s, a])]
        return m, table


def make_world(spec: str) -> World:
    if spec in ("perm4", "perm4reset"):
        items = list(permutations(range(4)))  # state = which item in each slot
        idx = {p: i for i, p in enumerate(items)}
        def swap(p, i, j):
            q = list(p); q[i], q[j] = q[j], q[i]; return tuple(q)
        def rot(p):
            return tuple(p[1:] + p[:1])
        acts = [lambda p: swap(p, 0, 1), lambda p: swap(p, 1, 2), lambda p: swap(p, 2, 3), rot]
        names = ["swap01", "swap12", "swap23", "rotate"]
        if spec == "perm4reset":
            acts.append(lambda p: (0, 1, 2, 3)); names.append("reset")
        step = np.array([[idx[f(p)] for f in acts] for p in items])
        obs = np.array([p[0] for p in items])
        return World(spec, len(acts), 4, items, step, obs, idx[(0, 1, 2, 3)],
                     {"actions": names, "reversible": spec == "perm4"})
    raise ValueError(spec)


def make_dataset(world: World, n_train: int, n_test: int, length: int, seed: int):
    rng = np.random.default_rng(seed)
    n = n_train + n_test
    acts = rng.integers(0, world.n_actions, (n, length))
    obs = world.run(acts)
    A = torch.tensor(acts, dtype=torch.long)
    O = torch.tensor(obs, dtype=torch.long)
    return (A[:n_train], O[:n_train]), (A[n_train:], O[n_train:])
