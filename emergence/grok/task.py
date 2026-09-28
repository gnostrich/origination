"""Binary-table tasks: (a, b) -> table[a, b] on n elements.

The default is modular addition, but the table can be any of the adversarial
controls below.  The extraction (``extract.py``) sees only tokens, sites and
behaviour; it never sees the table, the labels or the train/test split.

Task specs (``make_task``):

    zmod:97          (a + b) mod 97                    cyclic group
    zmod:89, zmod:101                                   other cyclic groups
    zprod:8x8        Z_8 x Z_8, 64 elements             non-cyclic abelian group
    sub:97           (a - b) mod 97                    non-associative, non-commutative quasigroup
    random:97        uniform random table               no structure at all
    scramble:zmod:97 input tokens and output tokens permuted *independently*
                     (the closed-loop table is an isotope of the group, not
                     an isomorphic copy)
    corrupt:0.1:zmod:97   10 % of table entries replaced by random values
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class Task:
    name: str
    n: int
    table: np.ndarray  # (n, n) int in 0..n-1
    # bookkeeping for the report only (never used by extraction)
    meta: dict

    def all_pairs(self) -> torch.Tensor:
        return all_pairs(self.n)

    def labels(self) -> torch.Tensor:
        return torch.tensor(self.table.ravel(), dtype=torch.long)

    def reference_partition(self) -> np.ndarray:
        """Inputs grouped by true output: the external reference partition."""
        return self.table.ravel()


def all_pairs(n: int) -> torch.Tensor:
    """All n^2 inputs as token rows ``[a, b, EQ]`` with ``EQ = n``."""
    a, b = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    x = np.stack([a.ravel(), b.ravel(), np.full(n * n, n)], axis=1)
    return torch.tensor(x, dtype=torch.long)


def labels(p: int) -> torch.Tensor:  # backwards compatible: modular addition
    return make_task(f"zmod:{p}").labels()


def split(n: int, train_frac: float, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n * n)
    k = int(round(train_frac * n * n))
    return torch.tensor(np.sort(perm[:k])), torch.tensor(np.sort(perm[k:]))


def _zmod(p):
    a = np.arange(p)
    return (a[:, None] + a[None, :]) % p


def make_task(spec: str, seed: int = 0) -> Task:
    parts = spec.split(":")
    kind = parts[0]
    if kind == "zmod":
        p = int(parts[1])
        return Task(spec, p, _zmod(p), {"kind": "group", "cyclic": True})
    if kind == "zprod":
        k1, k2 = (int(x) for x in parts[1].split("x"))
        n = k1 * k2
        i = np.arange(n)
        a1, a2 = i // k2, i % k2
        t = ((a1[:, None] + a1[None, :]) % k1) * k2 + (a2[:, None] + a2[None, :]) % k2
        return Task(spec, n, t, {"kind": "group", "cyclic": np.gcd(k1, k2) == 1})
    if kind == "sub":
        p = int(parts[1])
        a = np.arange(p)
        return Task(spec, p, (a[:, None] - a[None, :]) % p, {"kind": "quasigroup", "associative": False})
    if kind == "random":
        n = int(parts[1])
        rng = np.random.default_rng(10_000 + seed)
        return Task(spec, n, rng.integers(0, n, (n, n)), {"kind": "random"})
    if kind == "scramble":
        base = make_task(":".join(parts[1:]), seed)
        rng = np.random.default_rng(20_000 + seed)
        pin, pout = rng.permutation(base.n), rng.permutation(base.n)
        # token x stands for element pin^{-1}(x); output token pout(v) stands for value v
        inv = np.argsort(pin)
        t = pout[base.table[inv[:, None], inv[None, :]]]
        return Task(spec, base.n, t, {**base.meta, "scrambled": True, "pin": pin.tolist(), "pout": pout.tolist()})
    if kind == "corrupt":
        frac = float(parts[1])
        base = make_task(":".join(parts[2:]), seed)
        rng = np.random.default_rng(30_000 + seed)
        t = base.table.copy()
        m = rng.random(t.shape) < frac
        t[m] = rng.integers(0, base.n, m.sum())
        return Task(spec, base.n, t, {**base.meta, "corrupt_frac": frac, "n_corrupted": int(m.sum())})
    raise ValueError(f"unknown task spec {spec!r}")
