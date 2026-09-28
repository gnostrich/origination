"""Modular addition: (a, b) -> (a + b) mod p.

The task is chosen because the human-understood structure (a cyclic group)
is known, so we can ask afterwards whether the *extracted* structure is
equivalent to it.  The extraction itself (``extract.py``) never uses ``p``'s
arithmetic, the labels, or the train/test split.
"""

from __future__ import annotations

import numpy as np
import torch


def all_pairs(p: int) -> torch.Tensor:
    """All p^2 inputs as token rows ``[a, b, EQ]`` with ``EQ = p``."""
    a, b = np.meshgrid(np.arange(p), np.arange(p), indexing="ij")
    x = np.stack([a.ravel(), b.ravel(), np.full(p * p, p)], axis=1)
    return torch.tensor(x, dtype=torch.long)


def labels(p: int) -> torch.Tensor:
    x = all_pairs(p)
    return (x[:, 0] + x[:, 1]) % p


def split(p: int, train_frac: float, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    rng = np.random.default_rng(seed)
    n = p * p
    perm = rng.permutation(n)
    k = int(round(train_frac * n))
    return torch.tensor(np.sort(perm[:k])), torch.tensor(np.sort(perm[k:]))
