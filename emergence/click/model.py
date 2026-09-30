"""Learner: a plain MLP with two hidden sites.  Nothing about the circuit
(gates, arities, layers) is reflected in the architecture beyond depth 2
being *sufficient*; width 64 far exceeds the 6 hidden gates.
"""

from __future__ import annotations

import torch
import torch.nn as nn

SITES = ("h1", "h2")


class MLP(nn.Module):
    def __init__(self, n_in=12, width=64, n_out=6, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.l1 = nn.Linear(n_in, width)
        self.l2 = nn.Linear(width, width)
        self.l3 = nn.Linear(width, n_out)
        for lin in (self.l1, self.l2, self.l3):
            nn.init.kaiming_uniform_(lin.weight, a=5**0.5, generator=g)
            bound = 1 / lin.weight.shape[1] ** 0.5
            nn.init.uniform_(lin.bias, -bound, bound, generator=g)

    @staticmethod
    def encode(X):
        return torch.as_tensor(X, dtype=torch.float32) * 2 - 1

    def sites(self, x):
        h1 = torch.relu(self.l1(x))
        h2 = torch.relu(self.l2(h1))
        return {"h1": h1, "h2": h2}

    def from_site(self, site, z):
        """Continue the forward pass from a (possibly patched) site value."""
        if site == "h1":
            z = torch.relu(self.l2(z))
        return self.l3(z)

    def forward(self, x):
        return self.from_site("h2", self.sites(x)["h2"])
