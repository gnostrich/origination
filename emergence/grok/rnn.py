"""Recurrent substrate: a GRU reading action tokens and predicting observations.

Site interface (the only thing the extractor uses):

    forward(actions, h0=None, noise=None) -> (logits (B, L, n_obs), {"hidden": (B, L, d)})

``h0`` substitutes the hidden state before the first action of the batch
(``(B, d)`` or ``(1, d)`` broadcast); ``noise={"hidden": sigma}`` adds
Gaussian noise of ``sigma`` x RMS to ``h0``.  "Continue from configuration
h with suffix u" is therefore ``forward(u, h0=h)``.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class GRUWorldModel(nn.Module):
    site_names = ["hidden"]
    hidden_site = "hidden"

    def __init__(self, n_actions: int, n_obs: int, d: int = 128, d_embed: int = 32):
        super().__init__()
        self.d = d
        self.emb = nn.Embedding(n_actions, d_embed)
        self.gru = nn.GRU(d_embed, d, batch_first=True)
        self.out = nn.Linear(d, n_obs)
        self.h_init = nn.Parameter(torch.zeros(1, d))

    def forward(self, actions, h0=None, noise=None):
        B = actions.shape[0]
        h = self.h_init.expand(B, -1) if h0 is None else h0
        if h.shape[0] != B:
            h = h.expand(B, -1)
        if noise is not None and noise.get("hidden", 0) > 0:
            rms = h.detach().pow(2).mean(dim=-1, keepdim=True).sqrt().mean()
            h = h + noise["hidden"] * rms * torch.randn_like(h)
        x = self.emb(actions)
        hs, _ = self.gru(x, h.unsqueeze(0).contiguous())
        return self.out(hs), {"hidden": hs}

    def initial(self):
        return self.h_init.detach()
