"""Two substrates for the same task, exposing the same site interface.

``forward(tokens, patch=None, noise=None) -> (logits, sites)``

* ``patch``: ``{site: tensor}`` replaces the configuration at that site.
* ``noise``: ``{site: sigma}`` adds Gaussian noise of ``sigma`` times the
  site's per-batch RMS to the configuration at that site (after patching).
* ``sites``: ``{site: tensor}`` the configurations actually used downstream.

Sites are just hook points.  Their names are for bookkeeping; the extraction
treats them as opaque.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _apply(site, h, patch, noise, out):
    if patch is not None and site in patch:
        h = patch[site]
        if h.shape[0] != out["_B"]:
            h = h.expand(out["_B"], *h.shape[1:])
    if noise is not None and site in noise and noise[site] > 0:
        rms = h.detach().pow(2).mean(dim=-1, keepdim=True).sqrt().mean()
        h = h + noise[site] * rms * torch.randn_like(h)
    out[site] = h
    return h


class Transformer1L(nn.Module):
    """One attention block + MLP, no LayerNorm (as in the standard grokking setup).

    Sites: ``embed.0``, ``embed.1`` (token+position embeddings at the two
    operand positions), ``resid.attn`` (residual stream at the last position
    after attention), ``resid.mlp`` (after the MLP; this feeds the unembed),
    ``logits``.
    """

    site_names = ["embed.0", "embed.1", "resid.attn", "resid.mlp", "logits"]
    input_sites = ["embed.0", "embed.1"]
    hidden_site = "resid.mlp"

    def __init__(self, p: int, d_model: int = 128, n_heads: int = 4, d_mlp: int = 512, seq: int = 3):
        super().__init__()
        self.p = p
        self.d = d_model
        self.h = n_heads
        self.tok = nn.Embedding(p + 1, d_model)
        self.pos = nn.Parameter(torch.randn(seq, d_model) * 0.02)
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.o = nn.Linear(d_model, d_model, bias=False)
        self.mlp_in = nn.Linear(d_model, d_mlp)
        self.mlp_out = nn.Linear(d_mlp, d_model)
        self.unembed = nn.Linear(d_model, p, bias=False)

    def forward(self, tokens, patch=None, noise=None):
        B, L = tokens.shape
        out = {"_B": B}
        x = self.tok(tokens) + self.pos[:L][None]
        e0 = _apply("embed.0", x[:, 0], patch, noise, out)
        e1 = _apply("embed.1", x[:, 1], patch, noise, out)
        x = torch.stack([e0, e1, x[:, 2]], dim=1)
        q, k, v = self.qkv(x).split(self.d, dim=-1)
        hd = self.d // self.h
        q = q.view(B, L, self.h, hd).transpose(1, 2)
        k = k.view(B, L, self.h, hd).transpose(1, 2)
        v = v.view(B, L, self.h, hd).transpose(1, 2)
        att = (q @ k.transpose(-1, -2)) / hd**0.5
        mask = torch.triu(torch.ones(L, L, dtype=torch.bool, device=tokens.device), 1)
        att = att.masked_fill(mask, float("-inf")).softmax(-1)
        a = (att @ v).transpose(1, 2).reshape(B, L, self.d)
        x = x + self.o(a)
        r = _apply("resid.attn", x[:, -1], patch, noise, out)
        r = r + self.mlp_out(F.relu(self.mlp_in(r)))
        r = _apply("resid.mlp", r, patch, noise, out)
        logits = _apply("logits", self.unembed(r), patch, noise, out)
        del out["_B"]
        return logits, out


class MLPAdder(nn.Module):
    """Embeddings of the two operands concatenated into a 2-layer ReLU MLP.

    Sites: ``embed.0``, ``embed.1``, ``hidden`` (the last hidden layer, which
    feeds the readout), ``logits``.
    """

    site_names = ["embed.0", "embed.1", "hidden", "logits"]
    input_sites = ["embed.0", "embed.1"]
    hidden_site = "hidden"

    def __init__(self, p: int, d_embed: int = 128, d_hidden: int = 512):
        super().__init__()
        self.p = p
        self.tok = nn.Embedding(p + 1, d_embed)
        self.l1 = nn.Linear(2 * d_embed, d_hidden)
        self.l2 = nn.Linear(d_hidden, d_hidden)
        self.out = nn.Linear(d_hidden, p)

    def forward(self, tokens, patch=None, noise=None):
        B = tokens.shape[0]
        out = {"_B": B}
        e = self.tok(tokens)
        e0 = _apply("embed.0", e[:, 0], patch, noise, out)
        e1 = _apply("embed.1", e[:, 1], patch, noise, out)
        h = F.relu(self.l1(torch.cat([e0, e1], dim=-1)))
        h = F.relu(self.l2(h))
        h = _apply("hidden", h, patch, noise, out)
        logits = _apply("logits", self.out(h), patch, noise, out)
        del out["_B"]
        return logits, out


ARCHS = {"transformer": Transformer1L, "mlp": MLPAdder}


def build(arch: str, p: int, **kw) -> nn.Module:
    return ARCHS[arch](p, **kw)
