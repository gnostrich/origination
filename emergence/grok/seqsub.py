"""Sequence substrates behind one interface, for the blind sequence extractor.

A sequence substrate exposes *configurations*: whatever the substrate carries
forward after reading a prefix.  The extractor only needs to

    * read the configuration after each prefix         (run_prefixes)
    * continue from a configuration with more actions  (continue_)
    * perturb a configuration                          (noise)
    * pick a subset of a batch of configurations       (select)

For the GRU the configuration is the hidden vector.  For the transformer it
is the key/value cache of the prefix: a variable-size object, but exactly
what the substrate carries forward, so substitution, continuation and noise
have the same meaning.  Nothing else about either architecture is used.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .rnn import GRUWorldModel


class GRUSubstrate:
    def __init__(self, model: GRUWorldModel):
        self.model = model

    def run_prefixes(self, prefixes):
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
        return H

    def continue_(self, cfg, actions):
        with torch.no_grad():
            lg, sites = self.model(actions, h0=cfg)
        return lg, sites["hidden"][:, -1]

    def noise(self, cfg, sigma):
        rms = cfg.pow(2).mean(dim=-1, keepdim=True).sqrt().mean()
        return cfg + sigma * rms * torch.randn_like(cfg)

    def select(self, cfg, idx):
        return cfg[idx]

    def batch_size(self, cfg):
        return cfg.shape[0]


# ---------------------------------------------------------------------------


@dataclass
class KVCache:
    k: torch.Tensor  # (layers, B, Lmax, d)
    v: torch.Tensor  # (layers, B, Lmax, d)
    lens: torch.Tensor  # (B,)


class TransformerWorldModel(nn.Module):
    """Causal transformer reading action tokens; predicts the observation at
    every position.  Supports continuation from a KV cache."""

    def __init__(self, n_actions: int, n_obs: int, d: int = 128, n_heads: int = 4, n_layers: int = 2,
                 d_mlp: int = 256, max_len: int = 32):
        super().__init__()
        self.d, self.h, self.L = d, n_heads, n_layers
        self.emb = nn.Embedding(n_actions, d)
        self.pos = nn.Parameter(torch.randn(max_len, d) * 0.02)
        self.ln1 = nn.ModuleList([nn.LayerNorm(d) for _ in range(n_layers)])
        self.ln2 = nn.ModuleList([nn.LayerNorm(d) for _ in range(n_layers)])
        self.qkv = nn.ModuleList([nn.Linear(d, 3 * d, bias=False) for _ in range(n_layers)])
        self.o = nn.ModuleList([nn.Linear(d, d, bias=False) for _ in range(n_layers)])
        self.mlp = nn.ModuleList([nn.Sequential(nn.Linear(d, d_mlp), nn.GELU(), nn.Linear(d_mlp, d)) for _ in range(n_layers)])
        self.lnf = nn.LayerNorm(d)
        self.out = nn.Linear(d, n_obs)

    def forward(self, actions, cache: KVCache | None = None):
        B, L = actions.shape
        dev = actions.device
        if cache is None:
            lens = torch.zeros(B, dtype=torch.long, device=dev)
            Lc = 0
        else:
            lens = cache.lens
            Lc = cache.k.shape[2]
        pos = lens[:, None] + torch.arange(L, device=dev)[None, :]
        x = self.emb(actions) + self.pos[pos]
        new_k, new_v = [], []
        # mask over keys [cache positions 0..Lc-1 | new positions 0..L-1]
        key_idx = torch.arange(Lc + L, device=dev)[None, None, :]  # (1,1,Lc+L)
        q_pos = torch.arange(L, device=dev)[None, :, None]  # (1,L,1)
        allowed = (key_idx < lens[:, None, None]) | ((key_idx >= Lc) & (key_idx - Lc <= q_pos))  # (B,L,Lc+L)
        hd = self.d // self.h
        for l in range(self.L):
            hln = self.ln1[l](x)
            q, k, v = self.qkv[l](hln).split(self.d, dim=-1)
            new_k.append(k); new_v.append(v)
            if cache is not None:
                k = torch.cat([cache.k[l], k], dim=1)
                v = torch.cat([cache.v[l], v], dim=1)
            qh = q.view(B, L, self.h, hd).transpose(1, 2)
            kh = k.view(B, Lc + L, self.h, hd).transpose(1, 2)
            vh = v.view(B, Lc + L, self.h, hd).transpose(1, 2)
            att = (qh @ kh.transpose(-1, -2)) / hd**0.5
            att = att.masked_fill(~allowed[:, None], float("-inf")).softmax(-1)
            a = (att @ vh).transpose(1, 2).reshape(B, L, self.d)
            x = x + self.o[l](a)
            x = x + self.mlp[l](self.ln2[l](x))
        logits = self.out(self.lnf(x))
        nk = torch.stack(new_k); nv = torch.stack(new_v)  # (layers, B, L, d)
        if cache is None:
            new_cache = KVCache(nk, nv, lens + L)
        else:
            new_cache = KVCache(torch.cat([cache.k, nk], dim=2), torch.cat([cache.v, nv], dim=2), lens + L)
        return logits, new_cache


class TransformerSubstrate:
    def __init__(self, model: TransformerWorldModel):
        self.model = model

    def _pad(self, caches: list[KVCache]) -> KVCache:
        Lmax = max(c.k.shape[2] for c in caches)
        ks, vs, ls = [], [], []
        for c in caches:
            pad = Lmax - c.k.shape[2]
            ks.append(F.pad(c.k, (0, 0, 0, pad)))
            vs.append(F.pad(c.v, (0, 0, 0, pad)))
            ls.append(c.lens)
        return KVCache(torch.cat(ks, dim=1), torch.cat(vs, dim=1), torch.cat(ls))

    def run_prefixes(self, prefixes):
        by_len = {}
        for i, p in enumerate(prefixes):
            by_len.setdefault(len(p), []).append(i)
        n = len(prefixes)
        parts = [None] * n
        Lmax = max(len(p) for p in prefixes)
        d, nl = self.model.d, self.model.L
        K = torch.zeros(nl, n, Lmax, d); V = torch.zeros(nl, n, Lmax, d); lens = torch.zeros(n, dtype=torch.long)
        with torch.no_grad():
            for l, idx in by_len.items():
                if l == 0:
                    continue
                A = torch.tensor([prefixes[i] for i in idx], dtype=torch.long)
                _, c = self.model(A)
                K[:, idx, :l] = c.k; V[:, idx, :l] = c.v; lens[idx] = l
        return KVCache(K, V, lens)

    def continue_(self, cfg: KVCache, actions):
        B = actions.shape[0]
        if cfg.k.shape[1] != B:
            cfg = KVCache(cfg.k.expand(-1, B, -1, -1), cfg.v.expand(-1, B, -1, -1), cfg.lens.expand(B))
        with torch.no_grad():
            lg, new = self.model(actions, cache=cfg)
        return lg, new

    def noise(self, cfg: KVCache, sigma):
        valid = (torch.arange(cfg.k.shape[2])[None, :] < cfg.lens[:, None]).float()[None, :, :, None]
        def nz(t):
            rms = (t.pow(2).sum() / max(valid.sum() * t.shape[-1] * t.shape[0], 1)).sqrt()
            return t + sigma * rms * torch.randn_like(t) * valid
        return KVCache(nz(cfg.k), nz(cfg.v), cfg.lens)

    def select(self, cfg: KVCache, idx):
        idx = torch.as_tensor(idx)
        return KVCache(cfg.k[:, idx], cfg.v[:, idx], cfg.lens[idx])

    def batch_size(self, cfg):
        return cfg.k.shape[1]


def build_seq(arch: str, n_actions: int, n_obs: int):
    if arch == "gru":
        m = GRUWorldModel(n_actions, n_obs)
        return m, GRUSubstrate(m)
    if arch == "transformer":
        m = TransformerWorldModel(n_actions, n_obs)
        return m, TransformerSubstrate(m)
    raise ValueError(arch)
