"""Candidate interfaces: arbitrary combinations of hook sites and temporal positions.

A *cell* is ``(site, offset)``: the value carried at hook ``site`` for the
prefix position ``offset`` steps before the end of the prefix (offset 0 =
the last position).  A candidate *interface* is a set of cells.

Substitution.  Given a donor prefix ``p`` and a recipient prefix ``q`` of the
same length, the interface cells of ``p``'s execution trace are written into
``q``'s trace; whatever downstream part of the trace depends on the
substituted cells is recomputed (for the GRU the recurrence from the earliest
changed position; for the transformer the cache *is* what is carried
forward, so cache cells are patched directly and later positions attend to
the patched values).  Then both are continued with the same suffixes.

Sufficiency (label-free).  An interface ``I`` is sufficient when the
transplanted recipient behaves like the donor:

    S(I) = P_{p, q, suffixes}[ JS( behaviour(q <- I from p), behaviour(p) ) < eps ].

Nothing about the world's states or algebra is used.  Complexity is the
number of scalars the interface carries.  The search returns the smallest
interface with ``S >= target``.

The discovered interface is then wrapped as a sequence substrate
(``InterfaceSubstrate``) whose configuration is *only* the interface cells,
plugged into a fixed reference recipient of the same length, so that the
unchanged quotient/algebra extractor can be run on it.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from .extract import _js
from .seqsub import KVCache


# ---------------------------------------------------------------------------
# traces


@dataclass
class GRUTrace:
    e: torch.Tensor  # (B, L, d_e) action embeddings
    h: torch.Tensor  # (B, L + 1, d) hidden states, h[:, 0] = initial


class GRUTraceSubstrate:
    """Cells: ('h', off) for off in 0..L-1 (h after position L-1-off, i.e. off 0 = final hidden),
    ('e', off) the embedding of the action at position L-1-off."""

    def __init__(self, model):
        self.model = model
        self.sites = ["h", "e"]

    def cell_size(self, cell):
        return self.model.d if cell[0] == "h" else self.model.emb.embedding_dim

    def cells(self, L, max_off):
        return [(s, o) for o in range(min(L, max_off)) for s in ("h", "e")]

    def trace(self, actions):
        B, L = actions.shape
        with torch.no_grad():
            e = self.model.emb(actions)
            h = self.model.h_init.expand(B, -1).unsqueeze(1)
            hs = [h]
            for t in range(L):
                out, _ = self.model.gru(e[:, t : t + 1], hs[-1].transpose(0, 1).contiguous())
                hs.append(out)
            return GRUTrace(e, torch.cat(hs, dim=1))

    def substitute(self, rec: GRUTrace, don: GRUTrace, cells):
        L = rec.e.shape[1]
        e = rec.e.clone(); h = rec.h.clone()
        earliest = L + 1  # first hidden index that must be recomputed
        for site, off in cells:
            t = L - 1 - off
            if site == "e":
                e[:, t] = don.e[:, t]
                earliest = min(earliest, t + 1)
            else:
                h[:, t + 1] = don.h[:, t + 1]
                earliest = min(earliest, t + 2)
        with torch.no_grad():
            for idx in range(earliest, L + 1):
                out, _ = self.model.gru(e[:, idx - 1 : idx], h[:, idx - 1].unsqueeze(0).contiguous())
                h[:, idx] = out[:, 0]
        return GRUTrace(e, h)

    def continue_(self, tr: GRUTrace, actions):
        with torch.no_grad():
            lg, _ = self.model(actions, h0=tr.h[:, -1])
        return lg

    def noise(self, tr: GRUTrace, cells, sigma):
        L = tr.e.shape[1]
        e = tr.e.clone(); h = tr.h.clone()
        earliest = L + 1
        for site, off in cells:
            t = L - 1 - off
            if site == "e":
                x = e[:, t]; rms = x.pow(2).mean(-1, keepdim=True).sqrt().mean()
                e[:, t] = x + sigma * rms * torch.randn_like(x); earliest = min(earliest, t + 1)
            else:
                x = h[:, t + 1]; rms = x.pow(2).mean(-1, keepdim=True).sqrt().mean()
                h[:, t + 1] = x + sigma * rms * torch.randn_like(x); earliest = min(earliest, t + 2)
        with torch.no_grad():
            for idx in range(earliest, L + 1):
                out, _ = self.model.gru(e[:, idx - 1 : idx], h[:, idx - 1].unsqueeze(0).contiguous())
                h[:, idx] = out[:, 0]
        return GRUTrace(e, h)

    def select(self, tr: GRUTrace, idx):
        return GRUTrace(tr.e[idx], tr.h[idx])

    def expand(self, tr: GRUTrace, B):
        return GRUTrace(tr.e.expand(B, -1, -1), tr.h.expand(B, -1, -1))

    def batch_size(self, tr):
        return tr.e.shape[0]

    def length(self, tr):
        return tr.e.shape[1]

    def cat(self, trs):
        return GRUTrace(torch.cat([t.e for t in trs]), torch.cat([t.h for t in trs]))

    def extend(self, tr: GRUTrace, actions):
        with torch.no_grad():
            e2 = self.model.emb(actions)
            h = tr.h[:, -1]
            hs = []
            for t in range(actions.shape[1]):
                out, _ = self.model.gru(e2[:, t : t + 1], h.unsqueeze(0).contiguous())
                h = out[:, 0]; hs.append(h.unsqueeze(1))
        return GRUTrace(torch.cat([tr.e, e2], 1), torch.cat([tr.h] + hs, 1))

    def from_scratch(self, actions):
        return self.trace(actions)

    def continue_empty(self, actions):
        with torch.no_grad():
            lg, _ = self.model(actions)
        return lg


class TransformerTraceSubstrate:
    """Cells: ('kv0', off), ('kv1', off): keys+values of layer l at position L-1-off."""

    def __init__(self, model):
        self.model = model
        self.sites = [f"kv{l}" for l in range(model.L)]

    def cell_size(self, cell):
        return 2 * self.model.d

    def cells(self, L, max_off):
        return [(f"kv{l}", o) for o in range(min(L, max_off)) for l in range(self.model.L)]

    def trace(self, actions):
        with torch.no_grad():
            _, cache = self.model(actions)
        return cache

    def substitute(self, rec: KVCache, don: KVCache, cells):
        L = int(rec.lens[0])
        k = rec.k.clone(); v = rec.v.clone()
        for site, off in cells:
            l = int(site[2:]); t = L - 1 - off
            k[l, :, t] = don.k[l, :, t]; v[l, :, t] = don.v[l, :, t]
        return KVCache(k, v, rec.lens)

    def continue_(self, tr: KVCache, actions):
        with torch.no_grad():
            lg, _ = self.model(actions, cache=tr)
        return lg

    def noise(self, tr: KVCache, cells, sigma):
        L = int(tr.lens[0])
        k = tr.k.clone(); v = tr.v.clone()
        for site, off in cells:
            l = int(site[2:]); t = L - 1 - off
            for X in (k, v):
                x = X[l, :, t]; rms = x.pow(2).mean(-1, keepdim=True).sqrt().mean()
                X[l, :, t] = x + sigma * rms * torch.randn_like(x)
        return KVCache(k, v, tr.lens)

    def select(self, tr: KVCache, idx):
        idx = torch.as_tensor(idx)
        return KVCache(tr.k[:, idx], tr.v[:, idx], tr.lens[idx])

    def expand(self, tr: KVCache, B):
        return KVCache(tr.k.expand(-1, B, -1, -1), tr.v.expand(-1, B, -1, -1), tr.lens.expand(B))

    def batch_size(self, tr):
        return tr.k.shape[1]

    def length(self, tr):
        return int(tr.lens[0])

    def cat(self, trs):
        return KVCache(torch.cat([t.k for t in trs], 1), torch.cat([t.v for t in trs], 1), torch.cat([t.lens for t in trs]))

    def extend(self, tr: KVCache, actions):
        with torch.no_grad():
            _, cache = self.model(actions, cache=tr)
        return cache

    def from_scratch(self, actions):
        return self.trace(actions)

    def continue_empty(self, actions):
        with torch.no_grad():
            lg, _ = self.model(actions)
        return lg


def trace_substrate(arch, model):
    return GRUTraceSubstrate(model) if arch == "gru" else TransformerTraceSubstrate(model)


# ---------------------------------------------------------------------------
# sufficiency and search


def _behaviour(sub, tr, suffixes):
    n = sub.batch_size(tr)
    feats = []
    for s in range(suffixes.shape[0]):
        lg = sub.continue_(tr, suffixes[s][None].expand(n, -1))
        feats.append(F.softmax(lg, -1))
    return torch.cat(feats, dim=1).numpy()  # (n, S*Ls, n_obs)


def sufficiency(sub, cells, donors_tr, recips_tr, don_beh, suffixes, eps):
    """Fraction of (donor, recipient) pairs whose transplant reproduces the donor's behaviour."""
    if not cells:
        beh = _behaviour(sub, recips_tr, suffixes)
    else:
        beh = _behaviour(sub, sub.substitute(recips_tr, donors_tr, cells), suffixes)
    d = _js(beh, don_beh).mean(axis=1)
    return float(np.mean(d < eps)), float(d.mean())


def search_interface(sub, n_actions, L, rng, n_pairs=48, n_suffixes=8, suffix_len=5, eps=0.05,
                     target=0.95, max_off=None, exhaustive_up_to=3):
    """Smallest sufficient interface by complexity; exhaustive over small
    subsets, then greedy forward selection.  Returns the search log."""
    max_off = L if max_off is None else max_off
    cells = sub.cells(L, max_off)
    donors = torch.tensor(rng.integers(0, n_actions, (n_pairs, L)), dtype=torch.long)
    recips = torch.tensor(rng.integers(0, n_actions, (n_pairs, L)), dtype=torch.long)
    suffixes = torch.tensor(rng.integers(0, n_actions, (n_suffixes, suffix_len)), dtype=torch.long)
    dtr, rtr = sub.trace(donors), sub.trace(recips)
    don_beh = _behaviour(sub, dtr, suffixes)
    size = lambda I: sum(sub.cell_size(c) for c in I)
    log = []
    base_S, base_d = sufficiency(sub, [], dtr, rtr, don_beh, suffixes, eps)
    log.append({"cells": [], "complexity": 0, "sufficiency": base_S, "mean_js": base_d})
    best = None
    # exhaustive by subset size (candidates of equal size are compared by complexity)
    for k in range(1, min(exhaustive_up_to, len(cells)) + 1):
        cands = []
        for I in itertools.combinations(cells, k):
            S, d = sufficiency(sub, list(I), dtr, rtr, don_beh, suffixes, eps)
            log.append({"cells": list(I), "complexity": size(I), "sufficiency": S, "mean_js": d})
            if S >= target:
                cands.append((size(I), -S, list(I)))
        if cands:
            cands.sort()
            best = {"cells": cands[0][2], "complexity": cands[0][0], "sufficiency": -cands[0][1], "method": f"exhaustive size {k}"}
            break
    if best is None:
        I = []
        remaining = list(cells)
        while remaining:
            scored = []
            for c in remaining:
                S, d = sufficiency(sub, I + [c], dtr, rtr, don_beh, suffixes, eps)
                scored.append((S, -d, c))
            scored.sort(reverse=True)
            S, d, c = scored[0]
            I.append(c); remaining.remove(c)
            log.append({"cells": list(I), "complexity": size(I), "sufficiency": S, "mean_js": -d, "greedy": True})
            if S >= target:
                best = {"cells": list(I), "complexity": size(I), "sufficiency": S, "method": "greedy"}
                break
        if best is None:
            best = {"cells": list(I), "complexity": size(I), "sufficiency": S, "method": "greedy (target not reached)"}
    full_S, full_d = sufficiency(sub, cells, dtr, rtr, don_beh, suffixes, eps)
    return {"L": L, "all_cells": cells, "full_sufficiency": full_S, "full_complexity": size(cells),
            "baseline_sufficiency": base_S, "best": best, "log": log}


# ---------------------------------------------------------------------------
# the discovered interface as a sequence substrate for the quotient extractor


class InterfaceSubstrate:
    """Configuration = the interface cells of a prefix's trace, written into a
    fixed reference recipient of the same length (a fixed random action
    sequence per length).  If the interface is sufficient the reference does
    not matter; if it is not, the extractor will see it.  The empty prefix is
    the substrate's genuine initial configuration."""

    def __init__(self, sub, cells, n_actions, rng, max_len=24):
        self.sub = sub
        self.cells = list(cells)
        self.k = n_actions
        self.ref = {L: torch.tensor(rng.integers(0, n_actions, (1, L)), dtype=torch.long) for L in range(1, max_len + 1)}
        self._ref_tr = {}

    def _plug(self, tr):
        L = self.sub.length(tr)
        B = self.sub.batch_size(tr)
        if L not in self._ref_tr:
            self._ref_tr[L] = self.sub.trace(self.ref[L])
        ref = self.sub.expand(self._ref_tr[L], B)
        usable = [c for c in self.cells if c[1] < L]
        return self.sub.substitute(ref, tr, usable) if usable else ref

    def _split(self, tr):
        B = self.sub.batch_size(tr)
        return [self.sub.select(tr, torch.tensor([j])) for j in range(B)]

    def run_prefixes(self, prefixes):
        out = [None] * len(prefixes)
        by_len = {}
        for i, p in enumerate(prefixes):
            by_len.setdefault(len(p), []).append(i)
        for L, idx in by_len.items():
            if L == 0:
                continue
            A = torch.tensor([prefixes[i] for i in idx], dtype=torch.long)
            for i, item in zip(idx, self._split(self._plug(self.sub.trace(A)))):
                out[i] = item
        return _Batch(out)

    def continue_(self, cfg, actions):
        B = actions.shape[0]
        items = cfg.items if cfg.n() == B else [cfg.items[0]] * B
        groups = {}
        for b, it in enumerate(items):
            key = -1 if it is None else self.sub.length(it)
            groups.setdefault(key, []).append(b)
        logits = [None] * B
        new = [None] * B
        for L, bs in groups.items():
            A = actions[bs]
            if L == -1:
                lg = self.sub.continue_empty(A)
                tr = self.sub.from_scratch(A)
            else:
                base = self.sub.cat([items[b] for b in bs])
                lg = self.sub.continue_(base, A)
                tr = self.sub.extend(base, A)
            plugged = self._split(self._plug(tr))
            for j, b in enumerate(bs):
                logits[b] = lg[j]; new[b] = plugged[j]
        return torch.stack(logits), _Batch(new)

    def noise(self, cfg, sigma):
        groups = {}
        for b, it in enumerate(cfg.items):
            key = -1 if it is None else self.sub.length(it)
            groups.setdefault(key, []).append(b)
        out = [None] * cfg.n()
        for L, bs in groups.items():
            if L == -1:
                continue
            base = self.sub.cat([cfg.items[b] for b in bs])
            usable = [c for c in self.cells if c[1] < L]
            noisy = self._split(self.sub.noise(base, usable, sigma)) if usable else self._split(base)
            for j, b in enumerate(bs):
                out[b] = noisy[j]
        return _Batch(out)

    def select(self, cfg, idx):
        return _Batch([cfg.items[int(i)] for i in np.atleast_1d(idx)])

    def batch_size(self, cfg):
        return cfg.n()


class _Batch:
    def __init__(self, items):
        self.items = items

    def n(self):
        return len(self.items)
