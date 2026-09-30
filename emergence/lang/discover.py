"""Discovery: interfaces, fits, composition — from behaviour only.

This module never imports a world, a task or a teacher.  It receives a
substrate (pieces, interactions, behaviour), an unlabelled family of
contexts, and a resolution ε.
"""

from __future__ import annotations

from itertools import product

import numpy as np

from emergence.lang.model import Fit, InterfaceModel
from emergence.lang.substrate import js_distance


def all_chunks(n_actions, max_len):
    out = []
    for l in range(1, max_len + 1):
        out.extend(product(range(n_actions), repeat=l))
    return out


def discovery_pool(n_actions, rng, max_len=3, n_random=200, rand_len=8):
    """Unlabelled pieces: every prefix up to max_len plus random longer ones."""
    pre = [()] + all_chunks(n_actions, max_len)
    pre += [tuple(rng.integers(0, n_actions, rand_len)) for _ in range(n_random)]
    return pre


def leader_cluster(signatures, eps):
    """Greedy ε-typing of behaviour signatures (n, C, L, o) by mean JS distance to the leader."""
    n = signatures.shape[0]
    labels = np.full(n, -1, dtype=int)
    leaders = []
    for i in range(n):
        if leaders:
            d = np.nanmean(js_distance(signatures[leaders], signatures[i][None]), axis=(1, 2))
            j = int(np.argmin(d))
            if d[j] < eps:
                labels[i] = j
                continue
        leaders.append(i)
        labels[i] = len(leaders) - 1
    return labels, np.array(leaders)


def discover_interfaces(sub, prefixes, contexts, eps):
    """Interfaces = ε-classes of pieces under substitution across `contexts`."""
    X = sub.pieces(prefixes)
    S = sub.behave(X, contexts)                       # (n, C, L, o)
    labels, leaders = leader_cluster(S, eps)
    K = len(leaders)
    within = np.array([np.nanmean(js_distance(S[labels == k], S[leaders[k]][None])) for k in range(K)])
    # emissions: the effect of each interface under each primitive action = its leader's first-step output
    ctx_index = {c: i for i, c in enumerate(contexts)}
    n_actions = sub.n_actions
    em = np.zeros((K, n_actions, sub.n_obs))
    for a in range(n_actions):
        ci = ctx_index[(a,)]
        em[:, a] = S[leaders, ci, 0]
    model = InterfaceModel(eps=eps, n_actions=n_actions, n_obs=sub.n_obs, contexts=list(contexts),
                           leaders=S[leaders], realizations=[np.nonzero(labels == k)[0].tolist() for k in range(K)],
                           within=within, emissions=em)
    return model, X, S, labels


def discover_fits(sub, model, X, labels, chunks, contexts, max_real=12, rng=None):
    """For every interface and every chunk in `chunks`: continue up to `max_real`
    realisations, type the results, record the majority interface and confidence."""
    rng = rng or np.random.default_rng(0)
    for k in range(model.K):
        idx = np.array(model.realizations[k])
        if len(idx) > max_real:
            idx = rng.choice(idx, max_real, replace=False)
        for chunk in chunks:
            Y = sub.step(X[idx], chunk)
            Sy = sub.behave(Y, contexts)
            t, _ = model.classify(Sy)
            vals, counts = np.unique(t, return_counts=True)
            j = int(np.argmax(counts))
            model.fits[(k, tuple(chunk))] = Fit(k, tuple(chunk), int(vals[j]), float(counts[j] / len(t)), int(len(t)), True)
    return model


def compose(model, max_len=2):
    """Composition structure.

    * derived fits: chunks up to `max_len` not observed during discovery are
      predicted by chaining first-order fits (these are the compositional
      predictions);
    * second-order interfaces: chunks with the same induced map on interfaces;
    * second-order fits: (W1, W2) ⇝ W3 whenever the concatenation of members
      has the map of W3.
    """
    for chunk in all_chunks(model.n_actions, max_len):
        for k in range(model.K):
            if (k, chunk) not in model.fits:
                model.fits[(k, chunk)] = Fit(k, chunk, model.run_chunk_map(k, chunk), float("nan"), 0, False)
    maps = {}
    for chunk in all_chunks(model.n_actions, max_len):
        m = tuple(model.fits[(k, chunk)].dst for k in range(model.K))
        if m not in maps:
            maps[m] = len(maps)
        model.chunk_types[chunk] = maps[m]
    model.chunk_maps = {w: m for m, w in maps.items()}
    inv = {}
    for chunk, w in model.chunk_types.items():
        inv.setdefault(w, []).append(chunk)
    for w1, c1s in inv.items():
        for w2, c2s in inv.items():
            cat = c1s[0] + c2s[0]
            if cat in model.chunk_types:
                model.second_fits[(w1, w2)] = model.chunk_types[cat]
            else:  # compose the maps
                m1, m2 = model.chunk_maps[w1], model.chunk_maps[w2]
                m = tuple(-1 if m1[k] < 0 else m2[m1[k]] for k in range(model.K))
                if m in maps:
                    model.second_fits[(w1, w2)] = maps[m]
    return model


def extract(sub, eps, rng, ctx_len=2, withhold=None, max_real=12, pool_kw=None):
    """Full discovery pipeline.  `withhold`: chunks (tuples) excluded from fit
    observation; their fits are only derived by chaining."""
    prefixes = discovery_pool(sub.n_actions, rng, **(pool_kw or {}))
    contexts = all_chunks(sub.n_actions, ctx_len)
    model, X, S, labels = discover_interfaces(sub, prefixes, contexts, eps)
    withhold = set(map(tuple, withhold or []))
    observe = [c for c in all_chunks(sub.n_actions, 2) if c not in withhold]
    model.withheld = sorted(withhold)
    discover_fits(sub, model, X, labels, observe, contexts, max_real=max_real, rng=rng)
    compose(model, max_len=2)
    return model, dict(prefixes=prefixes, X=X, S=S, labels=labels, contexts=contexts)
