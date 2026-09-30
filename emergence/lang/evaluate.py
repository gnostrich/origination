"""Evaluation of an extracted interface model: counterfactual fidelity on a
held-out intervention suite, description complexity, baselines, causal
abstraction, compositional generalisation, basis robustness.

This module may unseal a reference world (`unseal`) — for description only.
"""

from __future__ import annotations

import warnings

import numpy as np

from emergence.lang.substrate import js_distance

warnings.filterwarnings("ignore", message="Mean of empty slice")
from emergence.lang.discover import all_chunks, extract

BITS_PER_PARAM = 32


# ------------------------------------------------------------ held-out suite
def heldout_suite(sub, rng, discovery_prefixes, n_pieces=120, piece_len=(5, 12), n_strings=40, string_len=8, withheld=()):
    """Fresh pieces (prefixes never in the discovery pool) × fresh action strings
    (length 8, never used as discovery contexts or fit chunks).  Also a
    compositional set: strings that contain withheld chunks."""
    seen = set(map(tuple, discovery_prefixes))
    pieces = []
    while len(pieces) < n_pieces:
        p = tuple(rng.integers(0, sub.n_actions, rng.integers(piece_len[0], piece_len[1] + 1)))
        if p not in seen:
            seen.add(p)
            pieces.append(p)
    strings = [tuple(rng.integers(0, sub.n_actions, string_len)) for _ in range(n_strings)]
    comp_strings, comp_pos = [], []
    withheld = list(map(tuple, withheld))
    for _ in range(n_strings):
        if not withheld:
            break
        s, pos = [], []
        while len(s) < string_len:
            if rng.random() < 0.5 and len(s) + 2 <= string_len:
                w = withheld[rng.integers(len(withheld))]
                s.extend(w)
                pos.append(len(s))       # the step index right after the withheld chunk completes
            else:
                s.append(int(rng.integers(0, sub.n_actions)))
        comp_strings.append(tuple(s))
        comp_pos.append([p for p in pos if p < string_len])
    X = sub.pieces(pieces)
    S = sub.behave(X, strings)                         # (n, Q, L, o)
    Sc = sub.behave(X, comp_strings) if comp_strings else None
    return dict(pieces=pieces, X=X, strings=strings, S=S, comp_strings=comp_strings, comp_pos=comp_pos, Sc=Sc)


# ------------------------------------------------------------ prediction
def abstract_states(model, sub, X, contexts):
    """The abstraction map α: type each piece by its behaviour in the discovery contexts."""
    sig = sub.behave(X, contexts)
    t, d = model.classify(sig)
    return t, d


def predict_intervention(model, state, string):
    """M(q) without the substrate: (L, o) array with NaN rows where M abstains."""
    out = model.run(state, string)
    P = np.full((len(string), model.n_obs), np.nan)
    for i, p in enumerate(out):
        if p is not None:
            P[i] = p
    return P


def _score(P_pred, P_true):
    """Per-step argmax agreement and 1 − JS distance; abstentions count as 0."""
    ok = ~np.isnan(P_pred[..., 0])
    agree = np.zeros(ok.shape)
    fid_js = np.zeros(ok.shape)
    if ok.any():
        agree[ok] = (P_pred[ok].argmax(-1) == P_true[ok].argmax(-1))
        fid_js[ok] = 1 - js_distance(P_pred[ok], P_true[ok])
    return agree, fid_js, ok


def evaluate_fidelity(model, sub, suite, contexts):
    states, _ = abstract_states(model, sub, suite["X"], contexts)
    n, Q, L, o = suite["S"].shape
    P = np.stack([np.stack([predict_intervention(model, states[i], s) for s in suite["strings"]]) for i in range(n)])
    agree, fid_js, ok = _score(P, suite["S"])
    res = dict(fidelity_argmax=float(agree.mean()), fidelity_js=float(fid_js.mean()), coverage=float(ok.mean()),
               fidelity_argmax_covered=float(agree[ok].mean()) if ok.any() else float("nan"),
               abstained_pieces=float((states < 0).mean()),
               by_step=[float(agree[:, :, t].mean()) for t in range(L)])
    if suite["Sc"] is not None:
        Pc = np.stack([np.stack([predict_intervention(model, states[i], s) for s in suite["comp_strings"]]) for i in range(n)])
        ac, jc, okc = _score(Pc, suite["Sc"])
        mask = np.zeros(ac.shape, dtype=bool)
        for q, pos in enumerate(suite["comp_pos"]):
            for p in pos:
                mask[:, q, p] = True
        res.update(comp_fidelity_argmax=float(ac[mask].mean()) if mask.any() else float("nan"),
                   comp_fidelity_js=float(jc[mask].mean()) if mask.any() else float("nan"),
                   comp_coverage=float(okc[mask].mean()) if mask.any() else float("nan"),
                   comp_fidelity_all_steps=float(ac.mean()))
    return res


# ------------------------------------------------------------ complexity
def measure_complexity(model):
    K, A, o = model.K, model.n_actions, model.n_obs
    first = [f for (s, c), f in model.fits.items() if len(c) == 1]
    bits_fits = len(first) * np.log2(K + 1)
    bits_emit_argmax = K * A * np.log2(o)
    bits_emit_dist = K * A * o * 8
    exceptions = sum(round((1 - f.confidence) * f.n) for f in first if np.isfinite(f.confidence))
    bits_exceptions = exceptions * (np.log2(K + 1) + np.log2(max(K * A, 2)))
    residual = float(np.mean([f.dst < 0 for f in first])) if first else 1.0
    return dict(K=K, n_first_fits=len(first), exceptions=int(exceptions), residual_undefined=residual,
                bits_interfaces=float(K * np.log2(K + 1)), bits_fits=float(bits_fits),
                bits_emissions_argmax=float(bits_emit_argmax), bits_emissions_dist=float(bits_emit_dist),
                bits_exceptions=float(bits_exceptions),
                total_bits_argmax=float(K * np.log2(K + 1) + bits_fits + bits_emit_argmax + bits_exceptions),
                total_bits_dist=float(K * np.log2(K + 1) + bits_fits + bits_emit_dist + bits_exceptions))


def substrate_bits(model_torch):
    return float(sum(p.numel() for p in model_torch.parameters()) * BITS_PER_PARAM)


# ------------------------------------------------------------ baselines
def random_abstraction(model, sub, disc, suite, rng):
    """Same number of interfaces and the same class sizes, realisations assigned
    at random; tables refitted by majority from the same observations."""
    from emergence.lang.model import InterfaceModel, Fit
    from emergence.lang.discover import discover_fits, compose
    labels = disc["labels"].copy()
    perm = rng.permutation(len(labels))
    labels = labels[perm]                                 # random partition with identical sizes
    S = disc["S"]
    K = model.K
    leaders = np.array([np.nanmean(S[labels == k], axis=0) for k in range(K)])
    real = [np.nonzero(labels == k)[0].tolist() for k in range(K)]
    within = np.array([np.nanmean(js_distance(S[labels == k], leaders[k][None])) for k in range(K)])
    ctx_index = {c: i for i, c in enumerate(model.contexts)}
    em = np.zeros((K, model.n_actions, model.n_obs))
    for a in range(model.n_actions):
        em[:, a] = leaders[:, ctx_index[(a,)], 0]
    rm = InterfaceModel(eps=np.inf, n_actions=model.n_actions, n_obs=model.n_obs, contexts=model.contexts,
                        leaders=leaders, realizations=real, within=within, emissions=em)
    # classification for the random model: nearest random-class centroid (no ε cutoff)
    observe = [c for c in all_chunks(model.n_actions, 2) if c not in set(model.withheld)]
    discover_fits(sub, rm, disc["X"], labels, observe, model.contexts, rng=rng)
    compose(rm, 2)
    return rm


def lookup_baseline(model, sub, disc, suite, contexts):
    """Non-compositional catalogue with the same observation budget: for each
    interface, the observed outputs along every observed chunk (length ≤ 2);
    a query is answered step by step while its prefix is an observed chunk of
    the piece's interface, then the model abstains.  Complexity: the stored
    observations."""
    states, _ = abstract_states(model, sub, suite["X"], contexts)
    table = {}
    obs_chunks = [c for c in all_chunks(model.n_actions, 2) if c not in set(model.withheld)]
    for k in range(model.K):
        idx = np.array(model.realizations[k])[:12]
        for c in obs_chunks:
            table[(k, c)] = sub.behave(disc["X"][idx], [c])[:, 0].mean(0)   # (L, o) mean observed output
    def predict(state, string):
        P = np.full((len(string), model.n_obs), np.nan)
        if state < 0:
            return P
        for t in range(len(string)):
            c = tuple(string[: t + 1])
            if len(c) > 2 or (state, c) not in table:
                break
            P[t] = table[(state, c)][t]
        return P
    n = suite["S"].shape[0]
    P = np.stack([np.stack([predict(states[i], s) for s in suite["strings"]]) for i in range(n)])
    agree, fid_js, ok = _score(P, suite["S"])
    res = dict(fidelity_argmax=float(agree.mean()), fidelity_js=float(fid_js.mean()), coverage=float(ok.mean()))
    if suite["Sc"] is not None:
        Pc = np.stack([np.stack([predict(states[i], s) for s in suite["comp_strings"]]) for i in range(n)])
        ac, jc, okc = _score(Pc, suite["Sc"])
        mask = np.zeros(ac.shape, dtype=bool)
        for q, pos in enumerate(suite["comp_pos"]):
            for p in pos:
                mask[:, q, p] = True
        res["comp_fidelity_argmax"] = float(ac[mask].mean()) if mask.any() else float("nan")
    res["bits"] = float(len(table) * 2 * np.log2(model.n_obs))   # argmax per observed step
    res["bits_dist"] = float(len(table) * 2 * model.n_obs * 8)
    return res


# ------------------------------------------------------------ causal abstraction
def causal_abstraction(model, sub, suite, contexts, n_real=5, n_strings=10, rng=None):
    """do(I = i) through several distinct realisations of i: agreement of the
    substrate's downstream behaviour across realisations (commutation of the
    square) and with M's abstract prediction."""
    rng = rng or np.random.default_rng(0)
    states, _ = abstract_states(model, sub, suite["X"], contexts)
    strings = suite["strings"][:n_strings]
    S = suite["S"][:, :n_strings]                       # (n, Q, L, o)
    pair_agree, abs_agree, per_type = [], [], {}
    for k in range(model.K):
        idx = np.nonzero(states == k)[0]
        if len(idx) < 2:
            continue
        idx = idx[:n_real]
        A = S[idx].argmax(-1)                            # (r, Q, L)
        pa = np.mean([(A[i] == A[j]).mean() for i in range(len(idx)) for j in range(i + 1, len(idx))])
        P = np.stack([predict_intervention(model, k, s) for s in strings])
        ok = ~np.isnan(P[..., 0])
        aa = np.mean([(np.where(ok, P.argmax(-1) == A[i], False)).mean() for i in range(len(idx))])
        pair_agree.append(pa); abs_agree.append(aa); per_type[k] = (float(pa), float(aa), int(len(idx)))
    return dict(commutation=float(np.mean(pair_agree)) if pair_agree else float("nan"),
                abstract_agreement=float(np.mean(abs_agree)) if abs_agree else float("nan"),
                n_types_tested=len(pair_agree), per_type=per_type)


# ------------------------------------------------------------ basis robustness
def basis_robustness(sub_plain, sub_basis, eps, seed, withhold):
    """Extract from the substrate and from its change-of-basis wrapper with the
    same seed; compare types and fit tables."""
    from emergence.grok.compare import adjusted_rand_index as ari
    m1, d1 = extract(sub_plain, eps, np.random.default_rng(seed), withhold=withhold)
    m2, d2 = extract(sub_basis, eps, np.random.default_rng(seed), withhold=withhold)
    a = float(ari(d1["labels"], d2["labels"]))
    same_K = m1.K == m2.K
    # fit tables compared through the type correspondence given by the labels
    corr = {}
    for l1, l2 in zip(d1["labels"], d2["labels"]):
        corr.setdefault(l1, l2)
    agree = np.mean([corr.get(f.dst, -2) == m2.fits[(corr[s], c)].dst for (s, c), f in m1.fits.items()
                     if len(c) == 1 and s in corr and (corr[s], c) in m2.fits]) if same_K else 0.0
    return dict(ari=a, same_K=bool(same_K), K1=m1.K, K2=m2.K, fit_table_agreement=float(agree))


# ------------------------------------------------------------ unsealing (description only)
def unseal(model, disc, world):
    """Describe the extracted language relative to the world's minimal automaton
    and transformation monoid.  Never used for selection."""
    from emergence.grok.compare import adjusted_rand_index as ari
    from emergence.grok.world import transformation_monoid
    m, table = world.minimal_automaton()
    # world state of each discovery prefix
    st = np.full(len(disc["prefixes"]), world.init_state)
    for i, p in enumerate(disc["prefixes"]):
        s = world.init_state
        for a in p:
            s = int(world.step_table[s, a])
        st[i] = s
    ref = transformation_monoid(table)
    # monoid generated by the discovered first-order maps on interfaces (if total)
    K = model.K
    T = np.array([[model.fits[(k, (a,))].dst for a in range(model.n_actions)] for k in range(K)])
    if (T >= 0).all():
        got = transformation_monoid(T)
    else:
        got = {"order": None, "note": "first-order fits not total"}
    return dict(ari_types_vs_world_states=float(ari(disc["labels"], st)), world_min_states=int(m), K=K,
                world_monoid=ref, discovered_monoid=got,
                n_second_order=len(model.chunk_maps))
