"""Post-hoc (after the frozen evaluation): each order-restricted language on
its OWN scope.  Re-extracts the ctx-3, seed-0 languages deterministically
and evaluates every language on order-restricted held-out suites (pieces
and continuations in one reading order), informative steps only."""
from __future__ import annotations
import json, os
from itertools import product
import numpy as np
from emergence.lang import discover as D, evaluate as E
from emergence.lang.m2 import withheld_chunks, informative_scores, EPS, OUT
from emergence.lang.systems import ReaderMachine, order_policy_prefixes, N_VAL, QUERY

def order_suite(sub, order, rng, n=120):
    slot = {"A": 0, "B": 1, "D": 2}
    o = [slot[c] for c in order]
    pieces, strings = [], []
    for _ in range(n):
        k = int(rng.integers(0, 3))                     # slots already set, in order
        p = tuple(o[j] * N_VAL + int(rng.integers(0, N_VAL)) for j in range(k))
        pieces.append(p)
        s = [(o[j] * N_VAL + int(rng.integers(0, N_VAL)), True) for j in range(k, 3)] + [(QUERY, False)]
        while len(s) < 6:                                # padding that stays in scope: queries, and no-op re-sets of slots already set at that point
            pos = int(rng.integers(0, len(s) + 1))
            set_before = k + sum(1 for a, orig in s[:pos] if orig)     # slots genuinely set before pos
            if rng.random() < 0.5 or set_before == 0:
                s.insert(pos, (QUERY, False))
            else:
                s.insert(pos, (o[int(rng.integers(0, set_before))] * N_VAL + int(rng.integers(0, N_VAL)), False))
        strings.append(tuple(a for a, _ in s[:6]))
    X = sub.pieces(pieces)
    S = np.stack([sub.run(X[i:i+1], strings[i])[0] for i in range(n)])[:, None]     # (n, 1, L, o)
    return dict(X=X, strings=strings, S=S, Sc=None, comp_pos=[])

def main():
    sub = ReaderMachine(seed=0)
    withhold = withheld_chunks(np.random.default_rng(11))
    langs = {}
    for policy in ("free", "ABD", "BDA", "ADB"):
        prefixes = order_policy_prefixes(policy, np.random.default_rng(100), 200)
        m, disc = D.extract(sub, EPS, np.random.default_rng(0), ctx_len=3, withhold=withhold, prefixes=prefixes)
        langs[policy] = (m, disc["contexts"])
    rows = []
    for order in ("ABD", "BDA", "ADB"):
        suite = order_suite(sub, order, np.random.default_rng(777))
        for policy, (m, contexts) in langs.items():
            states, _ = E.abstract_states(m, sub, suite["X"], contexts)
            # per-piece strings: score each piece with its own string
            P = np.stack([E.predict_intervention(m, states[i], suite["strings"][i]) for i in range(len(states))])[:, None]
            agree, _, ok = E._score(P, suite["S"])
            inf = suite["S"].argmax(-1) != 0
            rows.append(dict(suite_order=order, language=policy, K=m.K, coverage=float(ok.mean()),
                             informative_fidelity=float(agree[inf].mean()), n_informative=int(inf.sum())))
            print(f"suite {order}: language {policy:4s} K={m.K:2d} coverage={ok.mean():.3f} informative fidelity={agree[inf].mean():.3f}")
    json.dump(rows, open(os.path.join(OUT, "scope_results.json"), "w"))

if __name__ == "__main__":
    main()
