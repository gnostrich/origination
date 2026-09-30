"""Milestone 2 driver: non-unique ontology.

    python -m emergence.lang.m2 run      # discovery under several initialisations, evaluation, unsealing, report

Discovery initialisations (the "search over explanatory languages"):
reading-order policy of the unlabelled piece pool (free / ABD / BDA / ADB),
discovery seed (3), context length (2 / 3).  Every extracted language is
evaluated on ONE common held-out intervention suite (free order, never
used in discovery) and on withheld compositions; languages are compared by
a behavioural distance (disagreement of their predictions on the common
suite).  The sealed decompositions are unsealed only in the last step.
"""

from __future__ import annotations

import json
import os
from itertools import product

import numpy as np

from emergence.lang import discover as D
from emergence.lang import evaluate as E
from emergence.lang.systems import (ReaderMachine, order_policy_prefixes, sealed_decompositions,
                                    N_ACTIONS, N_VAL, QUERY, UNSET)

OUT = "results/lang/m2"
POLICIES = ("free", "ABD", "BDA", "ADB")
SEEDS = (0, 1, 2)
CTX_LENS = (2, 3)
EPS = 0.05
WITHHOLD_FRAC = 0.25


def withheld_chunks(rng, frac=WITHHOLD_FRAC):
    pairs = [c for c in D.all_chunks(N_ACTIONS, 2) if len(c) == 2]
    k = int(round(frac * len(pairs)))
    idx = rng.choice(len(pairs), k, replace=False)
    return [pairs[i] for i in sorted(idx)]


def common_suite(sub, rng, all_discovery_prefixes, withhold):
    """One held-out suite shared by every language: fresh pieces (free-order
    prefixes never in any discovery pool) × fresh strings, plus a
    compositional set, plus a near-exhaustive check: every reachable
    configuration × every string of length 3."""
    suite = E.heldout_suite(sub, rng, all_discovery_prefixes, n_pieces=120, piece_len=(4, 10), n_strings=40, string_len=6, withheld=withhold)
    # exhaustive block: all 64 configurations (as pieces) × all 1000 length-3 strings
    reach = [()]
    for a in range(N_ACTIONS):
        for b in range(N_ACTIONS):
            for c in range(N_ACTIONS):
                reach.append((a, b, c))
    X_all = sub.pieces(reach)
    # unique configurations
    idx = sub._decode(X_all)
    _, first = np.unique(idx, return_index=True)
    X_cfg = X_all[np.sort(first)]
    strings3 = D.all_chunks(N_ACTIONS, 3)
    strings3 = [c for c in strings3 if len(c) == 3]
    suite["X_cfg"] = X_cfg
    suite["strings3"] = strings3
    suite["S3"] = sub.behave(X_cfg, strings3)
    return suite


def informative_scores(predict, states, suite, withheld):
    """Fidelity restricted to informative steps: steps at which the substrate
    emits a non-⊥ observation (query on a complete assignment).  On this
    system ⊥ is emitted at most steps, so unrestricted argmax fidelity is
    inflated; these are the discriminating numbers.  Compositional version:
    informative steps at or after the completion of a withheld chunk."""
    from emergence.lang.systems import BOT
    out = {}
    for key, S, strings in (("", suite["S"], suite["strings"]), ("comp_", suite["Sc"], suite["comp_strings"])):
        if S is None:
            continue
        P = np.stack([np.stack([predict(states[i], s) for s in strings]) for i in range(len(states))])
        agree, _, ok = E._score(P, S)
        inf = S.argmax(-1) != BOT
        if key == "comp_":
            after = np.zeros(inf.shape, dtype=bool)
            for q, pos in enumerate(suite["comp_pos"]):
                if pos:
                    after[:, q, min(pos):] = True
            inf &= after
        out[key + "fidelity_informative"] = float(agree[inf].mean()) if inf.any() else float("nan")
        out[key + "n_informative"] = int(inf.sum())
    return out


def evaluate_language(model, sub, suite, contexts, withheld=()):
    fid = E.evaluate_fidelity(model, sub, suite, contexts)
    states, _ = E.abstract_states(model, sub, suite["X"], contexts)
    fid.update(informative_scores(lambda st, s: E.predict_intervention(model, st, s), states, suite, withheld))
    # exhaustive block
    states, _ = E.abstract_states(model, sub, suite["X_cfg"], contexts)
    P = np.stack([np.stack([E.predict_intervention(model, states[i], s) for s in suite["strings3"]]) for i in range(len(states))])
    agree, fjs, ok = E._score(P, suite["S3"])
    fid["exhaustive_fidelity"] = float(agree.mean())
    fid["exhaustive_coverage"] = float(ok.mean())
    from emergence.lang.systems import BOT
    inf = suite["S3"].argmax(-1) != BOT
    fid["exhaustive_fidelity_informative"] = float(agree[inf].mean())
    fid["exhaustive_n_informative"] = int(inf.sum())
    return fid, P


def lookup_informative(model, sub, disc, suite, contexts):
    """Informative-step scores for the lookup catalogue (same construction as evaluate.lookup_baseline)."""
    states, _ = E.abstract_states(model, sub, suite["X"], contexts)
    table = {}
    obs_chunks = [c for c in D.all_chunks(model.n_actions, 2) if c not in set(model.withheld)]
    for k in range(model.K):
        idx = np.array(model.realizations[k])[:12]
        for c in obs_chunks:
            table[(k, c)] = sub.behave(disc["X"][idx], [c])[:, 0].mean(0)
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
    return informative_scores(predict, states, suite, model.withheld)


def language_distance(P1, P2):
    """Disagreement of two languages' predictions on the exhaustive block
    (abstention by either counts as disagreement)."""
    ok = ~np.isnan(P1[..., 0]) & ~np.isnan(P2[..., 0])
    both = np.zeros(ok.shape, dtype=bool)
    both[ok] = P1[ok].argmax(-1) == P2[ok].argmax(-1)
    return float(1 - both.mean())


def unseal_language(model, sub, disc):
    """Describe each interface by the sealed partial assignments of its
    realisations: which slots are set, and which intermediate class (C/E/F)
    the realisations share."""
    dec = sealed_decompositions()
    assign = sub.decode_assignment(disc["X"])
    per = []
    for k in range(model.K):
        A = [assign[i] for i in model.realizations[k]]
        masks = {tuple(v != UNSET for v in a) for a in A}
        C = {dec["C"][(a[0], a[1])] for a in A if a[0] != UNSET and a[1] != UNSET}
        Ev = {dec["E"][(a[1], a[2])] for a in A if a[1] != UNSET and a[2] != UNSET}
        F = {dec["F"][(a[0], a[2])] for a in A if a[0] != UNSET and a[2] != UNSET}
        full = {(a[0] * a[1] + a[2]) % N_VAL for a in A if UNSET not in a}
        per.append(dict(k=k, n=len(A), set_masks=sorted(masks), C=sorted(C), E=sorted(Ev), F=sorted(F), Y=sorted(full)))
    def counts(sel):
        return sum(1 for p in per if len(p["set_masks"]) == 1 and p["set_masks"][0] == sel)
    n_C = sum(1 for p in per if p["set_masks"] == [(True, True, False)] and len(p["C"]) == 1)
    n_E = sum(1 for p in per if p["set_masks"] == [(False, True, True)] and len(p["E"]) == 1)
    n_F = sum(1 for p in per if p["set_masks"] == [(True, False, True)] and len(p["F"]) == 1)
    mixed = sum(1 for p in per if len(p["set_masks"]) > 1)
    return dict(per_interface=per, n_interfaces=model.K, sizes=dec["sizes"],
                n_AB_interfaces=counts((True, True, False)), n_BD_interfaces=counts((False, True, True)), n_AD_interfaces=counts((True, False, True)),
                n_pure_C=n_C, n_pure_E=n_E, n_pure_F=n_F, n_mixed_mask=mixed,
                n_full=counts((True, True, True)), n_empty=counts((False, False, False)),
                n_single=counts((True, False, False)) + counts((False, True, False)) + counts((False, False, True)))


def run():
    os.makedirs(OUT, exist_ok=True)
    sub = ReaderMachine(seed=0)
    withhold = withheld_chunks(np.random.default_rng(11))
    # discovery under every initialisation
    langs = []
    all_prefixes = set()
    for policy, seed, ctx in product(POLICIES, SEEDS, CTX_LENS):
        rng = np.random.default_rng(100 + seed)
        prefixes = order_policy_prefixes(policy, rng, 200)
        all_prefixes |= set(map(tuple, prefixes))
        sub.calls = 0
        model, disc = D.extract(sub, EPS, np.random.default_rng(seed), ctx_len=ctx, withhold=withhold, prefixes=prefixes)
        langs.append(dict(policy=policy, seed=seed, ctx=ctx, model=model, disc=disc, calls=sub.calls))
        print(f"discovered policy={policy} seed={seed} ctx={ctx}: K={model.K} pieces={len(prefixes)} calls={sub.calls}", flush=True)
    suite = common_suite(sub, np.random.default_rng(999), sorted(all_prefixes), withhold)
    # evaluation (frozen: no selection happens after this point)
    rows, preds = [], []
    for L in langs:
        m = L["model"]
        contexts = L["disc"]["contexts"]
        fid, P = evaluate_language(m, sub, suite, contexts, withhold)
        cx = E.measure_complexity(m)
        rand = E.random_abstraction(m, sub, L["disc"], suite, np.random.default_rng(5))
        fid_r, _ = evaluate_language(rand, sub, suite, contexts, withhold)
        look = E.lookup_baseline(m, sub, L["disc"], suite, contexts)
        look.update(lookup_informative(m, sub, L["disc"], suite, contexts))
        ca = E.causal_abstraction(m, sub, suite, contexts)
        row = dict(policy=L["policy"], seed=L["seed"], ctx=L["ctx"], K=m.K, n_fits=len([f for f in m.fits.values() if f.observed]),
                   n_second_order=len(m.chunk_maps), fidelity=fid, complexity=cx,
                   random=dict(fidelity=fid_r, complexity=E.measure_complexity(rand)), lookup=look, causal=ca,
                   discovery_calls=L["calls"], substrate_bits=sub.substrate_bits())
        rows.append(row); preds.append(P)
        name = f"{L['policy']}_s{L['seed']}_ctx{L['ctx']}"
        open(os.path.join(OUT, f"SPEC_{name}.md"), "w").write(m.spec(prefixes=L["disc"]["prefixes"]))
        json.dump(m.to_json(), open(os.path.join(OUT, f"model_{name}.json"), "w"))
        print(f"{name}: K={m.K} fid={fid['fidelity_argmax']:.3f} INF={fid['fidelity_informative']:.3f} exhINF={fid['exhaustive_fidelity_informative']:.3f} cov={fid['coverage']:.2f} "
              f"compINF={fid.get('comp_fidelity_informative', float('nan')):.3f} bits={cx['total_bits_argmax']:.0f} randINF={fid_r['fidelity_informative']:.3f} "
              f"lookupINF={look['fidelity_informative']:.3f} causal={ca['commutation']:.3f}/{ca['abstract_agreement']:.3f}", flush=True)
    # language distance matrix
    n = len(langs)
    Dm = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            Dm[i, j] = language_distance(preds[i], preds[j])
    # unsealing (diagnostic only, after everything above)
    for L, row in zip(langs, rows):
        row["unsealed"] = unseal_language(L["model"], sub, L["disc"])
    json.dump(dict(rows=rows, distance=Dm.tolist(), names=[f"{r['policy']}_s{r['seed']}_ctx{r['ctx']}" for r in rows],
                   withheld=[list(c) for c in withhold], sealed_sizes=sealed_decompositions()["sizes"]),
              open(os.path.join(OUT, "m2_results.json"), "w"), default=float)
    report(rows, Dm, withhold)


def report(rows, Dm, withhold):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = [f"{r['policy']}_s{r['seed']}_ctx{r['ctx']}" for r in rows]
    L = ["# Milestone 2: non-unique ontology — results", "",
         f"Withheld chunks ({len(withhold)} of 100 length-2 chunks): {withhold}", "",
         "## Extracted languages", "",
         "| language (policy, seed, ctx) | K | observed fits | 2nd-order | held-out fidelity all steps | **informative-step fidelity** (held-out / exhaustive) | coverage | **withheld-composition fidelity, informative** (model / lookup) | causal commutation / abstract | bits (model / lookup / substrate) | random abstraction (all / informative) | exceptions |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for n, r in zip(names, rows):
        f = r["fidelity"]
        L.append(f"| {n} | {r['K']} | {r['n_fits']} | {r['n_second_order']} | {f['fidelity_argmax']:.3f} | **{f['fidelity_informative']:.3f} / {f['exhaustive_fidelity_informative']:.3f}** | {f['coverage']:.2f} | "
                 f"**{f.get('comp_fidelity_informative', float('nan')):.3f} / {r['lookup'].get('comp_fidelity_informative', float('nan')):.3f}** | {r['causal']['commutation']:.3f} / {r['causal']['abstract_agreement']:.3f} | "
                 f"{r['complexity']['total_bits_argmax']:.0f} / {r['lookup']['bits']:.0f} / {r['substrate_bits']:.0f} | {r['random']['fidelity']['fidelity_argmax']:.3f} / {r['random']['fidelity']['fidelity_informative']:.3f} | {r['complexity']['exceptions']} |")
    L += ["", "## Language distance matrix (disagreement of predictions on the exhaustive block; 0 = behaviourally equivalent)", ""]
    L.append("| | " + " | ".join(names) + " |")
    L.append("|---|" + "---|" * len(names))
    for i, n in enumerate(names):
        L.append(f"| {n} | " + " | ".join(f"{Dm[i, j]:.3f}" for j in range(len(names))) + " |")
    L += ["", "## Unsealed (diagnostic only)", "",
          "| language | K | empty | single-slot | AB (pure C) | BD (pure E) | AD (pure F) | full | mixed-mask interfaces |", "|---|---|---|---|---|---|---|---|---|"]
    for n, r in zip(names, rows):
        u = r["unsealed"]
        L.append(f"| {n} | {u['n_interfaces']} | {u['n_empty']} | {u['n_single']} | {u['n_AB_interfaces']} ({u['n_pure_C']}) | {u['n_BD_interfaces']} ({u['n_pure_E']}) | {u['n_AD_interfaces']} ({u['n_pure_F']}) | {u['n_full']} | {u['n_mixed_mask']} |")
    L.append("")
    fig, ax = plt.subplots(figsize=(7, 5))
    for i, (n, r) in enumerate(zip(names, rows)):
        mk = {"free": "o", "ABD": "s", "BDA": "^", "ADB": "v"}[r["policy"]]
        ax.plot(r["complexity"]["total_bits_argmax"], r["fidelity"]["fidelity_informative"], mk, color="C0" if r["ctx"] == 3 else "C1", ms=7, alpha=0.7)
        ax.plot(r["lookup"]["bits"], r["lookup"]["fidelity_informative"], mk, color="C2", ms=5, alpha=0.5)
        ax.plot(r["random"]["complexity"]["total_bits_argmax"], r["random"]["fidelity"]["fidelity_informative"], "x", color="C3", ms=5, alpha=0.5)
    ax.plot([rows[0]["substrate_bits"]], [1.0], "*", color="k", ms=12)
    ax.set_xscale("log"); ax.set_xlabel("description complexity (bits)"); ax.set_ylabel("held-out informative-step fidelity (free-order suite)")
    ax.set_title("Milestone 2: blue = ctx 3, orange = ctx 2 (o free, s ABD, ^ BDA, v ADB); green lookup; red random; star substrate", fontsize=8)
    ax.set_ylim(0, 1.02)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "pareto_m2.png"), dpi=120)
    open(os.path.join(OUT, "M2_REPORT.md"), "w").write("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    run()
