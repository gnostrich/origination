"""Milestone 3 driver — executes MILESTONE3_PROTOCOL.md.

    python -m emergence.lang.m3 splits      # fixed splits, saved (before discovery)
    python -m emergence.lang.m3 discover    # ε × context × seed grid; validation scores
    python -m emergence.lang.m3 freeze      # primary model by validation rule; prospective predictions written
    python -m emergence.lang.m3 test        # substrate executed on sealed test; scoring; baselines (same order)
    python -m emergence.lang.m3 analyse     # nested contexts, reproducibility, interpretation (unsealed), report

Discovery code path: emergence.lang.{discover, model}.  The simulator
(`cart.py`) is imported here only for the alphabet sizes, the trained
model, and — in `analyse` — the interpretation step.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import time
from datetime import datetime, timezone
from itertools import product

import numpy as np

from emergence.lang import discover as D
from emergence.lang import evaluate as E
from emergence.lang.substrate import SeqSubstrate
from emergence.lang.cart import N_ACTIONS, N_OBS, load as load_cart

OUT = "results/lang/m3"
EPS_GRID = (0.05, 0.1, 0.2, 0.3, 0.5)
CTX = {"C1": 1, "C2": 2, "C3": 3}
SEEDS = (0, 1, 2)
N_WITHHELD = 4


def log(msg):
    stamp = datetime.now(timezone.utc).isoformat()
    line = f"[{stamp}] {msg}"
    print(line, flush=True)
    with open(os.path.join(OUT, "protocol_log.txt"), "a") as f:
        f.write(line + "\n")


def substrate():
    m, _ = load_cart(OUT)
    return SeqSubstrate(m, N_ACTIONS, N_OBS), m


def sha(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=float).encode()).hexdigest()[:16]


# ------------------------------------------------------------------ splits
def make_splits():
    rng = np.random.default_rng(2026)
    D_pool = [()] + D.all_chunks(N_ACTIONS, 3)
    D_pool += [tuple(int(a) for a in rng.integers(0, N_ACTIONS, int(rng.integers(4, 17)))) for _ in range(600)]
    seen = set(D_pool)
    pairs = [c for c in D.all_chunks(N_ACTIONS, 2) if len(c) == 2]
    W = [pairs[i] for i in sorted(rng.choice(len(pairs), N_WITHHELD, replace=False))]
    observed_chunks = [c for c in D.all_chunks(N_ACTIONS, 2) if c not in set(W)]

    def fresh_pieces(n, lo, hi):
        out = []
        while len(out) < n:
            p = tuple(int(a) for a in rng.integers(0, N_ACTIONS, int(rng.integers(lo, hi + 1))))
            if p not in seen:
                seen.add(p); out.append(p)
        return out
    V_pieces = fresh_pieces(120, 4, 12)
    V_strings = [tuple(int(a) for a in rng.integers(0, N_ACTIONS, 6)) for _ in range(30)]
    T_pieces = fresh_pieces(200, 4, 12)
    obs1 = [c for c in observed_chunks if len(c) == 1]
    obs2 = [c for c in observed_chunks if len(c) == 2]
    T1 = [obs1[int(rng.integers(len(obs1)))] if rng.random() < 0.3 else obs2[int(rng.integers(len(obs2)))] for _ in range(40)]
    T2 = []
    while len(T2) < 40:                       # length 3-4 from observed chunks, never itself a discovery chunk
        s = ()
        while len(s) < 3:
            s += observed_chunks[int(rng.integers(len(observed_chunks)))]
        s = s[:4] if len(s) > 4 else s
        if len(s) >= 3 and all(tuple(s[i:i + 2]) not in set(W) for i in range(len(s) - 1)):
            T2.append(tuple(s))
    T3 = []
    while len(T3) < 40:                       # length 3-6 containing a withheld chunk
        L = int(rng.integers(3, 7))
        w = W[int(rng.integers(len(W)))]
        pos = int(rng.integers(0, L - 1))
        s = [int(a) for a in rng.integers(0, N_ACTIONS, L)]
        s[pos:pos + 2] = list(w)
        T3.append(tuple(s))
    T4 = [tuple(int(a) for a in rng.integers(0, N_ACTIONS, int(rng.integers(8, 13)))) for _ in range(40)]
    splits = dict(D_pool=[list(p) for p in D_pool], W=[list(w) for w in W], observed_chunks=[list(c) for c in observed_chunks],
                  V_pieces=[list(p) for p in V_pieces], V_strings=[list(s) for s in V_strings],
                  T_pieces=[list(p) for p in T_pieces], T={"T1": [list(s) for s in T1], "T2": [list(s) for s in T2],
                                                           "T3": [list(s) for s in T3], "T4": [list(s) for s in T4]})
    json.dump(splits, open(os.path.join(OUT, "splits.json"), "w"))
    log(f"splits written: |D|={len(D_pool)} W={W} |V|={len(V_pieces)}x{len(V_strings)} |T pieces|={len(T_pieces)} T families 4x40; sha={sha(splits)}")
    return splits


def load_splits():
    s = json.load(open(os.path.join(OUT, "splits.json")))
    s["D_pool"] = [tuple(p) for p in s["D_pool"]]
    s["W"] = [tuple(w) for w in s["W"]]
    s["V_pieces"] = [tuple(p) for p in s["V_pieces"]]
    s["V_strings"] = [tuple(x) for x in s["V_strings"]]
    s["T_pieces"] = [tuple(p) for p in s["T_pieces"]]
    s["T"] = {k: [tuple(x) for x in v] for k, v in s["T"].items()}
    return s


# ------------------------------------------------------------------ discovery
def model_name(eps, ctx, seed):
    return f"eps{eps}_{ctx}_s{seed}"


def val_suite(sub, splits):
    X = sub.pieces(splits["V_pieces"])
    return dict(X=X, strings=splits["V_strings"], S=sub.behave(X, splits["V_strings"]), Sc=None, comp_pos=[])


def discover_all():
    sub, _ = substrate()
    splits = load_splits()
    vs = val_suite(sub, splits)
    results = {}
    os.makedirs(os.path.join(OUT, "models"), exist_ok=True)
    for eps, (cname, clen), seed in product(EPS_GRID, CTX.items(), SEEDS):
        t0 = time.time()
        model, disc = D.extract(sub, eps, np.random.default_rng(seed), ctx_len=clen, withhold=splits["W"], prefixes=splits["D_pool"])
        fid = E.evaluate_fidelity(model, sub, vs, disc["contexts"])
        cx = E.measure_complexity(model)
        name = model_name(eps, cname, seed)
        with open(os.path.join(OUT, "models", name + ".pkl"), "wb") as f:
            pickle.dump(dict(model=model, labels=disc["labels"], contexts=disc["contexts"], X=disc["X"]), f)
        open(os.path.join(OUT, "models", "SPEC_" + name + ".md"), "w").write(model.spec(prefixes=splits["D_pool"]))
        results[name] = dict(eps=eps, ctx=cname, seed=seed, K=model.K, val=fid, complexity=cx, seconds=time.time() - t0)
        log(f"discovered {name}: K={model.K} val fidelity={fid['fidelity_argmax']:.3f} js={fid['fidelity_js']:.3f} cov={fid['coverage']:.2f} "
            f"bits={cx['total_bits_argmax']:.0f} exceptions={cx['exceptions']} undefined={cx['residual_undefined']:.2f} ({time.time()-t0:.0f}s)")
        json.dump(results, open(os.path.join(OUT, "discovery_results.json"), "w"), default=float)
    return results


# ------------------------------------------------------------------ freeze + prospective predictions
def select_primary(results):
    cands = [(n, r) for n, r in results.items() if r["seed"] == 0]
    cands.sort(key=lambda nr: (-nr[1]["val"]["fidelity_argmax"], nr[1]["complexity"]["total_bits_argmax"]))
    return cands[0][0]


def load_model(name):
    with open(os.path.join(OUT, "models", name + ".pkl"), "rb") as f:
        return pickle.load(f)


def predict_all(model, contexts, sub, splits, tag):
    """Abstract predictions for every sealed test query; the substrate is used
    only for the abstraction map (classifying test pieces in the discovery
    contexts), never for the queries themselves."""
    X = sub.pieces(splits["T_pieces"])
    states, dist = E.abstract_states(model, sub, X, contexts)
    preds = {}
    for fam, strings in splits["T"].items():
        preds[fam] = [[E.predict_intervention(model, int(states[i]), s).tolist() for s in strings] for i in range(len(states))]
    out = dict(tag=tag, states=states.tolist(), abstract_dist=dist.tolist(), predictions=preds)
    return out


def freeze():
    results = json.load(open(os.path.join(OUT, "discovery_results.json")))
    primary = select_primary(results)
    log(f"PRIMARY MODEL selected by validation rule: {primary} (val fidelity {results[primary]['val']['fidelity_argmax']:.3f}, bits {results[primary]['complexity']['total_bits_argmax']:.0f}); frozen.")
    sub, _ = substrate()
    splits = load_splits()
    P = load_model(primary)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    pred = predict_all(P["model"], P["contexts"], sub, splits, primary)
    path = os.path.join(OUT, f"predictions_{stamp}.json")
    json.dump(pred, open(path, "w"))
    h = hashlib.sha256(open(path, "rb").read()).hexdigest()[:16]
    json.dump(dict(primary=primary, predictions_file=path, sha256_16=h, stamp=stamp), open(os.path.join(OUT, "frozen.json"), "w"))
    log(f"PROSPECTIVE PREDICTIONS written before substrate execution: {path} sha256[:16]={h}; test pieces classified: {int((np.array(pred['states'])>=0).sum())}/{len(pred['states'])} within ε")
    return primary, path


# ------------------------------------------------------------------ test execution + scoring + baselines
def score(pred_fam, S_fam):
    P = np.array(pred_fam)                      # (n, Q, L, o) with NaN for abstentions
    agree, fjs, ok = E._score(P, S_fam)
    by_step = [float(agree[:, :, t].mean()) for t in range(agree.shape[2])]
    full = float(np.mean(agree.reshape(-1, agree.shape[2]).all(axis=1)))     # strings predicted entirely correctly
    return dict(fidelity_argmax=float(agree.mean()), fidelity_js=float(fjs.mean()), coverage=float(ok.mean()),
                fidelity_covered=float(agree[ok].mean()) if ok.any() else float("nan"), by_step=by_step, whole_string=full)


def run_test():
    frozen = json.load(open(os.path.join(OUT, "frozen.json")))
    pred = json.load(open(frozen["predictions_file"]))
    sub, torch_model = substrate()
    splits = load_splits()
    log(f"SUBSTRATE EXECUTION on sealed test begins (after predictions {frozen['stamp']}, sha {frozen['sha256_16']})")
    X = sub.pieces(splits["T_pieces"])
    S = {fam: sub.behave(X, strings) for fam, strings in splits["T"].items()}
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    np.savez(os.path.join(OUT, f"substrate_{stamp}.npz"), **{fam: S[fam] for fam in S})
    log(f"substrate outputs written: substrate_{stamp}.npz")
    scores = {fam: score(pred["predictions"][fam], S[fam]) for fam in S}
    P = load_model(frozen["primary"])
    model = P["model"]
    # causal commutation on T4
    suite4 = dict(X=X, strings=splits["T"]["T4"], S=S["T4"])
    ca = E.causal_abstraction(model, sub, suite4, P["contexts"])
    # reuse / depth
    first = [f for (s_, c) in model.fits if len(c) == 1 for f in [model.fits[(s_, c)]]]
    land = {}
    for f in first:
        land.setdefault(f.dst, 0)
        land[f.dst] += 1
    states = np.array(pred["states"])
    fit_use = {}
    for fam, strings in splits["T"].items():
        for i in range(len(states)):
            st = int(states[i])
            for s in strings:
                for a in s:
                    if st < 0:
                        break
                    fit_use[(st, a)] = fit_use.get((st, a), 0) + 1
                    st = model.transition(st, a)
    reuse = dict(interface_reuse=float(np.mean(list(land.values()))) if land else 0.0,
                 fit_reuse=float(np.mean(list(fit_use.values()))) if fit_use else 0.0,
                 fits_used=len(fit_use), of_first_order=len(first))
    primary_res = dict(primary=frozen["primary"], K=model.K, complexity=E.measure_complexity(model), scores=scores,
                       causal=ca, reuse=reuse, substrate_bits=E.substrate_bits(torch_model))
    json.dump(primary_res, open(os.path.join(OUT, "test_primary.json"), "w"), default=float)
    log("primary scored: " + "; ".join(f"{fam}: fid={sc['fidelity_argmax']:.3f} js={sc['fidelity_js']:.3f} cov={sc['coverage']:.2f} whole={sc['whole_string']:.2f}" for fam, sc in scores.items())
        + f"; causal {ca['commutation']:.3f}/{ca['abstract_agreement']:.3f}")
    # ---- baselines, same order: predictions first, then scoring on the saved substrate outputs
    base = {}
    base["random"] = baseline_random(P, sub, splits, S)
    base["lookup"] = baseline_lookup(P, sub, splits, S)
    base["kmeans"] = baseline_geometric(P, sub, splits, S, pca=None)
    base["kmeans_pca8"] = baseline_geometric(P, sub, splits, S, pca=8)
    json.dump(base, open(os.path.join(OUT, "test_baselines.json"), "w"), default=float)
    for b, r in base.items():
        log(f"baseline {b}: " + "; ".join(f"{fam}: {r['scores'][fam]['fidelity_argmax']:.3f}" for fam in r["scores"]) + f"; bits {r.get('bits', float('nan')):.0f}")
    # all other discovered models on the sealed test (frontier; not used for selection)
    frontier = {}
    results = json.load(open(os.path.join(OUT, "discovery_results.json")))
    for name in results:
        Pm = load_model(name)
        pr = predict_all(Pm["model"], Pm["contexts"], sub, splits, name)
        frontier[name] = dict(K=Pm["model"].K, complexity=E.measure_complexity(Pm["model"]),
                              scores={fam: score(pr["predictions"][fam], S[fam]) for fam in S},
                              states=pr["states"])
    json.dump(frontier, open(os.path.join(OUT, "test_frontier.json"), "w"), default=float)
    log("frontier models scored on the sealed test")
    return primary_res, base


def _executable_from_partition(sub, X, labels, contexts, splits, model_template):
    """Build an executable model from an arbitrary partition of the discovery
    pieces (used by the random and geometric baselines): leaders = mean
    signatures, emissions from single-step contexts, fits by majority on the
    same observed chunks, classification by nearest centroid in the
    corresponding space (passed in as `classify`)."""
    from emergence.lang.model import InterfaceModel
    S = sub.behave(X, contexts)
    K = int(labels.max()) + 1
    leaders = np.array([np.nanmean(S[labels == k], axis=0) for k in range(K)])
    real = [np.nonzero(labels == k)[0].tolist() for k in range(K)]
    ctx_index = {c: i for i, c in enumerate(contexts)}
    em = np.zeros((K, N_ACTIONS, N_OBS))
    for a in range(N_ACTIONS):
        em[:, a] = leaders[:, ctx_index[(a,)], 0]
    m = InterfaceModel(eps=np.inf, n_actions=N_ACTIONS, n_obs=N_OBS, contexts=list(contexts), leaders=leaders,
                       realizations=real, within=np.zeros(K), emissions=em)
    m.withheld = list(splits["W"])
    return m


def baseline_random(P, sub, splits, S):
    from emergence.lang.discover import discover_fits, compose
    rng = np.random.default_rng(5)
    labels = P["labels"][rng.permutation(len(P["labels"]))]
    m = _executable_from_partition(sub, P["X"], labels, P["contexts"], splits, P["model"])
    discover_fits(sub, m, P["X"], labels, splits["observed_chunks"], P["contexts"], rng=rng)
    compose(m, 2)
    pr = predict_all(m, P["contexts"], sub, splits, "random")
    return dict(K=m.K, scores={fam: score(pr["predictions"][fam], S[fam]) for fam in S}, bits=E.measure_complexity(m)["total_bits_argmax"])


def baseline_lookup(P, sub, splits, S):
    model, contexts = P["model"], P["contexts"]
    X = sub.pieces(splits["T_pieces"])
    states, _ = E.abstract_states(model, sub, X, contexts)
    table = {}
    for k in range(model.K):
        idx = np.array(model.realizations[k])[:8]
        for c in splits["observed_chunks"]:
            table[(k, c)] = sub.behave(P["X"][idx], [c])[:, 0].mean(0)
    def predict(state, string):
        Pq = np.full((len(string), N_OBS), np.nan)
        if state < 0:
            return Pq
        for t in range(len(string)):
            c = tuple(string[: t + 1])
            if len(c) > 2 or (state, c) not in table:
                break
            Pq[t] = table[(state, c)][t]
        return Pq
    preds = {fam: [[predict(int(states[i]), s).tolist() for s in strings] for i in range(len(states))] for fam, strings in splits["T"].items()}
    return dict(K=model.K, scores={fam: score(preds[fam], S[fam]) for fam in S}, bits=float(len(table) * 2 * np.log2(N_OBS)))


def baseline_geometric(P, sub, splits, S, pca=None):
    """k-means on the discovery pieces' hidden vectors (optionally in the top-`pca`
    principal subspace), K = primary K; executable model from that partition;
    test pieces classified by nearest centroid in the same space."""
    from emergence.lang.discover import discover_fits, compose
    X = np.asarray(P["X"])
    K = P["model"].K
    Z = X - X.mean(0)
    if pca:
        _, _, Vt = np.linalg.svd(Z, full_matrices=False)
        proj = Vt[:pca]
        Z = Z @ proj.T
    rng = np.random.default_rng(7)
    cent = Z[rng.choice(len(Z), K, replace=False)]
    for _ in range(50):
        lab = np.argmin(((Z[:, None, :] - cent[None]) ** 2).sum(-1), axis=1)
        for k in range(K):
            if (lab == k).any():
                cent[k] = Z[lab == k].mean(0)
    # relabel to contiguous ids
    _, lab = np.unique(lab, return_inverse=True)
    cent = np.array([Z[lab == k].mean(0) for k in range(lab.max() + 1)])
    m = _executable_from_partition(sub, P["X"], lab, P["contexts"], splits, P["model"])
    discover_fits(sub, m, P["X"], lab, splits["observed_chunks"], P["contexts"], rng=rng)
    compose(m, 2)
    Xt = np.asarray(sub.pieces(splits["T_pieces"])) - X.mean(0)
    if pca:
        Xt = Xt @ proj.T
    states = np.argmin(((Xt[:, None, :] - cent[None]) ** 2).sum(-1), axis=1)
    preds = {fam: [[E.predict_intervention(m, int(states[i]), s).tolist() for s in strings] for i in range(len(states))] for fam, strings in splits["T"].items()}
    return dict(K=m.K, scores={fam: score(preds[fam], S[fam]) for fam in S}, bits=E.measure_complexity(m)["total_bits_argmax"], pca=pca)


# ------------------------------------------------------------------ analyses after freezing
def analyse():
    from emergence.grok.compare import adjusted_rand_index as ari
    from emergence.lang.cart import state_after
    frozen = json.load(open(os.path.join(OUT, "frozen.json")))
    results = json.load(open(os.path.join(OUT, "discovery_results.json")))
    frontier = json.load(open(os.path.join(OUT, "test_frontier.json")))
    primary = frozen["primary"]
    pe, pc, _ = results[primary]["eps"], results[primary]["ctx"], results[primary]["seed"]
    splits = load_splits()
    out = {}
    # nested contexts at the primary ε, seed 0
    nest = {}
    parts = {}
    for cname in CTX:
        name = model_name(pe, cname, 0)
        Pm = load_model(name)
        parts[cname] = np.asarray(Pm["labels"])
        nest[cname] = dict(K=Pm["model"].K, val=results[name]["val"]["fidelity_argmax"], bits=results[name]["complexity"]["total_bits_argmax"],
                           test={fam: frontier[name]["scores"][fam]["fidelity_argmax"] for fam in frontier[name]["scores"]})
    def refines(fine, coarse):
        ok = 0
        for k in np.unique(fine):
            ok += len(np.unique(coarse[fine == k])) == 1
        return float(ok / len(np.unique(fine)))
    nest["C2_refines_C1"] = refines(parts["C2"], parts["C1"]); nest["C3_refines_C2"] = refines(parts["C3"], parts["C2"])
    nest["ari_C1_C2"] = float(ari(parts["C1"], parts["C2"])); nest["ari_C2_C3"] = float(ari(parts["C2"], parts["C3"]))
    out["nested"] = nest
    # reproducibility at primary (ε, C)
    rep = {}
    names = [model_name(pe, pc, s) for s in SEEDS]
    labs = {n: np.asarray(load_model(n)["labels"]) for n in names}
    rep["ari"] = {f"{a}|{b}": float(ari(labs[a], labs[b])) for a in names for b in names if a < b}
    # behavioural distance on the sealed test predictions (frontier states+predictions were not saved per step; recompute)
    sub, _ = substrate()
    preds = {}
    for n in names:
        Pm = load_model(n)
        preds[n] = predict_all(Pm["model"], Pm["contexts"], sub, splits, n)["predictions"]
    def dist(p1, p2):
        tot = agree = 0
        for fam in p1:
            A1, A2 = np.array(p1[fam]), np.array(p2[fam])
            ok = ~np.isnan(A1[..., 0]) & ~np.isnan(A2[..., 0])
            agree += (A1[ok].argmax(-1) == A2[ok].argmax(-1)).sum(); tot += ok.size
        return float(1 - agree / tot)
    rep["behavioural_distance"] = {f"{a}|{b}": dist(preds[a], preds[b]) for a in names for b in names if a < b}
    out["reproducibility"] = rep
    # interpretation: unseal the simulator
    Pm = load_model(primary)
    model = Pm["model"]
    xv = np.array([state_after(p) for p in splits["D_pool"]])
    interp = []
    for k in range(model.K):
        idx = np.array(model.realizations[k])
        interp.append(dict(k=k, n=int(len(idx)), x_mean=float(xv[idx, 0].mean()), x_std=float(xv[idx, 0].std()),
                           v_mean=float(xv[idx, 1].mean()), v_std=float(xv[idx, 1].std()),
                           x_bin=int(np.floor(xv[idx, 0].mean() * N_OBS))))
    labels = np.asarray(Pm["labels"])
    # how much of (x, v) variance the interfaces explain
    def r2(col):
        tot = xv[:, col].var()
        within = np.mean([xv[labels == k, col].var() for k in range(model.K)] if model.K else [tot])
        return float(1 - within / tot) if tot > 0 else float("nan")
    vbins = np.digitize(xv[:, 1], np.linspace(-0.15, 0.15, 7))
    xbins = np.floor(xv[:, 0] * N_OBS).astype(int)
    out["interpretation"] = dict(per_interface=interp, R2_x=r2(0), R2_v=r2(1),
                                 x_bin_partition_ari=float(ari(labels, xbins)),
                                 xv_grid_ari=float(ari(labels, xbins * 10 + vbins)),
                                 n_xv_cells_occupied=int(len(np.unique(xbins * 10 + vbins))))
    json.dump(out, open(os.path.join(OUT, "analysis.json"), "w"), default=float)
    log("post-freeze analyses written (nested contexts, reproducibility, interpretation)")
    return out


if __name__ == "__main__":
    import sys
    os.makedirs(OUT, exist_ok=True)
    cmd = sys.argv[1]
    {"splits": make_splits, "discover": discover_all, "freeze": freeze, "test": run_test, "analyse": analyse}[cmd]()
