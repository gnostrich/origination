"""Backward tracing of the final interfaces of a GRU world-model run (S_4).

The final interfaces are the 24 behavioural classes of prefix configurations
found by the frozen sequence extractor at the last checkpoint (recomputed
with its own seed and checked against the stored partition).  The instance
pool is the extractor's prefix pool; it is split once into fit / eval
halves.  Contexts are suffix families: all length-1, all length-2, and two
disjoint random pools of length-5 suffixes.
"""

from __future__ import annotations

import json
from itertools import product
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from emergence.grok.world import make_world
from emergence.grok.rnn import GRUWorldModel
from emergence.grok.seqsub import GRUSubstrate
from emergence.grok.extract_seq import SeqExtractConfig, _all_prefixes, extract_seq
from emergence.grok.compare import adjusted_rand_index as ari
from emergence.trace import common as C

SEED = 0
N_JAC = 96


def families(k, rng):
    F1 = np.array(list(product(range(k), repeat=1)))
    F2 = np.array(list(product(range(k), repeat=2)))
    F5a = rng.integers(0, k, (24, 5))
    F5b = rng.integers(0, k, (24, 5))
    return {"F1": F1, "F2": F2, "F5a": F5a, "F5b": F5b}


def behaviour(model, Hcfg, suff):
    """Predicted observations (argmax) and probabilities of continuing each configuration with each suffix."""
    n = Hcfg.shape[0]
    S = torch.as_tensor(suff, dtype=torch.long)
    outs, probs = [], []
    with torch.no_grad():
        for s in range(len(S)):
            lg, _ = model(S[s][None].expand(n, -1), h0=Hcfg)
            outs.append(lg.argmax(-1))
            probs.append(F.softmax(lg, -1))
    return torch.stack(outs, 1).numpy(), torch.stack(probs, 1).numpy()   # (n, S, L), (n, S, L, o)


def agree(b1, b2):
    return float((b1 == b2).mean())


def trace(run_dir, out=None):
    d = torch.load(Path(run_dir) / "checkpoints.pt", weights_only=False)
    cfg, ckpts, log = d["config"], d["checkpoints"], d["log"]
    metrics = json.load(open(Path(run_dir) / "metrics.json"))["metrics"]
    world = make_world(cfg["world"])
    k = world.n_actions
    model = GRUWorldModel(k, world.n_obs)
    sub = GRUSubstrate(model)
    last = max(ckpts)
    # frozen final partition: rerun the frozen extractor at the last checkpoint with its own seed
    model.load_state_dict(ckpts[last]); model.eval()
    xcfg = SeqExtractConfig()
    rng_x = np.random.default_rng(1234)
    prefixes = _all_prefixes(k, xcfg.prefix_len)
    prefixes = prefixes + [tuple(rng_x.integers(0, k, xcfg.long_len)) for _ in range(xcfg.n_long_prefixes)]
    ext = extract_seq(sub, k, xcfg, np.random.default_rng(1234))
    Pi = np.asarray(ext["partition"])
    stored = np.asarray(metrics[str(last)]["partition"])
    match = float(ari(Pi, stored)) if len(stored) == len(Pi) else float("nan")
    _, Pi = np.unique(Pi, return_inverse=True)
    K = int(Pi.max()) + 1
    T_final = np.asarray(ext["_T"])            # (classes, actions) on the extractor's labels == Pi's labels (same order)
    n = len(prefixes)
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(n)
    fit, ev = np.sort(perm[: n // 2]), np.sort(perm[n // 2:])
    fam = families(k, np.random.default_rng(11))
    by_class = {c: fit[Pi[fit] == c] for c in range(K)}
    donor_same = np.array([rng.choice(by_class[Pi[i]]) if len(by_class[Pi[i]]) else -1 for i in ev])
    donor_rand = rng.choice(fit, len(ev))
    chunks2 = rng.integers(0, k, (len(ev), 4, 2))    # 2-step chunks per eval instance
    log_by_step = {e["step"]: e for e in log}
    prev_R, prev_theta = None, None
    rows = []
    for step in sorted(ckpts):
        sd = ckpts[step]
        model.load_state_dict(sd); model.eval()
        theta = np.concatenate([v.numpy().ravel() for v in sd.values()])
        H = sub.run_prefixes(prefixes)
        R = H.numpy().astype(np.float64)
        row = dict(step=step, **{q: log_by_step.get(step, {}).get(q) for q in ("train_loss", "test_loss", "train_acc", "test_acc")})
        row["identity_probe"], row["identity_probe_fit"] = C.linear_probe(R[fit], Pi[fit], R[ev], Pi[ev], K)
        row["identity_knn"] = C.knn_purity(R[fit], Pi[fit], R[ev], Pi[ev])
        geo = C.geometry(R, Pi, K)
        row.update(between_within=geo["between_within"], margin=geo["margin"])
        V, mu, _, _ = C.between_projector(R[fit], Pi[fit], K)
        Vt = torch.as_tensor(V, dtype=torch.float32); mu_t = torch.as_tensor(mu, dtype=torch.float32)
        Hev = H[ev]
        base = {f: behaviour(model, Hev, fam[f])[0] for f in fam}
        comp = (Hev - mu_t) @ Vt.T @ Vt
        row["causal_remove"] = 1 - agree(behaviour(model, Hev - comp, fam["F5a"])[0], base["F5a"])
        row["causal_keep"] = agree(behaviour(model, mu_t + comp, fam["F5a"])[0], base["F5a"])
        G = torch.linalg.qr(torch.randn(H.shape[1], V.shape[0], generator=torch.Generator().manual_seed(1)))[0].T
        row["causal_remove_rand"] = 1 - agree(behaviour(model, Hev - (Hev - mu_t) @ G.T @ G, fam["F5a"])[0], base["F5a"])
        ok = donor_same >= 0
        Hd = H[donor_same[ok]]
        for f in fam:
            row[f"subst_{f}"] = agree(behaviour(model, Hd, fam[f])[0], base[f][ok])
        row["subst"] = row["subst_F5a"]
        row["ctx_indep"] = min(row[f"subst_{f}"] for f in fam)
        row["ctx_spread"] = max(row[f"subst_{f}"] for f in fam) - row["ctx_indep"]
        row["subst_rand"] = agree(behaviour(model, H[donor_rand], fam["F5a"])[0], base["F5a"])
        # fit: composites of substitutable pieces are substitutable (x·a vs x'·a)
        acts = rng.integers(0, k, len(ev))
        A1 = torch.as_tensor(acts[ok], dtype=torch.long)[:, None]
        _, Hxa = sub.continue_(Hev[ok], A1)
        _, Hda = sub.continue_(Hd, A1)
        row["fit"] = agree(behaviour(model, Hxa, fam["F5a"])[0], behaviour(model, Hda, fam["F5a"])[0])
        # composition: (eventual class, action) lands in the eventual class of the final table
        comp_vals, per_ka = [], {}
        for a in range(k):
            A = torch.full((len(ev), 1), a, dtype=torch.long)
            _, Hx = sub.continue_(Hev, A)
            tgt = T_final[Pi[ev], a]
            good = tgt >= 0
            mem = np.array([rng.choice(by_class[t]) if t >= 0 and len(by_class[t]) else -1 for t in tgt])
            good &= mem >= 0
            if good.sum() == 0:
                continue
            ag = (behaviour(model, Hx[good], fam["F5a"])[0] == behaviour(model, H[mem[good]], fam["F5a"])[0]).mean(axis=(1, 2))
            comp_vals.append(ag)
            for c_, v in zip(Pi[ev][good], ag):
                per_ka.setdefault((int(c_), a), []).append(v)
        row["composition"] = float(np.mean(np.concatenate(comp_vals))) if comp_vals else float("nan")
        per_k = {}
        for (c_, a), v in per_ka.items():
            per_k.setdefault(c_, []).append(np.mean(v))
        row["reuse"] = float(np.mean([min(v) for v in per_k.values()])) if per_k else float("nan")
        # recursive: 2-step chunk lands in the eventual class of the composite of composites
        rec = []
        for j in range(chunks2.shape[1]):
            A2 = torch.as_tensor(chunks2[:, j], dtype=torch.long)
            _, Hx = sub.continue_(Hev, A2)
            t1 = T_final[Pi[ev], chunks2[:, j, 0]]
            tgt = np.where(t1 >= 0, T_final[np.maximum(t1, 0), chunks2[:, j, 1]], -1)
            mem = np.array([rng.choice(by_class[t]) if t >= 0 and len(by_class[t]) else -1 for t in tgt])
            good = (tgt >= 0) & (mem >= 0)
            if good.sum():
                rec.append((behaviour(model, Hx[good], fam["F5a"])[0] == behaviour(model, H[mem[good]], fam["F5a"])[0]).mean(axis=(1, 2)))
        row["recursive"] = float(np.mean(np.concatenate(rec))) if rec else float("nan")
        # microscopic
        row["param_norm"] = float(np.linalg.norm(theta))
        row.update(C.effective_rank(R))
        row["interference"] = C.interference(R[fit], Pi[fit], K)
        idx = ev[:N_JAC]
        Js = []
        for a in range(k):
            h = H[idx].clone().requires_grad_(True)
            lg, _ = model(torch.full((len(idx), 1), a, dtype=torch.long), h0=h)
            J = torch.stack([torch.autograd.grad(lg[:, 0, o].sum(), h, retain_graph=True)[0] for o in range(world.n_obs)], 1)
            Js.append(J.detach().numpy())
        Js = np.stack(Js, 1)   # (n, actions, obs, d)
        row["sens_between"] = C.subspace_energy(Js, V)
        row["sens_between_rand"] = C.subspace_energy(Js, G.numpy())
        flat = Js.reshape(len(idx), k, -1)
        flat = flat / (np.linalg.norm(flat, axis=2, keepdims=True) + 1e-9)
        cos = np.einsum("nad,nbd->nab", flat, flat)
        iu = np.triu_indices(k, 1)
        row["jac_ctx_dep"] = float(1 - cos[:, iu[0], iu[1]].mean())
        dd = model.d
        row["layer_align"] = C.align(sd["out.weight"].numpy(), sd["gru.weight_hh_l0"].numpy()[2 * dd:3 * dd])
        row["drift"] = float("nan") if prev_R is None else 1 - C.linear_cka(R, prev_R)
        row["weight_drift"] = float("nan") if prev_theta is None else float(np.linalg.norm(theta - prev_theta) / np.linalg.norm(theta))
        prev_R, prev_theta = R, theta
        m = metrics.get(str(step), {})
        row.update(q_classes=m.get("n_classes"), q_meta=m.get("metastability"), q_closure=m.get("closure"),
                   q_coherence=m.get("action_coherence"), q_lawfree=m.get("crystallization_lawfree"),
                   q_monoid=(m.get("monoid") or {}).get("monoid_order"), q_discrete=m.get("discreteness"))
        rows.append(row)
        print(f"{Path(run_dir).name} step {step:5d} test {row['test_acc']:.2f} id {row['identity_probe']:.2f} knn {row['identity_knn']:.2f} "
              f"caus {row['causal_remove']:.2f}/{row['causal_keep']:.2f} subst {row['subst']:.2f} ctx {row['ctx_indep']:.2f} fit {row['fit']:.2f} "
              f"comp {row['composition']:.2f} rec {row['recursive']:.2f} rank {row['eff_rank']:.1f} bw {row['between_within']:.2f} q {row['q_classes']}", flush=True)
    res = dict(run=str(run_dir), system="s4", K=K, partition_match_ari=match, rows=rows)
    if out:
        json.dump(res, open(out, "w"))
    return res
