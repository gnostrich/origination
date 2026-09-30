"""Backward tracing of the final interfaces of a modular-addition run.

The final interfaces are the behavioural classes of the hidden-site
configurations at the last checkpoint, found by the frozen extractor
(`grok/extract.py`; stored in metrics.json).  Their identities are frozen
as a partition Π of all p² inputs and traced backward through every
checkpoint.  Held-out instances are the task's own test split, so every
"eval" number is about inputs the model never trained on.

For a one-block model the hidden site feeds the readout directly, so the
substitution context is trivial there; context-dependent measures use the
closed loop (the model's output token used as an operand in a further
fit), which is the only context this substrate has.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from emergence.grok.models import build
from emergence.grok.task import all_pairs, make_task, split
from emergence.grok.extract import _forward, _js
from emergence.trace import common as C

N_CTX = 8          # closed-loop contexts per instance
N_TRIPLES = 4000
SEED = 0


def load_run(run_dir):
    d = torch.load(Path(run_dir) / "checkpoints.pt", weights_only=False)
    m = json.load(open(Path(run_dir) / "metrics.json"))["metrics"]
    return d["config"], d["checkpoints"], d["log"], m


def final_partition(metrics):
    last = max(metrics, key=int)
    part = np.asarray(metrics[last]["partition"])
    _, labels = np.unique(part, return_inverse=True)
    return labels, np.asarray(metrics[last]["op_table"])


def trace(run_dir, ref_dir=None, out=None):
    cfg, ckpts, log, metrics = load_run(run_dir)
    ref_metrics = metrics if ref_dir is None else load_run(ref_dir)[3]
    Pi, op_T = final_partition(ref_metrics)
    K = int(Pi.max()) + 1
    task = make_task(cfg.get("task", "zmod:97"), cfg["seed"])
    p = task.n
    X = all_pairs(p)
    tr, te = split(p, cfg["train_frac"], cfg["seed"])
    tr, te = tr.numpy(), te.numpy()
    model = build(cfg["arch"], p)
    hs = model.hidden_site
    rng = np.random.default_rng(SEED)
    # fixed donors: for each eval instance a same-class training instance and a random-class one
    by_class = {k: tr[Pi[tr] == k] for k in range(K)}
    donor_same = np.array([rng.choice(by_class[Pi[i]]) if len(by_class[Pi[i]]) else -1 for i in te])
    donor_rand = rng.choice(tr, len(te))
    ctx = rng.integers(0, p, (len(te), N_CTX))
    trip = rng.integers(0, p, (N_TRIPLES, 3))
    log_by_step = {e["step"]: e for e in log}
    prev_R, prev_theta = None, None
    rows = []
    for step in sorted(ckpts):
        sd = ckpts[step]
        model.load_state_dict(sd)
        model.eval()
        theta = np.concatenate([v.numpy().ravel() for v in sd.values()])
        logits, sites = _forward(model, X)
        H = sites[hs]
        R = H.numpy().astype(np.float64)
        pred = logits.argmax(-1).numpy()
        probs = F.softmax(logits, -1).numpy()
        row = dict(step=step, **{k: log_by_step.get(step, {}).get(k) for k in ("train_loss", "test_loss", "train_acc", "test_acc")})
        # ---- identity
        row["identity_probe"], row["identity_probe_fit"] = C.linear_probe(R[tr], Pi[tr], R[te], Pi[te], K)
        row["identity_knn"] = C.knn_purity(R[tr], Pi[tr], R[te], Pi[te])
        geo = C.geometry(R, Pi, K)
        row.update(between_within=geo["between_within"], margin=geo["margin"])
        # ---- causal: the between-final-class subspace of the representation
        V, mu, _, _ = C.between_projector(R[tr], Pi[tr], K)
        Vt = torch.as_tensor(V, dtype=torch.float32)
        mu_t = torch.as_tensor(mu, dtype=torch.float32)
        Hte = H[te]
        comp = (Hte - mu_t) @ Vt.T @ Vt
        def agree(Hp):
            lg, _ = _forward(model, X[te], patch={hs: Hp})
            return float((lg.argmax(-1).numpy() == pred[te]).mean())
        row["causal_remove"] = 1 - agree(Hte - comp)
        row["causal_keep"] = agree(mu_t + comp)
        G = torch.linalg.qr(torch.randn(H.shape[1], V.shape[0], generator=torch.Generator().manual_seed(1)))[0].T
        row["causal_remove_rand"] = 1 - agree(Hte - (Hte - mu_t) @ G.T @ G)
        # ---- substitutability (same eventual class, unseen recipient inputs)
        ok = donor_same >= 0
        lg_s, _ = _forward(model, X[te[ok]], patch={hs: H[donor_same[ok]]})
        row["subst"] = float((lg_s.argmax(-1).numpy() == pred[te[ok]]).mean())
        row["subst_js"] = float(np.mean(_js(F.softmax(lg_s, -1).numpy(), probs[te[ok]]) < 0.05))
        lg_r, _ = _forward(model, X[te], patch={hs: H[donor_rand]})
        row["subst_rand"] = float((lg_r.argmax(-1).numpy() == pred[te]).mean())
        # ---- closed-loop: context independence, composition, recursion
        op_t = pred.reshape(p, p)
        o_x, o_d = pred[te[ok]], pred[donor_same[ok]]
        row["subst_ctx"] = float(np.mean(op_t[o_x[:, None], ctx[ok]] == op_t[o_d[:, None], ctx[ok]]))
        a_te, b_te = X[te, 0].numpy(), X[te, 1].numpy()
        row["composition"] = float(np.mean(op_t[a_te, b_te] == op_T[a_te, b_te]))
        a, b, c = trip.T
        row["recursive"] = float(np.mean(op_t[op_t[a, b], c] == op_T[op_T[a, b], c]))
        row["assoc"] = float(np.mean(op_t[op_t[a, b], c] == op_t[a, op_t[b, c]]))
        # ---- microscopic
        row["param_norm"] = float(np.linalg.norm(theta))
        row.update(C.effective_rank(R))
        row["interference"] = C.interference(R[tr], Pi[tr], K)
        if cfg["arch"] == "mlp":
            J, W_prev = sd["out.weight"].numpy(), sd["l2.weight"].numpy()
        else:
            J, W_prev = sd["unembed.weight"].numpy(), sd["mlp_out.weight"].numpy()
        row["sens_between"] = C.subspace_energy(J, V)
        row["sens_between_rand"] = C.subspace_energy(J, G.numpy())
        row["layer_align"] = C.align(J, W_prev)
        row["jac_ctx_dep"] = 0.0   # one-block model: the readout Jacobian is context-free by construction
        row["drift"] = float("nan") if prev_R is None else 1 - C.linear_cka(R, prev_R)
        row["weight_drift"] = float("nan") if prev_theta is None else float(np.linalg.norm(theta - prev_theta) / np.linalg.norm(theta))
        prev_R, prev_theta = R, theta
        # ---- existing quotient measures at this checkpoint
        m = metrics.get(str(step), {})
        row.update(q_classes=m.get("n_classes"), q_stable=m.get("n_stable_classes"), q_meta=m.get("metastability"),
                   q_closure=m.get("closure"), q_assoc=(m.get("op") or {}).get("associativity"),
                   q_latin=(m.get("op") or {}).get("latin"), q_discrete=m.get("discreteness"))
        rows.append(row)
        print(f"{Path(run_dir).name} step {step:6d} test {row['test_acc']:.2f} id {row['identity_probe']:.2f} knn {row['identity_knn']:.2f} "
              f"caus {row['causal_remove']:.2f}/{row['causal_keep']:.2f} subst {row['subst']:.2f} comp {row['composition']:.2f} rec {row['recursive']:.2f} "
              f"rank {row['eff_rank']:.1f} bw {row['between_within']:.2f} sens {row['sens_between']:.2f}", flush=True)
    res = dict(run=str(run_dir), ref=str(ref_dir) if ref_dir else None, system="modadd", arch=cfg["arch"], K=K, rows=rows)
    if out:
        json.dump(res, open(out, "w"))
    return res
