"""Label-free extraction of the candidate emergent algebra at a checkpoint.

Given a substrate (a model with sites) and its input space, we recover

1. **effective objects at the hidden site** -- behavioural classes of the
   configurations found there, where two configurations are equivalent when
   substituting one for the other across sampled contexts leaves downstream
   behaviour unchanged (JS divergence of the output distribution < eps);
2. **discreteness** -- polarisation of pairwise behavioural distances
   (configurations are either the same thing or clearly different things);
3. **metastability** -- retention of the behavioural class under Gaussian
   noise at the hidden site (basin size in units of the site's own scale);
4. **clicks** -- for every tuple of input-site configurations, the stability
   of the composite configuration it produces (retention at a reference
   noise), giving the compatibility tensor ``C`` and its polarisation;
5. **closure** -- fraction of products that land in a *stable* class;
6. **the induced closed-loop operation** -- the world re-feeds the substrate's
   output as an input (the output alphabet is the input alphabet), so
   ``op(a, b) := argmax behaviour``; its coherence laws (associativity,
   commutativity, identity, inverses, Latin property) are measured on
   sampled tuples, with no reference to the true labels.

None of these use ``(a + b) mod p``, the labels or the train/test split.
The partition of inputs by hidden class is returned so that different seeds
and architectures can be compared as *quotients* (``compare.py``).

Important caveat, stated in the README: for one-block models the hidden
site's downstream map is context-free, so its behavioural classes coincide
with output-distribution classes.  The internal content of the measures is
in (2), (3), (4) and (5); (6) is an unsupervised self-consistency measure.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from .task import all_pairs


@dataclass
class ExtractConfig:
    eps_beh: float = 0.05  # JS divergence (bits) below which behaviours are "the same"
    d_far: float = 0.5  # JS above which behaviours are "clearly different"
    n_contexts: int = 2  # contexts a hidden configuration is substituted into
    n_pair_sample: int = 600  # states sampled for the pairwise distance matrix
    sigmas: tuple = (0.25, 0.5, 1.0, 2.0)  # noise scales (x site RMS)
    sigma_ref: float = 0.5
    n_noise: int = 4
    stable_retention: float = 0.75
    n_triples: int = 4000
    batch: int = 4096


def _js(P: np.ndarray, Q: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Row-wise Jensen-Shannon divergence in bits, P,Q (n, k) or broadcastable."""
    M = 0.5 * (P + Q)
    def kl(A, B):
        return np.sum(np.where(A > 0, A * (np.log2(A + eps) - np.log2(B + eps)), 0.0), axis=-1)
    return 0.5 * kl(P, M) + 0.5 * kl(Q, M)


def _forward(model, X, patch=None, noise=None, batch=4096):
    outs, sites = [], {}
    with torch.no_grad():
        for s in range(0, len(X), batch):
            xb = X[s : s + batch]
            pb = None if patch is None else {k: v[s : s + batch] for k, v in patch.items()}
            lg, st = model(xb, patch=pb, noise=noise)
            outs.append(lg)
            for k, v in st.items():
                sites.setdefault(k, []).append(v)
    return torch.cat(outs), {k: torch.cat(v) for k, v in sites.items()}


def behaviour_of(model, X, site, states, contexts, batch=4096) -> np.ndarray:
    """Behaviour vector of each configuration in ``states`` at ``site``:
    output distributions when substituted into each of ``contexts``,
    concatenated.  (n, n_ctx * p)."""
    n = states.shape[0]
    feats = []
    for c in contexts:
        Xc = c[None].expand(n, -1)
        lg, _ = _forward(model, Xc, patch={site: states}, batch=batch)
        feats.append(F.softmax(lg, -1).numpy())
    return np.concatenate(feats, axis=1)


def leader_cluster(B: np.ndarray, n_ctx: int, eps: float, batch: int = 2048):
    """Greedy behavioural clustering: mean-over-contexts JS to the leader < eps."""
    n, kk = B.shape
    p = kk // n_ctx
    B3 = B.reshape(n, n_ctx, p)
    labels = np.full(n, -1, dtype=int)
    leaders: list[np.ndarray] = []
    L = np.empty((0, n_ctx, p))
    for s in range(0, n, batch):
        blk = B3[s : s + batch]
        for i in range(blk.shape[0]):
            x = blk[i]
            if len(leaders):
                d = _js(L, x[None]).mean(axis=1)  # (n_leaders,)
                j = int(np.argmin(d))
                if d[j] < eps:
                    labels[s + i] = j
                    continue
            leaders.append(x)
            L = np.stack(leaders)
            labels[s + i] = len(leaders) - 1
    return labels, L


def extract(model, p: int, cfg: ExtractConfig, rng: np.random.Generator) -> dict:
    model.eval()
    X = all_pairs(p)
    n = len(X)
    hs = model.hidden_site
    logits, sites = _forward(model, X, batch=cfg.batch)
    probs = F.softmax(logits, -1).numpy()
    H = sites[hs]
    pred = logits.argmax(-1).numpy()
    crisp = probs.max(axis=1)

    # ---- 1. behavioural classes at the hidden site ------------------------
    ctx_idx = rng.choice(n, size=cfg.n_contexts, replace=False)
    contexts = [X[i] for i in ctx_idx]
    B = behaviour_of(model, X, hs, H, contexts, batch=cfg.batch)
    labels, leaders = leader_cluster(B, cfg.n_contexts, cfg.eps_beh)
    n_classes = int(labels.max()) + 1
    class_size = np.bincount(labels)

    # ---- 2. discreteness ------------------------------------------------
    m = min(cfg.n_pair_sample, n)
    sub = rng.choice(n, size=m, replace=False)
    Bs = B[sub].reshape(m, cfg.n_contexts, -1)
    D = np.zeros((m, m))
    for i in range(m):
        D[i] = _js(Bs, Bs[i][None]).mean(axis=1)
    iu = np.triu_indices(m, 1)
    d = D[iu]
    discreteness = float(np.mean(d < cfg.eps_beh) + np.mean(d > cfg.d_far))
    frac_same = float(np.mean(d < cfg.eps_beh))

    # ---- 3. metastability: retention under noise at the hidden site ----------
    retention = {}
    ret_pair_ref = np.zeros(n)
    for sigma in cfg.sigmas:
        keep_arg = np.zeros(n)
        keep_js = np.zeros(n)
        for _ in range(cfg.n_noise):
            lg, _ = _forward(model, X, noise={hs: float(sigma)}, batch=cfg.batch)
            pr = F.softmax(lg, -1).numpy()
            keep_arg += (lg.argmax(-1).numpy() == pred)
            keep_js += (_js(pr, probs) < cfg.eps_beh)
        keep_arg /= cfg.n_noise
        keep_js /= cfg.n_noise
        retention[sigma] = {"argmax": float(keep_arg.mean()), "class": float(keep_js.mean())}
        if sigma == cfg.sigma_ref:
            ret_pair_ref = keep_js
    sig = np.array(cfg.sigmas)
    r_arg = np.array([retention[s]["argmax"] for s in cfg.sigmas])
    # basin radius: largest sigma (log-interpolated) at which argmax retention >= 0.5
    if r_arg[0] < 0.5:
        radius = 0.0
    elif r_arg[-1] >= 0.5:
        radius = float(sig[-1])
    else:
        k = int(np.argmax(r_arg < 0.5))
        x0, x1, y0, y1 = np.log(sig[k - 1]), np.log(sig[k]), r_arg[k - 1], r_arg[k]
        radius = float(np.exp(x0 + (0.5 - y0) * (x1 - x0) / (y1 - y0)))
    metastability = float(retention[cfg.sigma_ref]["class"])

    # ---- 4. clicks: compatibility of input tuples = stability of the composite
    C = ret_pair_ref.reshape(p, p)
    v = C.ravel()
    polarization = float(1.0 - 2.0 * np.mean(np.minimum(v, 1 - v)))

    # ---- 5. closure: products land in stable classes ------------------------
    class_ret = np.zeros(n_classes)
    np.add.at(class_ret, labels, ret_pair_ref)
    class_ret /= np.maximum(class_size, 1)
    stable = class_ret >= cfg.stable_retention
    closure = float(np.mean(stable[labels]))
    n_stable_classes = int(stable.sum())

    # ---- 6. closed-loop induced operation and its laws (label-free) ----------
    op = pred.reshape(p, p)
    comm = float(np.mean(op == op.T))
    a = rng.integers(0, p, cfg.n_triples)
    b = rng.integers(0, p, cfg.n_triples)
    c = rng.integers(0, p, cfg.n_triples)
    assoc = float(np.mean(op[op[a, b], c] == op[a, op[b, c]]))
    ar = np.arange(p)
    id_score = np.array([0.5 * (np.mean(op[e] == ar) + np.mean(op[:, e] == ar)) for e in range(p)])
    e_best = int(id_score.argmax())
    identity = float(id_score.max())
    rows_perm = float(np.mean([len(set(op[i].tolist())) == p for i in range(p)]))
    cols_perm = float(np.mean([len(set(op[:, j].tolist())) == p for j in range(p)]))
    latin = 0.5 * (rows_perm + cols_perm)
    inv = float(np.mean([(op[i] == e_best).any() for i in range(p)]))
    # image size: how many distinct outputs the operation produces
    n_outputs = int(len(np.unique(op)))

    crystallization = float((max(metastability, 1e-9) * max(discreteness, 1e-9)
                             * max(closure, 1e-9) * max(assoc, 1e-9)) ** 0.25)
    return {
        "n_classes": n_classes,
        "n_stable_classes": n_stable_classes,
        "compression": float(np.clip(np.log(n / max(n_classes, 1)) / np.log(p), 0, 1)),
        "class_sizes_top": np.sort(class_size)[::-1][:10].tolist(),
        "discreteness": discreteness,
        "frac_same_pairs": frac_same,
        "retention": {str(k): v for k, v in retention.items()},
        "basin_radius": radius,
        "metastability": metastability,
        "polarization": polarization,
        "C_mean": float(v.mean()),
        "closure": closure,
        "crispness": float(crisp.mean()),
        "op": {
            "commutativity": comm,
            "associativity": assoc,
            "identity": identity,
            "identity_element": e_best,
            "inverses": inv,
            "latin": float(latin),
            "n_outputs": n_outputs,
        },
        "crystallization": crystallization,
        "partition": labels.tolist(),
        "op_table": op.tolist(),
    }
