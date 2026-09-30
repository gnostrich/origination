"""Behavioural interface discovery (BID): a label-free search over internal
subspaces for interface quality under transplantation.

Follow-up to the click experiment.  Nothing about the task, model, training
runs, checkpoints, pressure levels or the definition of an interface
changes; only the *candidate dictionary* does.  Instead of principal or
output-sensitivity directions, candidates are subspaces `U` (rank 1, 2, 4,
8) found by maximising, on a discovery pool, the properties that define an
interface operationally:

    s_mag   non-trivial behavioural effect
    s_count few (2..8) substitutable types
    s_res   type count stable across nearby resolutions
    s_part  type partition stable across nearby resolutions
    s_cov   type assignments cover held-out donors
    s_A     effects generalise to held-out recipient contexts
    s_bal   balanced type usage

    J(U) = geometric mean of the seven,

so every property is necessary.  `J` depends on `U` only through the
projector `UᵀU`: the basis inside `U` does not matter, and the search moves
on the Grassmannian (random orthonormal restarts, Gaussian steps followed
by re-orthonormalisation, annealed step size).  Sufficiency (B),
context-generality on an independent pool (C), reuse (D) and composition
(E) are NOT in the objective; they are held-out tests.

Three disjoint pools of inputs: discovery (training inputs) for the
search; validation (one half of the held-out inputs) for selecting
candidates by the *unchanged* acceptance criterion of the frozen
experiment; final (the other half) for A–E and the two-stage composition
tests, run exactly once.  Controls: matched random subspaces of the same
rank through the same validation and final tests; the same search on
shuffled-weight models.

The generating circuit is never read here.  `novelty.py` unseals it after
everything is frozen.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np

from emergence.click import interfaces as I
from emergence.click.model import SITES
from emergence.grok.compare import adjusted_rand_index as ari

RANKS = (1, 2, 4, 8)
EPS = (0.2, 0.3, 0.4)          # the frozen resolutions; 0.3 is the typing resolution
SEARCH = dict(restarts=8, iters=150, sigma0=0.3, sigma_min=0.02, decay=0.98)
POOL = 64
N_RAND = 20                    # matched random subspaces per accepted interface
CKPT_SUBSET = tuple(range(0, 31, 2))   # 16 of the 31 checkpoints, last included
COMP = dict(pred_min=0.8, nonadd_min=0.1)   # stage-2 thresholds (frozen)
CFG = I.CFG


# ----------------------------------------------------------------- pools
def make_pools(train_idx, test_idx, n=POOL, seed=2024):
    """Discovery from training inputs; validation and final from disjoint
    halves of the held-out inputs.  All fixed once."""
    rng = np.random.default_rng(seed)
    tr = rng.permutation(train_idx)
    te = rng.permutation(test_idx)
    half = len(te) // 2
    A, B = te[:half], te[half:]
    disc = dict(d_fit=tr[:n], d_gen=tr[n:2 * n], r_fit=tr[2 * n:3 * n], r_gen=tr[3 * n:4 * n])
    val = dict(d_fit=A[:n], d_gen=A[n:2 * n], r_fit=A[2 * n:3 * n], r_gen=A[3 * n:4 * n])
    fin = dict(d_ref=B[:n], r_ref=B[n:2 * n], d_new=B[2 * n:3 * n], r_new=B[3 * n:4 * n], r_alt=B[4 * n:5 * n])
    return disc, val, fin


# --------------------------------------------------------------- objective
def components(sub, site, U, pools):
    """The seven first-order properties of subspace U on a pool with
    d_fit / d_gen / r_fit / r_gen."""
    Z = sub.Z[site]
    Zd, Zdg, Zr, Zrg = (Z[pools[k]] for k in ("d_fit", "d_gen", "r_fit", "r_gen"))
    E = sub.effects(site, Zd, Zr, U)
    mag = I.rms(E)
    out = dict(mag=mag)
    if mag < 1e-9:
        out.update(N={e: 0 for e in EPS}, s=dict(mag=0, count=0, res=0, part=0, cov=0, A=0, bal=0), J=0.0)
        return out
    P = E.reshape(len(Zd), -1)
    Dm = rms_matrix(P, P)
    lab, lead = {}, {}
    for e in EPS:
        lab[e], lead[e] = leader_cluster_matrix(Dm, e * mag)
    N = {e: int(len(lead[e])) for e in EPS}
    labels, leaders = lab[0.3], lead[0.3]
    n = N[0.3]
    shares = np.bincount(labels) / len(labels)
    s_mag = min(1.0, mag / CFG["mag_min"])
    s_count = float(n >= 2) * np.exp(-max(0, n - CFG["max_types"]) / 4)
    s_res = 1 - (max(N.values()) - min(N.values())) / max(N.values())
    s_part = float(np.mean([ari(labels, lab[0.2]), ari(labels, lab[0.4])]))
    Pg = sub.effects(site, Zdg, Zr, U).reshape(len(Zdg), -1)
    Dg = rms_matrix(Pg, P[leaders])
    s_cov = float(np.mean(Dg.min(1) <= 0.3 * mag))
    tg = Dg.argmin(1)
    A = I.r2(sub.effects(site, Zd[leaders], Zrg, U)[tg], sub.effects(site, Zdg, Zrg, U))
    s_A = float(np.clip(A, 0, 1)) if np.isfinite(A) else 0.0
    s_bal = float(-(shares * np.log(shares)).sum() / np.log(n)) if n > 1 else 0.0
    s = dict(mag=s_mag, count=s_count, res=s_res, part=max(s_part, 0.0), cov=s_cov, A=s_A, bal=s_bal)
    J = float(np.prod([max(v, 1e-6) for v in s.values()]) ** (1 / len(s)))
    out.update(N=N, n_types=n, max_share=float(shares.max()), A=float(A), s=s, J=J)
    return out


def rms_matrix(A, Bm):
    """rms distance between every row of A and every row of Bm."""
    sq = (A * A).sum(1)[:, None] + (Bm * Bm).sum(1)[None] - 2 * A @ Bm.T
    return np.sqrt(np.maximum(sq, 0) / A.shape[1])


def leader_cluster_matrix(Dm, eps):
    """`interfaces.leader_cluster` computed from the pairwise rms-distance matrix (identical output)."""
    n = len(Dm)
    labels = -np.ones(n, dtype=int)
    leaders = []
    for i in range(n):
        if leaders:
            d = Dm[i, leaders]
            hit = np.nonzero(d <= eps)[0]
            if len(hit):
                labels[i] = int(hit[0])
                continue
        leaders.append(i)
        labels[i] = len(leaders) - 1
    return labels, np.array(leaders)


def random_subspace(rng, k, dim):
    return I.orthonormal(rng.normal(size=(k, dim)))


def search(sub, site, k, pools, rng, cfg=SEARCH):
    """Grassmannian local search from random restarts; returns the best U per restart."""
    dim = sub.Z[site].shape[1]
    results = []
    for r in range(cfg["restarts"]):
        U = random_subspace(rng, k, dim)
        best = components(sub, site, U, pools)
        sigma = cfg["sigma0"]
        for it in range(cfg["iters"]):
            V = I.orthonormal(U + sigma * rng.normal(size=U.shape))
            c = components(sub, site, V, pools)
            if c["J"] >= best["J"]:
                U, best = V, c
            sigma = max(cfg["sigma_min"], sigma * cfg["decay"])
        results.append(dict(rank=k, restart=r, U=U, disc=best))
    return results


# -------------------------------------------------------------- validation
def validate(sub, site, U, pools):
    """The frozen acceptance criterion, evaluated on the validation pool
    (d_fit/r_fit as reference, d_gen/r_gen as fresh) — unchanged from the
    first experiment, plus the objective components for reporting."""
    ref = dict(d_ref=pools["d_fit"], r_ref=pools["r_fit"], d_new=pools["d_gen"], r_new=pools["r_gen"],
               r_alt=pools["r_gen"])
    d = analyse_subspace(sub, site, U, ref)
    counts = [analyse_subspace(sub, site, U, ref, eps_rel=e)["n_types"] for e in CFG["eps_alt"]] + [d["n_types"]]
    d["plateau"] = bool(I.plateau(counts, CFG["plateau_tol"]))
    d["passed"] = bool(I.passes(d, CFG) and d["plateau"])
    d["comp"] = components(sub, site, U, pools)
    return {k: v for k, v in d.items() if not k.startswith("_")}


def analyse_subspace(sub, site, U, pools, cfg=CFG, eps_rel=None):
    """`interfaces.analyse_direction` for a subspace (U with orthonormal rows)."""
    eps_rel = cfg["eps_rel"] if eps_rel is None else eps_rel
    Z = sub.Z[site]
    Zd, Zr, Zdn, Zrn, Zra = (Z[pools[k]] for k in ("d_ref", "r_ref", "d_new", "r_new", "r_alt"))
    E_ref = sub.effects(site, Zd, Zr, U)
    mag = I.rms(E_ref)
    out = {"mag": mag, "outputs_moved": int((I.rms(E_ref, axis=(0, 1)) >= cfg["out_min"]).sum())}
    if mag <= 0:
        out.update(n_types=0, max_share=1.0, A=float("nan"), A_base=float("nan"), C=float("nan"))
        return out
    eps = eps_rel * mag
    P_ref = E_ref.reshape(len(Zd), -1)
    labels, leaders = I.leader_cluster(P_ref, eps)
    shares = np.bincount(labels) / len(labels)
    out.update(n_types=int(len(leaders)), max_share=float(shares.max()))
    P_new = sub.effects(site, Zdn, Zr, U).reshape(len(Zdn), -1)
    t_new = I.assign(P_new, leaders, P_ref)
    E_lead_new = sub.effects(site, Zd[leaders], Zrn, U)
    E_new_new = sub.effects(site, Zdn, Zrn, U)
    out["A"] = I.r2(E_lead_new[t_new], E_new_new)
    rng = np.random.default_rng(0)
    out["A_base"] = float(np.mean([I.r2(E_lead_new[rng.permutation(t_new)], E_new_new) for _ in range(cfg["n_shuffle"])]))
    P_alt = sub.effects(site, Zd, Zra, U).reshape(len(Zd), -1)
    labels_alt, _ = I.leader_cluster(P_alt, eps_rel * I.rms(P_alt))
    out["C"] = float(ari(labels, labels_alt))
    out["_labels"], out["_leaders"], out["_t_new"] = labels, leaders, t_new
    return out


def overlap(U, Us):
    """Fraction of U's energy inside the span of the subspaces in Us."""
    if not Us:
        return 0.0
    W = I.orthonormal(np.vstack(Us))
    return float(np.linalg.norm(U @ W.T) ** 2 / len(U))


def select(cands, thr=0.5):
    """Greedy selection of validated candidates by validation J, dropping
    subspaces mostly inside the span of those already kept."""
    ok = [c for c in cands if c["val"]["passed"]]
    ok.sort(key=lambda c: -c["val"]["comp"]["J"])
    keep = []
    for c in ok:
        if overlap(c["U"], [k["U"] for k in keep]) < thr:
            keep.append(c)
    return keep


# ---------------------------------------------------------- final tests
def final_tests(sub, site, acc, pools, cfg=CFG):
    """A–E on the final pool for an accepted set (run once)."""
    res = {"n_accepted": len(acc), "per": []}
    if not acc:
        res["flags"] = dict(A=False, B=False, C=False, D=False, E=False)
        return res
    per = []
    for c in acc:
        d = analyse_subspace(sub, site, c["U"], pools, cfg)
        d["passed"] = bool(I.passes(d, cfg))
        per.append(d)
    res["per"] = [{k: v for k, v in d.items() if not k.startswith("_")} for d in per]
    Uall = I.orthonormal(np.vstack([c["U"] for c in acc]))
    res.update(I.test_B(sub, site, Uall, pools, cfg))
    res["C_mean"] = float(np.mean([d["C"] for d in per]))
    res["D_max"] = int(max(d["outputs_moved"] for d in per))
    pairs = []
    for i, j in combinations(range(min(len(acc), 4)), 2):
        e = test_E_sub(sub, site, acc[i]["U"], acc[j]["U"], per[i], per[j], pools, cfg)
        e.update(i=i, j=j)
        pairs.append(e)
    res["pairs"] = pairs
    res["E_best"] = max((p["E"] for p in pairs), default=float("nan"))
    res["flags"] = dict(
        A=any(d["passed"] for d in per),
        B=res["B"] >= cfg["B_min"] and res["B"] - res["B_base"] >= cfg["B_margin"],
        C=res["C_mean"] >= cfg["C_min"], D=res["D_max"] >= cfg["D_min"],
        E=bool(pairs) and res["E_best"] >= cfg["E_min"])
    return res


def test_E_sub(sub, site, UI, UJ, dI, dJ, pools, cfg=CFG):
    U2 = I.orthonormal(np.vstack([UI, UJ]))
    Z = sub.Z[site]
    Zd, Zr, Zdn, Zrn = (Z[pools[k]] for k in ("d_ref", "r_ref", "d_new", "r_new"))
    E_ref = sub.effects(site, Zd, Zr, U2)
    P_ref = E_ref.reshape(len(Zd), -1)
    jl, jlead = I.leader_cluster(P_ref, cfg["eps_rel"] * I.rms(E_ref))
    pair_ref = dI["_labels"] * 1000 + dJ["_labels"]
    pair_new = dI["_t_new"] * 1000 + dJ["_t_new"]
    rep = {}
    for i, p in enumerate(pair_ref):
        rep.setdefault(p, i)
    P_new = sub.effects(site, Zdn, Zr, U2).reshape(len(Zdn), -1)
    idx, fallback = [], 0
    for i, p in enumerate(pair_new):
        if p in rep:
            idx.append(rep[p])
        else:
            fallback += 1
            idx.append(int(jlead[I.assign(P_new[i:i + 1], jlead, P_ref)[0]]))
    E = I.r2(sub.effects(site, Zd[np.array(idx)], Zrn, U2), sub.effects(site, Zdn, Zrn, U2))
    return dict(E=E, n_joint=int(len(jlead)), n_pairs_seen=int(len(rep)),
                product=int(dI["n_types"] * dJ["n_types"]), fallback=fallback)


# ------------------------------------------------------ stage 2: composition
def joint_effects(sub, site, Us, Zd, Zr):
    return sub.effects(site, Zd, Zr, I.orthonormal(np.vstack(Us)))


def compose(sub, site, parts, part_labels, part_tnew, pools, cfg=CFG):
    """Joint transplant of several interfaces (I, J[, K]) ↦ B.

    * pred     : the joint type of a fresh donor (assigned on reference
                 recipients) predicts the joint effect on fresh recipients (R²).
    * nonadd   : 1 − R² of the additive model  E_IJ ≈ Σ_i E_i  — how much of
                 the joint effect is interaction, not the sum of the parts.
    * pred_int : the interaction residual  E_IJ − Σ_i E_i  on fresh recipients
                 is predicted by the joint type (R²).
    A composite is *stable* when pred ≥ 0.8, nonadd ≥ 0.1 and pred_int ≥ 0.8.
    """
    Z = sub.Z[site]
    Zd, Zr, Zdn, Zrn = (Z[pools[k]] for k in ("d_ref", "r_ref", "d_new", "r_new"))
    E_ref = joint_effects(sub, site, parts, Zd, Zr)
    P_ref = E_ref.reshape(len(Zd), -1)
    labels, leaders = I.leader_cluster(P_ref, cfg["eps_rel"] * I.rms(E_ref))
    P_new = joint_effects(sub, site, parts, Zdn, Zr).reshape(len(Zdn), -1)
    t_new = I.assign(P_new, leaders, P_ref)
    E_new = joint_effects(sub, site, parts, Zdn, Zrn)
    E_lead = joint_effects(sub, site, parts, Zd[leaders], Zrn)
    pred = I.r2(E_lead[t_new], E_new)
    S_new = sum(sub.effects(site, Zdn, Zrn, U) for U in parts)
    S_lead = sum(sub.effects(site, Zd[leaders], Zrn, U) for U in parts)
    nonadd = 1 - I.r2(S_new, E_new)  # additive model's unexplained share
    R_new, R_lead = E_new - S_new, E_lead - S_lead
    pred_int = I.r2(R_lead[t_new], R_new)
    # the joint type as a function of the parts' types
    key_ref = sum(l * (100 ** i) for i, l in enumerate(part_labels))
    det = float(max(ari(labels, key_ref), 0.0))
    stable = bool(pred >= COMP["pred_min"] and nonadd >= COMP["nonadd_min"] and pred_int >= COMP["pred_min"])
    return dict(pred=float(pred), nonadd=float(nonadd), pred_int=float(pred_int), n_joint=int(len(leaders)),
                ari_with_parts=det, stable=stable, _labels=labels, _t_new=t_new)


def stage2(sub, site, acc, per, pools, cfg=CFG):
    """Pairs (I,J) ⇝ B over the accepted interfaces; for each stable
    composite B, triples (B,K) ⇝ C with a third interface K."""
    out = {"pairs": [], "triples": []}
    idx = list(range(min(len(acc), 4)))
    comps = {}
    for i, j in combinations(idx, 2):
        c = compose(sub, site, [acc[i]["U"], acc[j]["U"]], [per[i]["_labels"], per[j]["_labels"]],
                    [per[i]["_t_new"], per[j]["_t_new"]], pools, cfg)
        comps[(i, j)] = c
        out["pairs"].append({k: v for k, v in c.items() if not k.startswith("_")} | dict(i=i, j=j))
    for (i, j), c in comps.items():
        if not c["stable"]:
            continue
        for k in idx:
            if k in (i, j):
                continue
            t = compose(sub, site, [acc[i]["U"], acc[j]["U"], acc[k]["U"]], [c["_labels"], per[k]["_labels"]],
                        [c["_t_new"], per[k]["_t_new"]], pools, cfg)
            # additivity here is tested against  E_IJ + E_K  (composite as a unit), not the three parts
            Z = sub.Z[site]
            Zd, Zr, Zdn, Zrn = (Z[pools[q]] for q in ("d_ref", "r_ref", "d_new", "r_new"))
            UIJ = I.orthonormal(np.vstack([acc[i]["U"], acc[j]["U"]]))
            E_new = joint_effects(sub, site, [UIJ, acc[k]["U"]], Zdn, Zrn)
            S_new = sub.effects(site, Zdn, Zrn, UIJ) + sub.effects(site, Zdn, Zrn, acc[k]["U"])
            t["nonadd_unit"] = float(1 - I.r2(S_new, E_new))
            t["stable_unit"] = bool(t["pred"] >= COMP["pred_min"] and t["nonadd_unit"] >= COMP["nonadd_min"] and t["pred_int"] >= COMP["pred_min"])
            out["triples"].append({q: v for q, v in t.items() if not q.startswith("_")} | dict(i=i, j=j, k=k))
    out["n_stable_pairs"] = int(sum(p["stable"] for p in out["pairs"]))
    out["n_stable_triples"] = int(sum(t["stable_unit"] for t in out["triples"]))
    return out


# ----------------------------------------------------------- one cell
def run_site(sub, site, disc, val, fin, rng, ranks=RANKS, search_cfg=SEARCH):
    cands = []
    for k in ranks:
        for c in search(sub, site, k, disc, rng, search_cfg):
            c["val"] = validate(sub, site, c["U"], val)
            cands.append(c)
    acc = select(cands)
    fin_res = final_tests(sub, site, acc, fin)
    per = [analyse_subspace(sub, site, c["U"], fin) for c in acc]
    s2 = stage2(sub, site, acc, per, fin) if len(acc) >= 2 else {"pairs": [], "triples": [], "n_stable_pairs": 0, "n_stable_triples": 0}
    # matched random controls: same rank, validation criterion and final A
    ctrl = []
    dim = sub.Z[site].shape[1]
    for c in acc:
        vp, fp = 0, 0
        for _ in range(N_RAND):
            R = random_subspace(rng, len(c["U"]), dim)
            vp += validate(sub, site, R, val)["passed"]
            d = analyse_subspace(sub, site, R, fin)
            fp += I.passes(d)
        ctrl.append(dict(rank=int(len(c["U"])), val_pass=vp / N_RAND, final_A_pass=fp / N_RAND))
    # random-subspace validation pass rate per rank regardless of acceptance
    rand_rate = {}
    for k in ranks:
        rand_rate[k] = float(np.mean([validate(sub, site, random_subspace(rng, k, dim), val)["passed"] for _ in range(N_RAND)]))
    return dict(
        candidates=[dict(rank=c["rank"], restart=c["restart"], J_disc=c["disc"]["J"], s_disc=c["disc"]["s"],
                         J_val=c["val"]["comp"]["J"], s_val=c["val"]["comp"]["s"], val=_clean(c["val"]))
                    for c in cands],
        accepted=[dict(rank=c["rank"], restart=c["restart"], U=c["U"].tolist(), J_disc=c["disc"]["J"],
                       J_val=c["val"]["comp"]["J"], val=_clean(c["val"])) for c in acc],
        final=fin_res, stage2=s2, controls=ctrl, rand_val_rate=rand_rate,
        n_val_passed=int(sum(c["val"]["passed"] for c in cands)),
        best_J_val=float(max(c["val"]["comp"]["J"] for c in cands)),
        best_J_disc=float(max(c["disc"]["J"] for c in cands)),
    )


def _clean(d):
    return {str(k): (v if not isinstance(v, dict) else _clean(v)) for k, v in d.items()
            if not (isinstance(k, str) and k.startswith("_"))}


def shuffle_weights(model, seed):
    """Permute the entries of every weight matrix and bias (null model)."""
    import torch
    rng = np.random.default_rng(seed)
    sd = {}
    for k, v in model.state_dict().items():
        a = v.numpy().copy()
        sd[k] = torch.as_tensor(rng.permutation(a.ravel()).reshape(a.shape))
    model.load_state_dict(sd)
    return model
