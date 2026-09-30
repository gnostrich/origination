"""Frozen, label-free interface discovery for the click experiment.

Everything here was fixed before any run was inspected (see README, "The
click experiment").  Nothing in this module reads the generating circuit;
labels are used only for the *novelty* evaluation in `novelty()`, which
runs after discovery and never feeds back into it.

Candidate interfaces
    A candidate interface is a unit direction `u` in a site's activation
    space (never a neuron, never a cluster).  Two coordinate-free
    dictionaries per site: the leading principal directions of the site's
    activations, and the leading eigen-directions of the site's mean output
    sensitivity  E_x[J(x)ᵀJ(x)],  J = ∂logits/∂z.  A third dictionary of
    random unit directions is the baseline and goes through the identical
    pipeline.

Fit
    Transplanting donor d into recipient r along u:
        z ← z_r + u uᵀ (z_d − z_r)
    The *effect* of d on r along u is the change in the six output logits.
    A donor's *profile* along u is its effect on a fixed pool of reference
    recipients; donors are typed by leader clustering of profiles.

Tests (each yields a number; thresholds fixed in CFG)
    A  substitutability : a fresh donor's type (assigned from its profile on
       reference recipients) predicts its effects on fresh recipients
       through the type leader (R²), versus shuffled types.
    B  sufficiency      : transplanting only along the accepted directions
       reproduces the output bits of the full transplant, versus a random
       subspace of the same dimension.
    C  context-generality: types assigned with an independent recipient
       pool agree with the reference typing (ARI).
    D  reuse            : number of outputs the direction moves.
    E  composition      : a pair of accepted directions, typed jointly,
       predicts the joint transplant on fresh recipients (R²); the number of
       joint types is compared with the product of the factors' type counts.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import torch

from emergence.grok.compare import adjusted_rand_index as ari
from emergence.click.model import SITES

CFG = dict(
    n_dirs=16,      # per dictionary per site
    pool=64,        # donors / recipients per pool
    eps_rel=0.3,    # typing resolution, relative to the rms effect size
    eps_alt=(0.2, 0.4),
    mag_min=0.3,    # rms effect (logits) for a direction to count as doing anything
    min_types=2, max_types=8, nondeg=0.15,
    A_min=0.8, A_margin=0.3,
    B_min=0.9, B_margin=0.1,
    C_min=0.6,
    D_min=2, out_min=0.3,
    E_min=0.8,
    plateau_tol=1,  # max-min of the type count over eps 0.2/0.3/0.4
    dedup=0.5,      # residual norm below which an accepted direction is redundant
    n_shuffle=20, n_rand_basis=10,
    pool_seed=12345, rand_seed=777,
)


# ----------------------------------------------------------------- utilities
def rms(a, axis=None):
    return float(np.sqrt(np.mean(np.square(a), axis=axis))) if axis is None else np.sqrt(np.mean(np.square(a), axis=axis))


def orthonormal(rows):
    Q, _ = np.linalg.qr(np.asarray(rows, dtype=np.float64).T)
    return Q.T[: len(rows)]


def leader_cluster(P, eps):
    """First-come leader clustering with rms distance; returns labels, leaders."""
    labels = -np.ones(len(P), dtype=int)
    leaders = []
    for i, p in enumerate(P):
        for t, l in enumerate(leaders):
            if rms(p - P[l]) <= eps:
                labels[i] = t
                break
        else:
            leaders.append(i)
            labels[i] = len(leaders) - 1
    return labels, np.array(leaders)


def assign(P, L, ref):
    """Nearest-leader assignment of profiles P to leaders L (indices into ref)."""
    d = np.stack([rms(P - ref[l][None], axis=1) for l in L], 1)
    return d.argmin(1)


def r2(pred, actual):
    resid = np.mean(np.square(pred - actual))
    var = np.mean(np.square(actual - actual.mean(0, keepdims=True)))
    return float(1 - resid / var) if var > 0 else float("nan")


def make_pools(train_idx, test_idx, n, seed):
    """Reference pools from the training inputs; fresh pools from held-out inputs."""
    rng = np.random.default_rng(seed)
    tr = rng.permutation(train_idx)
    te = rng.permutation(test_idx)
    return dict(d_ref=tr[:n], r_ref=tr[n:2 * n], d_new=te[:n], r_new=te[n:2 * n], r_alt=te[2 * n:3 * n])


# ------------------------------------------------------------- the substrate
class Sub:
    """Wraps a model at fixed weights: site activations and continuation."""

    def __init__(self, model, X):
        self.m = model
        with torch.no_grad():
            self.x = model.encode(X)
            self.Z = {s: z.numpy().astype(np.float64) for s, z in model.sites(self.x).items()}
            self.logits = model(self.x).numpy().astype(np.float64)

    def cont(self, site, Z):
        with torch.no_grad():
            return self.m.from_site(site, torch.as_tensor(Z, dtype=torch.float32)).numpy().astype(np.float64)

    def effects(self, site, Zd, Zr, U):
        """Δlogits (D, R, n_out) of transplanting along the subspace spanned by U."""
        P = U.T @ U
        Zp = Zr[None] + (Zd[:, None, :] - Zr[None]) @ P
        D, R, W = Zp.shape
        out = self.cont(site, Zp.reshape(-1, W)).reshape(D, R, -1)
        return out - self.cont(site, Zr)[None]

    def sensitivity(self, site, idx):
        m = self.m
        W2, W3 = m.l2.weight.detach().numpy().astype(np.float64), m.l3.weight.detach().numpy().astype(np.float64)
        if site == "h2":
            J = np.broadcast_to(W3, (len(idx),) + W3.shape)
        else:
            pre = (self.Z["h1"][idx] @ W2.T + m.l2.bias.detach().numpy()) > 0
            J = (W3[None] * pre[:, None, :]) @ W2
        return np.einsum("nki,nkj->ij", J, J) / len(idx)

    def candidates(self, site, idx, n, rand_seed):
        Z = self.Z[site][idx]
        Zc = Z - Z.mean(0)
        _, _, Vt = np.linalg.svd(Zc, full_matrices=False)
        pca = Vt[:n]
        w, V = np.linalg.eigh(self.sensitivity(site, idx))
        sens = V[:, ::-1][:, :n].T
        rng = np.random.default_rng(rand_seed)
        R = rng.normal(size=(n, Z.shape[1]))
        rand = R / np.linalg.norm(R, axis=1, keepdims=True)
        return {"pca": pca, "sens": sens, "rand": rand}


# --------------------------------------------------------- single direction
def analyse_direction(sub, site, u, pools, cfg=CFG, eps_rel=None):
    eps_rel = cfg["eps_rel"] if eps_rel is None else eps_rel
    U = u[None] / np.linalg.norm(u)
    Z = sub.Z[site]
    Zd, Zr, Zdn, Zrn, Zra = (Z[pools[k]] for k in ("d_ref", "r_ref", "d_new", "r_new", "r_alt"))
    E_ref = sub.effects(site, Zd, Zr, U)
    mag = rms(E_ref)
    out = {"mag": mag, "outputs_moved": int((rms(E_ref, axis=(0, 1)) >= cfg["out_min"]).sum())}
    if mag <= 0:
        out.update(n_types=0, max_share=1.0, A=float("nan"), A_base=float("nan"), C=float("nan"))
        return out
    eps = eps_rel * mag
    P_ref = E_ref.reshape(len(Zd), -1)
    labels, leaders = leader_cluster(P_ref, eps)
    shares = np.bincount(labels) / len(labels)
    out.update(n_types=int(len(leaders)), max_share=float(shares.max()))
    # A: fresh donors typed on reference recipients, predicted on fresh recipients
    P_new = sub.effects(site, Zdn, Zr, U).reshape(len(Zdn), -1)
    t_new = assign(P_new, leaders, P_ref)
    E_lead_new = sub.effects(site, Zd[leaders], Zrn, U)          # (T, R, out)
    E_new_new = sub.effects(site, Zdn, Zrn, U)                    # (D, R, out)
    out["A"] = r2(E_lead_new[t_new], E_new_new)
    rng = np.random.default_rng(0)
    out["A_base"] = float(np.mean([r2(E_lead_new[rng.permutation(t_new)], E_new_new) for _ in range(cfg["n_shuffle"])]))
    # C: retype the reference donors with an independent recipient pool
    P_alt = sub.effects(site, Zd, Zra, U).reshape(len(Zd), -1)
    labels_alt, _ = leader_cluster(P_alt, eps_rel * rms(P_alt))
    out["C"] = float(ari(labels, labels_alt))
    out["_labels"] = labels
    out["_leaders"] = leaders
    out["_t_new"] = t_new
    return out


def passes(d, cfg=CFG):
    return (d["mag"] >= cfg["mag_min"] and cfg["min_types"] <= d["n_types"] <= cfg["max_types"]
            and 1 - d["max_share"] >= cfg["nondeg"] and d["A"] >= cfg["A_min"]
            and d["A"] - d["A_base"] >= cfg["A_margin"])


def plateau(counts, tol):
    """Resolution stability: the type count must not depend on the typing
    resolution.  A direction along which donors spread continuously has a
    count ~ 1/eps (halving from eps 0.2 to 0.4); one along which they fall
    into separated groups keeps its count.  Added after the smoke test,
    before the sweep: without it a one-dimensional transplant, which carries
    only a scalar of the donor, is 'substitutable' along *any* direction."""
    return max(counts) - min(counts) <= tol


# ------------------------------------------------------------- set-level tests
def dedup(dirs, scores, thr):
    """Greedy orthogonal selection in decreasing score order."""
    keep = []
    for i in np.argsort(-np.asarray(scores)):
        v = dirs[i].copy()
        for j in keep:
            v -= (v @ dirs[j]) * dirs[j]
        if np.linalg.norm(v) >= thr:
            keep.append(i)
    return keep


def test_B(sub, site, U, pools, cfg=CFG):
    Z = sub.Z[site]
    Zd, Zr = Z[pools["d_new"]], Z[pools["r_new"]]
    base_r = sub.cont(site, Zr)
    full = sub.cont(site, Zd) > 0                                  # (D, out)
    def agree(U_):
        E = sub.effects(site, Zd, Zr, U_)
        return float(np.mean((base_r[None] + E > 0) == full[:, None, :]))
    B = agree(U)
    rng = np.random.default_rng(1)
    base = []
    for _ in range(cfg["n_rand_basis"]):
        R = rng.normal(size=(len(U), Z.shape[1]))
        base.append(agree(orthonormal(R)))
    none = float(np.mean((base_r[None] > 0) == full[:, None, :]))
    return dict(B=B, B_base=float(np.mean(base)), B_none=none, dim=int(len(U)))


def test_E(sub, site, u, v, du, dv, pools, cfg=CFG):
    """Joint typing of a pair of accepted directions."""
    U2 = orthonormal([u, v])
    Z = sub.Z[site]
    Zd, Zr, Zdn, Zrn = (Z[pools[k]] for k in ("d_ref", "r_ref", "d_new", "r_new"))
    E_ref = sub.effects(site, Zd, Zr, U2)
    P_ref = E_ref.reshape(len(Zd), -1)
    joint_labels, joint_leaders = leader_cluster(P_ref, cfg["eps_rel"] * rms(E_ref))
    # pair type from the factors
    pair_ref = du["_labels"] * 1000 + dv["_labels"]
    pair_new = du["_t_new"] * 1000 + dv["_t_new"]
    rep = {}
    for i, p in enumerate(pair_ref):
        rep.setdefault(p, i)
    fallback = 0
    idx = []
    P_new = sub.effects(site, Zdn, Zr, U2).reshape(len(Zdn), -1)
    for i, p in enumerate(pair_new):
        if p in rep:
            idx.append(rep[p])
        else:  # unseen pair: nearest joint leader
            fallback += 1
            idx.append(int(joint_leaders[assign(P_new[i:i + 1], joint_leaders, P_ref)[0]]))
    E_pred = sub.effects(site, Zd[np.array(idx)], Zrn, U2)
    E_act = sub.effects(site, Zdn, Zrn, U2)
    return dict(E=r2(E_pred, E_act), n_joint=int(len(joint_leaders)),
                n_pairs_seen=int(len(rep)), product=int(du["n_types"] * dv["n_types"]),
                fallback=fallback)


# ----------------------------------------------------------- one checkpoint
def analyse_checkpoint(model, X, train_idx, test_idx, cfg=CFG):
    sub = Sub(model, X)
    pools = make_pools(train_idx, test_idx, cfg["pool"], cfg["pool_seed"])
    res = {"sites": {}}
    for site in SITES:
        dicts = sub.candidates(site, train_idx, cfg["n_dirs"], cfg["rand_seed"])
        per = []
        for name, D in dicts.items():
            for i, u in enumerate(D):
                d = analyse_direction(sub, site, u, pools, cfg)
                d.update(dict_=name, i=i)
                for e in cfg["eps_alt"]:
                    de = analyse_direction(sub, site, u, pools, cfg, eps_rel=e)
                    d[f"A@{e}"] = de["A"]; d[f"n_types@{e}"] = de["n_types"]; d[f"C@{e}"] = de["C"]
                    d[f"passed@{e}"] = bool(passes(de, cfg))
                d["plateau"] = bool(plateau([d["n_types"]] + [d[f"n_types@{e}"] for e in cfg["eps_alt"]], cfg["plateau_tol"]))
                d["passed_A"] = bool(passes(d, cfg))
                d["passed"] = d["passed_A"] and d["plateau"]
                d["_u"] = u / np.linalg.norm(u)
                per.append(d)
        acc = [d for d in per if d["passed"] and d["dict_"] != "rand"]
        keep = dedup([d["_u"] for d in acc], [d["A"] for d in acc], cfg["dedup"]) if acc else []
        acc = [acc[i] for i in keep]
        site_res = {
            "n_candidates": len(per),
            "n_pass_raw": int(sum(d["passed"] for d in per if d["dict_"] != "rand")),
            "n_pass_rand": int(sum(d["passed"] for d in per if d["dict_"] == "rand")),
            "n_passA_raw": int(sum(d["passed_A"] for d in per if d["dict_"] != "rand")),
            "n_passA_rand": int(sum(d["passed_A"] for d in per if d["dict_"] == "rand")),
            "n_plateau_raw": int(sum(d["plateau"] for d in per if d["dict_"] != "rand")),
            "n_plateau_rand": int(sum(d["plateau"] for d in per if d["dict_"] == "rand")),
            "n_accepted": len(acc),
            "accepted": [{k: v for k, v in d.items() if not k.startswith("_")} for d in acc],
            "accepted_dirs": [d["_u"].tolist() for d in acc],
            "accepted_labels": [d["_labels"].tolist() for d in acc],
            "candidates": [{k: (float(v) if isinstance(v, (np.floating, float)) else v)
                            for k, v in d.items() if not k.startswith("_")} for d in per],
        }
        if acc:
            site_res.update(test_B(sub, site, orthonormal([d["_u"] for d in acc]), pools, cfg))
            # minimal sufficient prefix (accepted directions are already in decreasing A order)
            site_res["B_prefix"] = [test_B(sub, site, orthonormal([d["_u"] for d in acc[:k]]), pools, cfg)["B"]
                                    for k in range(1, len(acc) + 1)]
            site_res["B_min_dim"] = next((k + 1 for k, b in enumerate(site_res["B_prefix"]) if b >= cfg["B_min"]), None)
            site_res["C_mean"] = float(np.mean([d["C"] for d in acc]))
            site_res["D_mean"] = float(np.mean([d["outputs_moved"] for d in acc]))
            site_res["D_max"] = int(max(d["outputs_moved"] for d in acc))
            pairs = []
            top = sorted(range(len(acc)), key=lambda i: -acc[i]["A"])[:4]
            for i, j in combinations(top, 2):
                e = test_E(sub, site, acc[i]["_u"], acc[j]["_u"], acc[i], acc[j], pools, cfg)
                e.update(i=i, j=j)
                pairs.append(e)
            site_res["pairs"] = pairs
            site_res["E_best"] = max((p["E"] for p in pairs), default=float("nan"))
        res["sites"][site] = site_res
    res["pools"] = {k: v.tolist() for k, v in pools.items()}
    return res


def flags(site_res, cfg=CFG):
    """Pass flags of the set-level tests for one site at one checkpoint."""
    if site_res.get("n_accepted", 0) == 0:
        return dict(A=False, B=False, C=False, D=False, E=False)
    return dict(
        A=True,
        B=site_res["B"] >= cfg["B_min"] and site_res["B"] - site_res["B_base"] >= cfg["B_margin"],
        C=site_res["C_mean"] >= cfg["C_min"],
        D=site_res["D_max"] >= cfg["D_min"],
        E=bool(site_res.get("pairs")) and site_res["E_best"] >= cfg["E_min"],
    )


# --------------------------------------------------------------- novelty
def novelty(model, X, circuit, site_res, site, pools):
    """Post-hoc only: how far each accepted interface is readable off the
    generator (gates), the outputs, or the inputs.  Uses labels."""
    sub = Sub(model, X)
    G, Y = circuit.hidden(X), circuit.outputs(X)
    out = []
    d_ref = np.asarray(pools["d_ref"])
    for u, labels in zip(site_res["accepted_dirs"], site_res["accepted_labels"]):
        p = sub.Z[site] @ np.asarray(u)
        def maxcorr(B):
            cs = [abs(np.corrcoef(p, b)[0, 1]) if b.std() > 0 else 0.0 for b in B.T]
            return float(np.nanmax(cs)), int(np.nanargmax(cs))
        lab = np.asarray(labels)
        def maxari(B):
            a = [ari(lab, b) for b in B[d_ref].T]
            return float(max(a)), int(np.argmax(a))
        out.append(dict(
            corr_gate=maxcorr(G), corr_out=maxcorr(Y), corr_in=maxcorr(X),
            ari_gate=maxari(G), ari_out=maxari(Y), ari_in=maxari(X),
        ))
    return out
