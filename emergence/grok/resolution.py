"""The resolution-dependent behavioural quotient.

Behavioural pseudometric on states/histories:

    d_B(h, h') = E_c [ JSdist( B(h, c), B(h', c) ) ]

with JSdist = sqrt(JS divergence in bits), a metric on distributions, so
d_B is a pseudometric and ε-closeness obeys the triangle inequality
(ε-close to ε-close is at most 2ε-close; approximate equivalence is *not*
transitive, so no cluster is called a quotient here).

At resolution ε:
* covering number N_cov(ε): size of a greedy ε-net (every state within ε of
  a centre) -- an upper bound on the minimal cover;
* packing number N_pack(ε): size of a greedy maximal ε-separated set;
  N_pack(2ε) ≤ N_cov(ε) ≤ N_pack(ε);
* the extractor's own leader count (order-dependent convention), for
  comparison with earlier results;
* plateau: the widest range of ε (in decades) over which N_cov is constant;
* scaling: the log-log slope of N_cov(ε) over the descending part, with R².

Congruence and closure at resolution ε, on held-out contexts:
* congruence defect: among pairs with d_B < ε (measured on one context
  pool), the fraction whose images under the same action are > ε apart on a
  *fresh* context pool, versus the same fraction for arbitrary pairs;
* closure: fraction of (ε-centre, action) images within ε of some centre;
* stability: fraction of states whose noisy copy stays within ε of the
  state's centre.

Nothing here uses labels or the world.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .extract import _js
from .extract_seq import _all_prefixes, _behaviour


# ---------------------------------------------------------------------------
# pseudometric


def pseudometric(B: np.ndarray, block: int = 256) -> np.ndarray:
    """B (n, C, o) behaviour distributions -> (n, n) mean JS distance."""
    n = B.shape[0]
    D = np.zeros((n, n))
    for i in range(0, n, block):
        Bi = B[i : i + block]
        for j in range(0, n, block):
            Bj = B[j : j + block]
            # JS divergence per context for all pairs in the blocks: (bi, bj, C)
            M = 0.5 * (Bi[:, None] + Bj[None])
            def kl(A, Mm):
                return np.sum(np.where(A > 0, A * (np.log2(A + 1e-12) - np.log2(Mm + 1e-12)), 0.0), axis=-1)
            js = 0.5 * kl(Bi[:, None], M) + 0.5 * kl(Bj[None], M)
            D[i : i + block, j : j + block] = np.sqrt(np.clip(js, 0, None)).mean(axis=-1)
    return 0.5 * (D + D.T)


def greedy_cover(D: np.ndarray, eps: float, order=None) -> np.ndarray:
    """Greedy ε-net: centres such that every point is within ε of a centre."""
    n = D.shape[0]
    order = np.arange(n) if order is None else order
    covered = np.zeros(n, dtype=bool)
    centres = []
    for i in order:
        if not covered[i]:
            centres.append(i)
            covered |= D[i] <= eps
    return np.array(centres)


def greedy_packing(D: np.ndarray, eps: float, order=None) -> np.ndarray:
    """Greedy maximal ε-separated set (pairwise > ε)."""
    n = D.shape[0]
    order = np.arange(n) if order is None else order
    chosen = []
    for i in order:
        if all(D[i, j] > eps for j in chosen):
            chosen.append(i)
    return np.array(chosen)


def counts_over_eps(D: np.ndarray, eps_grid, rng) -> dict:
    order = rng.permutation(D.shape[0])
    N_cov, N_pack, N_lead = [], [], []
    for e in eps_grid:
        N_cov.append(len(greedy_cover(D, e, order)))
        N_pack.append(len(greedy_packing(D, e, order)))
        N_lead.append(len(greedy_cover(D, e)))  # in index order: the extractor's convention
    return {"eps": [float(e) for e in eps_grid], "N_cov": N_cov, "N_pack": N_pack, "N_leader": N_lead}


def plateau(eps_grid, N) -> dict:
    """Widest run of constant N (with N ≥ 2) in decades of ε."""
    eps_grid = np.asarray(eps_grid); N = np.asarray(N)
    best = {"N": None, "eps_lo": None, "eps_hi": None, "decades": 0.0}
    i = 0
    while i < len(N):
        j = i
        while j + 1 < len(N) and N[j + 1] == N[i]:
            j += 1
        dec = float(np.log10(eps_grid[j] / eps_grid[i])) if j > i else 0.0
        if N[i] >= 2 and dec > best["decades"]:
            best = {"N": int(N[i]), "eps_lo": float(eps_grid[i]), "eps_hi": float(eps_grid[j]), "decades": dec}
        i = j + 1
    return best


def scaling(eps_grid, N, n_states) -> dict:
    """Log-log slope of N(ε) where 2 ≤ N ≤ n/2 (the regime not limited by
    sample size or degeneracy); R² says whether a power law fits at all."""
    e = np.asarray(eps_grid, float); N = np.asarray(N, float)
    m = (N >= 2) & (N <= n_states / 2)
    if m.sum() < 3:
        return {"slope": None, "r2": None, "n_points": int(m.sum())}
    x, y = np.log(e[m]), np.log(N[m])
    A = np.vstack([x, np.ones_like(x)]).T
    coef, res, *_ = np.linalg.lstsq(A, y, rcond=None)
    yhat = A @ coef
    ss_res = float(((y - yhat) ** 2).sum()); ss_tot = float(((y - y.mean()) ** 2).sum())
    return {"slope": float(-coef[0]), "r2": (1 - ss_res / ss_tot) if ss_tot > 0 else None, "n_points": int(m.sum())}


# ---------------------------------------------------------------------------
# congruence, closure, stability for sequence substrates


def seq_behaviour(sub, H, suffixes):
    return _behaviour(sub, H, suffixes)


def congruence_defect(sub, H, D, eps, n_actions, fresh_suffixes, rng, n_pairs=300) -> dict:
    """Pairs ε-close on the defining contexts: after the same random action,
    are they still ε-close on fresh contexts?  Compared with arbitrary pairs."""
    n = D.shape[0]
    iu = np.triu_indices(n, 1)
    close = np.flatnonzero(D[iu] < eps)
    if len(close) == 0:
        return {"n_close_pairs": 0}
    pick = rng.choice(close, size=min(n_pairs, len(close)), replace=False)
    ia, ib = iu[0][pick], iu[1][pick]
    acts = torch.tensor(rng.integers(0, n_actions, (len(pick), 1)), dtype=torch.long)
    _, Ha = sub.continue_(sub.select(H, torch.tensor(ia)), acts)
    _, Hb = sub.continue_(sub.select(H, torch.tensor(ib)), acts)
    Ba = seq_behaviour(sub, Ha, fresh_suffixes); Bb = seq_behaviour(sub, Hb, fresh_suffixes)
    d_img = np.sqrt(np.clip(_js(Ba, Bb), 0, None)).mean(axis=1)
    # baseline: arbitrary pairs
    ra, rb = rng.integers(0, n, len(pick)), rng.integers(0, n, len(pick))
    _, Hra = sub.continue_(sub.select(H, torch.tensor(ra)), acts)
    _, Hrb = sub.continue_(sub.select(H, torch.tensor(rb)), acts)
    d_base = np.sqrt(np.clip(_js(seq_behaviour(sub, Hra, fresh_suffixes), seq_behaviour(sub, Hrb, fresh_suffixes)), 0, None)).mean(axis=1)
    # and the pairs' own distance on the fresh contexts before the action (context transfer)
    Bfa = seq_behaviour(sub, sub.select(H, torch.tensor(ia)), fresh_suffixes)
    Bfb = seq_behaviour(sub, sub.select(H, torch.tensor(ib)), fresh_suffixes)
    d_fresh = np.sqrt(np.clip(_js(Bfa, Bfb), 0, None)).mean(axis=1)
    return {"n_close_pairs": int(len(close)), "defect": float(np.mean(d_img > eps)), "mean_image_dist": float(d_img.mean()),
            "baseline_frac_far": float(np.mean(d_base > eps)), "context_transfer_defect": float(np.mean(d_fresh > eps))}


def closure_at(sub, H, D, eps, n_actions, suffixes, rng) -> dict:
    centres = greedy_cover(D, eps, rng.permutation(D.shape[0]))
    Hc = sub.select(H, torch.tensor(centres))
    Bc = seq_behaviour(sub, Hc, suffixes)
    hits = []
    for a in range(n_actions):
        _, Hi = sub.continue_(Hc, torch.full((len(centres), 1), a, dtype=torch.long))
        Bi = seq_behaviour(sub, Hi, suffixes)
        for i in range(len(centres)):
            d = np.sqrt(np.clip(_js(Bc, Bi[i][None]), 0, None)).mean(axis=1)
            hits.append(d.min() <= eps)
    return {"n_centres": int(len(centres)), "closure": float(np.mean(hits))}


def stability_at(sub, H, B, D, eps, sigma, suffixes, rng) -> dict:
    centres = greedy_cover(D, eps, rng.permutation(D.shape[0]))
    # centre of each state: nearest centre
    cidx = centres[np.argmin(D[:, centres], axis=1)]
    Hn = sub.noise(H, sigma)
    Bn = seq_behaviour(sub, Hn, suffixes)
    d = np.sqrt(np.clip(_js(Bn, B[cidx]), 0, None)).mean(axis=1)
    return {"retention": float(np.mean(d <= eps))}


# ---------------------------------------------------------------------------
# systems


def load_world(run_dir: Path):
    from .seqsub import build_seq
    from .world import make_world
    d = torch.load(run_dir / "checkpoints.pt", weights_only=False)
    world = make_world(d["config"]["world"])
    model, sub = build_seq(d["config"].get("arch", "gru"), world.n_actions, world.n_obs)
    return d, model, sub, world.n_actions


def load_discover(run_dir: Path):
    from .rnn import GRUWorldModel
    from .seqsub import GRUSubstrate
    d = torch.load(run_dir / "checkpoints.pt", weights_only=False)
    V = d["config"]["V"]
    model = GRUWorldModel(V, V, d=d["config"]["d_student"])
    return d, model, GRUSubstrate(model), V


def analyse_sequence(run_dir: Path, kind: str, eps_grid, n_ckpt, out: Path, rng, n_prefix=500, congruence_eps=None):
    if kind == "world":
        d, model, sub, k = load_world(run_dir)
        prefixes = _all_prefixes(k, 4)
        prefixes += [tuple(rng.integers(0, k, 10)) for _ in range(max(0, n_prefix - len(prefixes)))]
        suffixes = torch.tensor(rng.integers(0, k, (24, 5)), dtype=torch.long)
        fresh = torch.tensor(rng.integers(0, k, (24, 5)), dtype=torch.long)
    else:
        d, model, sub, k = load_discover(run_dir)
        from .discover import Teacher, make_pools
        cfg = d["config"]
        T = Teacher(V=cfg["V"], d=cfg["d_teacher"], gain=cfg["gain"], seed=cfg["teacher_seed"])
        held = T.sample(2048, cfg["seq_len"], np.random.default_rng(1000 + cfg["teacher_seed"] + 77))
        prefixes, suffixes = make_pools(held, n_prefix, 24, 6, np.random.default_rng(7))
        _, fresh = make_pools(held, 2, 24, 6, np.random.default_rng(8))
    steps = sorted(d["checkpoints"])
    pick = sorted(set([steps[0], steps[-1]] + [steps[i] for i in np.unique(np.round(np.geomspace(1, len(steps) - 1, n_ckpt)).astype(int))]))
    tl = {x["step"]: x for x in d["log"]}
    rows = []
    congruence_eps = congruence_eps or [eps_grid[i] for i in (3, 6, 9)]
    for step in pick:
        model.load_state_dict(d["checkpoints"][step]); model.eval()
        H = sub.run_prefixes(prefixes)
        B = seq_behaviour(sub, H, suffixes)
        D = pseudometric(B)
        cnt = counts_over_eps(D, eps_grid, rng)
        row = {"step": step, "test": tl[step].get("test_acc", tl[step].get("test_loss")), "test_loss": tl[step].get("test_loss"),
               **cnt, "plateau": plateau(eps_grid, cnt["N_cov"]), "scaling": scaling(eps_grid, cnt["N_cov"], D.shape[0]),
               "congruence": {}, "closure": {}, "stability": {}}
        for e in congruence_eps:
            row["congruence"][str(round(e, 4))] = congruence_defect(sub, H, D, e, k, fresh, rng)
            row["closure"][str(round(e, 4))] = closure_at(sub, H, D, e, k, suffixes, rng)
            row["stability"][str(round(e, 4))] = stability_at(sub, H, B, D, e, 0.5, suffixes, rng)
        rows.append(row)
        print(f"[{run_dir.name}] step {step:6d} test {row['test']}: N_cov {cnt['N_cov']}  plateau {row['plateau']}  scaling {row['scaling']}", flush=True)
        for e in congruence_eps:
            c, cl, st = row["congruence"][str(round(e, 4))], row["closure"][str(round(e, 4))], row["stability"][str(round(e, 4))]
            print(f"      eps {e:.3f}: congruence defect {c.get('defect')} (baseline far {c.get('baseline_frac_far')}, context-transfer defect {c.get('context_transfer_defect')}, {c.get('n_close_pairs')} close pairs); closure {cl['closure']:.2f} over {cl['n_centres']} centres; stability {st['retention']:.2f}", flush=True)
    return rows


def analyse_grok(run_dir: Path, eps_grid, n_ckpt, rng, n_states=1200):
    """Feedforward substrate: states are hidden-site configurations of inputs;
    behaviour is the (context-free) output distribution.  Closure via the
    closed-loop map (output token re-fed); no context-dependent interaction, so
    no congruence measurement."""
    from .models import build
    from .task import make_task
    d = torch.load(run_dir / "checkpoints.pt", weights_only=False)
    cfg = d["config"]; task = make_task(cfg.get("task", "zmod:97"), cfg["seed"])
    model = build(cfg["arch"], cfg["p"])
    X = task.all_pairs()
    idx = rng.choice(len(X), n_states, replace=False)
    steps = sorted(d["checkpoints"])
    grok = next((r["step"] for r in d["log"] if r["test_acc"] >= 0.9), None)
    pick = sorted(set([steps[0], steps[-1]] + [steps[i] for i in np.unique(np.round(np.geomspace(1, len(steps) - 1, n_ckpt)).astype(int))]
                      + ([s for s in steps if grok and abs(s - grok) <= 1000] if grok else [])))
    tl = {x["step"]: x for x in d["log"]}
    rows = []
    for step in pick:
        model.load_state_dict(d["checkpoints"][step]); model.eval()
        with torch.no_grad():
            logits, _ = model(X)
        P = F.softmax(logits, -1).numpy()
        B = P[idx][:, None, :]
        D = pseudometric(B)
        cnt = counts_over_eps(D, eps_grid, rng)
        # closure: image of state (a,b) under "action c" = state of (argmax, c); within eps of a centre?
        pred = logits.argmax(-1).numpy(); p = cfg["p"]
        clos = {}
        for e in [eps_grid[i] for i in (3, 6, 9)]:
            centres = greedy_cover(D, e, rng.permutation(D.shape[0]))
            Pc = P[idx][centres]
            hits = []
            for c in rng.integers(0, p, 8):
                img = pred[idx[centres]] * p + c
                Pi = P[img]
                for i in range(len(centres)):
                    dd = np.sqrt(np.clip(_js(Pc, Pi[i][None]), 0, None))
                    hits.append(dd.min() <= e)
            clos[str(round(e, 4))] = {"n_centres": int(len(centres)), "closure": float(np.mean(hits))}
        row = {"step": step, "test": tl[step]["test_acc"], "train": tl[step]["train_acc"], **cnt,
               "plateau": plateau(eps_grid, cnt["N_cov"]), "scaling": scaling(eps_grid, cnt["N_cov"], D.shape[0]), "closure": clos}
        rows.append(row)
        print(f"[{run_dir.name}] step {step:6d} test {row['test']:.2f}: N_cov {cnt['N_cov']} plateau {row['plateau']} scaling {row['scaling']} closure {clos}", flush=True)
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", required=True, help="world | discover | grok")
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", default="results/resolution")
    ap.add_argument("--n_ckpt", type=int, default=6)
    ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args(argv)
    torch.set_num_threads(a.threads)
    rng = np.random.default_rng(0)
    eps_grid = np.geomspace(0.01, 1.0, 13)
    run = Path(a.run)
    if a.system == "grok":
        rows = analyse_grok(run, eps_grid, a.n_ckpt, rng)
    else:
        rows = analyse_sequence(run, a.system, eps_grid, a.n_ckpt, Path(a.out), rng)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    name = run.name if a.system != "discover" else f"{run.parent.name}_{run.name}"
    json.dump({"run": str(run), "system": a.system, "eps": [float(e) for e in eps_grid], "rows": rows},
              open(out / f"{name}.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
