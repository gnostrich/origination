"""Sealed novelty diagnostic for behavioural interface discovery.

Run ONLY after `run_bid search` has finished for every run and the A–E /
composition results are frozen.  This is the first and only place the
generating circuit is read in the follow-up.

For every accepted interface (final checkpoint unless `--all`):
* overlap with gate-decodable directions: logistic-probe directions for the
  six gates at the site; `||U w_g||²` per gate (fraction of the probe
  direction inside U) and the fraction of U inside the span of the six;
* whether the discovered behavioural types predict gate values: ARI and
  purity of each gate given the type; best single gate and best pair of
  gates (a type partition that matches a *pair* better than any single gate
  combines gates; a partition finer than the gate it matches splits it);
* whether different seeds find different decompositions: mean canonical
  correlation between the projections `P_U z(x)` of accepted subspaces of
  two seeds over all inputs;
* "valid but non-isomorphic": an interface that passed A on the final pool
  whose types are not a function of any single gate or pair of gates
  (best purity < 0.9).

    python -m emergence.click.novelty
"""

from __future__ import annotations

import argparse
import json
import os
from itertools import combinations

import numpy as np
import torch

from emergence.click import bid as B
from emergence.click import run as R
from emergence.click import run_bid as RB
from emergence.click import interfaces as I
from emergence.click.diagnose import probe_direction
from emergence.click.model import MLP, SITES
from emergence.grok.compare import adjusted_rand_index as ari


def purity(labels, g):
    """Accuracy of predicting g (0/1 or small int) from the type label by majority."""
    acc = 0
    for t in np.unique(labels):
        vals = g[labels == t]
        acc += np.bincount(vals).max()
    return acc / len(labels)


def canon_corr(Fa, Fb):
    """Mean canonical correlation between two feature sets over the same inputs."""
    Fa = Fa - Fa.mean(0)
    Fb = Fb - Fb.mean(0)
    Qa, _ = np.linalg.qr(Fa)
    Qb, _ = np.linalg.qr(Fb)
    s = np.linalg.svd(Qa.T @ Qb, compute_uv=False)
    return float(np.mean(np.clip(s, 0, 1)))


def analyse(all_ckpts=False):
    c, X, Y, tr, te = R.data()
    G = c.hidden(X)
    disc, val, fin = B.make_pools(tr, te)
    d_ref = fin["d_ref"]
    lines = ["# Sealed novelty diagnostic (behavioural interface discovery)", "",
             f"Generating circuit: gate inputs {[list(map(int, s)) for s in c.S]}, output gate sets {[list(map(int, r)) for r in c.R]}, reuse {c.reuse()}.", ""]
    feats = {}   # (wd, seed, site) -> list of projection feature matrices at the final checkpoint
    rows = []
    for wd in RB.WDS:
        for seed in R.SEEDS:
            an = RB.load(wd, seed)
            if an is None:
                continue
            ckpts = torch.load(os.path.join(R.run_dir(wd, seed), "checkpoints.pt"))
            by_step = {ck["step"]: ck for ck in ckpts}
            recs = an if all_ckpts else an[-1:]
            for a in recs:
                m = MLP()
                m.load_state_dict(by_step[a["step"]]["state"])
                m.eval()
                sub = I.Sub(m, X)
                for site in SITES:
                    acc = a["sites"][site]["accepted"]
                    if not acc:
                        continue
                    Z = sub.Z[site]
                    W = np.stack([probe_direction(Z, G[:, j], tr, te)[0] for j in range(6)])
                    Wq = I.orthonormal(W)
                    for i, cnd in enumerate(acc):
                        U = np.asarray(cnd["U"])
                        d = B.analyse_subspace(sub, site, U, fin)
                        lab = d["_labels"]
                        ov_gate = [float(np.linalg.norm(U @ W[j]) ** 2) for j in range(6)]
                        in_span = float(np.linalg.norm(U @ Wq.T) ** 2 / len(U))
                        ar = [float(ari(lab, G[d_ref, j])) for j in range(6)]
                        pu = [float(purity(lab, G[d_ref, j])) for j in range(6)]
                        pairs = {}
                        for j, k in combinations(range(6), 2):
                            gj = G[d_ref, j] * 2 + G[d_ref, k]
                            pairs[(j, k)] = (float(ari(lab, gj)), float(purity(lab, gj)))
                        bp = max(pairs, key=lambda p: pairs[p][0])
                        ar_out = [float(ari(lab, Y[d_ref, j])) for j in range(6)]
                        rec = dict(wd=wd, seed=seed, step=a["step"], site=site, idx=i, rank=len(U), n_types=d["n_types"],
                                   A_final=float(d["A"]), passed_A=bool(I.passes(d)),
                                   in_gate_span=in_span, overlap_gate=ov_gate, best_gate=int(np.argmax(ar)),
                                   ari_gate=max(ar), purity_gate=max(pu), best_pair=list(bp), ari_pair=pairs[bp][0], purity_pair=pairs[bp][1],
                                   ari_out=max(ar_out),
                                   combines=bool(pairs[bp][0] > max(ar) + 0.1), splits=bool(max(pu) >= 0.95 and d["n_types"] > 2),
                                   nonisomorphic_valid=bool(I.passes(d) and max(max(pu), pairs[bp][1]) < 0.9))
                        rows.append(rec)
                        if a["step"] == an[-1]["step"]:
                            feats.setdefault((wd, seed, site), []).append(Z @ U.T)
    lines += ["## Per accepted interface (final checkpoints" + (" and all analysed checkpoints" if all_ckpts else "") + ")", "",
              "| run | step | site | idx | rank | types | A(final) | frac U in gate span | max ‖U w_g‖² (gate) | ARI best gate | purity best gate | best pair ARI / purity | ARI best output | combines | splits | non-isomorphic & valid |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| wd={r['wd']} s={r['seed']} | {r['step']} | {r['site']} | {r['idx']} | {r['rank']} | {r['n_types']} | {r['A_final']:.2f} | {r['in_gate_span']:.2f} | "
                     f"{max(r['overlap_gate']):.2f} (g{int(np.argmax(r['overlap_gate']))}) | {r['ari_gate']:.2f} (g{r['best_gate']}) | {r['purity_gate']:.2f} | "
                     f"{r['ari_pair']:.2f} / {r['purity_pair']:.2f} (g{r['best_pair'][0]},g{r['best_pair'][1]}) | {r['ari_out']:.2f} | {r['combines']} | {r['splits']} | {r['nonisomorphic_valid']} |")
    lines += ["", "## Cross-seed correspondence (final checkpoints)", "",
              "Mean canonical correlation over all 4096 inputs between the projections of accepted subspaces of two seeds (same wd, same site).", ""]
    for wd in RB.WDS:
        for site in SITES:
            vals = []
            for a_, b_ in combinations(R.SEEDS, 2):
                if (wd, a_, site) in feats and (wd, b_, site) in feats:
                    Fa = np.hstack(feats[(wd, a_, site)])
                    Fb = np.hstack(feats[(wd, b_, site)])
                    vals.append(f"s{a_}–s{b_}: {canon_corr(Fa, Fb):.2f} (dims {Fa.shape[1]}, {Fb.shape[1]})")
            lines.append(f"- wd={wd} {site}: " + ("; ".join(vals) if vals else "fewer than two seeds with accepted interfaces"))
    lines.append("")
    n = len(rows)
    if n:
        lines += ["## Summary", "",
                  f"- accepted interfaces examined: {n}; passing A on the final pool: {sum(r['passed_A'] for r in rows)}",
                  f"- mean fraction of U inside the gate-probe span: {np.mean([r['in_gate_span'] for r in rows]):.2f} (a random rank-k subspace of R^64 has ≈ 6/64 = 0.09)",
                  f"- types matching a single gate (purity ≥ 0.9): {sum(r['purity_gate'] >= 0.9 for r in rows)}; matching a pair of gates better than any single gate: {sum(r['combines'] for r in rows)}",
                  f"- splitting a gate (purity ≥ 0.95 with > 2 types): {sum(r['splits'] for r in rows)}",
                  f"- valid (A passed) but not a function of any gate or gate pair: {sum(r['nonisomorphic_valid'] for r in rows)}", ""]
    else:
        lines += ["## Summary", "", "No accepted interfaces to examine.", ""]
    os.makedirs(R.ROOT, exist_ok=True)
    open(os.path.join(R.ROOT, "NOVELTY.md"), "w").write("\n".join(lines))
    json.dump(rows, open(os.path.join(R.ROOT, "novelty.json"), "w"))
    print("\n".join(lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    a = ap.parse_args()
    analyse(a.all)
