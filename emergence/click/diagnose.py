"""Post-hoc diagnostics for the click experiment (final checkpoints only).

These do NOT change the frozen acceptance criterion; they interpret its
outcome.

1. `N(ε)` per direction: the type count of the leading dictionary directions
   and of random directions across a decade of resolutions, to show whether
   *any* plateau exists (the frozen criterion looked only at 0.2/0.3/0.4).

2. LABELLED upper bound: logistic-probe directions for the six generating
   gates (and the six outputs) at each site, pushed through the same
   per-direction tests.  If those pass where the label-free dictionaries
   fail, the negative result is an extraction failure (wrong dictionary);
   if they fail too, no resolution-stable one-dimensional interface for the
   gates exists in the substrate at all.

    python -m emergence.click.diagnose
"""

from __future__ import annotations

import json
import os

import numpy as np
import torch

from emergence.click import run as R
from emergence.click import interfaces as I
from emergence.click.model import MLP, SITES

EPS = [0.05, 0.07, 0.1, 0.14, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0]


def probe_direction(Z, y, tr, te, steps=500):
    """Logistic probe z ↦ σ(wᵀz + b); returns unit w and held-out accuracy."""
    torch.manual_seed(0)
    Zt = torch.as_tensor(Z, dtype=torch.float32)
    yt = torch.as_tensor(y, dtype=torch.float32)
    lin = torch.nn.Linear(Z.shape[1], 1)
    opt = torch.optim.Adam(lin.parameters(), lr=1e-2, weight_decay=1e-3)
    for _ in range(steps):
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(lin(Zt[tr]).squeeze(1), yt[tr])
        loss.backward()
        opt.step()
    with torch.no_grad():
        acc = float(((lin(Zt[te]).squeeze(1) > 0).float() == yt[te]).float().mean())
        w = lin.weight.detach().numpy()[0].astype(np.float64)
    return w / np.linalg.norm(w), acc


def n_eps_curve(sub, site, u, pools):
    return [I.analyse_direction(sub, site, u, pools, eps_rel=e)["n_types"] for e in EPS]


def longest_plateau(counts):
    """Widest run of consecutive equal counts, in decades of ε."""
    best = 0.0
    i = 0
    while i < len(counts):
        j = i
        while j + 1 < len(counts) and counts[j + 1] == counts[i]:
            j += 1
        best = max(best, np.log10(EPS[j] / EPS[i]))
        i = j + 1
    return float(best)


def diagnose(wds=R.WDS + (1.0,), seeds=R.SEEDS):
    c, X, Y, tr, te = R.data()
    G = c.hidden(X)
    lines = ["# Click experiment: post-hoc diagnostics (final checkpoints)", "",
             f"Resolutions ε (relative): {EPS}", ""]
    out = {}
    for wd in wds:
        for seed in seeds:
            d = R.run_dir(wd, seed)
            p = os.path.join(d, "checkpoints.pt")
            if not os.path.exists(p):
                continue
            ck = torch.load(p)[-1]
            m = MLP()
            m.load_state_dict(ck["state"])
            m.eval()
            sub = I.Sub(m, X)
            pools = I.make_pools(tr, te, I.CFG["pool"], I.CFG["pool_seed"])
            rec = {}
            lines += [f"## wd={wd} seed={seed}", ""]
            for site in SITES:
                dicts = sub.candidates(site, tr, I.CFG["n_dirs"], I.CFG["rand_seed"])
                lines += [f"### {site}: N(ε) of leading directions (label-free)", "",
                          "| direction | " + " | ".join(f"{e}" for e in EPS) + " | widest plateau (decades) |",
                          "|" + "---|" * (len(EPS) + 2)]
                curves = {}
                for name in ("pca", "sens", "rand"):
                    for i in range(4):
                        u = dicts[name][i]
                        cu = n_eps_curve(sub, site, u, pools)
                        curves[f"{name}{i}"] = cu
                        lines.append(f"| {name}{i} | " + " | ".join(str(v) for v in cu) + f" | {longest_plateau(cu):.2f} |")
                # labelled probes
                Z = sub.Z[site]
                lines += ["", f"### {site}: LABELLED probe directions through the frozen per-direction tests", "",
                          "| probe | held-out acc | mag | types@0.2/0.3/0.4 | A | A_base | C | outputs | passes A-tests | resolution-stable | N(ε) |",
                          "|---|---|---|---|---|---|---|---|---|---|---|"]
                probes = {}
                for kind, B in (("gate", G), ("out", Y)):
                    for j in range(B.shape[1]):
                        u, acc = probe_direction(Z, B[:, j], tr, te)
                        dd = I.analyse_direction(sub, site, u, pools)
                        nts = [dd["n_types"]] + [I.analyse_direction(sub, site, u, pools, eps_rel=e)["n_types"] for e in I.CFG["eps_alt"]]
                        cu = n_eps_curve(sub, site, u, pools)
                        pa = I.passes(dd)
                        st = I.plateau(nts, I.CFG["plateau_tol"])
                        probes[f"{kind}{j}"] = dict(acc=acc, mag=dd["mag"], n_types=nts, A=dd["A"], A_base=dd["A_base"],
                                                    C=dd["C"], outputs=dd["outputs_moved"], passes_A=bool(pa), plateau=bool(st), curve=cu)
                        lines.append(f"| {kind}{j} | {acc:.3f} | {dd['mag']:.2f} | {nts[1]}/{nts[0]}/{nts[2]} | {dd['A']:.2f} | {dd['A_base']:.2f} | "
                                     f"{dd['C']:.2f} | {dd['outputs_moved']} | {pa} | {st} | {cu} |")
                lines.append("")
                rec[site] = dict(curves=curves, probes=probes)
            out[f"wd{wd}_s{seed}"] = rec
    os.makedirs(R.ROOT, exist_ok=True)
    open(os.path.join(R.ROOT, "DIAGNOSTICS.md"), "w").write("\n".join(lines))
    json.dump(out, open(os.path.join(R.ROOT, "diagnostics.json"), "w"))
    print("\n".join(lines))


if __name__ == "__main__":
    diagnose()
