"""Driver for behavioural interface discovery over the existing checkpoints.

    python -m emergence.click.run_bid search --wd 0.1 --seed 0          # 16 checkpoints, both sites, ranks 1/2/4/8
    python -m emergence.click.run_bid search --wd 0.1 --seed 0 --null   # shuffled-weight model, final checkpoint
    python -m emergence.click.run_bid report

Settings frozen after a smoke test that only checked execution (`--smoke`).
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import time

import numpy as np
import torch

from emergence.click import bid as B
from emergence.click import run as R
from emergence.click import interfaces as I
from emergence.click.model import MLP, SITES


def out_path(wd, seed, null=False, smoke=False):
    tag = "bid_null" if null else "bid"
    if smoke:
        tag += "_smoke"
    return os.path.join(R.run_dir(wd, seed), f"{tag}.json.gz")


def search(wd, seed, null=False, smoke=False):
    c, X, Y, tr, te = R.data()
    disc, val, fin = B.make_pools(tr, te)
    ckpts = torch.load(os.path.join(R.run_dir(wd, seed), "checkpoints.pt"))
    idx = [len(ckpts) - 1] if null else list(B.CKPT_SUBSET)
    ranks, scfg = B.RANKS, B.SEARCH
    if smoke:
        idx, ranks, scfg = [len(ckpts) - 1], (1, 2), dict(B.SEARCH, restarts=2, iters=3)
    out = []
    t0 = time.time()
    for i in idx:
        m = MLP()
        m.load_state_dict(ckpts[i]["state"])
        if null:
            B.shuffle_weights(m, seed=1000 + seed)
        m.eval()
        sub = I.Sub(m, X)
        rec = {"step": ckpts[i]["step"], "sites": {}}
        for site in SITES:
            rng = np.random.default_rng([wd == 1.0, int(wd * 100), seed, i, site == "h2", null])
            rec["sites"][site] = B.run_site(sub, site, disc, val, fin, rng, ranks, scfg)
        out.append(rec)
        msg = "  ".join(f"{s}: acc={rec['sites'][s]['final']['n_accepted']} valpass={rec['sites'][s]['n_val_passed']} "
                        f"bestJ={rec['sites'][s]['best_J_val']:.2f} {''.join(k for k, v in rec['sites'][s]['final']['flags'].items() if v) or '-'} "
                        f"pairs*={rec['sites'][s]['stage2']['n_stable_pairs']}" for s in SITES)
        print(f"wd={wd} s={seed}{' null' if null else ''} step {rec['step']:6d}  {msg}  ({time.time()-t0:.0f}s)", flush=True)
    with gzip.open(out_path(wd, seed, null, smoke), "wt") as f:
        json.dump(out, f)


def load(wd, seed, null=False):
    p = out_path(wd, seed, null)
    if not os.path.exists(p):
        return None
    with gzip.open(p, "rt") as f:
        return json.load(f)


WDS = R.WDS + (1.0,)


def report():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lines = ["# Behavioural interface discovery: results", "",
             "Per run and checkpoint: number of accepted interfaces at each site (validated on the validation pool, "
             "deduplicated), the final-pool flags A–E, the number of stable composites (I,J) ⇝ B and (B,K) ⇝ C, "
             "and the validation pass rate of matched random subspaces.", ""]
    summary = {}
    fig, axes = plt.subplots(len(WDS), 2, figsize=(12, 3.2 * len(WDS)), squeeze=False)
    for wi, wd in enumerate(WDS):
        for seed in R.SEEDS:
            an = load(wd, seed)
            if an is None:
                continue
            curve = json.load(open(os.path.join(R.run_dir(wd, seed), "curve.json")))
            cs = {e["step"]: e for e in curve}
            lines += [f"## wd={wd} seed={seed}", "",
                      "| step | test acc | " + " | ".join(f"{s}: acc (ranks) valpass rand% flags pairs* triples*" for s in SITES) + " |",
                      "|" + "---|" * (2 + len(SITES))]
            for a in an:
                row = f"| {a['step']} | {cs.get(a['step'], {}).get('test_acc', float('nan')):.3f} |"
                for s in SITES:
                    sr = a["sites"][s]
                    ranks = ",".join(str(c["rank"]) for c in sr["accepted"])
                    fl = "".join(k for k, v in sr["final"]["flags"].items() if v) or "-"
                    rr = np.mean(list(sr["rand_val_rate"].values())) * 100
                    row += (f" {sr['final']['n_accepted']} ({ranks}) {sr['n_val_passed']}/{len(sr['candidates'])} "
                            f"{rr:.0f}% {fl} {sr['stage2']['n_stable_pairs']} {sr['stage2']['n_stable_triples']} |")
                lines.append(row)
            lines.append("")
            fin = an[-1]
            for s in SITES:
                sr = fin["sites"][s]
                if sr["accepted"]:
                    lines.append(f"- final {s}: " + "; ".join(
                        f"[{i}] rank {c['rank']} J_val={c['J_val']:.2f} types={c['val']['n_types']} A={c['val']['A']:.2f} C={c['val']['C']:.2f} moves {c['val']['outputs_moved']}"
                        for i, c in enumerate(sr["accepted"])))
                    f_ = sr["final"]
                    lines.append(f"  final pool: A per interface {[round(d['A'], 2) for d in f_['per']]}, B={f_['B']:.2f} (rand {f_['B_base']:.2f}, none {f_['B_none']:.2f}, dim {f_['dim']}), "
                                 f"C_mean={f_['C_mean']:.2f}, D_max={f_['D_max']}, E_best={f_['E_best']:.2f}; controls (rank, val pass, final A pass) "
                                 f"{[(c['rank'], c['val_pass'], c['final_A_pass']) for c in sr['controls']]}")
                    for p in sr["stage2"]["pairs"]:
                        lines.append(f"  pair ({p['i']},{p['j']}): pred={p['pred']:.2f} nonadd={p['nonadd']:.2f} pred_int={p['pred_int']:.2f} joint types {p['n_joint']} ARI(parts)={p['ari_with_parts']:.2f} stable={p['stable']}")
                    for t in sr["stage2"]["triples"]:
                        lines.append(f"  triple (({t['i']},{t['j']}),{t['k']}): pred={t['pred']:.2f} nonadd_unit={t['nonadd_unit']:.2f} pred_int={t['pred_int']:.2f} joint types {t['n_joint']} stable={t['stable_unit']}")
            lines.append("")
            # temporal
            steps = [a["step"] for a in an]
            for s in SITES:
                n_acc = [a["sites"][s]["final"]["n_accepted"] for a in an]
                ok = [all(a["sites"][s]["final"]["flags"][k] for k in "ABCD") for a in an]
                first = R.first_true(steps, [n > 0 for n in n_acc])
                pers = R.first_persistent(steps, [n > 0 for n in n_acc])
                click = R.first_persistent(steps, ok)
                t_test = R.first_true([e["step"] for e in curve], [e["test_acc"] >= 0.99 for e in curve])
                lines.append(f"- {s}: interfaces first at {first}, persistent from {pers}; A–D persisting from {click}; test acc ≥ 0.99 at {t_test}; "
                             f"n_accepted over time {n_acc}; best J_val over time {[round(a['sites'][s]['best_J_val'], 2) for a in an]}")
                summary[(wd, seed, s)] = dict(first=first, pers=pers, click=click, t_test=t_test, n_acc=n_acc)
                ax = axes[wi, SITES.index(s)]
                ax.plot(steps, [a["sites"][s]["best_J_val"] for a in an], color=f"C{seed}", marker="o", ms=3, label=f"best J_val s{seed}")
                ax.plot(steps, np.array(n_acc) / 8, color=f"C{seed}", ls="--", lw=1, label=f"#accepted/8 s{seed}")
                ax.plot([e["step"] for e in curve], [e["test_acc"] for e in curve], color=f"C{seed}", lw=0.7, alpha=0.5)
                ax.set_xscale("symlog", linthresh=10)
                ax.set_ylim(-0.05, 1.05)
                ax.set_title(f"wd={wd} {s}: best validation J (solid), #accepted/8 (dashed), test acc (thin)", fontsize=8)
                if wi == 0 and seed == 0:
                    ax.legend(fontsize=6)
            lines.append("")
    # null models
    lines += ["## Shuffled-weight null models (final checkpoints, same search)", "",
              "| run | " + " | ".join(f"{s}: accepted, valpass/cands, best J_val, flags, pairs*" for s in SITES) + " |", "|---|---|---|"]
    for wd in WDS:
        for seed in R.SEEDS:
            an = load(wd, seed, null=True)
            if an is None:
                continue
            a = an[-1]
            row = f"| wd={wd} s={seed} |"
            for s in SITES:
                sr = a["sites"][s]
                fl = "".join(k for k, v in sr["final"]["flags"].items() if v) or "-"
                row += f" {sr['final']['n_accepted']}, {sr['n_val_passed']}/{len(sr['candidates'])}, {sr['best_J_val']:.2f}, {fl}, {sr['stage2']['n_stable_pairs']} |"
            lines.append(row)
    lines.append("")
    fig.tight_layout()
    fig.savefig(os.path.join(R.ROOT, "bid.png"), dpi=110)
    open(os.path.join(R.ROOT, "BID_REPORT.md"), "w").write("\n".join(lines))
    print("\n".join(lines))
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["search", "report"])
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--null", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.cmd == "search":
        search(a.wd, a.seed, a.null, a.smoke)
    else:
        report()
