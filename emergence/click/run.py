"""The click experiment: train / analyse / report.

    python -m emergence.click.run train   --wd 0.01 --seed 0
    python -m emergence.click.run analyse --wd 0.01 --seed 0
    python -m emergence.click.run report

Design frozen before the first run; see README "The click experiment".
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from emergence.click.task import Circuit, split
from emergence.click.model import MLP, SITES
from emergence.click import interfaces as I

ROOT = "results/click"
WDS = (0.0, 0.01, 0.1)
SEEDS = (0, 1, 2)
STEPS = 20000
LR = 1e-3
TRAIN_FRAC = 0.5
CIRCUIT_SEED = 0
SPLIT_SEED = 0


def ckpt_steps(steps=STEPS, n=30):
    s = np.unique(np.round(np.logspace(0, np.log10(steps), n)).astype(int))
    return [0] + [int(v) for v in s]


def run_dir(wd, seed):
    return os.path.join(ROOT, f"wd{wd}_s{seed}")


def data():
    c = Circuit(seed=CIRCUIT_SEED)
    X = c.all_inputs()
    Y = c.outputs(X)
    tr, te = split(len(X), TRAIN_FRAC, SPLIT_SEED)
    return c, X, Y, tr, te


def train(wd, seed):
    c, X, Y, tr, te = data()
    torch.manual_seed(seed)
    m = MLP(seed=seed)
    x, y = MLP.encode(X), torch.as_tensor(Y, dtype=torch.float32)
    opt = torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=wd)
    cs = set(ckpt_steps())
    ckpts, curve = [], []
    d = run_dir(wd, seed)
    os.makedirs(d, exist_ok=True)

    def evaluate(step):
        m.eval()
        with torch.no_grad():
            lg = m(x)
            loss = F.binary_cross_entropy_with_logits(lg, y, reduction="none").mean(1)
            bits = (lg > 0).float() == y
            rec = dict(step=step,
                       train_loss=float(loss[tr].mean()), test_loss=float(loss[te].mean()),
                       train_acc=float(bits[tr].float().mean()), test_acc=float(bits[te].float().mean()),
                       train_all=float(bits[tr].all(1).float().mean()), test_all=float(bits[te].all(1).float().mean()),
                       wnorm=float(sum((p ** 2).sum() for p in m.parameters()).sqrt()))
        m.train()
        return rec

    t0 = time.time()
    for step in range(STEPS + 1):
        if step in cs:
            ckpts.append(dict(step=step, state={k: v.clone() for k, v in m.state_dict().items()}))
        if step in cs or step % 200 == 0:
            curve.append(evaluate(step))
        if step == STEPS:
            break
        opt.zero_grad()
        loss = F.binary_cross_entropy_with_logits(m(x[tr]), y[tr])
        loss.backward()
        opt.step()
    torch.save(ckpts, os.path.join(d, "checkpoints.pt"))
    json.dump(curve, open(os.path.join(d, "curve.json"), "w"))
    print(f"trained wd={wd} seed={seed} in {time.time()-t0:.0f}s; final {curve[-1]}")


def analyse(wd, seed):
    c, X, Y, tr, te = data()
    d = run_dir(wd, seed)
    ckpts = torch.load(os.path.join(d, "checkpoints.pt"))
    out = []
    t0 = time.time()
    for ck in ckpts:
        m = MLP()
        m.load_state_dict(ck["state"])
        m.eval()
        r = I.analyse_checkpoint(m, X, tr, te)
        r["step"] = ck["step"]
        for s in SITES:
            r["sites"][s]["flags"] = I.flags(r["sites"][s])
            r["sites"][s]["novelty"] = I.novelty(m, X, c, r["sites"][s], s, r["pools"])
            # projection functions of accepted directions on all inputs, for cross-seed comparison
            sub = I.Sub(m, X)
            r["sites"][s]["proj"] = [(sub.Z[s] @ np.asarray(u)).tolist() for u in r["sites"][s]["accepted_dirs"]]
        out.append(r)
        fl = {s: "".join(k for k, v in r["sites"][s]["flags"].items() if v) or "-" for s in SITES}
        print(f"step {ck['step']:6d}  " + "  ".join(
            f"{s}: acc={r['sites'][s]['n_accepted']} rand={r['sites'][s]['n_pass_rand']} {fl[s]}" for s in SITES),
            f"  ({time.time()-t0:.0f}s)", flush=True)
    json.dump(out, open(os.path.join(d, "analysis.json"), "w"))


# ------------------------------------------------------------------ report
def first_persistent(steps, ok):
    """First step from which `ok` holds at every later checkpoint."""
    for i in range(len(ok)):
        if all(ok[i:]):
            return steps[i]
    return None


def first_true(steps, ok):
    for s, o in zip(steps, ok):
        if o:
            return s
    return None


def report(wds=WDS + (1.0,)):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    c, X, Y, tr, te = data()
    G = c.hidden(X)
    runs = {}
    for wd in wds:
        for seed in SEEDS:
            d = run_dir(wd, seed)
            if not os.path.exists(os.path.join(d, "analysis.json")):
                continue
            runs[(wd, seed)] = dict(curve=json.load(open(os.path.join(d, "curve.json"))),
                                    an=json.load(open(os.path.join(d, "analysis.json"))))
    lines = ["# Click experiment: results", "",
             f"Circuit seed {CIRCUIT_SEED} (accepted after {c.draws} draws), gate reuse {c.reuse()}, "
             f"train fraction {TRAIN_FRAC}, {STEPS} steps, lr {LR}.", "",
             "Flags per site: A substitutability (>=1 accepted direction), B sufficiency, C context-generality, "
             "D reuse, E composition.  'click' = first checkpoint from which A,B,C,D all hold and keep holding.",
             "wd 0 / 0.01 / 0.1 is the frozen sweep; wd 1.0 is the post-hoc extension of the same axis (everything else identical),",
             "added because the frozen levels produced nearly identical weight norms and trajectories.", ""]
    summary = {}
    for (wd, seed), r in runs.items():
        an, curve = r["an"], r["curve"]
        steps = [a["step"] for a in an]
        cs = {e["step"]: e for e in curve}
        t_test = first_true([e["step"] for e in curve], [e["test_acc"] >= 0.99 for e in curve])
        t_train = first_true([e["step"] for e in curve], [e["train_acc"] >= 0.99 for e in curve])
        lines += [f"## wd={wd} seed={seed}", "",
                  f"train_acc>=0.99 at step {t_train}; test_acc>=0.99 at step {t_test}; "
                  f"final test_acc {curve[-1]['test_acc']:.3f}, test_all {curve[-1]['test_all']:.3f}, "
                  f"‖w‖ {curve[-1]['wnorm']:.1f}", ""]
        hdr = "| step | train | test | " + " | ".join(f"{s}: acc/rand A B C D E" for s in SITES) + " |"
        lines += [hdr, "|" + "---|" * (3 + len(SITES))]
        per_site = {}
        for s in SITES:
            ok = [all(a["sites"][s]["flags"][k] for k in "ABCD") for a in an]
            per_site[s] = dict(
                click=first_persistent(steps, ok),
                first={k: first_true(steps, [a["sites"][s]["flags"][k] for a in an]) for k in "ABCDE"},
                persist={k: first_persistent(steps, [a["sites"][s]["flags"][k] for a in an]) for k in "ABCDE"},
            )
        for a in an:
            e = cs.get(a["step"], {})
            row = f"| {a['step']} | {e.get('train_acc', float('nan')):.3f} | {e.get('test_acc', float('nan')):.3f} |"
            for s in SITES:
                sr = a["sites"][s]
                fl = sr["flags"]
                def f(k):
                    v = sr.get(k, float("nan"))
                    return f"{v:.2f}" if isinstance(v, float) else str(v)
                row += (f" {sr['n_accepted']}/{sr['n_pass_rand']} "
                        f"A={max([d['A'] for d in sr['accepted']], default=float('nan')):.2f}{'✓' if fl['A'] else ' '} "
                        f"B={f('B')}(r{f('B_base')}){'✓' if fl['B'] else ' '} "
                        f"C={f('C_mean')}{'✓' if fl['C'] else ' '} "
                        f"D={sr.get('D_max', 0)}{'✓' if fl['D'] else ' '} "
                        f"E={f('E_best')}{'✓' if fl['E'] else ' '} |")
            lines.append(row)
        lines.append("")
        for s in SITES:
            p = per_site[s]
            lines.append(f"- {s}: click at {p['click']}; first pass A/B/C/D/E = "
                         + "/".join(str(p["first"][k]) for k in "ABCDE")
                         + "; persistent from " + "/".join(str(p["persist"][k]) for k in "ABCDE"))
        # novelty at the final checkpoint
        fin = an[-1]
        lines.append("")
        lines.append("Novelty at the final checkpoint (per accepted direction: max |corr| of the projection with any gate / output / input; ARI of the donor typing with the gate / output / input partitions):")
        for s in SITES:
            for k, (d, nv) in enumerate(zip(fin["sites"][s]["accepted"], fin["sites"][s]["novelty"])):
                lines.append(f"- {s}[{k}] ({d['dict_']}{d['i']}, {d['n_types']} types, A={d['A']:.2f}, moves {d['outputs_moved']} outputs): "
                             f"corr gate {nv['corr_gate'][0]:.2f}(g{nv['corr_gate'][1]}) out {nv['corr_out'][0]:.2f}(y{nv['corr_out'][1]}) in {nv['corr_in'][0]:.2f}; "
                             f"ARI gate {nv['ari_gate'][0]:.2f}(g{nv['ari_gate'][1]}) out {nv['ari_out'][0]:.2f} in {nv['ari_in'][0]:.2f}")
            pairs = fin["sites"][s].get("pairs", [])
            if pairs:
                lines.append(f"- {s} pairs: " + "; ".join(
                    f"({p['i']},{p['j']}) E={p['E']:.2f} joint types {p['n_joint']} vs product {p['product']} (pairs seen {p['n_pairs_seen']}, fallback {p['fallback']})" for p in pairs))
        # threshold robustness at the final checkpoint
        rob = []
        for s in SITES:
            cand = [d for d in fin["sites"][s]["candidates"] if d["dict_"] != "rand"]
            rand = [d for d in fin["sites"][s]["candidates"] if d["dict_"] == "rand"]
            def n_pass(ds, A_min=I.CFG["A_min"], key="A", nkey="n_types"):
                return sum(1 for d in ds if d["mag"] >= I.CFG["mag_min"] and 2 <= d[nkey] <= 8 and 1 - d["max_share"] >= 0.15
                           and d[key] >= A_min and d[key] - d["A_base"] >= I.CFG["A_margin"])
            rob.append(f"{s}: A_min 0.7/0.8/0.9 → {n_pass(cand, 0.7)}/{n_pass(cand, 0.8)}/{n_pass(cand, 0.9)} (rand {n_pass(rand, 0.7)}/{n_pass(rand, 0.8)}/{n_pass(rand, 0.9)}); "
                       f"eps 0.2/0.3/0.4 → {sum(d['passed@0.2'] for d in cand)}/{sum(d['passed'] for d in cand)}/{sum(d['passed@0.4'] for d in cand)}")
        lines += ["", "Threshold robustness (raw passing candidates before dedup, final checkpoint): " + "; ".join(rob), ""]
        summary[(wd, seed)] = dict(per_site=per_site, t_train=t_train, t_test=t_test,
                                   final={s: fin["sites"][s]["n_accepted"] for s in SITES})

    # cross-seed correspondence at the final checkpoint
    lines += ["## Cross-seed correspondence (final checkpoint)", "",
              "For each accepted direction of one seed, the max |corr| over all inputs of its projection with any accepted direction of another seed at the same site; mean over directions.", ""]
    for wd in wds:
        for s in SITES:
            vals = []
            for a in SEEDS:
                for b in SEEDS:
                    if a == b or (wd, a) not in runs or (wd, b) not in runs:
                        continue
                    Pa = np.array(runs[(wd, a)]["an"][-1]["sites"][s]["proj"])
                    Pb = np.array(runs[(wd, b)]["an"][-1]["sites"][s]["proj"])
                    if len(Pa) == 0 or len(Pb) == 0:
                        continue
                    C = np.abs(np.corrcoef(np.vstack([Pa, Pb]))[: len(Pa), len(Pa):])
                    vals.append(float(np.nanmean(np.nanmax(C, axis=1))))
            lines.append(f"- wd={wd} {s}: " + (", ".join(f"{v:.2f}" for v in vals) if vals else "n/a"))
    lines.append("")
    # figure
    fig, axes = plt.subplots(len(wds), 2, figsize=(12, 3.2 * len(wds)), squeeze=False)
    for i, wd in enumerate(wds):
        for j, s in enumerate(SITES):
            ax = axes[i, j]
            for seed in SEEDS:
                if (wd, seed) not in runs:
                    continue
                r = runs[(wd, seed)]
                st = [e["step"] for e in r["curve"]]
                ax.plot(st, [e["test_acc"] for e in r["curve"]], color=f"C{seed}", lw=1, label=f"test acc s{seed}")
                ax.plot(st, [e["train_acc"] for e in r["curve"]], color=f"C{seed}", lw=1, ls=":")
                steps = [a["step"] for a in r["an"]]
                ax.plot(steps, [sum(a["sites"][s]["flags"][k] for k in "ABCD") / 4 for a in r["an"]],
                        color=f"C{seed}", marker="o", ms=3, lw=0.8, ls="--", label=f"A-D fraction s{seed}")
                ax2 = ax.twinx() if seed == SEEDS[0] else ax2
                ax2.plot(steps, [a["sites"][s]["n_accepted"] for a in r["an"]], color=f"C{seed}", alpha=0.4, lw=2)
            ax.set_xscale("symlog", linthresh=10)
            ax.set_title(f"wd={wd} site {s}: acc (solid/dotted), A-D fraction (dashed), #accepted (thick, right)")
            ax.set_ylim(-0.05, 1.05)
            if i == 0 and j == 0:
                ax.legend(fontsize=6)
    fig.tight_layout()
    os.makedirs(ROOT, exist_ok=True)
    fig.savefig(os.path.join(ROOT, "click.png"), dpi=110)
    open(os.path.join(ROOT, "REPORT.md"), "w").write("\n".join(lines))
    print("\n".join(lines))
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "analyse", "report"])
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if a.cmd == "train":
        train(a.wd, a.seed)
    elif a.cmd == "analyse":
        analyse(a.wd, a.seed)
    else:
        report()
