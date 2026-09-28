"""CLI for the grokking experiment.

    python -m emergence.grok.run train   --arch transformer --seeds 0,1,2 --out results/grok
    python -m emergence.grok.run train   --arch mlp --seeds 0,1,2 --out results/grok
    python -m emergence.grok.run train   --arch transformer --seeds 0 --weight_decay 0 --tag nowd --out results/grok
    python -m emergence.grok.run extract --out results/grok
    python -m emergence.grok.run report  --out results/grok
    python -m emergence.grok.run all     --arch transformer,mlp --seeds 0,1,2 --out results/grok
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from pathlib import Path

import numpy as np
import torch

from .compare import METRICS, _get, cross_run, run_timing
from .extract import ExtractConfig, extract
from .models import build
from .task import labels
from .train import TrainConfig, train


def run_name(arch, seed, tag):
    return f"{arch}_s{seed}" + (f"_{tag}" if tag else "")


def _train_one(args):
    cfg, out_dir, resume = args
    train(cfg, out_dir, resume=resume)
    return str(out_dir)


def cmd_train(a):
    jobs = []
    for arch in a.arch.split(","):
        for seed in [int(s) for s in a.seeds.split(",")]:
            cfg = TrainConfig(arch=arch, p=a.p, train_frac=a.train_frac, seed=seed, lr=a.lr,
                              weight_decay=a.weight_decay, max_steps=a.max_steps, ckpt_every=a.ckpt_every,
                              stop_after_grok=a.stop_after_grok, threads=a.threads)
            jobs.append((cfg, Path(a.out) / run_name(arch, seed, a.tag), a.resume))
    if a.parallel > 1:
        with mp.get_context("spawn").Pool(a.parallel) as pool:
            for r in pool.imap_unordered(_train_one, jobs):
                print("trained", r, flush=True)
    else:
        for j in jobs:
            _train_one(j)


def cmd_extract(a):
    out = Path(a.out)
    ecfg = ExtractConfig()
    for run_dir in sorted(out.iterdir()):
        ck = run_dir / "checkpoints.pt"
        if not ck.exists():
            continue
        if (run_dir / "metrics.json").exists() and not a.force:
            print("skip (exists)", run_dir)
            continue
        data = torch.load(ck, weights_only=False)
        cfg = data["config"]
        torch.set_num_threads(a.threads)
        model = build(cfg["arch"], cfg["p"])
        metrics = {}
        steps = sorted(data["checkpoints"])
        if a.max_ckpts and len(steps) > a.max_ckpts:
            idx = np.unique(np.linspace(0, len(steps) - 1, a.max_ckpts).astype(int))
            steps = [steps[i] for i in idx]
        for step in steps:
            model.load_state_dict(data["checkpoints"][step])
            rng = np.random.default_rng(1234)  # same contexts / triples for every checkpoint
            m = extract(model, cfg["p"], ecfg, rng)
            metrics[step] = m
            print(f"[{run_dir.name}] step {step:6d} classes {m['n_classes']:5d} meta {m['metastability']:.2f} "
                  f"disc {m['discreteness']:.2f} clos {m['closure']:.2f} assoc {m['op']['associativity']:.2f} "
                  f"comm {m['op']['commutativity']:.2f} cryst {m['crystallization']:.2f}", flush=True)
        with open(run_dir / "metrics.json", "w") as f:
            json.dump({"config": cfg, "metrics": metrics}, f)


def _load_runs(out: Path) -> dict:
    runs = {}
    for run_dir in sorted(out.iterdir()):
        mf, lf = run_dir / "metrics.json", run_dir / "train_log.json"
        if mf.exists() and lf.exists():
            m = json.load(open(mf))
            l = json.load(open(lf))
            runs[run_dir.name] = {"config": m["config"], "metrics": {int(k): v for k, v in m["metrics"].items()},
                                  "log": l["log"], "grok_step": l.get("grok_step")}
    return runs


def cmd_report(a):
    out = Path(a.out)
    runs = _load_runs(out)
    if not runs:
        print("no runs with metrics found")
        return
    p = next(iter(runs.values()))["config"]["p"]
    reference = labels(p).numpy()
    timing = {n: run_timing(r) for n, r in runs.items()}
    # cross-run comparison among runs that share the task (all do) -- split by wd control
    main_runs = {n: r for n, r in runs.items() if r["config"]["weight_decay"] > 0}
    cross = cross_run(main_runs, reference) if len(main_runs) >= 2 else None
    summary = {"timing": timing, "cross": cross}
    with open(out / "report.json", "w") as f:
        json.dump(summary, f, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    lines = ["# Grokking as crystallisation: report", ""]
    lines.append("## Transition timing per run (steps)")
    lines.append("")
    hdr = ["run", "t_mem", "t_gen(0.9)", "t_gen_half"] + [f"{m} half-rise (lag)" for m in METRICS]
    lines.append("| " + " | ".join(hdr) + " |")
    lines.append("|" + "---|" * len(hdr))
    for n, t in timing.items():
        row = [n, str(t["t_mem"]), str(t["t_gen"]), str(t["t_gen_half"])]
        for m in METRICS:
            mm = t["metrics"][m]
            row.append(f"{mm['half_rise']} ({mm['lag_vs_gen']})")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    lines.append("## Plateau (memorised, not generalised) vs after generalisation")
    lines.append("")
    hdr = ["run"] + [f"{m}: plateau -> after" for m in METRICS]
    lines.append("| " + " | ".join(hdr) + " |")
    lines.append("|" + "---|" * len(hdr))
    for n, t in timing.items():
        row = [n]
        for m in METRICS:
            mm = t["metrics"][m]
            pl = "n/a" if mm["plateau_mean"] is None else f"{mm['plateau_mean']:.2f}"
            af = "n/a" if mm["after_mean"] is None else f"{mm['after_mean']:.2f}"
            row.append(f"{pl} -> {af} (final {mm['final']:.2f})")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    if cross:
        lines.append("## Cross-run equivalence of the extracted quotients")
        lines.append("")
        lines.append("| step | mean ARI (hidden partitions) | mean op-table agreement | ARI to true-sum partition (reference) |")
        lines.append("|---|---|---|---|")
        for s in cross["common_steps"]:
            e = cross["per_step"][s]
            ref = ", ".join(f"{k}:{v:.2f}" for k, v in e.get("ari_to_reference", {}).items())
            lines.append(f"| {s} | {e['mean_ari']:.3f} | {e['mean_op_agreement']:.3f} | {ref} |")
        lines.append("")
        lines.append("### Final structures")
        lines.append("")
        for k, v in cross["final_pairs"].items():
            lines.append(f"- {k}: ARI {v['ari']:.3f}, op agreement {v['op_agreement']:.3f}, "
                         f"isomorphic up to relabelling: {v['isomorphism']}")
        lines.append("")
        for k, v in cross["final_invariants"].items():
            lines.append(f"- {k} invariants: {v}")
    (out / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    try:
        _plot(runs, out)
    except Exception as e:  # matplotlib optional
        print("plot skipped:", e)


def _plot(runs, out: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(runs)
    fig, axes = plt.subplots(n, 1, figsize=(9, 3.2 * n), squeeze=False)
    for ax, (name, r) in zip(axes[:, 0], runs.items()):
        steps = sorted(r["metrics"])
        tl = {x["step"]: x for x in r["log"]}
        ax.plot(steps, [tl[s]["train_acc"] for s in steps], "k--", label="train acc")
        ax.plot(steps, [tl[s]["test_acc"] for s in steps], "k-", lw=2, label="test acc")
        for m, st in [("metastability", "-"), ("discreteness", "-"), ("closure", "-"),
                      ("op.associativity", "--"), ("compression", ":"), ("crystallization", "-")]:
            ax.plot(steps, [_get(r["metrics"][s], m) for s in steps], st, label=m, alpha=0.8,
                    lw=2.5 if m == "crystallization" else 1.2)
        ax.set_xscale("symlog", linthresh=100)
        ax.set_ylim(-0.02, 1.02)
        ax.set_title(name)
        ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=7, ncol=4, loc="lower right")
    axes[-1, 0].set_xlabel("training step")
    fig.tight_layout()
    fig.savefig(out / "crystallization.png", dpi=110)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="emergence.grok")
    ap.add_argument("cmd", choices=["train", "extract", "report", "all"])
    ap.add_argument("--out", default="results/grok")
    ap.add_argument("--arch", default="transformer")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--tag", default="")
    ap.add_argument("--p", type=int, default=97)
    ap.add_argument("--train_frac", type=float, default=0.3)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1.0)
    ap.add_argument("--max_steps", type=int, default=25000)
    ap.add_argument("--ckpt_every", type=int, default=500)
    ap.add_argument("--stop_after_grok", type=int, default=3000)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--parallel", type=int, default=1)
    ap.add_argument("--max_ckpts", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--resume", action="store_true", help="continue training from the saved checkpoints")
    a = ap.parse_args(argv)
    if a.cmd in ("train", "all"):
        cmd_train(a)
    if a.cmd in ("extract", "all"):
        cmd_extract(a)
    if a.cmd in ("report", "all"):
        cmd_report(a)


if __name__ == "__main__":
    main()
