"""CLI for the non-explicitly-algebraic task (world prediction with a GRU).

    python -m emergence.grok.run_world train   --world perm4 --seeds 0,1,2 --out results/world
    python -m emergence.grok.run_world extract --out results/world
    python -m emergence.grok.run_world report  --out results/world
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .compare import adjusted_rand_index, first_step, half_rise_step
from .extract_seq import SeqExtractConfig, _all_prefixes, extract_seq
from .seqsub import build_seq
from .world import make_dataset, make_world, transformation_monoid


def train_one(world_spec, seed, out_dir: Path, n_train, n_test, length, steps, ckpt_every, lr, wd, threads, arch="gru", batch_size=0):
    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    world = make_world(world_spec)
    (Atr, Otr), (Ate, Ote) = make_dataset(world, n_train, n_test, length, seed)
    model, _ = build_seq(arch, world.n_actions, world.n_obs)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = set([0, 10, 25, 50] + list(range(ckpt_every, steps + 1, ckpt_every)))
    log, ckpts = [], {}
    t0 = time.time()

    def ev():
        model.eval()
        with torch.no_grad():
            ltr, _ = model(Atr); lte, _ = model(Ate)
            r = {"train_loss": float(F.cross_entropy(ltr.reshape(-1, world.n_obs), Otr.reshape(-1))),
                 "test_loss": float(F.cross_entropy(lte.reshape(-1, world.n_obs), Ote.reshape(-1))),
                 "train_acc": float((ltr.argmax(-1) == Otr).float().mean()),
                 "test_acc": float((lte.argmax(-1) == Ote).float().mean())}
        model.train()
        return r

    for step in range(steps + 1):
        if step in sched:
            r = ev(); r["step"] = step; r["elapsed_s"] = time.time() - t0
            log.append(r)
            ckpts[step] = copy.deepcopy({k: v.detach().clone() for k, v in model.state_dict().items()})
            if step % (5 * ckpt_every) == 0 and step > 0:  # periodic save so a restart cannot lose the run
                out_dir.mkdir(parents=True, exist_ok=True)
                torch.save({"config": {"world": world_spec, "arch": arch, "seed": seed}, "checkpoints": ckpts, "log": log},
                           out_dir / "checkpoints.pt")
            print(f"[{arch} {world_spec} s{seed}] step {step:5d} train {r['train_acc']:.3f}/{r['train_loss']:.3f} "
                  f"test {r['test_acc']:.3f}/{r['test_loss']:.3f} ({r['elapsed_s']:.0f}s)", flush=True)
        if step == steps:
            break
        if batch_size and batch_size < n_train:
            bi = torch.randint(0, n_train, (batch_size,))
            xb, yb = Atr[bi], Otr[bi]
        else:
            xb, yb = Atr, Otr
        lg, _ = model(xb)
        loss = F.cross_entropy(lg.reshape(-1, world.n_obs), yb.reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = {"world": world_spec, "arch": arch, "seed": seed, "n_train": n_train, "n_test": n_test, "length": length,
           "steps": steps, "lr": lr, "weight_decay": wd, "batch_size": batch_size}
    torch.save({"config": cfg, "checkpoints": ckpts, "log": log}, out_dir / "checkpoints.pt")
    json.dump({"config": cfg, "log": log}, open(out_dir / "train_log.json", "w"), indent=1)


def cmd_train(a):
    for seed in [int(s) for s in a.seeds.split(",")]:
        name = f"{a.world}_s{seed}" if a.arch == "gru" else f"{a.arch}_{a.world}_s{seed}"
        train_one(a.world, seed, Path(a.out) / name, a.n_train, a.n_test, a.length,
                  a.steps, a.ckpt_every, a.lr, a.weight_decay, a.threads, a.arch, a.batch_size)


def cmd_extract(a):
    torch.set_num_threads(a.threads)
    cfg = SeqExtractConfig()
    for run_dir in sorted(Path(a.out).iterdir()):
        ck = run_dir / "checkpoints.pt"
        if not ck.exists() or ((run_dir / "metrics.json").exists() and not a.force):
            continue
        data = torch.load(ck, weights_only=False)
        world = make_world(data["config"]["world"])
        model, sub = build_seq(data["config"].get("arch", "gru"), world.n_actions, world.n_obs)
        model.eval()
        metrics = {}
        steps = sorted(data["checkpoints"])
        if a.max_ckpts and len(steps) > a.max_ckpts:
            idx = np.unique(np.round(np.geomspace(1, len(steps), a.max_ckpts)).astype(int) - 1)
            steps = sorted({steps[0], steps[-1]} | {steps[i] for i in idx})
        for step in steps:
            model.load_state_dict(data["checkpoints"][step])
            m = extract_seq(sub, world.n_actions, cfg, np.random.default_rng(1234))
            m = {k: v for k, v in m.items() if not k.startswith("_")}
            metrics[step] = m
            mon = m["monoid"]
            print(f"[{run_dir.name}] step {step:5d} classes {m['n_classes']:4d} reach {m['n_reachable_classes']:4d} "
                  f"meta {m['metastability']:.2f} clos {m['closure']:.2f} coh {m['action_coherence']:.2f} "
                  f"lawfree {m['crystallization_lawfree']:.2f} monoid {mon.get('monoid_order', 'n/a')} "
                  f"group {mon.get('is_group', 'n/a')} abelian {mon.get('abelian', 'n/a')}", flush=True)
        json.dump({"config": data["config"], "metrics": metrics}, open(run_dir / "metrics.json", "w"))


def cmd_report(a):
    out = Path(a.out)
    runs = {}
    for run_dir in sorted(out.iterdir()):
        if (run_dir / "metrics.json").exists():
            m = json.load(open(run_dir / "metrics.json")); l = json.load(open(run_dir / "train_log.json"))
            runs[run_dir.name] = {"config": m["config"], "metrics": {int(k): v for k, v in m["metrics"].items()}, "log": l["log"]}
    if not runs:
        print("no runs"); return
    lines = ["# World prediction: blind quotient extraction", ""]
    lines.append("| run | world | task learned (test acc) | ref. minimal states | ref. monoid (order/idem/const/units) | classes (reachable) | metastability | closure | quotient = world's minimal automaton | recovered monoid (order/idem/const/units) | units fraction | group | abelian | actions bijective | coherence (automatic) | law-free index |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    timing = {}
    refs = {}
    for n, r in runs.items():
        world = make_world(r["config"]["world"])
        m_ref, T_ref = world.minimal_automaton()
        refs[n] = (m_ref, T_ref)
        steps = sorted(r["metrics"]); last = r["metrics"][steps[-1]]
        tl = {x["step"]: x for x in r["log"]}
        mon = last["monoid"]
        ref_mon = transformation_monoid(T_ref)
        fmt = lambda d: f"{d.get('order', d.get('monoid_order', 'n/a'))}/{d.get('idempotents', 'n/a')}/{d.get('constant_maps', 'n/a')}/{d.get('units', 'n/a')}"
        from .extract_seq import canonical_automaton as _canon
        Cref, _ = _canon(T_ref, 0)
        C_last = np.array(last["canonical_table"])
        matches = C_last.shape == Cref.shape and np.array_equal(C_last, Cref)
        acc = tl[steps[-1]]["test_acc"]
        learned = "yes" if acc >= 0.95 else "NO"
        units_frac = (f"{mon['units'] / mon['monoid_order']:.2f}" if mon.get("total") and mon.get("monoid_order") and mon.get("units") is not None else "n/a")
        lines.append(f"| {n} | {r['config']['world']} | {learned} ({acc:.3f}) | {m_ref} | {fmt(ref_mon)} | "
                     f"{last['n_classes']} ({last['n_reachable_classes']}) | {last['metastability']:.2f} | {last['closure']:.2f} | "
                     f"{matches} | {fmt(mon) if mon.get('total') else 'not total'} | {units_frac} | "
                     f"{mon.get('is_group', 'n/a')} | {mon.get('abelian', 'n/a')} | {mon.get('actions_bijective', 'n/a')} | "
                     f"{last['action_coherence']:.2f} | {last['crystallization_lawfree']:.2f} |")
        test = [tl[s]["test_acc"] for s in steps]
        lf = [r["metrics"][s]["crystallization_lawfree"] for s in steps]
        ncl = [r["metrics"][s]["n_classes"] for s in steps]
        timing[n] = {"t_gen_half": half_rise_step(steps, test), "t_gen_0.9": first_step(steps, test, 0.9),
                     "lawfree_half_rise": half_rise_step(steps, lf),
                     "classes_by_step": dict(zip(steps, ncl)), "test_by_step": dict(zip(steps, test))}
    lines.append("")
    lines.append("## Timing")
    lines.append("")
    lines.append("| run | test acc half-rise | test acc ≥ 0.9 | law-free index half-rise |")
    lines.append("|---|---|---|---|")
    for n, t in timing.items():
        lines.append(f"| {n} | {t['t_gen_half']} | {t['t_gen_0.9']} | {t['lawfree_half_rise']} |")
    lines.append("")
    lines.append("## Trajectories (step: classes / test acc / law-free index)")
    lines.append("")
    for n, r in runs.items():
        steps = sorted(r["metrics"]); tl = {x["step"]: x for x in r["log"]}
        lines.append(f"- {n}: " + ", ".join(f"{s}: {r['metrics'][s]['n_classes']} / {tl[s]['test_acc']:.2f} / {r['metrics'][s]['crystallization_lawfree']:.2f}" for s in steps))
    lines.append("")
    # cross-seed: canonical automata equal? ARI of prefix partitions; comparison with the reference automaton
    names = sorted(runs)
    lines.append("## Cross-seed equivalence of the induced automata (final checkpoints)")
    lines.append("")
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            ri, rj = runs[names[i]], runs[names[j]]
            if ri["config"]["world"] != rj["config"]["world"]:
                continue
            si, sj = max(ri["metrics"]), max(rj["metrics"])
            Ci, Cj = np.array(ri["metrics"][si]["canonical_table"]), np.array(rj["metrics"][sj]["canonical_table"])
            same = Ci.shape == Cj.shape and np.array_equal(Ci, Cj)
            ari = adjusted_rand_index(ri["metrics"][si]["partition"], rj["metrics"][sj]["partition"])
            lines.append(f"- {names[i]} vs {names[j]}: canonical automata identical: {same} "
                         f"({Ci.shape[0]} vs {Cj.shape[0]} reachable classes); ARI of prefix partitions {ari:.3f}")
    lines.append("")
    lines.append("## Against the external reference (uses the world; report only)")
    lines.append("")
    for n, r in runs.items():
        world = make_world(r["config"]["world"])
        m_ref, T_ref = refs[n]
        s = max(r["metrics"])
        C = np.array(r["metrics"][s]["canonical_table"])
        # canonical relabel of the reference from its initial state
        from .extract_seq import canonical_automaton
        Cref, _ = canonical_automaton(T_ref, 0)
        # reference partition of the prefixes by minimal-automaton state
        prefixes = _all_prefixes(world.n_actions, SeqExtractConfig().prefix_len)
        rng = np.random.default_rng(1234)
        long = [tuple(rng.integers(0, world.n_actions, SeqExtractConfig().long_len)) for _ in range(SeqExtractConfig().n_long_prefixes)]
        prefixes = prefixes + long
        st = []
        for p in prefixes:
            x = world.init_state
            for a in p:
                x = int(world.step_table[x, a])
            st.append(x)
        # minimal-automaton class of each world state: rerun minimisation mapping
        ref_part = np.array(st)
        ari = adjusted_rand_index(r["metrics"][s]["partition"], ref_part)
        same = C.shape == Cref.shape and np.array_equal(C, Cref)
        lines.append(f"- {n}: reference minimal automaton has {m_ref} states; recovered canonical automaton identical to it: {same}; "
                     f"ARI(prefix partition, world-state partition) = {ari:.3f}")
    (out / "report.md").write_text("\n".join(lines) + "\n")
    json.dump({"timing": timing}, open(out / "report.json", "w"), indent=1, default=str)
    print("\n".join(lines))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "extract", "report", "all"])
    ap.add_argument("--out", default="results/world")
    ap.add_argument("--world", default="perm4")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--n_train", type=int, default=1024)
    ap.add_argument("--n_test", type=int, default=1024)
    ap.add_argument("--length", type=int, default=12)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--arch", default="gru", help="gru | transformer")
    ap.add_argument("--ckpt_every", type=int, default=200)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--weight_decay", type=float, default=0.1)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--batch_size", type=int, default=0, help="minibatch size for training (0 = full batch)")
    ap.add_argument("--max_ckpts", type=int, default=0, help="extract at most this many log-spaced checkpoints")
    a = ap.parse_args(argv)
    if a.cmd in ("train", "all"):
        cmd_train(a)
    if a.cmd in ("extract", "all"):
        cmd_extract(a)
    if a.cmd in ("report", "all"):
        cmd_report(a)


if __name__ == "__main__":
    main()
