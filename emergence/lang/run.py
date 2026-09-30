"""Driver: extract an interface model from an existing trained GRU world
model and evaluate it.

    python -m emergence.lang.run extract --run results/world/perm4_s0 --eps 0.05
    python -m emergence.lang.run pareto  --run results/world/perm4_s0
    python -m emergence.lang.run report
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np
import torch

from emergence.lang.substrate import SeqSubstrate, random_basis
from emergence.lang import discover as D
from emergence.lang import evaluate as E

OUT = "results/lang"
EPS_GRID = (0.01, 0.02, 0.05, 0.1, 0.2, 0.4)
WITHHOLD_FRAC = 0.25
SEED = 0


def load_substrate(run_dir, basis_seed=None):
    from emergence.grok.rnn import GRUWorldModel
    from emergence.grok.world import make_world
    d = torch.load(Path(run_dir) / "checkpoints.pt", weights_only=False)
    world = make_world(d["config"]["world"])                     # evaluation only: used for n_actions/n_obs and unsealing
    model = GRUWorldModel(world.n_actions, world.n_obs)
    model.load_state_dict(d["checkpoints"][max(d["checkpoints"])])
    model.eval()
    basis = random_basis(model.d, basis_seed) if basis_seed is not None else None
    return SeqSubstrate(model, world.n_actions, world.n_obs, basis=basis), world, model


def withheld_chunks(n_actions, rng, frac=WITHHOLD_FRAC):
    pairs = D.all_chunks(n_actions, 2)
    pairs = [c for c in pairs if len(c) == 2]
    k = max(1, int(round(frac * len(pairs))))
    idx = rng.choice(len(pairs), k, replace=False)
    return [pairs[i] for i in sorted(idx)]


def extract_and_evaluate(run_dir, eps, tag=None, verbose=True):
    name = Path(run_dir).name
    sub, world, torch_model = load_substrate(run_dir)
    rng = np.random.default_rng(SEED)
    withhold = withheld_chunks(sub.n_actions, np.random.default_rng(SEED + 1))
    model, disc = D.extract(sub, eps, rng, withhold=withhold)
    calls_discovery = sub.calls
    contexts = disc["contexts"]
    suite = E.heldout_suite(sub, np.random.default_rng(SEED + 2), disc["prefixes"], withheld=withhold)
    fid = E.evaluate_fidelity(model, sub, suite, contexts)
    cx = E.measure_complexity(model)
    rand = E.random_abstraction(model, sub, disc, suite, np.random.default_rng(SEED + 3))
    fid_rand = E.evaluate_fidelity(rand, sub, suite, contexts)
    look = E.lookup_baseline(model, sub, disc, suite, contexts)
    ca = E.causal_abstraction(model, sub, suite, contexts)
    res = dict(run=name, eps=eps, K=model.K, withheld=[list(c) for c in withhold], discovery_calls=int(calls_discovery),
               fidelity=fid, complexity=cx, substrate_bits=E.substrate_bits(torch_model),
               random_abstraction=dict(fidelity=fid_rand, complexity=E.measure_complexity(rand)),
               lookup=look, causal_abstraction=ca, unsealed=E.unseal(model, disc, world))
    odir = os.path.join(OUT, name)
    os.makedirs(odir, exist_ok=True)
    tag = tag or f"eps{eps}"
    json.dump(res, open(os.path.join(odir, f"eval_{tag}.json"), "w"), default=float)
    json.dump(model.to_json(), open(os.path.join(odir, f"model_{tag}.json"), "w"))
    open(os.path.join(odir, f"SPEC_{tag}.md"), "w").write(model.spec(prefixes=disc["prefixes"]))
    if verbose:
        print(f"{name} eps={eps}: K={model.K} fidelity argmax={fid['fidelity_argmax']:.3f} js={fid['fidelity_js']:.3f} cov={fid['coverage']:.2f} "
              f"comp={fid.get('comp_fidelity_argmax', float('nan')):.3f} bits={cx['total_bits_argmax']:.0f} | random {fid_rand['fidelity_argmax']:.3f} "
              f"| lookup {look['fidelity_argmax']:.3f} (comp {look.get('comp_fidelity_argmax', float('nan')):.3f}, bits {look['bits']:.0f}) "
              f"| causal commutation {ca['commutation']:.3f} abstract {ca['abstract_agreement']:.3f} | ARI(world) {res['unsealed']['ari_types_vs_world_states']:.2f}", flush=True)
    return res


def basis_check(run_dir, eps):
    sub, _, _ = load_substrate(run_dir)
    subb, _, _ = load_substrate(run_dir, basis_seed=7)
    withhold = withheld_chunks(sub.n_actions, np.random.default_rng(SEED + 1))
    r = E.basis_robustness(sub, subb, eps, SEED, withhold)
    print(f"{Path(run_dir).name} basis robustness at eps={eps}: {r}")
    return r


def report():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = []
    for f in sorted(glob.glob(os.path.join(OUT, "*", "eval_*.json"))):
        rows.append(json.load(open(f)))
    runs = sorted({r["run"] for r in rows})
    L = ["# Interface-language extraction on existing trained GRU world models", "",
         "Discovery sees pieces (hidden configurations after prefixes), interactions (continuations) and behaviour; "
         "the world is unsealed only in the last column.  Fidelity: per-step argmax agreement with the substrate on "
         "120 unseen pieces × 40 unseen length-8 action strings (abstentions count as errors); comp = zero-shot fidelity "
         "at steps that complete a chunk withheld from discovery.  Bits: argmax interface model "
         "(interfaces + first-order fits + emissions + listed exceptions).", "",
         "| run | ε | K | fidelity argmax | fidelity JS | coverage | comp. fidelity (model / lookup) | bits (model / lookup / substrate) | random abstraction fidelity | causal commutation / abstract agreement | ARI vs world states | discovered monoid order (world) |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        u = r["unsealed"]
        L.append(f"| {r['run']} | {r['eps']} | {r['K']} | {r['fidelity']['fidelity_argmax']:.3f} | {r['fidelity']['fidelity_js']:.3f} | {r['fidelity']['coverage']:.2f} | "
                 f"{r['fidelity'].get('comp_fidelity_argmax', float('nan')):.3f} / {r['lookup'].get('comp_fidelity_argmax', float('nan')):.3f} | "
                 f"{r['complexity']['total_bits_argmax']:.0f} / {r['lookup']['bits']:.0f} / {r['substrate_bits']:.2e} | {r['random_abstraction']['fidelity']['fidelity_argmax']:.3f} | "
                 f"{r['causal_abstraction']['commutation']:.3f} / {r['causal_abstraction']['abstract_agreement']:.3f} | {u['ari_types_vs_world_states']:.2f} | "
                 f"{u['discovered_monoid'].get('order')} ({u['world_monoid']['order']}) |")
    L.append("")
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    for i, run in enumerate(runs):
        rr = sorted([r for r in rows if r["run"] == run], key=lambda r: r["complexity"]["total_bits_argmax"])
        ax.plot([r["complexity"]["total_bits_argmax"] for r in rr], [r["fidelity"]["fidelity_argmax"] for r in rr], "o-", color=f"C{i}", label=f"{run} (interface model, ε sweep)")
        ax.plot([r["lookup"]["bits"] for r in rr], [r["lookup"]["fidelity_argmax"] for r in rr], "s", color=f"C{i}", alpha=0.5, label=f"{run} lookup")
        ax.plot([r["random_abstraction"]["complexity"]["total_bits_argmax"] for r in rr], [r["random_abstraction"]["fidelity"]["fidelity_argmax"] for r in rr], "x", color=f"C{i}", alpha=0.7, label=f"{run} random abstraction")
        ax.plot([rr[0]["substrate_bits"]], [1.0], "*", color=f"C{i}", ms=10)
    ax.set_xscale("log")
    ax.set_xlabel("description complexity (bits)")
    ax.set_ylabel("held-out counterfactual fidelity (argmax)")
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=6)
    ax.set_title("fidelity vs complexity; stars = full substrate")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "pareto.png"), dpi=120)
    open(os.path.join(OUT, "REPORT.md"), "w").write("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["extract", "pareto", "basis", "report"])
    ap.add_argument("--run")
    ap.add_argument("--eps", type=float, default=0.05)
    a = ap.parse_args()
    torch.set_num_threads(1)
    if a.cmd == "extract":
        extract_and_evaluate(a.run, a.eps)
    elif a.cmd == "pareto":
        for eps in EPS_GRID:
            extract_and_evaluate(a.run, eps)
    elif a.cmd == "basis":
        basis_check(a.run, a.eps)
    else:
        report()
