"""Label-free interface search on a trained world model, then quotient
extraction on the discovered interface.

    python -m emergence.grok.run_interface --run results/world/counter_s0 --L 6
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .compare import adjusted_rand_index
from .extract_seq import SeqExtractConfig, canonical_automaton, extract_seq
from .interface import InterfaceSubstrate, search_interface, trace_substrate
from .seqsub import build_seq
from .world import make_world, transformation_monoid


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--L", type=int, default=6, help="prefix length used for the sufficiency search")
    ap.add_argument("--max_off", type=int, default=6)
    ap.add_argument("--target", type=float, default=0.95)
    ap.add_argument("--exhaustive", type=int, default=3)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", default="results/interface")
    a = ap.parse_args(argv)
    torch.set_num_threads(a.threads)
    run = Path(a.run)
    data = torch.load(run / "checkpoints.pt", weights_only=False)
    cfg = data["config"]
    world = make_world(cfg["world"])
    arch = cfg.get("arch", "gru")
    model, _ = build_seq(arch, world.n_actions, world.n_obs)
    step = max(data["checkpoints"])
    model.load_state_dict(data["checkpoints"][step]); model.eval()
    sub = trace_substrate(arch, model)
    rng = np.random.default_rng(7)

    print(f"[{run.name}] {arch} on {cfg['world']}, checkpoint {step}: searching interfaces at L={a.L}", flush=True)
    res = search_interface(sub, world.n_actions, a.L, rng, target=a.target, max_off=a.max_off,
                           exhaustive_up_to=a.exhaustive)
    best = res["best"]
    print(f"  baseline (no cells) sufficiency {res['baseline_sufficiency']:.2f}; full interface "
          f"({res['full_complexity']} scalars) sufficiency {res['full_sufficiency']:.2f}", flush=True)
    print(f"  best: {best['cells']} complexity {best['complexity']} sufficiency {best['sufficiency']:.2f} ({best['method']})", flush=True)
    # single-cell sufficiencies, for the record
    singles = sorted([e for e in res["log"] if len(e["cells"]) == 1], key=lambda e: -e["sufficiency"])
    for e in singles:
        print(f"    cell {e['cells'][0]}: sufficiency {e['sufficiency']:.2f} mean JS {e['mean_js']:.3f}", flush=True)

    # quotient extraction on the discovered interface
    isub = InterfaceSubstrate(sub, best["cells"], world.n_actions, np.random.default_rng(11))
    m = extract_seq(isub, world.n_actions, SeqExtractConfig(), np.random.default_rng(1234))
    m_ref, T_ref = world.minimal_automaton()
    Cref, _ = canonical_automaton(T_ref, 0)
    C = np.array(m["canonical_table"])
    matches = C.shape == Cref.shape and np.array_equal(C, Cref)
    # reference partition of the extractor's prefixes by world state (report only)
    from .extract_seq import _all_prefixes
    prefixes = _all_prefixes(world.n_actions, SeqExtractConfig().prefix_len)
    r2 = np.random.default_rng(1234)
    prefixes += [tuple(r2.integers(0, world.n_actions, SeqExtractConfig().long_len)) for _ in range(SeqExtractConfig().n_long_prefixes)]
    st = []
    for p in prefixes:
        x = world.init_state
        for act in p:
            x = int(world.step_table[x, act])
        st.append(x)
    ari = adjusted_rand_index(m["partition"], np.array(st))
    mon = m["monoid"]
    print(f"  extraction on interface: classes {m['n_classes']} (reachable {m['n_reachable_classes']}), "
          f"metastability {m['metastability']:.2f}, closure {m['closure']:.2f}, coherence {m['action_coherence']:.2f}, "
          f"monoid {mon.get('monoid_order','n/a')}/{mon.get('idempotents','n/a')}/{mon.get('constant_maps','n/a')}/{mon.get('units','n/a')}, "
          f"quotient = world's minimal automaton: {matches}, ARI to world states {ari:.3f}", flush=True)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    summary = {"run": run.name, "arch": arch, "world": cfg["world"], "checkpoint": step, "search": res,
               "extraction": {k: v for k, v in m.items() if k not in ("partition",) and not k.startswith("_")},
               "quotient_matches_reference": bool(matches), "ari_to_world_states": ari,
               "reference": {"minimal_states": m_ref, "monoid": transformation_monoid(T_ref)}}
    json.dump(summary, open(out / f"{run.name}.json", "w"), indent=1, default=str)
    return summary


if __name__ == "__main__":
    main()
