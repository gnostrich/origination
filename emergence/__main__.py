"""Command line entry point.

    python -m emergence run   --landscape designed --out results/designed
    python -m emergence run   --landscape random --seed 3 --T 0.2 --g 2.0
    python -m emergence sweep --landscape random --T 0.05,0.1,0.2,0.4 --g 1,2,4 --seeds 0,1,2
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

from .experiment import ExperimentConfig, run_experiment


def _add_config_args(p: argparse.ArgumentParser, skip=("out", "lean_out")):
    for f in dataclasses.fields(ExperimentConfig):
        if f.name in skip:
            continue
        t = f.type
        if t == "bool" or t is bool:
            p.add_argument(f"--{f.name}", type=lambda s: s.lower() in ("1", "true", "yes"), default=f.default)
        elif t in ("int", int):
            p.add_argument(f"--{f.name}", type=int, default=f.default)
        elif t in ("float", float):
            p.add_argument(f"--{f.name}", type=float, default=f.default)
        else:
            p.add_argument(f"--{f.name}", type=str, default=f.default)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="emergence")
    sub = ap.add_subparsers(dest="cmd", required=True)

    pr = sub.add_parser("run", help="run one experiment")
    _add_config_args(pr)
    pr.add_argument("--out", default="results/run")
    pr.add_argument("--lean-out", default=None, help="write the discovered table as a Lean file")
    pr.add_argument("--quiet", action="store_true")

    ps = sub.add_parser("sweep", help="sweep T, g, seeds and tabulate the composite score")
    _add_config_args(ps, skip=("out", "lean_out", "T", "g", "seed"))
    ps.add_argument("--T", dest="T_list", default="0.05,0.1,0.2,0.4")
    ps.add_argument("--g", dest="g_list", default="1,2,4")
    ps.add_argument("--seeds", default="0,1,2")
    ps.add_argument("--out", default="results/sweep")

    a = ap.parse_args(argv)
    names = {f.name for f in dataclasses.fields(ExperimentConfig)}

    if a.cmd == "run":
        kw = {k: v for k, v in vars(a).items() if k in names}
        kw["out"] = a.out
        kw["lean_out"] = a.lean_out
        cfg = ExperimentConfig(**kw)
        res = run_experiment(cfg, verbose=not a.quiet)
        print(json.dumps(res["score"], indent=1))
        return 0

    if a.cmd == "sweep":
        kw = {k: v for k, v in vars(a).items() if k in names and k not in ("T", "g", "seed")}
        Ts = [float(x) for x in a.T_list.split(",")]
        gs = [float(x) for x in a.g_list.split(",")]
        seeds = [int(x) for x in a.seeds.split(",")]
        rows = []
        out = Path(a.out)
        out.mkdir(parents=True, exist_ok=True)
        for T in Ts:
            for g in gs:
                for seed in seeds:
                    cfg = ExperimentConfig(**kw, T=T, g=g, seed=seed, out=str(out / f"T{T}_g{g}_s{seed}"))
                    print(f"=== T={T} g={g} seed={seed} ===", flush=True)
                    res = run_experiment(cfg, verbose=False)
                    sc = res["score"]
                    row = {
                        "T": T,
                        "g": g,
                        "seed": seed,
                        "n_basins": len(res.get("basins", [])),
                        "n_types": res.get("types", {}).get("n_types", 0),
                        "n_clicks": res.get("closure", {}).get("n_clicks", 0),
                        "irreducible_ternary": len((res.get("ternary") or {}).get("irreducible_ternary", [])),
                        **{k: sc.get(k, float("nan")) for k in
                           ("metastability", "polarization", "type_compression", "closure",
                            "substitutability", "associativity", "composite")},
                    }
                    rows.append(row)
                    print(json.dumps(row), flush=True)
        with open(out / "sweep.json", "w") as f:
            json.dump(rows, f, indent=1)
        # text table
        cols = ["T", "g", "seed", "n_basins", "n_types", "n_clicks", "irreducible_ternary",
                "metastability", "polarization", "type_compression", "closure",
                "substitutability", "associativity", "composite"]
        lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
        for r in rows:
            lines.append("| " + " | ".join(
                f"{r[c]:.2f}" if isinstance(r[c], float) else str(r[c]) for c in cols) + " |")
        (out / "sweep.md").write_text("\n".join(lines) + "\n")
        print("\n".join(lines))
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
