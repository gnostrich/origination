"""Unit test battery for the continuous substrate, plus evaluation against
what is analytically known (evaluation only; never enters extraction).

    python -m emergence.cont.run battery  --out results/cont
    python -m emergence.cont.run landscape --n 3 --wells 8 --seeds 0,1,2 --out results/cont
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from .extract_cont import ContConfig, extract_cont
from .systems import RandomLandscape, make_system


def evaluate_1d(res: dict, system: str) -> dict:
    """Compare discovered classes with the analytic basins of the 1-D systems."""
    X = np.array(res["states"])[:, 0]
    lab = np.array(res["labels"])
    if system == "double_well":
        truth = (X > 0).astype(int)  # basin of +1 vs -1 (x = 0 is the separatrix)
    else:
        truth = np.zeros_like(lab)
    # agreement: adjusted Rand index between discovered classes and analytic basins
    from ..grok.compare import adjusted_rand_index
    ari = adjusted_rand_index(lab, truth) if len(set(truth.tolist())) > 1 else float("nan")
    reps = np.array(res["rep_states"]).reshape(-1)
    # ARI restricted to states in stable classes (the discovered objects)
    st = set(res["stable_class_ids"]); mask = np.array([l in st for l in lab])
    ari_stable = adjusted_rand_index(lab[mask], truth[mask]) if mask.sum() > 1 and len(set(truth[mask].tolist())) > 1 else float("nan")
    return {"ari_vs_analytic_basins": ari, "ari_stable_states": ari_stable, "frac_states_in_stable_classes": float(mask.mean()),
            "stable_rep_states": reps.tolist(),
            "class_mean_x": [float(X[lab == c].mean()) for c in range(res["n_classes"])],
            "class_x_range": [[float(X[lab == c].min()), float(X[lab == c].max())] for c in range(res["n_classes"])]}


def evaluate_landscape(res: dict, sys: RandomLandscape, rng) -> dict:
    mins = sys.minima(rng)
    X = np.array(res["states"]); lab = np.array(res["labels"])
    # analytic basin of each state: descend
    Y = X.copy()
    for _ in range(4000):
        Y = Y + 0.01 * sys.f(Y)
    truth = np.array([int(np.argmin(np.linalg.norm(mins - y, axis=1))) for y in Y])
    from ..grok.compare import adjusted_rand_index
    # which analytic minimum each discovered stable object sits in (evaluation only)
    reps = np.array(res["rep_states"]).reshape(-1, sys.n) if len(res["rep_states"]) else np.zeros((0, sys.n))
    Z = reps.copy()
    for _ in range(4000):
        Z = Z + 0.01 * sys.f(Z)
    obj_min = [int(np.argmin(np.linalg.norm(mins - z, axis=1))) for z in Z] if len(mins) else []
    st = set(res["stable_class_ids"]); mask = np.array([l in st for l in lab])
    ari_stable = adjusted_rand_index(lab[mask], truth[mask]) if mask.sum() > 1 and len(set(truth[mask].tolist())) > 1 else float("nan")
    return {"n_minima_analytic": int(len(mins)), "minima": mins.tolist(),
            "n_basins_visited_by_states": int(len(set(truth.tolist()))),
            "ari_vs_analytic_basins": adjusted_rand_index(lab, truth), "ari_stable_states": ari_stable,
            "frac_states_in_stable_classes": float(mask.mean()),
            "object_to_minimum": obj_min, "n_distinct_minima_found": int(len(set(obj_min))),
            "V_at_minima": sys.V(mins).tolist() if len(mins) else []}


def run_one(system: str, cfg: ContConfig, out: Path, name: str, verbose=True, system_seed: int | None = None) -> dict:
    """``cfg.seed`` seeds the extraction; ``system_seed`` (default: the same) seeds a random landscape."""
    sys = make_system(system, cfg.seed if system_seed is None else system_seed)
    res = extract_cont(sys, cfg)
    ev = evaluate_1d(res, system) if system in ("double_well", "single_well") else \
        evaluate_landscape(res, sys, np.random.default_rng(99))
    mon = res["monoid"]
    row = {"name": name, "system": system, **{k: getattr(cfg, k) for k in ("dt", "K", "u_sigma", "T_skip", "D", "U", "seed", "eps")},
           "n_classes": res["n_classes"], "n_stable": res["n_stable_classes"], "stable_sizes": res["stable_class_sizes"],
           "class_sizes": res["class_sizes"], "discreteness": res["discreteness"],
           "metastability": res["metastability"], "n_operations": res["n_operations"],
           "frac_pulses_classified": res["frac_pulses_classified"], "closure": res["closure"],
           "coherence": res["coherence"], "monoid_order": mon.get("monoid_order"), "monoid_idempotents": mon.get("idempotents"),
           "monoid_constants": mon.get("constant_maps"), "monoid_units": mon.get("units"),
           "ari": ev.get("ari_vs_analytic_basins"), "eval": {k: v for k, v in ev.items() if k not in ("minima",)}}
    if verbose:
        print(f"[{name}] {system} dt={cfg.dt} K={cfg.K} u={cfg.u_sigma} Tskip={cfg.T_skip} D={cfg.D} seed={cfg.seed}: "
              f"classes={res['n_classes']} stable={res['n_stable_classes']} {res['stable_class_sizes']} disc={res['discreteness']:.2f} "
              f"meta={res['metastability']:.2f} ops={res['n_operations']} classified={res['frac_pulses_classified']:.2f} "
              f"closure={res['closure']:.2f} monoid={mon.get('monoid_order')}/{mon.get('idempotents')}/{mon.get('constant_maps')}/{mon.get('units')} "
              f"ARI={row['ari']} eval={ {k: v for k, v in ev.items() if k in ('n_minima_analytic', 'object_to_minimum', 'class_mean_x')} }", flush=True)
    out.mkdir(parents=True, exist_ok=True)
    json.dump({"config": asdict(cfg), "result": {k: v for k, v in res.items() if k not in ("states", "labels")},
               "evaluation": ev}, open(out / f"{name}.json", "w"), default=float)
    return row


def cmd_battery(a):
    out = Path(a.out)
    rows = []
    base = ContConfig()
    # main: double well, three seeds
    for s in (0, 1, 2):
        rows.append(run_one("double_well", replace(base, seed=s), out, f"dw_seed{s}"))
    # negative control: single well
    for s in (0, 1, 2):
        rows.append(run_one("single_well", replace(base, seed=s), out, f"sw_seed{s}"))
    # timestep
    for dt in (0.1, 0.05, 0.002):
        rows.append(run_one("double_well", replace(base, dt=dt), out, f"dw_dt{dt}"))
    # sampling density
    for K in (3, 40):
        rows.append(run_one("double_well", replace(base, K=K), out, f"dw_K{K}"))
    # control magnitude (suffix controls) and pulse range
    for u in (0.05, 1.0, 2.0):
        rows.append(run_one("double_well", replace(base, u_sigma=u), out, f"dw_u{u}"))
    # behaviour window
    for ts in (0.0, 2.0, 10.0):
        rows.append(run_one("double_well", replace(base, T_skip=ts), out, f"dw_Tskip{ts}"))
    # process noise
    for D in (0.01, 0.1, 0.5):
        rows.append(run_one("double_well", replace(base, D=D), out, f"dw_D{D}"))
    # tolerance
    for e in (0.03, 0.3):
        rows.append(run_one("double_well", replace(base, eps=e), out, f"dw_eps{e}"))
    json.dump(rows, open(out / "battery.json", "w"), indent=1, default=float)
    _table(rows, out / "battery.md", "Continuous substrate: double-well unit test and controls")


def cmd_landscape(a):
    out = Path(a.out)
    rows = []
    base = ContConfig(U=4.0)
    for s in [int(x) for x in a.seeds.split(",")]:
        spec = f"landscape:{a.n}:{a.wells}"
        for ex_seed in (0, 1):  # two independent extractions of the same landscape
            rows.append(run_one(spec, replace(base, seed=100 * s + ex_seed), out,
                                f"land{a.n}d_w{a.wells}_s{s}_x{ex_seed}", system_seed=s))
    json.dump(rows, open(out / f"landscape_{a.n}d.json", "w"), indent=1, default=float)
    _table(rows, out / f"landscape_{a.n}d.md", f"Random {a.n}-D landscape: blind quotient")


def _table(rows, path, title):
    cols = ["name", "system", "dt", "K", "u_sigma", "T_skip", "D", "eps", "seed", "n_classes", "n_stable", "stable_sizes", "discreteness",
            "metastability", "n_operations", "frac_pulses_classified", "closure", "coherence", "monoid_order",
            "monoid_idempotents", "monoid_constants", "monoid_units", "ari", "n_minima_analytic", "objects_to_minima"]
    lines = [f"# {title}", "", "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        cells = []
        for c in cols:
            v = r.get(c)
            cells.append(f"{v:.2f}" if isinstance(v, float) else str(v))
        lines.append("| " + " | ".join(cells) + " |")
    Path(path).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["battery", "landscape"])
    ap.add_argument("--out", default="results/cont")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--wells", type=int, default=8)
    ap.add_argument("--seeds", default="0,1,2")
    a = ap.parse_args(argv)
    if a.cmd == "battery":
        cmd_battery(a)
    else:
        cmd_landscape(a)


if __name__ == "__main__":
    main()
