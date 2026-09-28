"""End-to-end v0 pipeline and the composite score

    script C = metastability x polarization x type-compression x closure

together with the two validity checks the Lean development needs as
hypotheses: substitutability (clicking respects behavioural equivalence) and
associativity up to behavioural equivalence.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from itertools import combinations_with_replacement
from pathlib import Path

import numpy as np

from .basins import BasinSet, discover_basins
from .compat import ClickResult, click_tuples, pair_matrix, polarization, strain_summary
from .compose import (
    ClickTable,
    associativity,
    build_click_table,
    closure_score,
    exact_types,
    register_products,
    substitutability,
)
from .energy import Energy, MLPEnergy, designed_landscape
from .export import export_lean
from .types import fingerprints, type_classes, type_summary


@dataclass
class ExperimentConfig:
    landscape: str = "designed"  # "designed" (Experiment A) or "random" (Experiment B)
    N: int = 64
    seed: int = 0
    T: float = 0.1  # temperature
    dt: float = 0.01
    g: float = 2.0  # interaction strength
    # random MLP energy
    hidden: int = 128
    depth: int = 2
    scale: float = 4.0
    confine: float = 0.25
    input_gain: float = 1.0
    # basin discovery
    n_traj: int = 512
    steps: int = 3000
    record_every: int = 50
    init_scale: float = 2.0
    merge_tol: float = 0.3
    lump_p: float = 0.2
    lump_lag: int = 4
    min_occupancy: int = 5
    R_min: float = 10.0
    quench_lr: float = 0.2
    quench_steps: int = 300
    n_esc: int = 32
    esc_horizon: int = 2000
    check_every: int = 50
    relax_steps: int = 1000
    max_basins: int = 24
    max_esc_horizon: int = 20000
    # compatibility
    n_traj_click: int = 16
    horizon: int = 1000
    click_check_every: int = 50
    init_noise: float = 0.05
    c_thresh: float = 0.5
    strain_tol: float = 5.0  # click iff strain <= strain_tol * T
    eps: float = 0.5
    # arity / composition
    do_ternary: bool = True
    max_ternary_basins: int = 10
    max_rounds: int = 3
    # output
    out: str = "results/run"
    lean_out: str | None = None

    def build_energy(self) -> tuple[Energy, dict | None]:
        if self.landscape == "designed":
            return designed_landscape(self.N, seed=self.seed)
        if self.landscape == "random":
            e = MLPEnergy(
                N=self.N,
                hidden=self.hidden,
                depth=self.depth,
                seed=self.seed,
                scale=self.scale,
                confine=self.confine,
                input_gain=self.input_gain,
            )
            return e, None
        raise ValueError(f"unknown landscape {self.landscape!r}")


def _log(msg, t0):
    print(f"[{time.time() - t0:7.1f}s] {msg}", flush=True)


def match_truth(bset: BasinSet, truth: dict | None, energy) -> list[str]:
    """Name each basin by the nearest designed well (report only)."""
    if truth is None:
        return [f"b{b.id}" for b in bset]
    centers = energy.centers
    names = truth["names"]
    out = []
    for b in bset:
        d = np.linalg.norm(centers - b.rep[None, :], axis=1)
        j = int(np.argmin(d))
        out.append(names[j] if d[j] < 1.0 else f"?{b.id}")
    return out


def run_experiment(cfg: ExperimentConfig, verbose: bool = True) -> dict:
    t0 = time.time()
    log = (lambda m: _log(m, t0)) if verbose else (lambda m: None)
    rng = np.random.default_rng(cfg.seed + 1000)
    energy, truth = cfg.build_energy()
    log(f"energy: {energy.describe()}")

    # ---- Step 1: things -------------------------------------------------
    bset, diag = discover_basins(
        energy,
        cfg.T,
        cfg.dt,
        rng,
        n_traj=cfg.n_traj,
        steps=cfg.steps,
        record_every=cfg.record_every,
        init_scale=cfg.init_scale,
        merge_tol=cfg.merge_tol,
        lump_p=cfg.lump_p,
        lump_lag=cfg.lump_lag,
        min_occupancy=cfg.min_occupancy,
        R_min=cfg.R_min,
        quench_lr=cfg.quench_lr,
        quench_steps=cfg.quench_steps,
        n_esc=cfg.n_esc,
        esc_horizon=cfg.esc_horizon,
        check_every=cfg.check_every,
        relax_steps=cfg.relax_steps,
        max_basins=cfg.max_basins,
        max_esc_horizon=cfg.max_esc_horizon,
    )
    n0 = len(bset)
    log(
        f"discovery: {diag['n_microstates']} microstates -> {diag['n_macro_candidates']} candidates "
        f"-> {n0} metastable (R>={cfg.R_min})"
    )
    for b in bset:
        log(f"  basin {b.id}: E={b.energy:.3f} occ={b.occupancy} tau_relax={b.tau_relax:.3f} "
            f"tau_esc={b.tau_escape:.2f}{'+' if b.escape_censored else ''} R={b.R:.1f}")

    result: dict = {"config": asdict(cfg), "energy": energy.describe(), "discovery": diag}
    if n0 == 0:
        result["score"] = {"composite": 0.0, "reason": "no metastable basins"}
        _save(result, cfg)
        return result

    # ---- Steps 2 + 5: pairwise clicks, products, closure (iterate) ----------
    measured: dict[tuple[int, int], ClickResult] = {}
    rounds = 0
    while True:
        k = len(bset)
        todo = [(i, j) for i in range(k) for j in range(k) if (i, j) not in measured]
        if not todo:
            break
        log(f"round {rounds}: measuring {len(todo)} pairs over {k} basins")
        res = click_tuples(
            bset, todo, cfg.g, cfg.T, cfg.dt, rng,
            n_traj=cfg.n_traj_click, horizon=cfg.horizon, check_every=cfg.click_check_every,
            init_noise=cfg.init_noise, quench_lr=cfg.quench_lr, quench_steps=cfg.quench_steps,
            strain_tol=cfg.strain_tol,
        )
        new = register_products(
            bset, res, cfg.c_thresh, cfg.T, cfg.dt, rng, cfg.R_min,
            cfg.n_esc, cfg.esc_horizon, cfg.check_every, cfg.relax_steps, cfg.max_basins,
        )
        for r in res:
            measured[r.members] = r
        n_def = sum(1 for r in res if r.C >= cfg.c_thresh)
        log(f"  {n_def} clicks defined, {len(new)} new metastable products registered")
        for b in new:
            log(f"  product basin {b.id}: E={b.energy:.3f} tau_relax={b.tau_relax:.3f} "
                f"tau_esc={b.tau_escape:.2f}{'+' if b.escape_censored else ''} R={b.R:.1f}")
        rounds += 1
        if not new or rounds >= cfg.max_rounds:
            break
    k = len(bset)
    # pairs left unmeasured if we hit max_rounds: measure once more without registering
    todo = [(i, j) for i in range(k) for j in range(k) if (i, j) not in measured]
    if todo:
        log(f"final pass: measuring {len(todo)} remaining pairs (no new registrations)")
        res = click_tuples(
            bset, todo, cfg.g, cfg.T, cfg.dt, rng,
            n_traj=cfg.n_traj_click, horizon=cfg.horizon, check_every=cfg.click_check_every,
            init_noise=cfg.init_noise, quench_lr=cfg.quench_lr, quench_steps=cfg.quench_steps,
            strain_tol=cfg.strain_tol,
        )
        for r in res:
            ids = bset.match_minima(r.product_minima)
            known = ids[ids >= 0]
            if r.C >= cfg.c_thresh and len(known) and len(known) >= 0.5 * len(ids):
                r.product_id = int(np.bincount(known).argmax())
                r.product_metastable = True
            measured[r.members] = r
    results = list(measured.values())
    C_pair = pair_matrix(results, k)
    pol = polarization(C_pair)
    surv = np.full((k, k), np.nan)
    for r in results:
        surv[r.members] = r.survival
    strain = strain_summary(results, cfg.T, cfg.strain_tol)
    if strain["n"]:
        log(f"strain/T over {strain['n']} surviving trajectories: quantiles {np.round(strain['quantiles_over_T'], 2).tolist()}; "
            f"{strain['frac_below_tol']:.2f} below tol, {strain['frac_above_5tol']:.2f} above 5x tol")
    table = build_click_table(results, k, cfg.c_thresh)
    clos = closure_score(results, cfg.c_thresh)
    log(f"pairs: bimodality={pol['bimodality']:.3f} near0={pol['frac_near_0']:.2f} "
        f"near1={pol['frac_near_1']:.2f}; closure {clos['n_closed']}/{clos['n_clicks']}")

    # ---- Step 4: ternary clicks ------------------------------------------
    tern = None
    if cfg.do_ternary and k >= 3:
        m = min(k, cfg.max_ternary_basins)
        triples = list(combinations_with_replacement(range(m), 3))
        log(f"ternary: measuring {len(triples)} triples over first {m} basins")
        tres = click_tuples(
            bset, triples, cfg.g, cfg.T, cfg.dt, rng,
            n_traj=max(4, cfg.n_traj_click // 2), horizon=cfg.horizon,
            check_every=cfg.click_check_every, init_noise=cfg.init_noise,
            quench_lr=cfg.quench_lr, quench_steps=cfg.quench_steps, strain_tol=cfg.strain_tol,
        )
        C3 = np.full((k, k, k), np.nan)
        irreducible = []
        tern_clicks = []
        for r in tres:
            a, b, c = r.members
            for p in {(a, b, c), (a, c, b), (b, a, c), (b, c, a), (c, a, b), (c, b, a)}:
                C3[p] = r.C
            if r.C >= cfg.c_thresh and r.n_clicked:
                # products of ternary clicks are matched against known basins (not registered in v0)
                ids = bset.match_minima(r.product_minima)
                known = ids[ids >= 0]
                pid = int(np.bincount(known).argmax()) if len(known) else -1
                tern_clicks.append({"members": [a, b, c], "C": r.C, "product": pid})
            pairs_lo = all(C_pair[x, y] < cfg.c_thresh for x, y in ((a, b), (a, c), (b, c)))
            if r.C >= cfg.c_thresh and pairs_lo and len({a, b, c}) == 3:
                irreducible.append((a, b, c, r.C))
        tern = {
            "n_triples": len(triples),
            "n_defined": int(sum(r.C >= cfg.c_thresh for r in tres)),
            "polarization": polarization(np.array([r.C for r in tres])),
            "strain": strain_summary(tres, cfg.T, cfg.strain_tol),
            "clicks": tern_clicks,
            "irreducible_ternary": irreducible,
            "C3_measured_basins": m,
        }
        log(f"ternary: {tern['n_defined']} defined, {len(irreducible)} irreducible (pairwise-incompatible)")
    else:
        C3 = None

    # ---- Step 3: types ---------------------------------------------------
    V = fingerprints(C_pair)  # pairwise behaviour only: types must be comparable across arities
    labels = type_classes(V, cfg.eps)
    tsum = type_summary(labels)
    log(f"types: {tsum['n_basins']} basins -> {tsum['n_types']} types; classes {tsum['classes']}")

    # ---- Step 5: substitutability + associativity ----------------------------
    subst = substitutability(table, labels)
    assoc = associativity(table, labels)
    # the same checks with *exact* table-level equivalence: what Lean's `decide` will verify
    ex_labels = exact_types(table)
    ex_tsum = type_summary(ex_labels)
    ex_subst = substitutability(table, ex_labels)
    ex_assoc = associativity(table, ex_labels)
    log(f"exact table types: {ex_tsum['n_types']} (classes {ex_tsum['classes']}); "
        f"respects={ex_subst['substitutability']:.3f} weak-assoc={ex_assoc['associativity']:.3f}")
    log(f"substitutability: {subst['substitutability']:.3f} over {subst['n_instances']} instances"
        f"{' (vacuous)' if subst['vacuous'] else ''}")
    log(f"associativity: {assoc['n_behaviourally_equal']}/{assoc['n_both_defined']} both-defined triples "
        f"behaviourally equal ({assoc['n_microscopically_equal']} microscopically); "
        f"{assoc['n_one_sided']} one-sided")

    names = match_truth(bset, truth, energy)
    metast = diag["metastability_score"]
    composite = metast * pol["bimodality"] * pol["nontrivial"] * tsum["compression"] * clos["closure"]
    score = {
        "metastability": metast,
        "polarization": pol["bimodality"] * pol["nontrivial"],
        "type_compression": tsum["compression"],
        "closure": clos["closure"],
        "composite": composite,
        "substitutability": subst["substitutability"],
        "associativity": assoc["associativity"],
        "exact_respects": ex_subst["substitutability"],
        "exact_weak_assoc": ex_assoc["associativity"],
    }
    log(f"composite score C = {composite:.3f}  "
        f"(metastability {metast:.2f} x polarization {score['polarization']:.2f} "
        f"x compression {tsum['compression']:.2f} x closure {clos['closure']:.2f})")

    result.update(
        basins=[{**b.to_dict(), "name": names[b.id], "type": int(labels[b.id])} for b in bset],
        C_pair=np.nan_to_num(C_pair, nan=-1.0).round(4).tolist(),
        survival_pair=np.nan_to_num(surv, nan=-1.0).round(4).tolist(),
        strain=strain,
        product=table.product.tolist(),
        polarization=pol,
        closure=clos,
        ternary=tern,
        types=tsum,
        substitutability=subst,
        associativity={k_: v for k_, v in assoc.items()},
        exact={"types": ex_tsum, "substitutability": ex_subst,
               "associativity": {k_: v for k_, v in ex_assoc.items()},
               "lean_certifiable": bool(ex_subst["substitutability"] == 1.0 and ex_assoc["associativity"] == 1.0)},
        score=score,
        elapsed_s=time.time() - t0,
    )
    _save(result, cfg)
    if cfg.lean_out:
        p = export_lean(table, labels, names, Path(cfg.lean_out))
        log(f"lean export: {p}")
    return result


def _save(result: dict, cfg: ExperimentConfig):
    out = Path(cfg.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out.with_suffix(".json"), "w") as f:
        json.dump(result, f, indent=1, default=_json_default)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, tuple):
        return list(o)
    raise TypeError(str(type(o)))
