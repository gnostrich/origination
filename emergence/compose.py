"""Step 5 -- closure, substitutability and associativity of clicks.

A click ``(A, B) ~> D`` is *defined* when ``C(A,B) >= c_thresh`` and the
product of the surviving trajectories lands in a metastable basin ``D``
(discovered or newly registered).  The product is treated exactly like every
other basin: it gets timescales, a fingerprint, a type.

* closure:          fraction of defined clicks whose product is metastable
* substitutability: for A ~ A', is (A,B) defined iff (A',B) defined, and are
                    the products behaviourally equal?
* associativity:    for triples where both (A*B)*C and A*(B*C) are defined,
                    are the results behaviourally equal?
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .basins import Basin, BasinSet, cluster_minima, measure_timescales
from .compat import ClickResult, click_tuples


@dataclass
class ClickTable:
    """Finite partial binary operation on basin ids."""

    k: int
    C: np.ndarray  # (k,k) probabilities
    product: np.ndarray  # (k,k) int, -1 if undefined
    c_thresh: float

    def defined(self, i, j) -> bool:
        return self.product[i, j] >= 0

    def click(self, i, j) -> int:
        return int(self.product[i, j])


def register_products(
    bset: BasinSet,
    results: list[ClickResult],
    c_thresh: float,
    T: float,
    dt: float,
    rng: np.random.Generator,
    R_min: float,
    n_esc: int,
    esc_horizon: int,
    check_every: int,
    relax_steps: int,
    max_basins: int,
) -> list[Basin]:
    """Assign product ids to clicks; register new metastable products.

    Returns the list of newly added basins.
    """
    new_basins: list[Basin] = []
    pending: list[tuple[ClickResult, np.ndarray, float, int]] = []
    for r in results:
        if r.C < c_thresh or r.n_clicked == 0:
            continue
        ids = bset.match_minima(r.product_minima)
        known = ids[ids >= 0]
        if len(known) and len(known) >= 0.5 * len(ids):
            pid = int(np.bincount(known).argmax())
            r.product_id = pid
            r.product_metastable = True
            continue
        # Majority of products are unknown: cluster them and take the biggest.
        unk = r.product_minima[ids < 0]
        unkE = r.product_energy[ids < 0]
        labels, centers, cE = cluster_minima(unk, unkE, bset.merge_tol)
        big = int(np.bincount(labels).argmax())
        pending.append((r, centers[big], float(cE[big]), int((labels == big).sum())))

    # Merge pending candidates among themselves (several clicks may share a product).
    cand: list[Basin] = []
    for r, c, e, n in pending:
        hit = None
        for cb in cand:
            if np.linalg.norm(cb.rep - c) < bset.merge_tol:
                hit = cb
                break
        if hit is None:
            if len(bset) + len(cand) >= max_basins:
                continue
            hit = Basin(id=-1, rep=c, energy=e, minima=c[None, :].copy(), occupancy=n, origin="product")
            cand.append(hit)
        else:
            hit.occupancy += n
        r._pending = hit  # type: ignore[attr-defined]

    if cand:
        tmp = BasinSet(bset.energy, bset.merge_tol, bset.quench_lr, bset.quench_steps)
        # classification during escape must know about all basins
        for b in bset.basins:
            tmp.add(Basin(**{**b.__dict__}))
        for cb in cand:
            tmp.add(cb)
        measure_timescales(tmp, cand, T, dt, rng, n_esc, esc_horizon, check_every, relax_steps)
        for cb in cand:
            if np.isfinite(cb.R) and cb.R >= R_min:
                bset.add(Basin(**{**cb.__dict__}))
                new_basins.append(bset.basins[-1])
                cb.id = bset.basins[-1].id
            else:
                cb.id = -2  # not metastable

    for r, *_ in pending:
        hit = getattr(r, "_pending", None)
        if hit is None:
            continue
        if hit.id >= 0:
            r.product_id = hit.id
            r.product_metastable = True
        else:
            r.product_id = -1
            r.product_metastable = False
    return new_basins


def build_click_table(results: list[ClickResult], k: int, c_thresh: float) -> ClickTable:
    C = np.full((k, k), np.nan)
    P = np.full((k, k), -1, dtype=int)
    for r in results:
        i, j = r.members
        C[i, j] = r.C
        if r.C >= c_thresh and r.product_metastable:
            P[i, j] = r.product_id
    return ClickTable(k=k, C=C, product=P, c_thresh=c_thresh)


def closure_score(results: list[ClickResult], c_thresh: float) -> dict:
    defined = [r for r in results if r.C >= c_thresh and r.n_clicked > 0]
    closed = [r for r in defined if r.product_metastable]
    return {
        "n_clicks": len(defined),
        "n_closed": len(closed),
        "closure": (len(closed) / len(defined)) if defined else 0.0,
    }


def exact_types(table: ClickTable) -> np.ndarray:
    """Exact behavioural equivalence of the *thresholded* table: A ~ A' iff for
    every C, (A,C) is defined iff (A',C) is, and (C,A) iff (C,A').  This is
    literally `Interaction.BehEq` of `lean/Emergence/Basic.lean`; the
    epsilon-types of `types.py` are its noise-tolerant relaxation."""
    k = table.k
    D = table.product >= 0
    sig = [tuple(D[i].tolist()) + tuple(D[:, i].tolist()) for i in range(k)]
    seen: dict[tuple, int] = {}
    out = np.empty(k, dtype=int)
    for i, s in enumerate(sig):
        if s not in seen:
            seen[s] = len(seen)
        out[i] = seen[s]
    return out


def substitutability(table: ClickTable, types: np.ndarray) -> dict:
    """Does clicking respect behavioural equivalence?  (Lean: `Respects`)"""
    k = table.k
    n_inst = n_def_ok = n_prod_ok = 0
    for a in range(k):
        for a2 in range(k):
            if a2 == a or types[a] != types[a2]:
                continue
            for b in range(k):
                da, da2 = table.defined(a, b), table.defined(a2, b)
                if not (da or da2):
                    continue
                n_inst += 1
                if da == da2:
                    n_def_ok += 1
                    if types[table.click(a, b)] == types[table.click(a2, b)]:
                        n_prod_ok += 1
    return {
        "n_instances": n_inst,
        "definedness_respected": (n_def_ok / n_inst) if n_inst else 1.0,
        "products_respected": (n_prod_ok / n_inst) if n_inst else 1.0,
        "substitutability": (n_prod_ok / n_inst) if n_inst else 1.0,
        "vacuous": n_inst == 0,
    }


def associativity(table: ClickTable, types: np.ndarray) -> dict:
    """(A*B)*C ~ A*(B*C) up to behavioural equivalence."""
    k = table.k
    n_both = n_eq = n_one = n_micro_eq = 0
    witnesses = []
    for a in range(k):
        for b in range(k):
            for c in range(k):
                lhs = rhs = -1
                if table.defined(a, b):
                    ab = table.click(a, b)
                    if table.defined(ab, c):
                        lhs = table.click(ab, c)
                if table.defined(b, c):
                    bc = table.click(b, c)
                    if table.defined(a, bc):
                        rhs = table.click(a, bc)
                if lhs >= 0 and rhs >= 0:
                    n_both += 1
                    if types[lhs] == types[rhs]:
                        n_eq += 1
                        if len(witnesses) < 8:
                            witnesses.append((a, b, c, lhs, rhs))
                    if lhs == rhs:
                        n_micro_eq += 1
                elif lhs >= 0 or rhs >= 0:
                    n_one += 1
    return {
        "n_both_defined": n_both,
        "n_one_sided": n_one,
        "n_behaviourally_equal": n_eq,
        "n_microscopically_equal": n_micro_eq,
        "associativity": (n_eq / n_both) if n_both else 0.0,
        "definedness_agreement": (n_both / (n_both + n_one)) if (n_both + n_one) else 0.0,
        "witnesses": witnesses,
    }
