"""Step 1 -- discover "things": metastable basins of the Langevin dynamics.

Two points are in the same basin when they rapidly interconvert relative to
the escape time.  Operationally:

1. Many Langevin trajectories are run and their recorded states quenched to
   local minima (microstates).  Minima closer than ``merge_tol`` are one
   microstate.
2. Microstates that interconvert quickly at the recording lag are lumped
   (union-find on the symmetrised lag-transition frequency).  The lumps are
   candidate basins.
3. For each candidate the relaxation time inside it and the escape time out
   of it are measured; ``R = tau_escape / tau_relax``.  Candidates with
   ``R >= R_min`` are the emergent discrete states.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .dynamics import langevin, quench, relaxation_time
from .energy import Energy


@dataclass
class Basin:
    id: int
    rep: np.ndarray
    energy: float
    minima: np.ndarray  # (m, N) member microstate minima
    occupancy: int
    tau_relax: float = float("nan")
    tau_escape: float = float("nan")
    escape_censored: bool = False
    R: float = float("nan")
    origin: str = "discovered"  # or "product"

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "energy": float(self.energy),
            "n_minima": int(self.minima.shape[0]),
            "occupancy": int(self.occupancy),
            "tau_relax": float(self.tau_relax),
            "tau_escape": float(self.tau_escape),
            "escape_censored": bool(self.escape_censored),
            "R": float(self.R),
            "origin": self.origin,
            "rep_norm": float(np.linalg.norm(self.rep)),
        }


def cluster_minima(mins: np.ndarray, Es: np.ndarray, tol: float):
    """Greedy tolerance clustering.  Returns (labels, centers, center_E)."""
    order = np.argsort(Es)
    centers: list[np.ndarray] = []
    center_E: list[float] = []
    labels = np.empty(len(mins), dtype=int)
    C = np.empty((0, mins.shape[1]))
    for i in order:
        x = mins[i]
        if len(centers):
            d = np.linalg.norm(C - x[None, :], axis=1)
            j = int(np.argmin(d))
            if d[j] < tol:
                labels[i] = j
                continue
        centers.append(x)
        center_E.append(float(Es[i]))
        C = np.stack(centers)
        labels[i] = len(centers) - 1
    return labels, C, np.array(center_E)


class _UnionFind:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, a):
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


class BasinSet:
    """The discovered finite state space, with a classifier X -> basin id."""

    def __init__(self, energy: Energy, merge_tol: float, quench_lr: float, quench_steps: int):
        self.energy = energy
        self.merge_tol = merge_tol
        self.quench_lr = quench_lr
        self.quench_steps = quench_steps
        self.basins: list[Basin] = []
        self._all_minima = np.empty((0, energy.N))
        self._minima_owner = np.empty(0, dtype=int)

    def __len__(self):
        return len(self.basins)

    def __iter__(self):
        return iter(self.basins)

    def __getitem__(self, i):
        return self.basins[i]

    def add(self, basin: Basin) -> Basin:
        basin.id = len(self.basins)
        self.basins.append(basin)
        self._all_minima = np.concatenate([self._all_minima, basin.minima])
        self._minima_owner = np.concatenate([self._minima_owner, np.full(len(basin.minima), basin.id)])
        return basin

    def match_minima(self, mins: np.ndarray) -> np.ndarray:
        """Assign already-quenched minima to basins (-1 if none within tol)."""
        if len(self._all_minima) == 0:
            return np.full(len(mins), -1)
        out = np.full(len(mins), -1, dtype=int)
        for s in range(0, len(mins), 2048):
            blk = mins[s : s + 2048]
            d2 = (
                np.sum(blk * blk, axis=1)[:, None]
                - 2.0 * blk @ self._all_minima.T
                + np.sum(self._all_minima * self._all_minima, axis=1)[None, :]
            )
            j = np.argmin(d2, axis=1)
            ok = np.sqrt(np.maximum(d2[np.arange(len(blk)), j], 0.0)) < self.merge_tol
            out[s : s + 2048] = np.where(ok, self._minima_owner[j], -1)
        return out

    def classify(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Quench states and assign to basins.  Returns (ids, minima)."""
        mins, _ = quench(self.energy, X, lr=self.quench_lr, max_steps=self.quench_steps)
        return self.match_minima(mins), mins

    def reps(self) -> np.ndarray:
        return np.stack([b.rep for b in self.basins])


def measure_timescales(
    bset: BasinSet,
    basins: list[Basin],
    T: float,
    dt: float,
    rng: np.random.Generator,
    n_esc: int,
    esc_horizon: int,
    check_every: int,
    relax_steps: int,
    horizon_mult: float | None = None,
    max_esc_horizon: int = 20000,
):
    """Fill tau_relax, tau_escape, R for the given basins (batched escape run)."""
    if not basins:
        return
    for b in basins:
        b.tau_relax = relaxation_time(bset.energy, b.rep, T, dt, rng, n_traj=32, max_steps=relax_steps)
    # The escape horizon must be long enough that R = tau_esc / tau_relax can
    # actually reach the threshold; otherwise every R is censored at
    # horizon / tau_relax.  Extend it to ``horizon_mult`` relaxation times.
    if horizon_mult is not None:
        need = int(horizon_mult * max(b.tau_relax for b in basins) / dt)
        esc_horizon = min(max(esc_horizon, need), max_esc_horizon)

    reps = np.stack([b.rep for b in basins])
    ids = np.array([b.id for b in basins])
    X = np.repeat(reps, n_esc, axis=0)
    home = np.repeat(ids, n_esc)
    first_exit = np.full(len(X), -1, dtype=int)

    def cb(step, Xc):
        if step % check_every:
            return
        alive = first_exit < 0
        if not alive.any():
            return
        cur, _ = bset.classify(Xc[alive])
        left = cur != home[alive]
        idx = np.flatnonzero(alive)[left]
        first_exit[idx] = step

    langevin(bset.energy, X, esc_horizon, dt, T, rng, callback=cb)
    for k, b in enumerate(basins):
        fe = first_exit[k * n_esc : (k + 1) * n_esc]
        censored = fe < 0
        t = np.where(censored, esc_horizon, fe).astype(float) * dt
        b.tau_escape = float(t.mean())
        b.escape_censored = bool(censored.mean() > 0.5)
        b.R = b.tau_escape / b.tau_relax


def discover_basins(
    energy: Energy,
    T: float,
    dt: float,
    rng: np.random.Generator,
    n_traj: int = 512,
    steps: int = 2000,
    record_every: int = 50,
    burn_frac: float = 0.25,
    init_scale: float = 1.0,
    merge_tol: float = 0.3,
    lump_p: float = 0.2,
    lump_lag: int = 4,
    min_occupancy: int = 5,
    R_min: float = 10.0,
    quench_lr: float = 0.2,
    quench_steps: int = 300,
    n_esc: int = 32,
    esc_horizon: int = 2000,
    check_every: int = 50,
    relax_steps: int = 1000,
    max_basins: int = 24,
    max_esc_horizon: int = 20000,
) -> tuple[BasinSet, dict]:
    """Run the discovery pipeline.  Returns (BasinSet, diagnostics)."""
    N = energy.N
    X0 = rng.standard_normal((n_traj, N)) * init_scale
    _, rec = langevin(energy, X0, steps, dt, T, rng, record_every=record_every)
    n_rec = rec.shape[0]
    burn = int(burn_frac * n_rec)
    pts = rec[burn:]  # (n_rec', n_traj, N)
    nr, nt, _ = pts.shape
    flat = pts.reshape(-1, N)
    mins, Es = quench(energy, flat, lr=quench_lr, max_steps=quench_steps)
    labels, centers, center_E = cluster_minima(mins, Es, merge_tol)
    labels = labels.reshape(nr, nt)
    n_micro = len(centers)

    # Transition counts at lag ``lump_lag`` records.
    lag = max(1, min(lump_lag, nr - 1))
    counts = np.zeros((n_micro, n_micro))
    a = labels[:-lag].ravel()
    b = labels[lag:].ravel()
    np.add.at(counts, (a, b), 1.0)
    occ = np.bincount(labels.ravel(), minlength=n_micro).astype(float)
    occ_lag = np.bincount(a, minlength=n_micro).astype(float)

    # Lump rapidly interconverting microstates: i and j are one metastable
    # state when, starting in one of them, the chance of being found in the
    # other one lag later exceeds ``lump_p`` (fast-exit microstates are
    # absorbed by their destination).
    uf = _UnionFind(n_micro)
    with np.errstate(divide="ignore", invalid="ignore"):
        q = counts / np.maximum(occ_lag, 1.0)[:, None]  # q[i, j] = P(j at t+lag | i at t)
    q = np.maximum(q, q.T)
    ii, jj = np.nonzero(np.triu(q > lump_p, k=1))
    for i, j in zip(ii.tolist(), jj.tolist()):
        uf.union(i, j)
    roots = np.array([uf.find(i) for i in range(n_micro)])
    macro_ids = {r: k for k, r in enumerate(sorted(set(roots.tolist())))}
    macro = np.array([macro_ids[r] for r in roots])
    n_macro = len(macro_ids)

    bset = BasinSet(energy, merge_tol, quench_lr, quench_steps)
    candidates: list[Basin] = []
    for m in range(n_macro):
        members = np.flatnonzero(macro == m)
        occupancy = int(occ[members].sum())
        if occupancy < min_occupancy:
            continue
        best = members[int(np.argmin(center_E[members]))]
        candidates.append(
            Basin(
                id=-1,
                rep=centers[best].copy(),
                energy=float(center_E[best]),
                minima=centers[members].copy(),
                occupancy=occupancy,
            )
        )
    candidates.sort(key=lambda bb: -bb.occupancy)
    candidates = candidates[:max_basins]
    for c in candidates:
        bset.add(c)

    measure_timescales(bset, bset.basins, T, dt, rng, n_esc, esc_horizon, check_every, relax_steps,
                       horizon_mult=2.0 * R_min, max_esc_horizon=max_esc_horizon)

    # Keep the metastable ones; rebuild the set so ids are contiguous.
    kept = [c for c in bset.basins if np.isfinite(c.R) and c.R >= R_min]
    final = BasinSet(energy, merge_tol, quench_lr, quench_steps)
    for c in kept:
        final.add(c)
    diag = {
        "n_recorded_points": int(flat.shape[0]),
        "n_microstates": int(n_micro),
        "n_macro_candidates": int(n_macro),
        "n_candidates_after_occupancy": int(len(candidates)),
        "n_metastable": int(len(kept)),
        "candidate_R": [float(c.R) for c in candidates],
        "candidate_occupancy": [int(c.occupancy) for c in candidates],
        "candidate_censored": [bool(c.escape_censored) for c in candidates],
        "candidate_tau_relax": [float(c.tau_relax) for c in candidates],
        "metastability_score": (len(kept) / len(candidates)) if candidates else 0.0,
    }
    return final, diag
