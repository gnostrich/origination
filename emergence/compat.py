"""Step 2 / Step 4 -- discover "clicking" of arbitrary arity.

For a tuple of basins ``(A_1..A_k)`` the parts are initialised at their
representatives (plus a small kick), the joint dynamics are run for a
horizon much longer than the relaxation time, and

    C(A_1..A_k) = P( every part is still classified as its own basin at
                     every checkpoint up to the horizon )

i.e. the probability that the tuple forms a long-lived joint metastable
state.  For the surviving trajectories the joint configuration is quenched
under the joint energy and its superposition is quenched under ``E``: that
minimum is the *product* of the click, a candidate new thing.

Survival alone is not enough.  Because the coupling is a smooth energy, an
incompatible pair usually still has a joint local minimum in which the parts
are *strained* (pulled off their wells so that their superposition lands
near some well).  The measured quantity that separates a genuine click from
such a strained contact is the **strain energy** of the joint minimum,

    strain = sum_i [E(x_i*) - E(quench(x_i*))] + g [E(s*) - E(quench(s*))]  >= 0,

zero exactly when every part sits at its own minimum and the superposition
is itself a minimum.  A trajectory counts as a click when it survives and its
strain is below ``strain_tol * T``.  The raw strain distribution is kept so
that polarisation can be judged before any threshold is applied.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .basins import BasinSet, cluster_minima
from .coupling import JointEnergy
from .dynamics import langevin, quench


@dataclass
class ClickResult:
    members: tuple[int, ...]
    C: float
    n_traj: int
    n_survived: int
    product_minima: np.ndarray  # (n_clicked, N) quenched superpositions of clicked trajectories
    product_energy: np.ndarray
    strain: np.ndarray = field(default_factory=lambda: np.empty(0))  # per surviving trajectory
    n_clicked: int = 0
    product_id: int = -1  # filled in by registration
    product_metastable: bool = False

    @property
    def survival(self) -> float:
        return self.n_survived / self.n_traj

    @property
    def mean_strain(self) -> float:
        return float(self.strain.mean()) if self.strain.size else float("nan")


def click_tuples(
    bset: BasinSet,
    tuples: list[tuple[int, ...]],
    g: float,
    T: float,
    dt: float,
    rng: np.random.Generator,
    n_traj: int = 16,
    horizon: int = 1000,
    check_every: int = 50,
    init_noise: float = 0.05,
    quench_lr: float = 0.2,
    quench_steps: int = 300,
    strain_tol: float = 5.0,
) -> list[ClickResult]:
    """Measure C for a list of tuples of equal arity (batched)."""
    if not tuples:
        return []
    k = len(tuples[0])
    assert all(len(t) == k for t in tuples)
    N = bset.energy.N
    joint = JointEnergy(bset.energy, k, g)
    reps = bset.reps()
    n_tup = len(tuples)
    members = np.array(tuples)  # (n_tup, k)
    X0 = reps[members].reshape(n_tup, k * N)
    X0 = np.repeat(X0, n_traj, axis=0)
    X0 = X0 + init_noise * rng.standard_normal(X0.shape)
    home = np.repeat(members, n_traj, axis=0)  # (B, k)
    B = X0.shape[0]
    alive = np.ones(B, dtype=bool)

    def cb(step, X):
        if step % check_every:
            return
        idx = np.flatnonzero(alive)
        if len(idx) == 0:
            return
        parts = X[idx].reshape(-1, N)
        ids, _ = bset.classify(parts)
        ids = ids.reshape(len(idx), k)
        dead = (ids != home[idx]).any(axis=1)
        alive[idx[dead]] = False

    Xf = langevin(joint, X0, horizon, dt, T, rng, callback=cb)

    # Survivors: quench jointly, measure strain, quench the superposition.
    results: list[ClickResult] = []
    surv_idx = np.flatnonzero(alive)
    delta = strain_tol * T
    if len(surv_idx):
        Xj, _ = quench(joint, Xf[surv_idx], lr=quench_lr / k, max_steps=quench_steps)
        parts = joint.parts(Xj).reshape(-1, N)
        Ep = bset.energy.value(parts)
        _, Ep0 = quench(bset.energy, parts, lr=quench_lr, max_steps=quench_steps)
        part_strain = np.maximum(Ep - Ep0, 0.0).reshape(len(surv_idx), k).sum(axis=1)
        S = joint.superposition(Xj)
        Es = bset.energy.value(S)
        Pm, PE = quench(bset.energy, S, lr=quench_lr, max_steps=quench_steps)
        strain_all = part_strain + g * np.maximum(Es - PE, 0.0)
    pos = {int(i): n for n, i in enumerate(surv_idx)}
    for t_i, tup in enumerate(tuples):
        rows = np.arange(t_i * n_traj, (t_i + 1) * n_traj)
        srows = np.array([pos[int(r)] for r in rows if alive[r]], dtype=int)
        n_s = len(srows)
        st = strain_all[srows] if n_s else np.empty(0)
        ok = srows[st <= delta] if n_s else srows
        results.append(
            ClickResult(
                members=tuple(int(m) for m in tup),
                C=len(ok) / n_traj,
                n_traj=n_traj,
                n_survived=n_s,
                product_minima=Pm[ok] if len(ok) else np.empty((0, N)),
                product_energy=PE[ok] if len(ok) else np.empty(0),
                strain=st,
                n_clicked=int(len(ok)),
            )
        )
    return results


def pair_matrix(results: list[ClickResult], k: int) -> np.ndarray:
    C = np.full((k, k), np.nan)
    for r in results:
        i, j = r.members
        C[i, j] = r.C
    return C


def strain_summary(results: list[ClickResult], T: float, strain_tol: float) -> dict:
    """Distribution of strain over all surviving trajectories, in units of T."""
    st = np.concatenate([r.strain for r in results]) if results else np.empty(0)
    if st.size == 0:
        return {"n": 0}
    u = st / T
    return {
        "n": int(st.size),
        "frac_below_tol": float(np.mean(u <= strain_tol)),
        "frac_above_5tol": float(np.mean(u > 5 * strain_tol)),
        "median_over_T": float(np.median(u)),
        "quantiles_over_T": [float(x) for x in np.quantile(u, [0.05, 0.25, 0.5, 0.75, 0.95])],
        "per_pair_mean_over_T": [round(r.mean_strain / T, 3) if r.strain.size else None for r in results],
    }


def polarization(C: np.ndarray) -> dict:
    """How close the compatibility probabilities are to {0, 1}."""
    v = C[np.isfinite(C)].ravel()
    if v.size == 0:
        return {"bimodality": 0.0, "frac_near_0": 0.0, "frac_near_1": 0.0, "nontrivial": 0.0, "mean": float("nan")}
    bim = 1.0 - 2.0 * float(np.mean(np.minimum(v, 1.0 - v)))
    f0 = float(np.mean(v <= 0.2))
    f1 = float(np.mean(v >= 0.8))
    return {
        "bimodality": bim,
        "frac_near_0": f0,
        "frac_near_1": f1,
        "nontrivial": 1.0 if (f0 > 0 and f1 > 0) else 0.0,
        "mean": float(v.mean()),
    }
