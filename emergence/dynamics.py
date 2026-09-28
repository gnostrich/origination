"""Overdamped Langevin dynamics and zero-temperature quenches.

    dx = -grad E(x) dt + sqrt(2 T) dW

Euler-Maruyama, batched over trajectories.  ``quench`` is plain gradient
descent with early stopping and is used to assign a point to the local
minimum it descends to.
"""

from __future__ import annotations

import numpy as np

from .energy import Energy


def langevin(
    energy: Energy,
    X0: np.ndarray,
    steps: int,
    dt: float,
    T: float,
    rng: np.random.Generator,
    record_every: int | None = None,
    callback=None,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Run ``steps`` Euler-Maruyama steps from ``X0`` (B,N).

    Returns the final states.  If ``record_every`` is set, also returns an
    array ``(n_rec, B, N)`` of recorded states (the final state is always the
    last record).  ``callback(step, X)`` may be used for online checks.
    """
    X = np.array(X0, dtype=float, copy=True)
    noise_scale = np.sqrt(2.0 * T * dt)
    records = []
    for s in range(1, steps + 1):
        g = energy.grad(X)
        X -= dt * g
        if T > 0:
            X += noise_scale * rng.standard_normal(X.shape)
        if record_every and (s % record_every == 0 or s == steps):
            records.append(X.copy())
        if callback is not None:
            callback(s, X)
    if record_every:
        return X, np.stack(records)
    return X


def quench(
    energy: Energy,
    X0: np.ndarray,
    lr: float = 0.2,
    max_steps: int = 400,
    tol: float = 1e-4,
) -> tuple[np.ndarray, np.ndarray]:
    """Gradient descent to a local minimum.  Returns (X_min, E_min).

    A simple backtracking rule halves the step of any trajectory whose energy
    increased, so the quench is robust on rough MLP landscapes.
    """
    X = np.array(X0, dtype=float, copy=True)
    E, g = energy.value_and_grad(X)
    step = np.full(X.shape[0], lr)
    for _ in range(max_steps):
        gnorm = np.linalg.norm(g, axis=1)
        act = np.flatnonzero((gnorm > tol) & (step > 1e-6))
        if len(act) == 0:
            break
        # Only converged-or-not rows are evaluated: big saving on rough landscapes.
        Xn = X[act] - step[act, None] * g[act]
        En, gn = energy.value_and_grad(Xn)
        worse = En > E[act]
        acc = act[~worse]
        X[acc] = Xn[~worse]
        E[acc] = En[~worse]
        g[acc] = gn[~worse]
        step[act[worse]] *= 0.5
        step[acc] = np.minimum(step[acc] * 1.1, lr)
    return X, E


def relaxation_time(
    energy: Energy,
    rep: np.ndarray,
    T: float,
    dt: float,
    rng: np.random.Generator,
    n_traj: int = 64,
    max_steps: int = 2000,
) -> float:
    """Empirical relaxation time (in units of t = steps*dt) inside a basin.

    Trajectories start at the representative, so the mean squared distance to
    it rises from 0 to its thermal plateau.  The relaxation time is where it
    first reaches ``1 - 1/e`` of the plateau (plateau estimated from the
    last 20% of the run).
    """
    X = np.repeat(rep[None, :], n_traj, axis=0)
    _, rec = langevin(energy, X, max_steps, dt, T, rng, record_every=max(1, max_steps // 200))
    d2 = np.mean(np.sum((rec - rep[None, None, :]) ** 2, axis=2), axis=1)  # (n_rec,)
    plateau = d2[int(0.8 * len(d2)) :].mean()
    if plateau <= 0:
        return dt
    target = (1.0 - np.exp(-1.0)) * plateau
    idx = int(np.argmax(d2 >= target))
    rec_every = max(1, max_steps // 200)
    return max(dt, (idx + 1) * rec_every * dt)
