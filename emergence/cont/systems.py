"""Continuous controlled dynamical systems  dx/dt = f(x) + u(t)  on R^n.

* ``double_well``: f(x) = x - x^3            (unit test; 1-D)
* ``single_well``: f(x) = -x                 (negative control; 1-D)
* ``random_landscape``: f(x) = -grad V(x) with V a random smooth mixture of
  Gaussian wells plus a weak confining term, in n dimensions.  The number,
  location and depth of its minima are drawn at random and are *not* given to
  the extractor; they are computed afterwards (``minima``) for evaluation only.

Integration is RK4 (deterministic part) plus Euler-Maruyama process noise,
batched over trajectories; the timestep is a parameter that the controls
vary.  Controls are piecewise constant over chunks of duration ``tau``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class System:
    n: int  # state dimension
    m: int  # control dimension (= n here: additive control on every coordinate)

    def f(self, X: np.ndarray) -> np.ndarray:  # (B, n) -> (B, n)
        raise NotImplementedError

    def observe(self, X: np.ndarray) -> np.ndarray:  # (B, n) -> (B, o)
        return X

    def initial_states(self, B: int, rng: np.random.Generator) -> np.ndarray:
        raise NotImplementedError


class DoubleWell(System):
    n = m = 1

    def f(self, X):
        return X - X**3

    def initial_states(self, B, rng):
        return rng.uniform(-2.0, 2.0, (B, 1))


class SingleWell(System):
    n = m = 1

    def f(self, X):
        return -X

    def initial_states(self, B, rng):
        return rng.uniform(-2.0, 2.0, (B, 1))


class RandomLandscape(System):
    """V(x) = (kappa/2)|x|^2 - sum_i a_i exp(-|x - c_i|^2 / (2 s_i^2)).

    Centres are drawn in a ball of radius ``R``, depths ``a_i`` and widths
    ``s_i`` at random.  Wells that are deep and separated relative to their
    widths give distinct minima; shallow or overlapping ones merge.  How many
    survive is not chosen by us."""

    def __init__(self, n=3, n_wells=8, R=3.0, seed=0, kappa=0.3):
        rng = np.random.default_rng(seed)
        self.n = self.m = n
        self.c = rng.standard_normal((n_wells, n))
        self.c *= R * rng.uniform(0.3, 1.0, (n_wells, 1)) / np.linalg.norm(self.c, axis=1, keepdims=True)
        self.a = rng.uniform(1.0, 4.0, n_wells)
        self.s = rng.uniform(0.5, 1.2, n_wells)
        self.kappa = kappa
        self.R = R

    def V(self, X):
        d2 = ((X[:, None, :] - self.c[None]) ** 2).sum(-1)  # (B, W)
        return 0.5 * self.kappa * (X**2).sum(-1) - (self.a[None] * np.exp(-d2 / (2 * self.s[None] ** 2))).sum(-1)

    def f(self, X):
        diff = X[:, None, :] - self.c[None]  # (B, W, n)
        d2 = (diff**2).sum(-1)
        g = self.a[None] * np.exp(-d2 / (2 * self.s[None] ** 2)) / self.s[None] ** 2  # (B, W)
        gradV = self.kappa * X + (g[:, :, None] * diff).sum(1)
        return -gradV

    def initial_states(self, B, rng):
        return rng.uniform(-self.R, self.R, (B, self.n))

    def minima(self, rng, n_start=2000, steps=4000, dt=0.01, tol=0.05):
        """Evaluation only: distinct minima found by descent from random points."""
        X = self.initial_states(n_start, rng)
        for _ in range(steps):
            X = X + dt * self.f(X)
        mins = []
        for x in X:
            if all(np.linalg.norm(x - m) > tol for m in mins):
                mins.append(x)
        return np.array(mins)


def make_system(spec: str, seed: int = 0) -> System:
    if spec == "double_well":
        return DoubleWell()
    if spec == "single_well":
        return SingleWell()
    if spec.startswith("landscape"):
        parts = spec.split(":")
        n = int(parts[1]) if len(parts) > 1 else 3
        W = int(parts[2]) if len(parts) > 2 else 8
        return RandomLandscape(n=n, n_wells=W, seed=seed)
    raise ValueError(spec)


def integrate(sys: System, X0: np.ndarray, U: np.ndarray, tau: float, dt: float, D: float,
              rng: np.random.Generator, sample_times: np.ndarray | None = None):
    """Integrate from X0 (B, n) under piecewise-constant controls U (B, L, m),
    each held for duration ``tau``; process noise intensity ``D``.

    Returns (X_final, Y_samples) where Y_samples (B, K, o) are observations at
    ``sample_times`` (absolute times in [0, L*tau]) if given."""
    B, L, m = U.shape
    X = X0.astype(float).copy()
    n_sub = max(1, int(round(tau / dt)))
    h = tau / n_sub
    samples = []
    if sample_times is not None:
        st = np.asarray(sample_times)
        next_sample = 0
    t = 0.0
    for l in range(L):
        u = U[:, l]
        for _ in range(n_sub):
            k1 = sys.f(X) + u
            k2 = sys.f(X + 0.5 * h * k1) + u
            k3 = sys.f(X + 0.5 * h * k2) + u
            k4 = sys.f(X + h * k3) + u
            X = X + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            if D > 0:
                X = X + np.sqrt(2 * D * h) * rng.standard_normal(X.shape)
            t += h
            if sample_times is not None:
                while next_sample < len(st) and t >= st[next_sample] - 1e-9:
                    samples.append(sys.observe(X).copy())
                    next_sample += 1
    Y = np.stack(samples, axis=1) if samples else None
    return X, Y
