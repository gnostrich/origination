"""Energy functions E_theta : R^N -> R with analytic gradients.

Two families:

* ``MLPEnergy``  -- a small randomly initialised tanh MLP plus a weak
  quadratic confinement.  No structure is supplied (Experiment B).
* ``WellEnergy`` -- a soft-min of Gaussian wells at chosen centres.  Used to
  hand-design a landscape whose basins are known to click and compose
  (Experiment A), so the measurement machinery can be validated.

Everything is batched: ``value(X)`` and ``grad(X)`` take ``X`` of shape
``(batch, N)``.
"""

from __future__ import annotations

import numpy as np


class Energy:
    """Abstract batched energy on R^N."""

    N: int

    def value(self, X: np.ndarray) -> np.ndarray:  # (B,N) -> (B,)
        raise NotImplementedError

    def grad(self, X: np.ndarray) -> np.ndarray:  # (B,N) -> (B,N)
        raise NotImplementedError

    def value_and_grad(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self.value(X), self.grad(X)

    def describe(self) -> dict:
        return {"kind": type(self).__name__, "N": self.N}


class MLPEnergy(Energy):
    """E(x) = scale * MLP(x) + (confine/2) |x|^2.

    ``MLP`` is ``depth`` tanh layers of width ``hidden`` followed by a linear
    read-out.  Weights are drawn once from a seed; nothing about basins,
    types or interactions is encoded.
    """

    def __init__(
        self,
        N: int = 64,
        hidden: int = 128,
        depth: int = 2,
        seed: int = 0,
        scale: float = 4.0,
        confine: float = 0.25,
        input_gain: float = 1.0,
    ):
        self.N = N
        self.hidden = hidden
        self.depth = depth
        self.seed = seed
        self.scale = scale
        self.confine = confine
        self.input_gain = input_gain
        rng = np.random.default_rng(seed)
        self.Ws: list[np.ndarray] = []
        self.bs: list[np.ndarray] = []
        d_in = N
        for _ in range(depth):
            # Orthogonal-ish random features with gain so that pre-activations
            # are O(1) for |x| ~ O(1).
            W = rng.standard_normal((d_in, hidden)) / np.sqrt(d_in)
            if d_in == N:
                W *= input_gain
            self.Ws.append(W)
            self.bs.append(rng.standard_normal(hidden) * 0.5)
            d_in = hidden
        self.w_out = rng.standard_normal(hidden) / np.sqrt(hidden)
        self.b_out = 0.0

    def _forward(self, X):
        acts = [X]
        h = X
        for W, b in zip(self.Ws, self.bs):
            h = np.tanh(h @ W + b)
            acts.append(h)
        out = acts[-1] @ self.w_out + self.b_out
        return out, acts

    def value(self, X):
        out, _ = self._forward(X)
        return self.scale * out + 0.5 * self.confine * np.sum(X * X, axis=1)

    def value_and_grad(self, X):
        out, acts = self._forward(X)
        E = self.scale * out + 0.5 * self.confine * np.sum(X * X, axis=1)
        # backprop
        g = np.broadcast_to(self.w_out, acts[-1].shape) * self.scale  # dE/dh_last
        for layer in range(self.depth - 1, -1, -1):
            h = acts[layer + 1]
            g = g * (1.0 - h * h)  # through tanh
            g = g @ self.Ws[layer].T
        g = g + self.confine * X
        return E, g

    def grad(self, X):
        return self.value_and_grad(X)[1]

    def describe(self):
        d = super().describe()
        d.update(
            hidden=self.hidden,
            depth=self.depth,
            seed=self.seed,
            scale=self.scale,
            confine=self.confine,
            input_gain=self.input_gain,
        )
        return d


class WellEnergy(Energy):
    """E(x) = -tau * log sum_i w_i exp(-|x-p_i|^2 / (2 sigma^2)) + (confine/2)|x|^2.

    With ``tau = sigma^2`` every well has unit curvature at its bottom, and the
    energy grows like ``min_i |x-p_i|^2 / 2`` away from the wells, so every
    trajectory descends to some well.  The barrier between wells at distance
    ``d`` is roughly ``d^2/8``.
    """

    def __init__(
        self,
        centers: np.ndarray,
        sigma: float = 1.0,
        tau: float | None = None,
        confine: float = 0.0,
        weights: np.ndarray | None = None,
    ):
        self.centers = np.asarray(centers, dtype=float)
        self.N = self.centers.shape[1]
        self.sigma = sigma
        self.tau = sigma * sigma if tau is None else tau
        self.confine = confine
        k = self.centers.shape[0]
        self.logw = np.zeros(k) if weights is None else np.log(np.asarray(weights, float))

    def _logits(self, X):
        # (B, k) : -|x-p_i|^2 / (2 sigma^2) + log w_i
        d2 = (
            np.sum(X * X, axis=1, keepdims=True)
            - 2.0 * X @ self.centers.T
            + np.sum(self.centers * self.centers, axis=1)[None, :]
        )
        return -d2 / (2.0 * self.sigma**2) + self.logw[None, :]

    def value(self, X):
        L = self._logits(X)
        m = L.max(axis=1, keepdims=True)
        lse = m[:, 0] + np.log(np.sum(np.exp(L - m), axis=1))
        return -self.tau * lse + 0.5 * self.confine * np.sum(X * X, axis=1)

    def value_and_grad(self, X):
        L = self._logits(X)
        m = L.max(axis=1, keepdims=True)
        P = np.exp(L - m)
        Z = P.sum(axis=1, keepdims=True)
        E = -self.tau * (m[:, 0] + np.log(Z[:, 0])) + 0.5 * self.confine * np.sum(X * X, axis=1)
        R = P / Z  # responsibilities (B,k)
        # d/dx of -tau * lse = tau/sigma^2 * sum_i R_i (x - p_i)
        g = (self.tau / self.sigma**2) * (X - R @ self.centers) + self.confine * X
        return E, g

    def grad(self, X):
        return self.value_and_grad(X)[1]

    def describe(self):
        d = super().describe()
        d.update(n_wells=int(self.centers.shape[0]), sigma=self.sigma, tau=self.tau, confine=self.confine)
        return d


# ---------------------------------------------------------------------------
# Experiment A: a hand-designed landscape with known compositional structure.
# ---------------------------------------------------------------------------


def superpose(parts: np.ndarray) -> np.ndarray:
    """Superposition of k parts (k, N) -> (N): sum / sqrt(k).

    This is the *one* structural assumption of the whole experiment: when
    several things occupy the substrate together, the composite configuration
    is their normalised sum, and it is judged by the same energy E.
    """
    parts = np.asarray(parts)
    return parts.sum(axis=0) / np.sqrt(parts.shape[0])


def designed_landscape(N: int = 64, radius: float = 4.0, sigma: float = 0.5, seed: int = 0):
    """Build a WellEnergy whose wells realise a known interaction structure.

    Primitive things live on orthonormal directions u_1..u_8 at distance
    ``radius`` from the origin.  Product wells are placed exactly at the
    superposition of the parts that should click, so the joint state of a
    compatible pair is a genuine joint minimum, while incompatible pairs have
    a superposition far from any well.

    Ground truth (names are for the report only; the pipeline never sees them):

    * A1, A2 -- two microscopically different things with identical behaviour
      (both click with B and nothing else): should become one *type*.
    * B, C   -- distinct things.
    * D1 = A1*B, D2 = A2*B   -- products, both click with C  (one type).
    * Dp = B*C               -- clicks with A1, A2.
    * E1 = D1*C, E2 = D2*C, Ep1 = A1*Dp, Ep2 = A2*Dp -- terminal products.
      Associativity: (A1*B)*C = E1 ~ Ep1 = A1*(B*C) *behaviourally* (all
      terminal), although E1 != Ep1 microscopically.
    * P, Q, R -- pairwise incompatible, but P*Q*R = S is a well: irreducible
      ternary click.
    * J -- junk: clicks with nothing.
    """
    rng = np.random.default_rng(seed)
    # Random orthonormal frame.
    M = rng.standard_normal((N, N))
    Qm, _ = np.linalg.qr(M)
    u = lambda i: radius * Qm[:, i]

    prim = {
        "A1": u(0),
        "A2": u(1),
        "B": u(2),
        "C": u(3),
        "P": u(4),
        "Q": u(5),
        "R": u(6),
        "J": u(7),
    }
    wells = dict(prim)
    wells["D1"] = superpose([wells["A1"], wells["B"]])
    wells["D2"] = superpose([wells["A2"], wells["B"]])
    wells["Dp"] = superpose([wells["B"], wells["C"]])
    wells["E1"] = superpose([wells["D1"], wells["C"]])
    wells["E2"] = superpose([wells["D2"], wells["C"]])
    wells["Ep1"] = superpose([wells["A1"], wells["Dp"]])
    wells["Ep2"] = superpose([wells["A2"], wells["Dp"]])
    wells["S"] = superpose([wells["P"], wells["Q"], wells["R"]])

    names = list(wells.keys())
    centers = np.stack([wells[n] for n in names])
    energy = WellEnergy(centers, sigma=sigma, confine=0.0)
    truth = {
        "names": names,
        "types": {
            "A": ["A1", "A2"],
            "B": ["B"],
            "C": ["C"],
            "D": ["D1", "D2"],
            "Dp": ["Dp"],
            "terminal": ["E1", "E2", "Ep1", "Ep2", "J", "S"],
            "P": ["P"],
            "Q": ["Q"],
            "R": ["R"],
        },
        "pair_clicks": {
            ("A1", "B"): "D1",
            ("A2", "B"): "D2",
            ("B", "C"): "Dp",
            ("D1", "C"): "E1",
            ("D2", "C"): "E2",
            ("A1", "Dp"): "Ep1",
            ("A2", "Dp"): "Ep2",
        },
        "ternary_click": (("P", "Q", "R"), "S"),
    }
    return energy, truth
