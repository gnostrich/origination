"""Joint dynamics of several things occupying the substrate together.

The single structural assumption of the experiment: when ``k`` parts
``x_1..x_k`` (each in R^N) interact, the joint energy is

    E_k(x_1..x_k) = sum_i E(x_i) + g * E( (x_1 + ... + x_k) / sqrt(k) )

i.e. the composite configuration is the normalised superposition of the
parts and it is judged by the *same* energy that judges the parts.  No
notion of port, arity, edge or type is introduced: ``g`` is the interaction
strength and ``k`` is just how many things were put into the box.

The ``1/sqrt(k)`` keeps the superposition on the same scale as the parts
when they are near-orthogonal (which random basin representatives in high
dimension are), and makes the self-pair ``(A, A)`` non-trivial (its
superposition is ``sqrt(2) A``, not ``A``).
"""

from __future__ import annotations

import numpy as np

from .energy import Energy


class JointEnergy(Energy):
    def __init__(self, base: Energy, k: int, g: float):
        self.base = base
        self.k = k
        self.g = g
        self.N = base.N * k

    def parts(self, X: np.ndarray) -> np.ndarray:
        """(B, kN) -> (B, k, N)"""
        return X.reshape(X.shape[0], self.k, self.base.N)

    def superposition(self, X: np.ndarray) -> np.ndarray:
        return self.parts(X).sum(axis=1) / np.sqrt(self.k)

    def value(self, X):
        P = self.parts(X)
        flat = P.reshape(-1, self.base.N)
        Ep = self.base.value(flat).reshape(X.shape[0], self.k).sum(axis=1)
        return Ep + self.g * self.base.value(self.superposition(X))

    def value_and_grad(self, X):
        B = X.shape[0]
        P = self.parts(X)
        flat = P.reshape(-1, self.base.N)
        Ep, gp = self.base.value_and_grad(flat)
        Ep = Ep.reshape(B, self.k).sum(axis=1)
        gp = gp.reshape(B, self.k, self.base.N)
        S = P.sum(axis=1) / np.sqrt(self.k)
        Es, gs = self.base.value_and_grad(S)
        g = gp + (self.g / np.sqrt(self.k)) * gs[:, None, :]
        return Ep + self.g * Es, g.reshape(B, self.N)

    def grad(self, X):
        return self.value_and_grad(X)[1]
