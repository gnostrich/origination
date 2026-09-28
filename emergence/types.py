"""Step 3 -- types emerge from interchangeability.

Each basin gets a compatibility fingerprint (its row and column of the pair
matrix, and optionally its slices of the ternary tensor).  Two basins are
behaviourally equivalent when no partner tells them apart:

    A ~_eps A'   iff   max_B |C(A,B) - C(A',B)| < eps   (and same for columns)

The max-norm is used rather than the Euclidean norm because the relevant
question is whether *some* partner distinguishes them.  Equivalence classes
are computed by complete-linkage clustering so that every pair inside a
class is within eps.
"""

from __future__ import annotations

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist


def fingerprints(C_pair: np.ndarray, C_tern: np.ndarray | None = None) -> np.ndarray:
    k = C_pair.shape[0]
    C = np.nan_to_num(C_pair, nan=0.0)
    feats = [C, C.T]
    if C_tern is not None:
        Ct = np.nan_to_num(C_tern, nan=0.0)
        feats.append(Ct.reshape(k, -1))
    return np.concatenate(feats, axis=1)


def type_classes(V: np.ndarray, eps: float = 0.5) -> np.ndarray:
    """Return an integer type label per basin (labels 0..n_types-1)."""
    k = V.shape[0]
    if k == 1:
        return np.zeros(1, dtype=int)
    D = pdist(V, metric="chebyshev")
    Z = linkage(D, method="complete")
    labels = fcluster(Z, t=eps, criterion="distance") - 1
    # relabel in order of first appearance
    seen = {}
    out = np.empty(k, dtype=int)
    for i, l in enumerate(labels):
        if l not in seen:
            seen[l] = len(seen)
        out[i] = seen[l]
    return out


def type_summary(labels: np.ndarray) -> dict:
    k = len(labels)
    n_types = int(labels.max()) + 1 if k else 0
    classes = [sorted(np.flatnonzero(labels == t).tolist()) for t in range(n_types)]
    return {
        "n_basins": int(k),
        "n_types": n_types,
        "compression": (1.0 - n_types / k) if k else 0.0,
        "classes": classes,
        "n_nontrivial_classes": int(sum(len(c) > 1 for c in classes)),
    }
