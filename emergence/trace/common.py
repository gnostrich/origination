"""Shared measurements for backward tracing of final interfaces.

Everything here is coordinate-invariant in the representation (probes,
between/within subspaces, CKA, effective rank) or refers only to the frozen
final partition Π of a fixed instance pool.
"""

from __future__ import annotations

import numpy as np
import torch


# ------------------------------------------------------------- identity
def linear_probe(Xf, yf, Xe, ye, n_classes, steps=300, lr=0.05, wd=1e-4, seed=0):
    """Multinomial logistic probe fitted on (Xf, yf), accuracy on (Xe, ye)."""
    torch.manual_seed(seed)
    mu, sd = Xf.mean(0, keepdims=True), Xf.std(0, keepdims=True) + 1e-6
    Xf_t = torch.as_tensor((Xf - mu) / sd, dtype=torch.float32)
    Xe_t = torch.as_tensor((Xe - mu) / sd, dtype=torch.float32)
    yf_t = torch.as_tensor(yf, dtype=torch.long)
    lin = torch.nn.Linear(Xf.shape[1], n_classes)
    opt = torch.optim.Adam(lin.parameters(), lr=lr, weight_decay=wd)
    for _ in range(steps):
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(lin(Xf_t), yf_t)
        loss.backward()
        opt.step()
    with torch.no_grad():
        acc = float((lin(Xe_t).argmax(1).numpy() == ye).mean())
        acc_fit = float((lin(Xf_t).argmax(1).numpy() == yf).mean())
    return acc, acc_fit


def knn_purity(Xf, yf, Xe, ye):
    """1-nearest-neighbour (cosine) class agreement of eval instances with fit instances."""
    A = Xf / (np.linalg.norm(Xf, axis=1, keepdims=True) + 1e-9)
    B = Xe / (np.linalg.norm(Xe, axis=1, keepdims=True) + 1e-9)
    nn_ = (B @ A.T).argmax(1)
    return float((yf[nn_] == ye).mean())


def class_means(X, y, n_classes):
    M = np.zeros((n_classes, X.shape[1]))
    cnt = np.zeros(n_classes)
    np.add.at(M, y, X)
    np.add.at(cnt, y, 1)
    ok = cnt > 0
    M[ok] /= cnt[ok, None]
    return M, ok


def between_projector(X, y, n_classes):
    """Orthonormal basis of the between-class subspace (class means − grand mean)."""
    M, ok = class_means(X, y, n_classes)
    mu = X.mean(0)
    D = M[ok] - mu
    U, s, Vt = np.linalg.svd(D, full_matrices=False)
    keep = s > 1e-8 * max(s.max(), 1e-12)
    return Vt[keep], mu, M, ok


def geometry(X, y, n_classes):
    """Between/within variance ratio and nearest-centroid margin."""
    M, ok = class_means(X, y, n_classes)
    mu = X.mean(0)
    within = np.mean(np.sum((X - M[y]) ** 2, axis=1))
    cnt = np.bincount(y, minlength=n_classes)
    between = np.sum(cnt[ok] * np.sum((M[ok] - mu) ** 2, axis=1)) / len(X)
    d_own = np.linalg.norm(X - M[y], axis=1)
    Mo = M[ok]
    sq = (X * X).sum(1)[:, None] + (Mo * Mo).sum(1)[None] - 2 * X @ Mo.T
    D = np.sqrt(np.maximum(sq, 0))
    d_own_s, ys = d_own, y
    cls = np.nonzero(ok)[0]
    own_col = np.searchsorted(cls, ys)
    D[np.arange(len(ys)), own_col] = np.inf
    d_other = D.min(1)
    scale = np.sqrt(within) + 1e-9
    margin = float(np.median((d_other - d_own_s) / scale))
    return dict(between_within=float(between / (within + 1e-12)), margin=margin)


def interference(X, y, n_classes):
    """Mean |cos| between the class-mean directions of different final classes."""
    M, ok = class_means(X, y, n_classes)
    D = M[ok] - X.mean(0)
    D = D / (np.linalg.norm(D, axis=1, keepdims=True) + 1e-9)
    C = np.abs(D @ D.T)
    k = len(D)
    return float((C.sum() - k) / (k * (k - 1))) if k > 1 else float("nan")


# ------------------------------------------------------------ spectrum
def effective_rank(X):
    Xc = X - X.mean(0)
    s = np.linalg.svd(Xc, compute_uv=False)
    p = s ** 2
    pr = float(p.sum() ** 2 / (np.sum(p ** 2) + 1e-12))
    top5 = float(p[:5].sum() / (p.sum() + 1e-12))
    return dict(eff_rank=pr, top5_share=top5)


def linear_cka(X, Y):
    Xc = X - X.mean(0)
    Yc = Y - Y.mean(0)
    num = np.linalg.norm(Xc.T @ Yc) ** 2
    den = np.linalg.norm(Xc.T @ Xc) * np.linalg.norm(Yc.T @ Yc)
    return float(num / (den + 1e-12))


def align(A, B):
    """Normalised Frobenius norm of the product of adjacent weight matrices A (out) and B (in)."""
    return float(np.linalg.norm(A @ B) / (np.linalg.norm(A) * np.linalg.norm(B) + 1e-12))


def subspace_energy(J, V):
    """Fraction of the Frobenius energy of J (…, d) lying in the row space V (r, d)."""
    Jm = J.reshape(-1, J.shape[-1])
    return float(np.linalg.norm(Jm @ V.T) ** 2 / (np.linalg.norm(Jm) ** 2 + 1e-12))


# ------------------------------------------------------ transition timing
def rise_times(steps, values, lo_q=0.1, hi_q=0.9, direction=None):
    """Persistent crossing times of a series at 10/50/90 % of its rise
    between the 10th and 90th percentiles of its values.  A crossing is
    persistent when the series stays beyond the level for the rest of the
    run.  `direction` +1 (rising) / −1 (falling) / None (largest change)."""
    steps = np.asarray(steps)
    v = np.asarray(values, dtype=float)
    ok = np.isfinite(v)
    if ok.sum() < 3:
        return dict(t10=None, t50=None, t90=None, width=None, lo=None, hi=None, direction=0)
    lo, hi = np.nanquantile(v[ok], lo_q), np.nanquantile(v[ok], hi_q)
    if direction is None:
        first = np.nanmean(v[ok][: max(1, ok.sum() // 5)])
        last = np.nanmean(v[ok][-max(1, ok.sum() // 5):])
        direction = 1 if last >= first else -1
    r = (v - lo) / (hi - lo + 1e-12) if direction > 0 else (hi - v) / (hi - lo + 1e-12)
    out = {"lo": float(lo), "hi": float(hi), "direction": int(direction)}
    for name, level in (("t10", 0.1), ("t50", 0.5), ("t90", 0.9)):
        t = None
        for i in range(len(r)):
            seg = r[i:]
            if np.all(seg[np.isfinite(seg)] >= level) and np.isfinite(r[i]):
                t = int(steps[i])
                break
        out[name] = t
    out["width"] = (out["t90"] - out["t10"]) if (out["t90"] is not None and out["t10"] is not None) else None
    out["change"] = float(abs(hi - lo))
    return out
