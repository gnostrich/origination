"""Blind behavioural quotient of a continuous controlled system.

The extractor receives: a substrate that maps (state, control suffix) to a
future observation trajectory; a set of states (points of R^n, sampled from
the system's own trajectories and uniformly over the region they visit);
continuous control suffixes; continuous control pulses.  It receives no
labels, counts, locations or partitions.

Steps (the same principles as ``extract_seq``):

1. behaviour(x) = observation trajectories under S sampled control suffixes,
   sampled at K times inside the window [T_skip, T] (the part of the future
   beyond the initial transient; the window is varied as a control);
2. behavioural distance d(x, x') = RMS trajectory difference, relative to
   the overall spread of observations; classes by leader clustering at
   relative tolerance ``eps``; discreteness = polarisation of d;
3. metastability = retention of the class under Gaussian perturbation of the
   state (relative to the spread), averaged over a noise grid;
4. controls as actions: pulses of value u (continuous, sampled in [-U, U]^m)
   held for duration ``tau_pulse``; for each class representative the pulse
   is applied and the result classified.  Grouping pulse values by the
   class-map they induce gives the discovered *operations*; polarisation of
   the map over u says whether the continuous control acts discretely;
5. the discovered operations generate a transformation monoid on the
   classes (``monoid_analysis``), and closure/coherence are measured as
   before.

Everything is label-free; ``evaluate`` (in ``run.py``) compares the result
with what is analytically known afterwards.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..grok.extract_seq import canonical_automaton, monoid_analysis
from .systems import System, integrate


@dataclass
class ContConfig:
    n_states: int = 240  # states handed to the extractor (half from trajectories, half uniform)
    dt: float = 0.01
    D: float = 0.0  # process noise intensity during behaviour evaluation
    tau: float = 0.5  # control chunk duration
    T: float = 12.0  # behaviour horizon
    T_skip: float = 6.0  # start of the behaviour window
    K: int = 12  # samples inside the window
    n_suffixes: int = 8
    u_sigma: float = 0.3  # control suffix magnitude (piecewise-constant N(0, u_sigma^2))
    eps: float = 0.1  # relative behavioural tolerance
    d_far: float = 0.5
    sigmas: tuple = (0.05, 0.1, 0.2, 0.4)  # state perturbation, relative to spread
    n_noise: int = 3
    n_pulses: int = 40  # continuous pulse values
    n_realisations: int = 6  # noise realisations averaged when D > 0
    stable_retention: float = 0.75
    U: float = 3.0  # pulse range
    tau_pulse: float = 1.0
    burn_in: float = 3.0
    seed: int = 0


class ContSubstrate:
    def __init__(self, sys: System, cfg: ContConfig, rng):
        self.sys, self.cfg, self.rng = sys, cfg, rng
        L = int(np.ceil(cfg.T / cfg.tau))
        self.L = L
        self.sample_times = np.linspace(cfg.T_skip, cfg.T, cfg.K)

    def suffixes(self, S):
        return self.rng.normal(0, self.cfg.u_sigma, (S, self.L, self.sys.m))

    def behaviour(self, X, suffixes):
        """(n, d) states x (S, L, m) suffixes -> (n, S*K*o) future observations.
        With process noise the behaviour is the *expected* future: realisations
        are averaged (``n_realisations``)."""
        n = X.shape[0]
        R = self.cfg.n_realisations if self.cfg.D > 0 else 1
        feats = []
        for s in range(suffixes.shape[0]):
            U = np.repeat(suffixes[s][None], n, axis=0)
            acc = 0.0
            for _ in range(R):
                _, Y = integrate(self.sys, X, U, self.cfg.tau, self.cfg.dt, self.cfg.D, self.rng, self.sample_times)
                acc = acc + Y.reshape(n, -1)
            feats.append(acc / R)
        return np.concatenate(feats, axis=1)

    def pulse(self, X, u):
        """Apply constant control u (m,) for tau_pulse, then let the system run
        control-free for the transient part of the horizon is *not* included:
        the resulting state is handed back as is."""
        U = np.repeat(u[None, None, :], X.shape[0], axis=0)
        Xn, _ = integrate(self.sys, X, U, self.cfg.tau_pulse, self.cfg.dt, self.cfg.D, self.rng)
        return Xn


def sample_states(sub: ContSubstrate, n, rng):
    """Half from the system's own trajectories (random initial conditions,
    random controls, burn-in), half uniform over the box the trajectories visit."""
    cfg = sub.cfg
    X0 = sub.sys.initial_states(n // 2, rng)
    Lb = int(np.ceil(cfg.burn_in / cfg.tau))
    U = rng.normal(0, cfg.u_sigma, (n // 2, Lb, sub.sys.m))
    Xt, _ = integrate(sub.sys, X0, U, cfg.tau, cfg.dt, cfg.D, rng)
    lo, hi = Xt.min(0) - 0.5, Xt.max(0) + 0.5
    Xu = rng.uniform(lo, hi, (n - n // 2, sub.sys.n))
    return np.concatenate([Xt, Xu]), (lo, hi)


def rel_dist(B, Bref, scale):
    """RMS difference of behaviour vectors relative to the observation spread."""
    return np.sqrt(((B - Bref) ** 2).mean(axis=-1)) / scale


def leader_cluster_cont(B, scale, eps):
    n = B.shape[0]
    labels = np.full(n, -1, dtype=int)
    L = np.empty((n, B.shape[1])); k = 0
    for i in range(n):
        if k:
            d = rel_dist(L[:k], B[i][None], scale)
            j = int(np.argmin(d))
            if d[j] < eps:
                labels[i] = j; continue
        L[k] = B[i]; labels[i] = k; k += 1
    return labels, L[:k]


def classify(Bnew, leaders, scale, eps):
    out = np.full(Bnew.shape[0], -1, dtype=int)
    for i in range(Bnew.shape[0]):
        d = rel_dist(leaders, Bnew[i][None], scale)
        j = int(np.argmin(d))
        if d[j] < eps:
            out[i] = j
    return out


def extract_cont(sys: System, cfg: ContConfig) -> dict:
    rng = np.random.default_rng(cfg.seed)
    sub = ContSubstrate(sys, cfg, rng)
    X, box = sample_states(sub, cfg.n_states, rng)
    n = X.shape[0]
    suff = sub.suffixes(cfg.n_suffixes)
    B = sub.behaviour(X, suff)
    scale = float(B.std()) if B.std() > 0 else 1.0  # overall spread of observed futures

    # 1-2. classes and discreteness
    labels, leaders = leader_cluster_cont(B, scale, cfg.eps)
    n_classes = int(labels.max()) + 1
    class_size = np.bincount(labels)
    m = min(n, 200)
    smp = rng.choice(n, size=m, replace=False)
    d = np.concatenate([rel_dist(B[smp[i + 1:]], B[smp[i]][None], scale) for i in range(m - 1)])
    discreteness = float(np.mean(d < cfg.eps) + np.mean(d > cfg.d_far))
    rep_idx = np.array([int(np.argmax(labels == c)) for c in range(n_classes)])

    # 3. metastability under state perturbation
    spread = X.std(axis=0) + 1e-12
    retention = {}
    ret_avg = np.zeros(n)  # per-state retention averaged over the noise grid
    for sigma in cfg.sigmas:
        keep = np.zeros(n)
        for _ in range(cfg.n_noise):
            Xn = X + sigma * spread * rng.standard_normal(X.shape)
            Bn = sub.behaviour(Xn, suff)
            keep += (rel_dist(Bn, B, scale) < cfg.eps)
        keep /= cfg.n_noise
        retention[sigma] = float(keep.mean())
        ret_avg += keep / len(cfg.sigmas)
    metastability = float(np.mean(list(retention.values())))
    class_ret = np.zeros(n_classes); np.add.at(class_ret, labels, ret_avg); class_ret /= np.maximum(class_size, 1)

    # stable objects: classes that survive perturbation and are reproducible
    # (the same criterion as in the earlier experiments; label-free)
    stable = (class_ret >= cfg.stable_retention) & (class_size >= 2)
    stable_ids = np.flatnonzero(stable)
    n_stable = int(len(stable_ids))
    remap = {int(c): i for i, c in enumerate(stable_ids)}

    # 4. continuous pulses as actions -> induced maps on the stable classes
    Xrep = X[rep_idx[stable_ids]] if n_stable else X[:0]
    pulses = rng.uniform(-cfg.U, cfg.U, (cfg.n_pulses, sys.m))
    maps = np.full((cfg.n_pulses, n_stable), -1, dtype=int)
    for j, u in enumerate(pulses):
        if not n_stable:
            break
        Xp = sub.pulse(Xrep, u)
        lab_p = classify(sub.behaviour(Xp, suff), leaders, scale, cfg.eps)
        maps[j] = np.array([remap.get(int(c), -1) for c in lab_p])
    known = maps >= 0
    n_classes_all = n_classes
    class_ret_stable = class_ret[stable_ids]
    frac_known = float(known.mean()) if known.size else 0.0
    # discovered operations: distinct induced maps (only fully classified pulses)
    ops = {}
    for j in range(cfg.n_pulses):
        if known[j].all():
            ops.setdefault(tuple(maps[j].tolist()), []).append(j)
    op_list = sorted(ops.items(), key=lambda kv: -len(kv[1]))
    # closure: expected stability of the class a pulse lands in
    closure = float(np.mean(np.where(known, class_ret_stable[np.maximum(maps, 0)], 0.0))) if n_stable else 0.0
    # polarisation of the control: does u act discretely?  fraction of pulses whose
    # map is one of the discovered ops (i.e. fully classified) and the number of ops
    T = np.array([np.array(k) for k, _ in op_list]).T if op_list else np.zeros((n_stable, 0), dtype=int)
    if T.shape[1] and n_stable:
        # canonical automaton from the largest stable class (there is no distinguished initial state)
        start = int(np.argmax(class_size[stable_ids]))
        C, order = canonical_automaton(T, start)
        mon = monoid_analysis(C, 5000)
    else:
        C, mon = np.zeros((0, 0), dtype=int), {"total": False}
    # coherence: (x.u).v vs class of applying both pulses in sequence
    n_chk = n_coh = 0
    for _ in range(min(12, len(op_list) ** 2 if (op_list and n_stable) else 0)):
        (ka, ja), (kb, jb) = (op_list[rng.integers(len(op_list))] for _ in range(2))
        ua, ub = pulses[ja[0]], pulses[jb[0]]
        direct = np.array([remap.get(int(c), -1) for c in classify(sub.behaviour(sub.pulse(sub.pulse(Xrep, ua), ub), suff), leaders, scale, cfg.eps)])
        via = np.array([kb[ka[c]] if ka[c] >= 0 else -1 for c in range(n_stable)])
        ok = (direct >= 0) & (via >= 0)
        n_chk += int(ok.sum()); n_coh += int((direct[ok] == via[ok]).sum())
    coherence = (n_coh / n_chk) if n_chk else float("nan")
    nondegenerate = float(1.0 - class_size.max() / n)
    return {
        "n_states": n, "box": [box[0].tolist(), box[1].tolist()],
        "n_classes": n_classes, "class_sizes": class_size.tolist(),
        "n_stable_classes": n_stable, "stable_class_ids": stable_ids.tolist(),
        "stable_class_sizes": class_size[stable_ids].tolist(),
        "discreteness": discreteness,
        "dist_quantiles": [float(q) for q in np.quantile(d, [0.05, 0.25, 0.5, 0.75, 0.95])],
        "retention": {str(k): v for k, v in retention.items()}, "metastability": metastability,
        "class_retention": class_ret.tolist(),
        "pulses": pulses.tolist(), "pulse_maps": maps.tolist(), "frac_pulses_classified": frac_known,
        "n_operations": len(op_list), "operations": [{"map": list(k), "n_pulses": len(v),
                                                       "u_values": pulses[v].tolist()} for k, v in op_list],
        "closure": closure, "coherence": coherence, "nondegenerate": nondegenerate,
        "monoid": mon, "canonical_table": C.tolist(),
        "states": X.tolist(), "labels": labels.tolist(), "rep_states": Xrep.tolist(),
        "rep_states_all": X[rep_idx].tolist(),
    }
