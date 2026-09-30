"""The executable interface model  M = (𝓘, 𝓕, 𝓒, ε, residual).

`InterfaceModel.run` predicts the substrate's behaviour along any requested
action string from an abstract initial state without running the substrate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product

import numpy as np

from emergence.lang.substrate import js_distance


@dataclass
class Fit:
    src: int            # interface index
    chunk: tuple        # interaction pattern (action tuple)
    dst: int            # resulting interface (-1 = no interface within ε)
    confidence: float   # fraction of realisations landing in dst
    n: int              # realisations tested
    observed: bool = True   # observed during discovery (else derived by chaining)


@dataclass
class InterfaceModel:
    eps: float
    n_actions: int
    n_obs: int
    contexts: list                       # discovery context family (action tuples)
    leaders: np.ndarray                  # (K, C, L, o) effect of each interface in each context
    realizations: list                   # per interface: indices of discovery pieces
    within: np.ndarray                   # per interface: mean JS distance of realisations to the leader
    emissions: np.ndarray                # (K, n_actions, o): next-output distribution of interface under each action
    fits: dict = field(default_factory=dict)        # (src, chunk) -> Fit
    chunk_types: dict = field(default_factory=dict) # chunk -> second-order interface id
    chunk_maps: dict = field(default_factory=dict)  # second-order id -> tuple of dst per interface
    second_fits: dict = field(default_factory=dict) # (W1, W2) -> W3 (second-order composition)
    withheld: list = field(default_factory=list)    # chunks withheld from discovery

    @property
    def K(self):
        return len(self.leaders)

    # ---------------------------------------------------------- abstraction map α
    def classify(self, signatures):
        """Nearest leader within ε (mean JS distance over discovery contexts); -1 otherwise."""
        n = signatures.shape[0]
        out = np.full(n, -1, dtype=int)
        dist = np.full(n, np.inf)
        for k in range(self.K):
            d = np.nanmean(js_distance(signatures, self.leaders[k][None]), axis=(1, 2))
            better = d < dist
            dist[better] = d[better]
            out[better] = k
        out[dist >= self.eps] = -1
        return out, dist

    # ---------------------------------------------------------- execution
    def transition(self, state, action):
        f = self.fits.get((state, (action,)))
        return -1 if f is None else f.dst

    def run(self, state, string):
        """Predicted output distribution at every step of `string`; None where the model abstains."""
        out = []
        s = state
        for a in string:
            if s < 0:
                out.append(None)
                continue
            out.append(self.emissions[s, a])
            s = self.transition(s, a)
        return out

    def run_chunk_map(self, state, chunk):
        """Chain first-order fits along a chunk (derived prediction)."""
        s = state
        for a in chunk:
            if s < 0:
                return -1
            s = self.transition(s, a)
        return s

    # ---------------------------------------------------------- specification
    def spec(self, max_real=4, prefixes=None):
        L = [f"# Extracted interface model (ε = {self.eps}, {self.K} interfaces, {len(self.fits)} fits)", ""]
        L.append("## Interfaces")
        for k in range(self.K):
            r = self.realizations[k]
            ex = ", ".join(str(prefixes[i]) if prefixes is not None else str(i) for i in r[:max_real])
            em = "; ".join(f"a{a}→o{int(np.argmax(self.emissions[k, a]))} ({self.emissions[k, a].max():.2f})" for a in range(self.n_actions))
            L.append(f"- Interface I{k}: {len(r)} substitutable realisations (e.g. {ex}); accepted contexts: all {len(self.contexts)} discovery contexts; "
                     f"within-class disagreement {self.within[k]:.3f}; effect: {em}")
        L += ["", "## First-order fits  (I, action) ⇝ I'"]
        for (src, chunk), f in sorted(self.fits.items(), key=lambda kv: (len(kv[0][1]), kv[0][1], kv[0][0])):
            if len(chunk) == 1:
                L.append(f"- (I{src}, a{chunk[0]}) ⇝ {'I' + str(f.dst) if f.dst >= 0 else '∅'}   confidence {f.confidence:.2f} (n={f.n})")
        obs2 = [(k, f) for k, f in self.fits.items() if len(k[1]) > 1 and f.observed]
        if obs2:
            L += ["", f"## Observed multi-step fits ({len(obs2)}; the rest are derived by chaining)"]
            for (src, chunk), f in sorted(obs2)[:12]:
                L.append(f"- (I{src}, {chunk}) ⇝ {'I' + str(f.dst) if f.dst >= 0 else '∅'}   confidence {f.confidence:.2f}")
            if len(obs2) > 12:
                L.append(f"- … {len(obs2) - 12} more")
        L += ["", f"## Second-order interfaces (chunk types): {len(self.chunk_maps)}"]
        for w, m in self.chunk_maps.items():
            members = [c for c, t in self.chunk_types.items() if t == w]
            L.append(f"- W{w}: chunks {members[:6]}{' …' if len(members) > 6 else ''}; induced map " + " ".join(f"I{i}→{'I' + str(d) if d >= 0 else '∅'}" for i, d in enumerate(m)))
        L += ["", f"## Second-order fits  (W, W') ⇝ W''  ({len(self.second_fits)})"]
        for (w1, w2), w3 in list(self.second_fits.items())[:40]:
            L.append(f"- (W{w1}, W{w2}) ⇝ W{w3}")
        if len(self.second_fits) > 40:
            L.append(f"- … {len(self.second_fits) - 40} more")
        if self.withheld:
            L += ["", f"Withheld from discovery (predicted only by chaining): {self.withheld}"]
        return "\n".join(L)

    def to_json(self):
        return dict(eps=self.eps, n_actions=self.n_actions, n_obs=self.n_obs, K=self.K,
                    contexts=[list(c) for c in self.contexts], within=self.within.tolist(),
                    emissions=self.emissions.tolist(),
                    fits=[dict(src=f.src, chunk=list(f.chunk), dst=f.dst, confidence=f.confidence, n=f.n, observed=f.observed) for f in self.fits.values()],
                    chunk_types={str(list(c)): t for c, t in self.chunk_types.items()},
                    chunk_maps={str(w): list(m) for w, m in self.chunk_maps.items()},
                    second_fits=[[w1, w2, w3] for (w1, w2), w3 in self.second_fits.items()],
                    withheld=[list(c) for c in self.withheld],
                    realization_counts=[len(r) for r in self.realizations])
