"""The abstract substrate.

A substrate `S_theta` is anything whose internal configurations can be
*observed at sites* and *substituted at sites*.  That is all the theory
assumes.  Nodes, edges, objects, types, ports, arities, modules and
mathematical operations are never primitives: they are candidate emergent
structures that the extraction procedure (``emergence.grok.extract``,
``emergence.experiment``) may or may not recover.

Formally a substrate exposes

* ``sites``: a finite list of names.  A site is a place where a
  configuration can be read off and written back.  Nothing is assumed about
  what a site *means*; it is a tensor-valued hook point (a layer, a
  position, a slot of a joint system, a set of coordinates, a sub-hypergraph).
* ``sample_contexts(n)``: draws contexts (inputs / environments / the rest of
  the world).  A context fixes the configurations at every site other than
  the ones being substituted.
* ``run(contexts, patch)``: runs the substrate on the contexts with the
  configurations at ``patch``'s sites *replaced* by the given ones, and
  returns (a) the downstream **behaviour** (a vector per context that is
  compared with a divergence) and (b) the resulting configuration at every
  site.

Everything downstream is defined in terms of substitution:

* two configurations `h, h'` at site `s` are **behaviourally equivalent**
  when substituting one for the other across sampled contexts does not
  materially change downstream behaviour (``BehEq`` in the Lean files);
* the **effective objects / types** at `s` are the equivalence classes;
* a tuple of configurations at sites `(s_1..s_k)` **clicks** when the
  substrate run on it settles into a *stable* configuration (robust to
  perturbation, consistent across residual contexts) at a downstream site;
* the **operation induced** by a click sends the classes of the parts to the
  class of the product; a product is treated exactly like any other
  configuration (recursion), so closure, composition and coherence laws can
  be tested;
* the induced structure is the **behavioural quotient**
  ``M_theta = S_theta / ~``.

Two instances live in this repository:

``emergence.experiment`` (Langevin energy landscape)
    sites: the `k` slots of the joint system (``JointEnergy``);
    contexts: thermal noise realisations;
    configurations: points of R^N; run = relax the joint dynamics;
    behaviour: which basin each slot and the superposition quench to;
    stability: escape time / relaxation time and strain.

``emergence.grok`` (trained networks)
    sites: named hook points of a torch model (embeddings, residual stream,
    hidden layer, logits); contexts: inputs; run = forward pass with
    activation patching; behaviour: output distribution; stability:
    retention of the behavioural class under internal noise.
"""

from __future__ import annotations

from typing import Any, Protocol


class Substrate(Protocol):
    sites: list[str]

    def sample_contexts(self, n: int, rng: Any) -> Any: ...

    def run(self, contexts: Any, patch: dict[str, Any] | None = None, noise: dict[str, float] | None = None) -> tuple[Any, dict[str, Any]]:
        """Return (behaviour, {site: configuration}) for the contexts, with the
        configurations at ``patch``'s sites substituted and Gaussian noise of
        relative scale ``noise[site]`` added at those sites."""
        ...


class TorchSubstrate:
    """Adapter for a torch model exposing ``model.forward(tokens, patch, noise)``
    that returns ``(logits, {site: tensor})``.  Inputs are integer token
    arrays; contexts are rows of the model's input space.

    The extraction code never looks inside the model: only the site names and
    the substitution interface are used, so any architecture (MLP, GNN,
    transformer, RNN, ...) that offers hook points is a valid substrate.
    """

    def __init__(self, model, all_inputs, site_names: list[str] | None = None):
        import torch  # local import: torch is optional for the Langevin part

        self.torch = torch
        self.model = model
        self.inputs = all_inputs  # (n_contexts, seq) long tensor
        self.sites = list(site_names or model.site_names)

    def sample_contexts(self, n: int, rng):
        idx = rng.choice(len(self.inputs), size=min(n, len(self.inputs)), replace=False)
        return self.inputs[idx]

    def run(self, contexts, patch=None, noise=None):
        with self.torch.no_grad():
            logits, sites = self.model(contexts, patch=patch, noise=noise)
        return logits, sites
