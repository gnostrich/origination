# The interface language of a trained system: the target, stated operationally

**Project question.**  Can the endogenous interface language of an arbitrary
trained computational system be recovered purely from behavioural
substitution, interaction and composition, without specifying its internal
ontology in advance?

    trained substrate → behavioural interfaces/types → fits/interactions
                      → composition → higher-order interfaces
                      → an executable abstract description of the system

This is reverse engineering a program without assuming its programming
language.  Everything earlier in this repository (emergence, grokking,
clicking, discreteness, algebras) is retained as method and controls; none
of it is the target.

---

## 1. Substrate

A substrate `S` exposes, and only exposes:

* **pieces** — internal objects that can be read out and re-inserted:
  configurations at sites (a hidden state, a residual vector, a KV cache, a
  slot of a coupled system).  Their coordinates are available for
  *intervention*, never for *definition*;
* **interactions** — the ways pieces meet: insertion of a piece into a
  context (a continuation, a partner piece, an input), producing behaviour
  and, where the substrate carries state forward, new pieces;
* **behaviour** — the observable output of an interaction (a distribution
  over outputs, a sequence of them);
* **unlabelled contexts** — a family `𝒞` of interactions that discovery may
  draw from.

Discovery never receives: teacher or world states, generating gates,
automaton states, human labels, a known algebra, or any decomposition of
the architecture beyond what is unavoidable to *access* pieces (a hook
point is a place, not a meaning).

## 2. Interface (type)

For pieces `x, y` and a context family `𝒞`,

    x ~_𝒞 y   ⟺   for every c ∈ 𝒞, behaviour(x in c) ≈ behaviour(y in c)

where `≈` is closeness at a stated resolution ε in a stated behavioural
metric (here: Jensen–Shannon distance between output distributions, mean
over the steps of a continuation).  An **interface** is an equivalence
class under `~_𝒞` — the set of pieces that are substitutable across `𝒞`.

An interface is **not** a neuron, a direction, a subspace, a cluster of
activations, a probe-decodable concept or a human-labelled feature.  Any of
those may *realise* an interface, and an interface may be realised by
pieces with no coordinate similarity at all.  The definition refers to
behaviour only.

Allowed from the start:

* **approximate** equivalence (ε > 0; ε-types are reported with their
  resolution, and their stability across resolutions is a measured
  property, not an assumption);
* **partial** membership and **overlap**: a piece may satisfy `~_𝒞` for one
  context family and not another; a piece may belong to several
  context-relative interfaces;
* **finite or infinite** families: the number of interfaces at resolution
  ε is an output (`N(ε)`), never a target;
* **multiple valid decompositions**: two different interface languages may
  both be compact and faithful; neither is "the" ontology.

An interface record carries: its substitutable realisations, the contexts
under which they were found substitutable (accepted contexts), and its
behavioural effect (what any realisation does in each accepted context),
with the resolution and the residual disagreement inside the class.

## 3. Fit

A **fit** is a reproducible interaction among interfaces:

    (A₁, …, Aₙ) ⇝ B

holds when representatives of `A₁ … Aₙ`, inserted into an interaction
pattern, reproducibly produce behaviour — or a resulting piece — that
belongs to interface `B`.  "Reproducibly" is measured: the fraction of
representative choices that land in `B` (confidence), and the
disagreement among them (error).  Fits may be

* **partial** — defined for some representatives or contexts only;
* **multiway** — any arity, including arity one (an action on a piece);
* **probabilistic** — `B` is a distribution over interfaces with a stated
  entropy;
* **approximate** — landing within ε of `B`'s effect.

The essential requirement is **recursive reuse**: `B` must itself be an
interface that can enter further fits,

    (A, B) ⇝ C,    (C, D) ⇝ E,

so that a fit's result is a first-class piece of the language, not a
terminal effect.  This is what distinguishes a *language* from a catalogue
of intervention effects.

## 4. Higher-order interfaces

Interaction patterns are pieces too.  Two patterns (e.g. two action chunks,
two partner pieces, two contexts) are equivalent when they induce the same
fits on the first-order interfaces:

    w ~ w'   ⟺   for every interface A, (A, w) ⇝ B  and  (A, w') ⇝ B  for the same B.

Their classes are **second-order interfaces**, and fits among them
(`(W₁, W₂) ⇝ W₃` when composing the patterns induces the composed map) are
second-order fits.  The construction repeats at every order with the same
definition (`lean/Emergence/Interfaces.lean`: the interface of an assembly
is its action on interfaces).  The result is a typed hypergraph — nodes are
interfaces of any order, hyperedges are fits — not a flat partition.  An
operad, a category, a monoid or an automaton may turn out to describe it;
that is decided after discovery, by the behavioural facts.

## 5. The deliverable: an executable interface model

Discovery returns an object

    M = (𝓘, 𝓕, 𝓒, ε, residual)

* `𝓘` — the discovered interfaces (with realisations, accepted contexts,
  effects, resolution);
* `𝓕` — the discovered fits, with confidence and error;
* `𝓒` — the composition structure: how fits chain, and the higher-order
  interfaces and fits;
* `residual` — behaviour the model does not explain (abstentions,
  exceptions, unexplained variance).

`M` must be **executable**: given an initial abstract state (the interface
of a piece, obtained by classifying it in the discovery contexts) and a
requested network of interactions, `M` predicts the substrate's behaviour
*without running the substrate*, wherever the abstraction applies.  It is
written out as a specification:

    Interface A:  realisations …  accepted contexts …  effect …  (ε, residual)
    Fit F1:       (A, w) ⇝ B      confidence …  error …
    Fit F2:       (W₁, W₂) ⇝ W₃   (second order)

## 6. Evaluation: counterfactual fidelity against description complexity

The central measure is no longer recovery of known structure.  It is

    Fidelity(M, S) = 1 − E_q [ D( M(q), S(q) ) ]

over a **held-out intervention suite** `q` never used during discovery:
unseen pieces, unseen contexts, unseen compositions, and abstract
interventions `do(I = i)`.  `D` is a stated behavioural distance (0/1
disagreement of predicted outputs; JS distance of predicted
distributions).  Fidelity is reported together with the model's
**description complexity**, and the deliverable is the Pareto curve

    description complexity   vs   counterfactual fidelity.

Complexity is accounted separately for: the number and size of interfaces,
the number and size of fits, exceptions (listed disagreements), residual
(unexplained behaviour), and the parameters needed to predict effects.  No
single formula is fixed in advance; the principle is fixed:

    good explanation = high counterfactual fidelity + small endogenous language.

The full substrate has fidelity 1 and the complexity of its weights; a
lookup table of observed effects has the complexity of its observations
and no compositional reach.  The target is the smallest compositional
description that keeps predictive power.

## 7. The decisive tests

1. **Held-out intervention fidelity** — unseen pieces and unseen contexts.
2. **Compositional generalisation** — combinations withheld during
   discovery (`(A, B) ⇝ C` and `(C, D) ⇝ E` observed; `(A, B, D)` never
   observed) must be predicted through the discovered intermediate.
   A catalogue cannot do this; a language can.
3. **Causal abstraction** — for each interface `I` and abstract
   intervention `do(I = i)`, several *distinct* realisations of `i` are
   inserted; the square

        neural state ──neural intervention──▶ neural behaviour
             │                                     │
        interface state ──abstract intervention──▶ abstract behaviour

   must commute up to a measured error.
4. **Compression** — fidelity per bit against random abstractions of
   matched complexity and lookup models of matched observation budget.
5. **Coordinate robustness** — the extracted language must be unchanged
   under a change of basis of the exposed pieces; discovery that never
   reads coordinates satisfies this by construction, and it is checked.
6. **Executability** — the model runs on its own.

## 8. Discovery / evaluation separation

Discovery code receives only what Section 1 lists.  Evaluation code may,
where a reference exists, unseal known structure (automaton states, a
generating circuit) — but only to *describe* the extracted language
relative to it, never to select or score it.  The separation is enforced
in the code layout (`emergence/lang/discover.py` never imports a world or
a task; `emergence/lang/evaluate.py` may).

## 9. Benchmark ladder (validation before the target)

* Level 0 — known finite structures (S_4, reset monoid, counter monoid) on
  existing trained GRUs: can the language reproduce essentially all
  held-out counterfactual behaviour compactly?
* Level 1 — distributed realisation: the same computations under a change
  of basis of the exposed pieces.
* Level 2 — multiple valid decompositions: a system with several internal
  implementations of one behaviour; any compact, faithful language counts.
* Level 3 — a small learned system on an ordinary task with no privileged
  ontology, judged only by fidelity, compositional generalisation,
  compactness and reproducibility.

Levels 0–2 are validation.  Level 3 is the question.  Level 3 is not
attempted until Levels 0–2 show high held-out fidelity, strong compression
against the baselines, compositional prediction on withheld combinations,
and basis robustness; failure there is a reason to improve the definitions
and extraction, not to change the question.
