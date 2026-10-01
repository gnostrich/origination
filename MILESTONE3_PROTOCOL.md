# Milestone 3 protocol — frozen before training and discovery

Question.  Can the interface-language method extract a compact executable
language from a learned system whose correct internal objects are not
known, and can that language predict substrate computations that were
never observed?

This file is committed before the substrate is trained.  After it is
committed: no change of task, architecture, primary metrics, sealed test
set, or discovery objective.  Implementation bugs may be fixed; every fix
is recorded in Section 10 and all affected comparisons are rerun
symmetrically.  Any analysis not listed here is exploratory and excluded
from the verdict.

## 1. System

**Task.**  Next-observation prediction for a one-dimensional kinematic
cart driven by discrete actions.  The simulator (sealed; `cart.py`):

    state   (x, v),  x ∈ [0, 1],  v ∈ [−0.15, 0.15];  initial (0.5, 0)
    actions 0 push-left  v ← clip(0.95·v − 0.05)
            1 push-right v ← clip(0.95·v + 0.05)
            2 coast      v ← 0.95·v
            3 brake      v ← 0.5·v
            then x ← x + v, elastic reflection at 0 and 1 (x mirrored, v negated)
    observation after each action: the position bin ⌊8·x⌋ ∈ {0, …, 7}

Sequences of 16 random actions; the learner sees only action tokens and
must predict the observation bin after every action.  Train 20 000
sequences, held-out 2 000 of length 16 and 2 000 of length 24
(generalisation to unseen sequences and to lengths beyond training).

**Architecture.**  One GRU world model, the existing `GRUWorldModel`
(embedding 32, hidden 128, linear readout), one training run (seed 0),
AdamW lr 2·10⁻³, weight decay 0.1, batches of 512, 4 000 steps.  No
architecture or hyperparameter sweep.  If the model does not generalise
(held-out next-bin accuracy below 0.9 at length 16), that is recorded and
the milestone proceeds on the model as trained.

**Why this is an unknown ontology.**
* The generator's latent variables (x, v) are continuous; nothing finite
  is planted.  There is no automaton, algebra, table or circuit whose
  recovery could count as success, and the simulator is never shown to
  discovery.
* The trained GRU's internal objects are not known: whether its 128-d
  hidden state admits a compact substitutability quotient at any
  resolution, whether that quotient is finite, approximate, continuous or
  hierarchical, is the open question.  The method is allowed to answer
  "no compact language" (Outcome C).
* The task is ordinary (world modelling under control), compositional
  (actions compose; pieces are reused across continuations), generalises
  non-trivially (unseen sequences, longer horizons), and a lookup table is
  unattractive (4⁸ strings of length 8 from a continuum of states).
* Interventions are cheap and extensive: every hidden state can be
  transplanted and continued with any action string.

## 2. What discovery receives

Only, through `emergence/lang`: the trained substrate; its hidden
configuration after unlabelled action prefixes (pieces); continuation with
action strings (interactions); the softmax output at every step
(behaviour); unlabelled context families (action chunks).  Never: the
simulator, (x, v), bins as semantic quantities, probes against semantic
variables, or any decomposition.  The simulator is opened only in
Section 9 (interpretation) after every number of Sections 5–8 is frozen.

## 3. Resolution and ontology

Interfaces are ε-classes under the behavioural distance of
`INTERFACE_LANGUAGE.md`: the largest Jensen–Shannon distance over all
contexts and steps of the context family.  No finite ontology is assumed:
the number of interfaces at each (ε, context family) is an output; fits
carry confidence (fraction of realisations landing in the majority
interface) and may be partial (no interface within ε); the model may
abstain.  A clean finite quotient is not forced; if the frontier shows a
quasi-continuous family, that is reported as the result.

## 4. Splits, fixed before discovery (seeded, saved)

* **Discovery set.**  Piece pool D: all action prefixes of length ≤ 3 (85)
  plus 600 random prefixes of length 4–16.  Fit observation: all 4
  single-action chunks and the 16 two-action chunks minus a withheld set
  W of 4 (25 %), drawn once.  Context families, nested:
  C₁ = chunks of length 1 (4);  C₂ = lengths ≤ 2 (20);  C₃ = lengths ≤ 3 (84).
* **Validation set.**  120 fresh pieces (prefixes of length 4–12 not in
  D) × 30 fresh strings of length 6.  Used only to select ε and the context
  family of the primary model (Section 6).
* **Sealed test set** — generated and saved before discovery, never read
  until the primary model is frozen.  200 fresh pieces (not in D or
  validation) and four query families:
  T1 *interpolation* — strings of length 1–2 consisting of observed chunks
     (unseen pieces, observed interactions);
  T2 *unseen compositions of known fits* — strings of length 3–4 built from
     observed chunks whose concatenation was never a discovery chunk;
  T3 *withheld compositions* — strings of length 3–6 containing a withheld
     chunk from W;
  T4 *longer compositions* — strings of length 8–12, longer than any
     discovery context or fit chunk.
  40 strings per family.

## 5. Discovery

For every (ε, C) in ε ∈ {0.05, 0.1, 0.2, 0.3, 0.5} × {C₁, C₂, C₃} and
discovery seed s ∈ {0, 1, 2} (the seed samples which realisations are
used to observe fits), run the existing pipeline unchanged:
`discover_interfaces → discover_fits → compose`, producing
`M = (𝓘, 𝓕, 𝓒, ε, residual)` and its specification.

## 6. Primary model selection (validation only)

The primary model is the (ε, C, s = 0) model with the highest validation
fidelity (per-step argmax agreement with the substrate, abstention = error);
ties broken by lower description complexity.  All other models remain on
the reported frontier.  Nothing about the test set is consulted.

## 7. Prospective prediction (the centrepiece)

After the primary model is frozen: (i) write its predictions for every
sealed test query to `results/lang/m3/predictions_<timestamp>.json`,
with a hash of the file; (ii) only then run the substrate on the same
queries and write `substrate_<timestamp>.json`; (iii) score.  The log
records the two timestamps.  Predictions are made by the abstract model
alone (`InterfaceModel.run`) from the abstract state of each test piece
(its classification under the primary model's context family); the
substrate is not executed for any prediction.

Metrics per family T1–T4: argmax fidelity, JS fidelity (1 − JS distance),
coverage (non-abstained steps), fidelity on covered steps.  Overall:
description complexity L(M) (argmax and distribution variants), exceptions,
residual (undefined fits), interface reuse (mean number of distinct
(source, action) fits landing in an interface), fit reuse (mean number of
test steps per first-order fit), depth of successful composition (longest
test string predicted entirely correctly, and fidelity as a function of
step index in T4), causal commutation (do(I = i) through up to five
distinct realisations on T4 strings).

## 8. Baselines, same discovery budget

* Random abstraction: same number of interfaces and class sizes as the
  primary model, realisations assigned at random, tables refitted.
* Non-compositional lookup: catalogue of observed (interface, chunk)
  behaviours; abstains beyond length 2.
* Geometric clustering: k-means on the same discovery pieces' hidden
  vectors (Euclidean), K equal to the primary model's K; fits and
  emissions fitted from the same discovery observations; classification
  of test pieces by nearest centroid.  Also k-means in the top-8 PCA
  subspace of the same vectors.
* Full substrate: fidelity 1, complexity of its parameters.
Each baseline is scored on the same sealed test set, prospectively in the
same order (predictions written before substrate execution).

## 9. After freezing: analyses that do not feed back

* Nested contexts: for C₁ ⊂ C₂ ⊂ C₃ at the primary ε, report K,
  validation and test fidelity, complexity, and whether the finer partition
  refines the coarser one (fraction of C₂ interfaces contained in a single
  C₁ interface, likewise C₃ in C₂).
* Reproducibility: seeds 0–2 at the primary (ε, C): representational
  agreement (ARI of partitions on the discovery pool) and behavioural
  distance (prediction disagreement on T1–T4).
* Interpretation (simulator unsealed): per interface, the (x, v) of its
  realisations (mean, spread); whether interfaces are position cells,
  velocity cells, both, or neither; recurring fits; higher-order interfaces
  (chunk types).

## 10. Pre-registered verdicts

A — genuine discovery: the primary model beats random, geometric and lookup
baselines substantially on sealed prediction, predicts T3 and T4 well
above baselines, commutes causally (≥ 0.9), compresses the substrate
substantially (L(M) < 10 % of the substrate's bits) and was not specified
through task ontology.
B — useful partial language: causally real reusable interfaces and fits,
but substantial residual (coverage or fidelity on some family < 0.8) or
limited composition depth; the abstracted portion is characterised.
C — behavioural geometry without compact language: no (ε, C) gives a
compact model with fidelity near the substrate; fidelity rises only with
K approaching the number of pieces.  The geometry is reported.
D — no advantage: the primary model does not beat matched geometric,
lookup or random baselines on sealed prediction.

Record of fixes after commit: (none yet).
