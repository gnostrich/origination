# origination — emergent discreteness, types and composition from continuous substrates

This repository is a small, runnable v0 of one idea:

> Mathematics / discreteness may arise when continuously changing degrees of
> freedom **click** into stable, mutually compatible interfaces.  A jigsaw
> piece moves continuously, but *fit* is effectively discrete.  A
> continuously trained system may likewise develop components that at some
> point form a closed, composable mechanism, producing an apparently sharp
> behavioural transition.

The central object recovered from any substrate `S_θ` is its **behavioural
quotient**

    M_θ = S_θ / ~

where internal configurations are equivalent when they are *substitutable
across sampled internal contexts without materially changing downstream
behaviour*.  From `M_θ` we try to **discover, not impose**: stable discrete
objects, behavioural types, compatibility ("click") relations of arbitrary
arity, the operations induced by stable interactions, closure, composition,
approximate coherence laws, and the recursive treatment of a composite as a
new interface.  Nothing in the substrate is assumed to be a node, edge,
object, type, port, arity, module or mathematical operation: those are
candidate emergent structures.

Two substrates are implemented under one abstraction (`emergence/substrate.py`):

| substrate | sites | contexts | run | behaviour | stability |
|---|---|---|---|---|---|
| Langevin energy landscape (`emergence/experiment.py`) | slots of the joint system | thermal noise realisations | relax the joint dynamics | which basin each slot / the superposition quenches to | escape time / relaxation time, strain |
| trained networks (`emergence/grok/`) | hook points of a torch model | inputs | forward pass with activation patching | output distribution | retention of the behavioural class under internal noise |

and one Lean development (`lean/`) formalises the substrate-independent
layer: behavioural equivalence, quotient types, partial and multiary
operations, the conditions under which they descend to the quotient, and the
conditions under which *approximate* substrate-level laws become *exact*
after quotienting.

```
continuous / noisy substrate → stable fits → behavioural equivalence → quotient → exact structure
```

---

## Part I — Langevin substrate (Experiment A/B)

The first instance tests exactly one proposition: can continuous, initially
untyped dynamics generate metastable discrete states whose interaction
behaviour induces types, and whose stable multiway interactions compose?

The only primitive is an energy `E_θ : R^N → R` and overdamped Langevin
dynamics `dx = -∇E dt + sqrt(2T) dW`.  The single structural assumption is
that when `k` things occupy the substrate together the joint energy is
`Σ E(xᵢ) + g·E((Σxᵢ)/√k)`: the composite is the normalised superposition of
the parts, judged by the same energy.

Pipeline: Langevin trajectories → quench to microstates → lump microstates
that interconvert within a few relaxation times → keep lumps with
`R = τ_esc/τ_relax ≥ R_min` (things) → joint dynamics of every tuple of
representatives → compatibility `C(A,B,…)` = P(parts survive the horizon
**and** the joint minimum has strain ≤ `strain_tol·T`) → products (quenched
superpositions, treated as new things) → fingerprints → ε-types →
closure / substitutability / associativity → finite table exported to Lean.

Why strain: with a smooth coupling an incompatible pair still has a joint
local minimum in which the parts are pulled off their wells (verified:
raising `g` from 2 to 8 changes nothing), so survival alone says "everything
clicks".  The strain energy of the joint minimum is what is bimodal.

**Experiment A** (`designed_landscape`, wells placed so that a known
structure exists; the pipeline never sees the names) — `python -m emergence run --landscape designed`:

| measurement | outcome |
|---|---|
| things | 16 microstates → 14 metastable basins (E1/Ep1, E2/Ep2 lumped: 1.2 apart, interconvert fast), all `R ≥ 27` |
| clicks | exactly the 7 designed pairs out of 196; `C` bimodality 0.999; strain bimodal (10 % at ≈0, 63 % above 25 T) |
| types | 6 classes: `{A1,A2}`, `{D1,D2}`, `{B}`, `{C}`, `{Dp}`, terminal — microscopically different things become one type because the world interacts with them identically |
| arity | 1 irreducible ternary click `P·Q·R` (all pairs incompatible) among 220 triples |
| composition | closure 14/14; substitutability 1.0; associativity 4/4 both-defined triples, 8 one-sided |
| Lean | exported table certified by `decide`: `Respects` and `WeakAssocUpToBehEq` |

One-sided associativity is correct: `(B·C)·A1` is defined but `B·(C·A1)` is
not because `C` and `A1` do not click; the structure is a partial commutative
magma, not a category.

**Experiment B** (random 2-layer tanh MLP energy, no supervision).  A smooth
random MLP has a single basin.  Rougher settings (`scale 8, input_gain 2`)
give hundreds of shallow minima that lump into one dominant funnel holding
> 95 % of samples at `T ∈ {0.03, 0.1, 0.3}`: a glassy funnel, no set of
distinct metastable things, hence no click structure.  This is the honest
v0 finding for unstructured random landscapes; the phase-diagram sweep
(`python -m emergence sweep`) is wired up but was not run at scale.

---

## Part II — Grokking as mathematical crystallisation

### Hypothesis (to be falsified, not assumed)

Generalisation may coincide with continuous training producing a
sufficiently **closed internal algebra / compositional interface structure**.
Grokking would be one visible case: continuous microscopic change crosses a
compositional/coherence threshold and produces a sharp macroscopic
generalisation transition.

### Task and substrates

Modular addition `(a, b) ↦ a + b mod p`, `p = 97`, 30 % of pairs for
training, full-batch AdamW.  Substrates: a 1-block transformer without
LayerNorm and a 2-layer ReLU MLP on concatenated embeddings; several seeds
each; plus a **control** transformer trained without weight decay, which
memorises and does not grok.  The human-understood structure (a cyclic
group) is known, so we can ask afterwards whether the *extracted* structure
is equivalent to it — but the extraction never uses `p`'s arithmetic, the
labels, or the train/test split.

### What is extracted at every checkpoint (label-free)

`emergence/grok/extract.py` sees only sites, substitution and behaviour:

| quantity | definition |
|---|---|
| effective objects at the hidden site | behavioural classes of hidden configurations: `h ~ h'` iff substituting one for the other into sampled contexts changes the output distribution by JS < ε |
| discreteness | polarisation of pairwise behavioural distances: fraction that are "same" (JS < ε) or "clearly different" (JS > 0.5) |
| metastability | retention of the behavioural class under Gaussian noise at the hidden site, in units of the site's RMS; basin radius = noise at which retention drops to 0.5 |
| clicks `C(a,b)` | stability (retention at the reference noise) of the composite produced by input configurations `a`, `b`; polarisation of `C` |
| closure | fraction of products landing in a *stable* class (class retention ≥ 0.75) |
| induced operation | the world re-feeds the output as an input: `op(a,b) := argmax behaviour`; associativity on sampled triples, commutativity, identity, inverses, Latin property |
| crystallisation index | law-free: `√(metastability × closure) × (1 − largest class share)`; the original index also included discreteness and closed-loop associativity and is kept for reference |

Train/test accuracy are computed with labels only as the **external
reference** the measures are compared against.

Caveat stated up front: for one-block models the hidden site's downstream
map is context-free, so its behavioural classes coincide with
output-distribution classes.  The internal content of the measures is in
discreteness, retention/basin radius, click polarisation and closure; the
closed-loop associativity is an unsupervised self-consistency measure and
for this task is expected to track correctness closely.

### Falsification criteria

The hypothesis predicts, for every seed and both architectures:

1. on the memorisation plateau (train acc ≈ 1, test acc low) the
   crystallisation measures are low;
2. they rise sharply, and their half-rise step coincides with (or precedes)
   the half-rise of test accuracy within a few checkpoints;
3. the no-weight-decay control, which memorises without generalising, never
   shows the rise;
4. across seeds and architectures the hidden-site quotients converge to
   mutually equivalent structures (adjusted Rand index of the partitions
   → 1, operation tables isomorphic) at the transition, while on the
   plateau they are idiosyncratic (low ARI).

It is falsified if any measure is already high while memorising, rises only
long after generalisation, rises in the control, or if the quotients remain
idiosyncratic after generalisation.

### Idiosyncratic internal mathematics

Different seeds/substrates may solve the task with very different internal
representations.  Success is *not* judged by recovering Fourier features or
modular-arithmetic variables.  `emergence/grok/compare.py` compares the
extracted quotients directly: ARI between hidden-site partitions of the
same inputs (architecture-independent), agreement of induced operation
tables, algebraic invariants (associativity, commutativity, identity,
inverses, Latin), and an isomorphism test up to relabelling (generator
propagation).  The three layers are

    microscopic parameters → idiosyncratic emergent algebra → task-level equivalence class

and the Lean file `Equiv.lean` says exactly what "equivalent at a higher
level" means.

### Results

Full tables: `results/grok/report.md`; curves: `results/grok/crystallization.png`.

Seven runs, `p = 97`, 30 % training pairs, checkpoints every 500 steps
(denser before 2000).  All six weight-decay runs grokked; the control did not.

| run | train acc → 1 | test acc → 0.9 | plateau length |
|---|---|---|---|
| transformer s0 / s1 / s2 (wd 1) | 300–400 | 20500 / 15000 / 16500 | ~15k–20k steps |
| mlp s0 / s1 / s2 (wd 1) | 50 | 9000 / 8500 / 9000 | ~8.5k steps |
| transformer s0, no weight decay (control) | 750 | never (0.27 at 25k, test loss 130) | — |

**1. Plateau vs. after generalisation** (means over the plateau `t_mem ≤ t < t_gen`
and after `t_gen`, then the range over the six grokked runs):

| measure | memorised, not generalised | after generalisation | control at 25k |
|---|---|---|---|
| behavioural classes at the hidden site | 2000–6000 | 97 (exactly `p`) | 612 |
| compression `log(p²/n)/log p` | 0.11–0.29 | 0.88–0.93 | 0.60 |
| metastability (class retention under internal noise) | 0.21–0.30 | 0.53–0.62 | 0.05 |
| closure (stability of the class a product lands in) | 0.25–0.35 | 0.68–0.84 | 0.04 |
| discreteness (polarised behavioural distances) | 0.99 | 1.00 | 1.00 |
| closed-loop associativity | 0.04–0.12 | 0.97–1.00 | 0.07 |
| closed-loop commutativity | 0.15–0.85 | 0.99–1.00 | 0.78 |
| crystallisation index (with associativity) | 0.22–0.28 | 0.74–0.83 | 0.11 |
| crystallisation index (law-free) | 0.23–0.32 | 0.59–0.71 | 0.04 |

Discreteness is uninformative for this task: a memorising network's output
behaviours are already crisp and mutually far.  What the memoriser lacks is
*compression* (thousands of behaviourally distinct internal things instead
of 97), *stability* (its things fall apart under internal noise) and
*closure*.  The control shows that compression alone is not the signal: its
class count also shrinks over training (2710 → 612) while its things stay
unstable (retention 0.05) and its operation stays incoherent (associativity
0.07).  Around step 5500 the control's metastability and closure drop from
≈0.4 to ≈0.05 with no change in test accuracy: its internal things collapse
without any macroscopic event, the opposite of crystallisation.

**2. Timing.**  Half-rise step of each measure minus the half-rise step of
test accuracy (500-step resolution; six grokked runs):

| measure | lag (steps) |
|---|---|
| commutativity | −1000 … 0 |
| crystallisation index (with associativity) | 0 … +1000 |
| crystallisation index (law-free) | +1000 … +2500 |
| associativity | +500 … +1000 |
| compression | +500 … +2000 |
| closure | +1000 … +2500 |
| metastability | +1500 … +2500 |

The measures rise sharply *with* the transition, never during the
plateau and never in the control.  At this resolution they trail test
accuracy by one to five checkpoints rather than preceding it; only
commutativity (a self-consistency symmetry of the induced operation) rises
at or slightly before the test-accuracy half-rise.  So the data support
"generalisation coincides with crystallisation" and do **not** support the
stronger "crystallisation is a leading indicator" at 500-step resolution.
Denser checkpoints around the transition are the obvious next experiment.

**3. Idiosyncratic algebras, task-level equivalence.**  On the plateau the
hidden-site partitions of different seeds are idiosyncratic (pairwise ARI
≈ 0.10–0.15; ARI to the true-sum partition 0.15 for the MLPs, 0.30 for the
transformers).  After grokking, every fully converged run — three MLP seeds
and two transformer seeds, two different architectures — induces the
**same** partition of the 9409 inputs (pairwise ARI 1.000, ARI to the sum
partition 1.000), the same operation table (agreement 1.000), and the
tables are isomorphic up to relabelling (generator 1 ↦ 1) with invariants
associativity = commutativity = identity = Latin = 1: a cyclic group of
order 97, recovered without ever looking for one.  (Transformer seed 2's
last checkpoint falls in a brief post-grok instability — train acc 0.973 —
and gives ARI 0.93; its checkpoints 17000–19500 have 97–106 classes.)

Microscopic parameters differ across all seven runs; the induced quotients
of the six grokked runs are one algebra.

**4. Verdict against the falsification criteria.**  (1) low on the plateau:
yes.  (2) sharp rise at the transition: yes, coincident, not leading.
(3) absent in the control: yes — one control passes one falsification test;
it does not confirm the general hypothesis.  (4) quotients converge to one
equivalence class after generalisation and are idiosyncratic before: yes.

Stated conservatively, the observation is: memorisation → thousands of
unstable behavioural classes → 97 stable, closed classes → a near-exact
associative algebra, across two architectures, while the non-grokking
control does not undergo the transition.  Whether that is more than
"modular addition is an associative algebra on 97 elements, and the network
learned modular addition" is exactly what the adversarial controls below
test.

Caveats: one task; the closed-loop associativity is expected to track
correctness for this task and is reported as self-consistency, not as an
independent signal; retention-based measures are noisy on the plateau
(0.1–0.5) and the index fluctuates with them; class thresholds (JS < 0.05,
noise grid) were fixed a priori and not tuned.


### Adversarial controls on the extractor

The clean result (thousands of classes → exactly 97, ARI 1, associativity ≈ 1)
has an obvious deflationary explanation: 97 is the cardinality of the task
algebra and modular addition is associative.  Before the interpretation is
accepted the extractor has to be attacked.  `emergence/grok/task.py` provides
tables that change the external structure without telling the extractor,
and `compare.algebra_analysis` classifies the blindly recovered closed-loop
table without assuming a group (Latin property; exact associativity; the
principal loop isotope, which by Albert's theorem is isomorphic to a group
iff the table is a group *up to independent relabelling of inputs and
outputs*; element-order spectrum, which separates non-isomorphic groups of
the same order).

| control | external table | what the extractor must recover to survive |
|---|---|---|
| 1 random table | uniform random 97×97 | no coherent algebra, no stable closed classes |
| 2 other groups | Z_89, Z_101, Z_8×Z_8 | 89, 101, 64 elements; Z_8×Z_8 non-cyclic (order spectrum {1,2³,4¹²,8⁴⁸}) |
| 3 scrambled presentation | Z_97 with input tokens and output tokens permuted independently | 97 stable classes; the closed-loop table is an *isotope* of Z_97, non-associative, isotopic to a cyclic group — so closed-loop associativity must **drop** although the quotient crystallises |
| 4 non-associative magma | a − b mod 97 | 97 stable classes and a Latin, non-associative, non-commutative table (isotopic to a cyclic group); if associativity ≈ 1 is reported, the metric bakes composition in |
| 5 corrupted algebra | Z_97 with 5 % / 15 % of entries randomised | graded degradation of closure and coherence, not recovery of the intended group |

Learnable non-associative tables are necessarily group isotopes here (any
table a small network groks on has low complexity); a random Latin square
is not a group isotope but is not learnable either.

**Results** (MLP substrate, one seed each, weight decay 1; `results/grok_controls/report.md`):

| external table | test acc | classes at hidden site | blindly recovered closed-loop algebra | assoc | comm | Latin | ARI to true-output partition | law-free index plateau → after |
|---|---|---|---|---|---|---|---|---|
| random 97×97 | 0.01 | 5064, unstable | no coherent algebra | 0.01 | 0.02 | 0.00 | 0.15 | never rises (0.29 final) |
| Z_89 | 1.00 | **89** | group, cyclic | 1.00 | 1.00 | 1.00 | 1.00 | 0.27 → 0.72 |
| Z_101 | 1.00 | **101** | group, cyclic | 1.00 | 1.00 | 1.00 | 1.00 | 0.30 → 0.63 |
| Z_8 × Z_8 | 1.00 | **64** | group, **non-cyclic**, order spectrum {1, 2³, 4¹², 8⁴⁸} | 1.00 | 1.00 | 1.00 | 1.00 | 0.32 → 0.79 |
| Z_97, tokens scrambled independently | 1.00 | **97** | quasigroup, **non-associative** (isotopic to a cyclic group) | 0.02 | 1.00 | 1.00 | 1.00 | 0.28 → 0.68 |
| a − b mod 97 | 1.00 | **97** | quasigroup, **non-associative, non-commutative**, right identity only (isotopic to a cyclic group) | 0.01 | 0.01 | 1.00 | 1.00 | 0.28 → 0.76 |
| Z_97, 5 % corrupted | 0.94 | 451 | no exact algebra (Latin 0.24) | 0.93 | 0.96 | 0.24 | 0.89 | 0.31 → 0.54 |
| Z_97, 15 % corrupted | 0.04 | 5015, unstable | no coherent algebra | 0.03 | 0.12 | 0.00 | 0.15 | never rises (0.23 final) |

Every control comes out the way it must for the extractor to be trusted:

* The random table produces thousands of unstable classes and no algebra, so
  the extractor does not manufacture 97 closed classes from a 97×97 domain.
* Changing the group without telling the extractor changes the recovered
  cardinality (89, 101, 64) and the recovered isomorphism class: Z_8 × Z_8 is
  reported as a non-cyclic group whose element-order spectrum is exactly that
  of Z_8 × Z_8 (Z_64 would show elements of order 64).
* With input and output tokens scrambled independently, the hidden-site
  quotient still crystallises to 97 stable classes (ARI 1.0), but the
  closed-loop table is no longer associative: the "world" now identifies
  outputs with inputs through a different bijection, and the extractor
  reports the isotope it actually sees (Latin, commutative, non-associative,
  isotopic to a cyclic group).  Associativity is therefore measured, not
  built in.
* Subtraction is recovered as a Latin, non-associative, non-commutative
  operation with only a right identity — i.e. as what it is.
* Corruption degrades continuously: 5 % noise gives 451 classes (97 plus
  classes for corrupted entries), Latin 0.24 and closure 0.43; 15 % noise is
  not learnable and looks like the random table.  Nothing recovers the
  intended group.

A lesson from the battery: the first crystallisation index included
closed-loop associativity, so it wrongly scored the scrambled and
subtraction runs low (0.26, 0.31) although their quotients crystallised.
The index now used (`crystallization_lawfree` = √(metastability × closure) ×
non-degeneracy) contains no particular law; coherence laws are reported
alongside it.  With it, all twelve grokked runs across three groups, one
non-associative quasigroup, two presentations and two architectures move
from 0.22–0.32 on the plateau to 0.54–0.79 after generalisation, with
half-rise lags of +500 to +2500 steps behind test accuracy, while the
non-grokking runs (no weight decay, random table, 15 % corruption) stay at
0.04–0.29.



### A non-explicitly-algebraic task: predicting a small world

Everything above has mathematics on the outside: an operation table is the
objective.  The stronger thesis is that a **non-mathematically-presented
objective** can produce **internally discovered compositional
mathematics**.  `emergence/grok/world.py` and `run_world.py` test the
smallest version.

*World.* Four items in four slots.  Actions: swap slots 0-1, swap 1-2,
swap 2-3, rotate.  The observation after each action is only *which item is
in slot 0*.  A GRU (`rnn.py`) reads random action sequences (length 12; 1024
fixed training sequences, 1024 held out) and predicts the observation after
every action.  No table, no state label, no operation is ever presented; the
algebra (S_4 acting on itself: order 24, non-abelian) is implicit in the
dynamics.  A second world adds a `reset` action, so its dynamics form a
transformation monoid that is *not* a group.

*Extraction* (`extract_seq.py`) is the same blind quotient construction on
the recurrent site: things are the hidden configurations after prefixes of
actions; two are equivalent when continuing from either with the same
sampled suffixes gives the same predicted observations (Myhill-Nerode on
behaviour); clicks are (configuration, action-chunk) and their products are
configurations at the same site, so recursion is automatic; metastability
and closure as before.  The induced automaton on classes is canonically
labelled from the initial configuration (so seeds can be compared without
any alignment) and the transformation monoid it generates is computed:
whether each action is a bijection, the monoid's order, whether it is a
group, whether it is abelian.  Associativity of the action is automatic for
any deterministic sequential substrate and is therefore not claimed as a
discovery; what is *not* automatic is that a finite, stable, closed set of
classes exists at all, its cardinality, and the group it generates.

*Results* (`results/world/report.md`; three seeds of the reversible world,
two of the reset world, 6000 steps):

| run | test acc | classes (reachable) | metastability | closure | monoid order | group | abelian | actions bijective | canonical automaton = world's minimal automaton | ARI to world states |
|---|---|---|---|---|---|---|---|---|---|---|
| perm4 s0 / s1 / s2 | 1.000 | **24 (24)** | 0.72 | 0.98 | **24** | **yes** | **no** | all | **yes** (all three seeds identical) | 1.000 |
| perm4reset s0 / s1 | 0.993 | 40 (27) / 37 (25) | 0.72 | 0.80 | not total | — | — | — | no (not converged) | 0.98 / 0.97 |

From a prediction objective alone, three independently trained substrates
induce the same 24-element non-abelian group acting on the same 24 things,
with nothing in the extractor knowing what a permutation is.

The timing is different from grokking and, for the thesis, more
interesting.  Test accuracy is 0.98 by step 200, when the extractor still
sees ~90 unstable classes; the quotient then *compresses* over the next
2400 steps (96 → 73 → 51 → 37 → 28 → 24 classes) while accuracy stays at
1.0, and the law-free index rises from 0.45 to 0.77 during that phase.
Here the discrete closed algebra crystallises **after** the task is solved
behaviourally — accuracy first, algebra later — rather than coinciding with
a generalisation jump.  The reset world is slower (non-invertible actions
make more prefix configurations behaviourally distinct until late) and was
not converged at 6000 steps; the extractor correctly reports no total
algebra for it rather than a group.


### Round 2: a non-group world and a second architecture

The object being extracted is now called the substrate's **behavioural
algebra**: the discrete compositional structure obtained by quotienting the
continuous substrate by substitutability / indistinguishable future
consequences.  The hypothesis under test is

    world/task  →  minimal behavioural quotient (its algebra)
    architecture + seed  →  one continuous realisation of it

and it remains a hypothesis: a substrate that solves the task with a
*different* stable algebra, or with none, is a result to preserve, not to
tune away.  Task-learning failure ≠ extraction failure.

*Worlds.* `counter`: a saturating counter (levels 0–3) with a toggle flag;
actions `inc`, `dec` (saturating, non-invertible), `flip` (order 2),
`reset` (constant map); observation `(level ≥ 2, flag)`.  Its minimal
predictive quotient has 8 states and generates a transformation monoid of
**order 44 with 18 idempotents, 8 constant maps and a unit group of order 2,
non-commutative** — deliberately not a group.  `perm4reset` (S_4 plus a
reset): 24 states, monoid of order 48 (24 units + 24 constant maps).

*Second substrate.* A 2-layer causal transformer whose configuration after a
prefix is its key/value cache (`seqsub.py`); continuation, substitution and
noise act on that cache, so the extractor is unchanged in meaning.

*Per-cell results* (`results/world/report.md`; final checkpoints):

| world | substrate | task learned (test acc) | classes (reachable) | metastability / closure | quotient = world's minimal automaton | monoid order / idempotents / constants / units | bijective actions | cross-seed |
|---|---|---|---|---|---|---|---|---|
| perm4 | GRU ×3 | yes (1.000) | 24 (24) | 0.72 / 0.98 | **yes** | 24 / 1 / 0 / 24 (group, non-abelian) | 4 of 4 | identical, ARI 1.000 |
| perm4reset | GRU ×2, 12k steps | yes (0.999) | 29, 25 (24) | 0.74 / 0.94 | **yes** | **48 / 25 / 24 / 24** (= world's) | 4 of 5 | identical, ARI 0.993 |
| counter | GRU ×3 | yes (1.000) | 8 (8) | 0.96 / 1.00 | **yes** | **44 / 18 / 8 / 2** (= world's), non-abelian | 1 of 4 (`flip`) | identical, ARI 1.000 |
| counter | transformer ×2 | yes (0.977, 0.993) | 114, 75 (1) | 0.5 / 0.0 | no | transitions not total | — | ARI 0.83 to each other, 0.40–0.48 to world states |
| perm4 | transformer ×2, 1024 sequences | **NO** (0.61, 0.60) | 596 (1) | 0.01 / 0.00 | — | — | — | no conclusion about algebra |

Associativity of the action is automatic for a deterministic sequential
substrate and is not scored; `action coherence` (that `(c·a)·b` and `c·(ab)`
land in the same class) is 1.00 for every GRU cell and is a check on the
quotient, not a discovery.

*What the cells say.*

* **The extractor recovers the non-group monoid exactly.**  On the counter
  world every GRU seed yields the 8-state quotient and the 44-element monoid
  with the world's own idempotent/constant/unit counts; on the reset world
  the 48-element monoid.  Nothing pushed the representation toward a group:
  the same machinery that returned S_4 returns a monoid with a zero and
  eighteen idempotents when that is what the world's quotient is.
* **Timing on the counter world** is like the permutation world: the task
  is solved by step 50 and the 8-class algebra is present from step 150;
  there is no long plateau to speak of (the world is small).
* **Transformer, counter world: outcome (3).**  The transformer solves the
  task to 98–99 % but its KV-cache configurations do not quotient to a
  stable closed algebra at the standard resolution (JS < 0.05): 75–114
  classes, closure 0, products of the initial class not classifiable.  A
  resolution sweep (`results/world/eps_sweep.log`) shows it is not a
  threshold artefact in the simple sense: at JS < 0.3 the transformer
  collapses to 8–9 classes but its products still land between classes
  (transitions not total), whereas the GRU at the same coarse resolution
  over-merges to 4–6 classes with total transitions.  The transformer's
  predicted distributions are simply less crisp (test loss 0.03–0.15), and
  its cache configurations after different prefixes that are equivalent in
  the world are not behaviourally identical to the extractor.  Equivalent
  external behaviour, no stable closed quotient at this site and training
  stage.  This is preserved as a positive finding, not a failed run; whether
  longer training crystallises it is an open question this round did not
  test.
* **Transformer, permutation world: task-learning failure** with 1024
  training sequences (test 0.60 with train ≈ 1.0).  No claim about algebra
  recovery is made for that cell.  One principled retry with 4× the training
  data is reported below when complete.

Architecture independence is therefore *not* established by this round:
it holds for three GRU seeds on three worlds, and the transformer cells are
either a task failure or an outcome-(3) case.

TRANSFORMER_RERUN_PLACEHOLDER

### Running

```
pip install -e .[test,grok]
python -m emergence.grok.run train   --arch transformer,mlp --seeds 0,1,2 --out results/grok --parallel 3
python -m emergence.grok.run train   --arch transformer --seeds 0 --weight_decay 0 --tag nowd --out results/grok
python -m emergence.grok.run extract --out results/grok
python -m emergence.grok.run report  --out results/grok
results/grok_controls/battery.sh          # adversarial controls (tasks: zmod:89, zprod:8x8, sub:97, scramble:..., corrupt:..., random:97)
results/world/run_all.sh                  # world prediction (perm4, perm4reset), GRU substrate, sequence extraction
```

---

## Part III — Lean: the substrate-independent layer

Lean 4 core only (no Mathlib); `cd lean && lake build`.

`Emergence/Basic.lean` (binary clicks)
* `Interaction B` — a partial click `B → B → Option B`.
* `BehEq a b` — `∀ c, (a ⋈ c ↔ b ⋈ c) ∧ (c ⋈ a ↔ c ⋈ b)`; an equivalence.
* `EmergentType S := Quotient S.behEqSetoid` — **Type = Basin / interaction-indistinguishability**.
* `Respects` — clicking respects `BehEq` in each argument (measured by `substitutability`).
* `EmergentType.click` — descent of the click to the quotient; `click ⟦a⟧ ⟦b⟧ = (S.click a b).map ⟦·⟧`.
* `WeakAssocUpToBehEq` → `EmergentType.click_assoc_of_some`: if both bracketings are
  defined on emergent types they are **equal**.  `AssocUpToBehEq` (definedness agrees too)
  → `EmergentType.click_assoc`, full equality; the designed structure provably fails it.

`Emergence/Multi.lean` (arbitrary arity, substitution across contexts)
* `MultiInteraction B` — `op : List B → Option B`.
* `Context`, `plug`, `Fits`, `BehEq a b := ∀ c, Fits c a ↔ Fits c b` — literally the extraction's definition.
* `Respects` (one-slot substitution) ⇒ `Respects.list` (all slots).
* `EmergentType.op` — the multiary operation descends (`tupleOfTypes` turns a tuple of
  quotients into a quotient of tuples); `EmergentType.op_map`.
* `EmergentType.law_of_some` — a coherence law that holds up to `~` when both sides are
  defined holds exactly on types.

`Emergence/Equiv.lean` (task-level equivalence of idiosyncratic algebras)
* `Hom S₁ S₂` — a map preserving clicks up to `~` and reflecting/preserving `~`.
* `Hom.onTypes` — the induced map on emergent types; injective; commutes with the
  descended clicks (`onTypes_click`).
* `TaskEquiv S₁ S₂` — homs both ways inducing mutually inverse maps on types.

`Emergence/Finite.lean` — every hypothesis is decidable over `Fin k`; the
hand-written designed table is certified by `decide`.
`emergence/export.py` writes a discovered table to `Emergence/Discovered.lean`;
`lake build Emergence.Discovered` fails on the violated hypothesis if the
discovered structure does not descend.

---

## Layout

```
emergence/substrate.py    abstract substrate: sites, contexts, substitution
emergence/energy.py       energies (random MLP, designed wells), superposition
emergence/dynamics.py     Langevin, quench, relaxation time
emergence/basins.py       metastability detection (things), R_B
emergence/coupling.py     joint energy of k things
emergence/compat.py       k-ary clicks, strain, polarisation
emergence/types.py        fingerprints -> ε-types
emergence/compose.py      products, closure, substitutability, associativity, exact types
emergence/experiment.py   Langevin pipeline + composite score
emergence/export.py       finite table -> Lean certificate
emergence/grok/           table tasks + world prediction; MLP/transformer/GRU substrates; label-free extraction; comparison
lean/Emergence/*.lean     Basic, Finite, Multi, Equiv, Discovered (generated)
tests/                    gradient checks, algebra checks, comparison utilities
```

## Known limitations of v0

* Strain threshold and retention thresholds sit inside the click definitions;
  raw distributions are reported so polarisation can be judged before thresholding.
* One-block networks make the hidden site's downstream map context-free (see caveat).
* Ternary products are matched but not registered as new things in the Langevin pipeline.
* The isomorphism test propagates from a single generator; it certifies cyclic
  structures and reports failure otherwise rather than searching exhaustively.
* Not yet epiplexity; not yet an operad.  First establish the phenomenon.
