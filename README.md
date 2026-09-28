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
| crystallisation index | geometric mean of metastability, discreteness, closure, associativity |

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
| crystallisation index | 0.22–0.28 | 0.74–0.83 | 0.11 |

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
| crystallisation index | 0 … +1000 |
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
(3) absent in the control: yes.  (4) quotients converge to one equivalence
class after generalisation and are idiosyncratic before: yes.

Caveats: one task; the closed-loop associativity is expected to track
correctness for this task and is reported as self-consistency, not as an
independent signal; retention-based measures are noisy on the plateau
(0.1–0.5) and the index fluctuates with them; class thresholds (JS < 0.05,
noise grid) were fixed a priori and not tuned.


### Running

```
pip install -e .[test,grok]
python -m emergence.grok.run train   --arch transformer,mlp --seeds 0,1,2 --out results/grok --parallel 3
python -m emergence.grok.run train   --arch transformer --seeds 0 --weight_decay 0 --tag nowd --out results/grok
python -m emergence.grok.run extract --out results/grok
python -m emergence.grok.run report  --out results/grok
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
emergence/grok/           modular-addition substrates, training, label-free extraction, comparison
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
