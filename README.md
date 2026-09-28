# origination — v0: discrete states, clicks, types and composition from continuous dynamics

This repository is a deliberately small v0 of the ontology-emergence programme.
It tests exactly one proposition:

> Can continuous, initially untyped dynamics generate metastable discrete
> states whose interaction behaviour induces types, and whose stable multiway
> interactions compose?

Nothing about nodes, edges, objects, types, ports or arities is supplied.
The only primitive is an energy function `E_θ : R^N → R` and overdamped
Langevin dynamics

    dx = -∇E_θ(x) dt + sqrt(2T) dW.

Everything discrete is *measured*, never declared.  The Python side discovers
a finite interaction structure; the Lean side proves what such a structure
induces once you quotient by interaction-indistinguishability.

```
random / designed energy E on R^N          emergence/energy.py
        ↓ Langevin trajectories             emergence/dynamics.py
        ↓ metastability detection           emergence/basins.py       (Step 1: things)
basins B₁…Bₖ, R_B = τ_escape / τ_relax
        ↓ joint dynamics of k reps          emergence/coupling.py
        ↓ compatibility tensor C(i,j,…)     emergence/compat.py       (Steps 2, 4: clicks, arity)
        ↓ fingerprints → ε-classes          emergence/types.py        (Step 3: types)
        ↓ products, closure, substitution,
          associativity                     emergence/compose.py      (Step 5: composition)
        ↓ finite table                      emergence/export.py
Lean: BehEq, quotient, descent, exactness   lean/Emergence/*.lean
```

## The one structural assumption

The substrate is `R^N`.  A *thing* is a metastable basin of `E`.  When `k`
things are put together, the joint energy is

    E_k(x₁…x_k) = Σᵢ E(xᵢ) + g · E( (x₁ + … + x_k) / √k ).

That is: the composite configuration is the normalised superposition of the
parts and it is judged by the **same** energy that judges the parts.  `g` is
the interaction strength; `k` is just how many things were put in the box.
No port, arity, edge or type is introduced.  (The `1/√k` keeps the
superposition on the parts' scale for near-orthogonal parts, which random
basin representatives in high dimension are, and makes `(A, A)` non-trivial.)

The product of a click is the quenched superposition, a point of `R^N` that
is then treated *exactly* like every other discovered basin: it gets
timescales, a fingerprint, a type, and is itself clicked with everything.

## What is measured

| step | quantity | definition |
|---|---|---|
| 1 things | basins, `R_B` | quench trajectory samples to minima, lump minima that interconvert within a few relaxation times (lag-transition frequency), keep lumps with `R_B = τ_esc/τ_relax ≥ R_min` |
| 2 clicks | `C(A,B)` | fraction of joint trajectories from `(a,b)` in which every part is still classified as its basin at every checkpoint over the horizon **and** the joint minimum has strain `≤ strain_tol·T` |
| — | strain | `Σᵢ [E(xᵢ*) − E(quench xᵢ*)] + g [E(s*) − E(quench s*)] ≥ 0`; zero iff every part sits at its own minimum and the superposition is itself a minimum |
| 3 types | `[A]_∼` | `A ∼ A'` iff `max_B |C(A,B) − C(A',B)| < ε` on rows and columns (complete linkage); the exact table-level relation (`Lean: BehEq`) is also computed |
| 4 arity | `C(A,B,C)` | same as pairs with `k = 3`; a triple is *irreducible* when it clicks and all three pairs do not |
| 5 composition | closure, substitutability, associativity | fraction of clicks whose product is metastable; `A ∼ A' ⇒ A∘B` defined iff `A'∘B` defined and `A∘B ∼ A'∘B`; `(A∘B)∘C ∼ A∘(B∘C)` whenever both are defined |

Composite score: `𝒞 = metastability × polarization × type-compression × closure`,
where polarization is the bimodality of `C` (zero if all entries are 0.5,
one if all are in {0,1}; multiplied by 0 if only one class is present) and
type-compression is `1 − n_types / n_basins`.

Why strain and not survival alone: with a smooth coupling an incompatible
pair usually still has a joint local minimum in which the parts are pulled
off their wells so that the superposition lands near *some* well.  Those
strained contacts survive the horizon at any `g` (verified: raising `g` from
2 to 8 changes nothing), so survival alone says "everything clicks".  The
strain energy of the joint minimum is what is bimodal: exactly 0 for a
genuine click, of order the well depth for a strained contact.  The raw
strain distribution is reported before any threshold is applied.

## Experiment A — hand-designed landscape (validates the machinery)

`designed_landscape()` places Gaussian wells (softmin energy, unit curvature)
so that a known structure exists: primitives `A1 A2 B C P Q R J` on
orthogonal directions; product wells exactly at the superpositions
`D1 = A1·B`, `D2 = A2·B`, `Dp = B·C`, `E1 = D1·C`, `E2 = D2·C`,
`Ep1 = A1·Dp`, `Ep2 = A2·Dp`, and a ternary-only `S = P·Q·R`.
The pipeline never sees the names.

Result (`python -m emergence run --landscape designed`, N = 64, T = 0.1, g = 2):

| measurement | outcome |
|---|---|
| things | 16 microstates → 14 metastable basins (E1/Ep1 and E2/Ep2 each lumped: they are 1.2 apart and interconvert quickly), all with `R ≥ 27` |
| clicks | exactly the 7 designed pairs (14 symmetric entries) out of 196; `C` bimodality 0.999 |
| strain | bimodal: 10 % of surviving joint trajectories at strain ≈ 0, 63 % above 25 T |
| types | 6 classes: `{A1,A2}`, `{D1,D2}`, `{B}`, `{C}`, `{Dp}`, terminal `{P,Q,R,J,E1,E2,S}` — microscopically different things become one type because the world interacts with them identically |
| arity | 3 ternary clicks among 220 triples; 1 irreducible: `P·Q·R` with all pairs incompatible |
| composition | closure 14/14; substitutability 1.0 over 6 instances; associativity 4/4 both-defined triples (`(A1·B)·C = E1 = A1·(B·C)`), 8 one-sided |
| Lean | exported table certified: `Respects` and `WeakAssocUpToBehEq` proved by `decide` |
| score | `𝒞 = 0.57` (1.00 × 1.00 × 0.57 × 1.00) |

Everything the landscape was designed to contain is recovered, and nothing
else.  One-sided associativity is expected and correct: `(B·C)·A1 = Ep1` is
defined but `B·(C·A1)` is not because `C` and `A1` do not click.  So the
structure is a partial commutative magma, not a category; see the Lean note
below.

## Experiment B — random landscape (no structural supervision)

`MLPEnergy` is a random 2-layer tanh MLP plus weak quadratic confinement.
A random MLP at its smoothest setting has a single basin; roughness is set by
`input_gain` (feature length scale) and `scale` (energy amplitude).  The
sweep tabulates `𝒞` and its factors against `T` and `g` for several seeds:

    python -m emergence sweep --landscape random --T 0.1,0.3,1.0 --g 1,4 --seeds 0,1,2

Results of the sweep run in this repository are in `results/sweep/sweep.md`
and summarised in the section **Findings** below.

## Lean: from approximate physical sameness to exact algebra

`lean/Emergence/Basic.lean` (Lean 4 core only, no Mathlib):

* `Interaction B` — a partial click `B → B → Option B`.
* `BehEq a b` — `∀ c, (a ⋈ c ↔ b ⋈ c) ∧ (c ⋈ a ↔ c ⋈ b)`; proved an equivalence.
* `EmergentType S := Quotient S.behEqSetoid` — **Type = Basin / interaction-indistinguishability**.
* `Respects` — clicking respects `BehEq` in each argument (measured by `substitutability`).
* `EmergentType.click` — the click descends to the quotient (`Quotient.lift₂`), with
  `click ⟦a⟧ ⟦b⟧ = (S.click a b).map ⟦·⟧`.
* `WeakAssocUpToBehEq` — `(a∘b)∘c ∼ a∘(b∘c)` whenever both are defined (measured by `associativity`).
* `EmergentType.click_assoc_of_some` — under `Respects` and weak associativity, if both
  bracketings are defined on emergent types they are **equal**.
* `AssocUpToBehEq` / `EmergentType.click_assoc` — the strong version (definedness agrees too)
  gives full equality of the two `Option`s; the designed structure provably does *not*
  satisfy it (`example : ¬ designed.AssocUpToBehEq := by decide`).

`lean/Emergence/Finite.lean` makes every hypothesis decidable over `Fin k` and
certifies the hand-written designed table.  `emergence/export.py` writes the
*discovered* table to `lean/Emergence/Discovered.lean` in the same form; if
the discovered structure violates a hypothesis, `lake build Emergence.Discovered`
fails on that theorem (this was observed for the early survival-only click
criterion, whose table was neither substitution-respecting nor associative).

## Running

```
pip install -e .[test]            # numpy, scipy, pytest
pytest
python -m emergence run --landscape designed --out results/designed \
       --lean-out lean/Emergence/Discovered.lean
python -m emergence run --landscape random --seed 0 --T 0.3 --g 4
python -m emergence sweep --landscape random --T 0.1,0.3,1.0 --g 1,4 --seeds 0,1,2

cd lean && lake build && lake build Emergence.Discovered   # Lean 4 v4.22.0 via elan
```

A designed run takes ~3 min on a laptop CPU; all heavy steps are batched numpy.

## Known artifacts and limitations of v0

* **Strain threshold.** `C` is a probability over trajectories, but a threshold
  `strain ≤ strain_tol·T` sits inside it.  The raw strain distribution is
  reported so polarization can be judged before thresholding.
* **Absorption clicks.** Under superposition coupling a strained contact
  often relaxes so that one part shrinks and the other grows until the
  superposition *is* one of the parts (`Q·J → J`).  The strain criterion
  rejects these; without it they dominate.
* **Entropic barriers.** At `N = 64`, `T = 0.1` the thermal cloud radius
  `sqrt(N T) ≈ 2.5` exceeds half the well spacing; metastability is largely
  entropic.  `R_B` is measured, not assumed.
* **Censored escape times.** `τ_esc` is capped at the escape horizon; `R` is
  then a lower bound (flagged `escape_censored`).
* **Types use pairwise behaviour only**, so that products of any arity are
  comparable; ternary behaviour is reported separately.
* **Ternary products are matched but not registered** as new basins in v0.
* Not yet epiplexity; not yet an operad.  First establish the phenomenon.
