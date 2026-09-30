# Conclusions

This document closes the research phase.  It separates what the
experiments demonstrated from what remains conjecture, states the terminal
conclusion on "clicking", and grades each arrow of the thesis

    learning → interfaces → fits → composition → higher-order interfaces

by the evidence actually obtained.  It proposes no further experiment.

Everything referred to here is in the repository: the extractors
(`emergence/grok`, `emergence/cont`, `emergence/click`, `emergence/trace`),
the Lean formalisation (`lean/Emergence`), and the results directories
named in each section.  README.md holds the per-experiment detail.

---

## 1. The final experiment: what changes inside a learner when interfaces form

Design (`emergence/trace/`, results in `results/trace/`).  For each positive
run the final interfaces were taken from the frozen extractor at the last
checkpoint (97 behavioural classes of the hidden configuration for modular
addition; the 24 classes of prefix configurations for the S_4 world) and
their identities were frozen as a partition of a fixed instance pool.  No
re-clustering of earlier checkpoints was done.  At every earlier checkpoint,
on held-out instances, the following were measured against that frozen
partition: identity (linear probe accuracy and 1-NN purity of the eventual
class), causal effect (behaviour change when the between-eventual-class
subspace of the representation is removed or kept, against a random
subspace of the same dimension), substitutability (a same-eventual-class
donor from the other split replaces the instance; behaviour agreement),
context independence (S_4: agreement over four disjoint suffix families;
modular addition: agreement through the closed loop), fit (S_4: composites
of substitutable pieces are substitutable), composition and recursion
(the eventual class of a composite is reached by the composite at that
checkpoint; two-step chunks likewise), together with the existing quotient
measures and a preregistered set of microscopic quantities (parameter norm,
effective rank and top-5 share of the representation spectrum,
between/within variance ratio and margin of the eventual types, readout
Jacobian energy in the eventual-type subspace, Jacobian context dependence,
adjacent-layer weight alignment, class-direction interference, CKA drift of
the representation between checkpoints, weight drift).  For every series
the persistent 10/50/90 % crossing steps and the width t90 − t10 were
computed.  Ten runs: three MLP and three transformer modular-addition runs
(weight decay 1), the matched transformer run without weight decay (never
generalises), and three S_4 GRU runs.

### 1.1 Modular addition (regime A)

In all six positive runs every interface property measured on held-out
inputs rises with test accuracy: same t50 to within one checkpoint (500
steps), same width (1500 steps for the MLPs, 2500–4000 for the
transformers), and they persist.  Identity (a linear probe for the eventual
class, fitted on training inputs, evaluated on unseen inputs) is 0.00
throughout memorisation and rises with accuracy; during the transition it
exceeds the model's own accuracy by 0.06–0.20 (MLP) and 0.01–0.12
(transformer), a probe advantage, not an earlier phase.  So the eventual
interfaces are *not* decodable before they are used: information,
causal specialisation, substitutability, composition and recursion appear
together, inside the generalisation transition.  The quotient's class
count compresses to 97 with t50 within one checkpoint of accuracy and t90
one to two checkpoints after; metastability and closure rise one to four
checkpoints after that, gradually.  The causal measures at the hidden site
are uninformative in a one-block model, as anticipated in the README:
removing the between-eventual-class subspace destroys behaviour from step
25 onward (0.99) because the class-mean directions of training inputs are
the readout directions of a memorising network too.

Microscopic quantities.  The effective rank of the hidden representation
falls before generalisation (t50 lead 500–4000 steps) and collapses
transiently at the transition (to 2.7–4.9 from 60–120), with a spike of
representational drift (CKA drift 0.3–0.6) at the same checkpoints.  This
looked like a candidate signature until the matched negative run was
examined: without weight decay the same transformer, which never leaves 27 %
test accuracy and never forms the interfaces (identity 0.28, substitutability
0.27, equal to its accuracy, at every checkpoint), also collapses to
effective rank 1.4–2.0 (from 42 at step 5500) and shows drift spikes of
0.6–0.7 at four checkpoints.  Parameter norm decays monotonically from the
first step under weight decay.  Interference between the eventual class
directions rises slowly before grokking (0.10 → 0.25) and equally in the
negative run (0.13 → 0.4).  No microscopic quantity changes specifically at
interface formation rather than with training time, weight decay or
accuracy.

### 1.2 S_4 world model (regime B)

Accuracy reaches 0.98 at step 200 and 0.995 at 800.  Every interface
property makes its large move with accuracy (between steps 50 and 200:
identity 0.24 → 0.90, substitutability 0.49 → 0.91, composition 0.51 →
0.94, recursion 0.46 → 0.88).  What happens after accuracy is what the
earlier report called crystallisation: substitutability, context
independence, fit, composition and recursion go from 0.95–0.98 to 1.00,
and the quotient compresses from 96 classes to 24.  Traced with frozen
identities this consolidation is:

* gradual — widths 1000–1800 steps for the interface properties and
  1400–2800 for the class count, with a maximum per-checkpoint increment of
  0.02–0.03 in substitutability; nothing sharp;
* tied to the continuing decrease of loss — Spearman correlation between
  log test loss and the substitutability deficit 0.76–1.00, and between log
  test loss and the class count 0.75–0.98, across checkpoints after
  accuracy; the class count is a threshold statistic (behavioural
  equivalence at JS < 0.05) and its compression is what a sharpening output
  distribution does to that threshold;
* microscopically featureless — representational drift is 0.00 (CKA) from
  step 400 onward, so the representation is fixed up to linear
  transformation while the between/within ratio doubles, the margin grows,
  the effective rank falls from 11 to 8 and the parameter norm from 30 to 24,
  all monotonically over 2400–4000 steps.  Jacobian context dependence stays
  at 0.45–0.52.  Nothing reorganises; something contracts.

So in the one system where interface crystallisation was known to follow
task accuracy, the follow-on phase is ordinary continued optimisation under
weight decay, seen through the thresholds of the extractor.

### 1.3 Boolean circuit (regime C, from the earlier experiments)

Perfect accuracy, perfectly decodable gates at both hidden sites, and no
resolution-stable, substitutable, sufficient interface set at any
checkpoint, under either the fixed dictionaries or a label-free behavioural
search over subspaces whose best result equals that of weight-shuffled
networks.  Task solution and decodable reusable information without
interface formation.

### 1.4 The three regimes together

    task solution  ≠  representation of reusable information  ≠  interface formation

holds as three inequalities: the circuit shows the first two without the
third; the no-weight-decay transformer memorises without either the second
(on held-out inputs) or the third; modular addition acquires all three at
once; S_4 acquires them together with accuracy and then tightens.  The
inequalities are demonstrated.  What is *not* demonstrated is that the
third has a dynamics of its own.

### 1.5 Intervention

None was performed.  The rule was one mechanistically motivated
intervention if a quantity clearly and reproducibly changed immediately
before crystallisation across positive runs.  The only candidates
(effective-rank decline, drift spike) occur equally in the matched negative
run, and the S_4 consolidation has no microscopic event at all.  Inventing
an intervention was excluded by design.

### 1.6 Terminal conclusion: C

**No independently identifiable click.**  Backward tracing with frozen
final identities shows, in modular addition, that every interface property
undergoes one transition and that transition is the generalisation
transition itself, with the same timing and width and with no earlier
decodable phase; in S_4, a gradual post-accuracy consolidation that tracks
the loss and passes through the extractor's thresholds, with no
reorganisation of the representation; and, in the matched negative, the
microscopic signatures without the interfaces.  The project provides a
method for extracting behavioural interfaces and compositional quotients
when they exist, but does not establish "clicking" as a learning
phenomenon distinct from task and generalisation dynamics.

---

## 2. Demonstrated results

1. **Label-free extraction works where structure exists.**  The
   behavioural quotient (substitutability across sampled contexts, then
   closed-loop or action-induced operations) recovers, blind, the designed
   Langevin structure (Lean-certified), Z_97 / Z_89 / Z_101 / Z_8×Z_8 and
   their isotopes and corruptions, S_4 (order 24), the reset monoid (order
   48), the counter monoid (order 44), and the analytic minima of continuous
   landscapes, identically across seeds and across MLP/transformer for
   modular addition.  Idiosyncratic presentations compare correctly at the
   quotient level.
2. **The extractor does not manufacture structure.**  Random tables,
   memorising networks, a single well, a random landscape, the shuffled
   controls and the weight-shuffled nulls all come out empty.
3. **Interfaces are not required for generalisation.**  The Boolean-circuit
   MLPs generalise perfectly, encode the generating gates linearly at both
   sites, and form no stable substitutable interface at any pressure level
   or checkpoint, under fixed or searched dictionaries.
4. **Exact structure appears only where the task's predictive quotient is
   finite.**  The resolution-dependent quotient N(ε) develops a
   decade-wide plateau with zero congruence defect only for finite-quotient
   tasks; the random-source students and the circuit are continuous
   behavioural objects (power-law N(ε), no plateau) at every checkpoint.
5. **Timing.**  In modular addition all interface properties transition
   with generalisation (Section 1.1); in the finite worlds the quotient
   tightens after accuracy, gradually, following the loss (Section 1.2).
6. **The exact layer is substrate-independent.**  Behavioural equivalence
   is always an equivalence; congruence is automatic when contexts are
   closed under the interaction (so coherence in a deterministic sequential
   substrate is not a discovery); congruent actions, partial operations and
   multiway assemblies descend to the quotient; laws that hold up to
   equivalence hold exactly on types; the interface of an assembly is its
   action on interfaces, so the notion applies recursively
   (`lean/Emergence/*.lean`, no Mathlib).

## 3. Conjecture (not supported by these experiments)

* That some ordinary task without a finite predictive quotient drives a
  learner to a finite plateau.  Every attempt to find one (random recurrent
  source, Boolean circuit) produced none.
* That compression pressure alone produces interfaces.  Weight decay up to
  1.0 on the circuit produced a labelled trend (gate directions becoming
  resolution-stable) that no label-free search could find and that never
  became sufficient or compositional.
* That interface formation has a mechanism distinct from optimisation
  under weight decay.  Nothing measured supports it.

---

## 4. The arrows, graded

| arrow | grade | basis |
|---|---|---|
| learning → interfaces | **demonstrated only on structured tasks; negative counterexample exists** | Finite-quotient tasks (modular addition, S_4, reset, counter) yield stable substitutable interfaces; the Boolean circuit and the random recurrent source yield none, at any checkpoint or pressure, under fixed or searched dictionaries. |
| interfaces → fits | **demonstrated only on structured tasks** | Where interfaces exist their fits are reproducible across contexts (context independence and fit 1.00 in S_4; congruence defect 0 in the finite worlds).  Where they do not exist there is nothing to fit. |
| fits → composition / reuse | **demonstrated only on structured tasks** | Closed-loop and action-induced operations are total on the quotient and coherent (associativity 1.00, action coherence 1.00); the same class participates in all actions (reuse 1.00 in S_4).  The Boolean circuit never produced two interfaces to compose. |
| composition → higher-order interfaces | **demonstrated only on structured tasks (and exactly in Lean)** | Two-step composites land in the eventual class of the composite of composites (recursion 1.00 in S_4; associativity in modular addition); in Lean the interface of an assembly is exactly its action on interfaces. Empirically observed only for the finite algebras that were already dictated by the task. |
| "clicking" as a distinct event | **unsupported** | Section 1. |

## 5. The seven questions

1. **What did grokking contribute?**  A clean before/after: thousands of
   unstable classes during memorisation, 97 stable, closed, associative
   classes after generalisation, identical across seeds and architectures.
   Traced backward it contributed one more thing: the demonstration that
   in this system the interface transition *is* the generalisation
   transition — same t50, same width, no earlier decodable phase — so
   grokking cannot be used as evidence that interfaces have a dynamics of
   their own.
2. **Is interface formation distinct from ordinary representation
   learning?**  Not as observed.  The three inequalities of Section 1.4
   hold, so the *outcomes* are distinct, but the *process* that produced
   interfaces where they appeared is indistinguishable from generalisation
   (modular addition) or from continued loss decrease under weight decay
   (S_4).  No microscopic quantity singled it out.
3. **Is interface formation necessary for generalisation?**  No.  The
   Boolean-circuit MLPs generalise perfectly without forming any.
4. **Is interface formation sufficient for generalisation?**  Not tested
   directly, and no case supports it: every run that formed interfaces
   had already generalised or was doing so at the same time.  In the
   no-weight-decay transformer no interfaces form and no generalisation
   occurs; that is consistent with sufficiency but is equally consistent
   with both being effects of weight decay.
5. **When exact algebra appeared, was it invented or recovered?**
   Recovered.  Every exact algebra found (Z_97 and the other cyclic groups,
   Z_8×Z_8, the quasigroup of subtraction, S_4, the reset and counter
   monoids, the double-well operations) is the task's own predictive
   quotient; the extractor found it blind, but the task had put it there.
   The idiosyncratic presentations (scrambled tokens) are isotopes of the
   same object, not new ones.
6. **Did we ever observe internally invented compositional organisation
   not dictated by the task?**  No.  The random-source students developed
   predictive, interventionally real types that fragmented with training
   and never closed into a canonical algebra.  The Boolean circuit, built so
   that its decomposition was underdetermined, produced no compositional
   organisation at all; the four isolated acceptances of the behavioural
   search were unrelated to the circuit and also unstable, insufficient and
   non-compositional.
7. **What remains of "things click together and mathematics appears"?**
   The second half survives in a weak form: where a task has a finite
   predictive quotient, a learner that generalises comes to represent it
   as exactly that quotient, with exact laws on the quotient, and this can
   be extracted without labels and certified.  The first half does not
   survive: nothing was seen to click.  The transition is the
   generalisation transition where there is one, and a gradual contraction
   where there is not; in the one task where the mature structure was not
   dictated in advance, no structure formed; and no internal mechanism was
   found that produces interfaces rather than accuracy.  The mathematics
   that appeared was recovered, not made.

---

## 6. Limitations that bound these conclusions

* Positive systems are small (one-block models, a 128-unit GRU) and the
  finite-quotient tasks are all algebraic; the causal measures at a
  readout-adjacent hidden site are uninformative for one-block models.
* The interface criteria use fixed resolutions and JS thresholds; the
  resolution analysis mitigates but does not remove this.
* "No interfaces" on the circuit and the random source is a negative
  result under two dictionaries (fixed and behaviourally searched) and one
  interface definition; it is not a proof of absence.
* Time series have checkpoint spacing 200–500 steps; transitions narrower
  than that are not resolved.
* Cross-seed sample sizes are three; no statistical test beyond
  agreement across runs was attempted.
