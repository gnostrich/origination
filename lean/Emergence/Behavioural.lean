import Emergence.Basic

/-!
# Behavioural systems: the substrate-independent layer

A *behavioural system* is a set of states or histories `H`, a set of contexts
`C` (future continuations, interactions, probes) and an observation
`obs : H → C → O`.  Nothing else about the substrate is assumed.  Everything
the experiments measure is a finite-sample, ε-approximate version of the
exact notions defined here.

* `BehEq h h'` — exact behavioural equivalence: indistinguishable in every
  context.  It is always an equivalence relation, so the quotient
  `H / ~` (`Quot`) exists unconditionally: **types are quotients of states
  by indistinguishability**.
* `Congruent act` — an interaction respects `~`.  `congruent_of_closed`
  makes precise when this is *automatic*: if the contexts are closed under
  pre-composition with the interaction (every "act then observe in `c`" is
  itself some context `c'`), congruence holds for free.  This is the exact
  form of the caveat that associativity/congruence of a deterministic
  sequential substrate is not a discovery.  The empirical congruence defect
  measures the failure of this closure for *sampled* contexts.
* Descent: a congruent action, and a congruent partial binary operation,
  descend to `Quot`; laws that hold up to `~` on `H` hold exactly on `Quot`
  (`actQ_law`, `opQ_assoc_of_some`).
* `Hom` — a structure-preserving map between two systems (a map on states
  with a pull-back on contexts preserving observations) descends to the
  quotients; with a surjective pull-back it is injective on quotients.
  `QuotEquiv` — two systems with equivalent behavioural quotients.
* The four notions distinguished in the write-up:
  - A `FiniteQuotient`: finitely many exact types (a finite covering list);
  - C `ApproxCovered Close`: finitely many ε-types for a closeness relation;
    `finiteQuotient_approxCovered` — a finite exact quotient is finitely
    covered at every coarser resolution;
  - B `TotallyBounded`: finitely covered at *every* resolution but not
    necessarily finite — a continuous behavioural object, not "no structure";
  - D `ExactAlgebra`: descent + laws, i.e. the quotient carries an exact
    operation (`actQ` / `opQ`).
-/

universe u v w u₁ u₂ v₁ v₂

namespace Emergence

structure BehSystem (H : Type u) (C : Type v) (O : Type w) where
  obs : H → C → O

namespace BehSystem

variable {H : Type u} {C : Type v} {O : Type w} (S : BehSystem H C O)

/-- Exact behavioural equivalence. -/
def BehEq (h h' : H) : Prop := ∀ c, S.obs h c = S.obs h' c

theorem BehEq.refl (h : H) : S.BehEq h h := fun _ => rfl
theorem BehEq.symm {h h' : H} (e : S.BehEq h h') : S.BehEq h' h := fun c => (e c).symm
theorem BehEq.trans {a b d : H} (e₁ : S.BehEq a b) (e₂ : S.BehEq b d) : S.BehEq a d :=
  fun c => (e₁ c).trans (e₂ c)

def setoid : Setoid H := ⟨S.BehEq, ⟨BehEq.refl S, fun e => BehEq.symm S e, fun e₁ e₂ => BehEq.trans S e₁ e₂⟩⟩

/-- The behavioural quotient `H / ~`. -/
def Quot : Type u := Quotient S.setoid

def typeOf (h : H) : S.Quot := Quotient.mk S.setoid h

theorem typeOf_eq_iff {h h' : H} : S.typeOf h = S.typeOf h' ↔ S.BehEq h h' :=
  ⟨fun e => Quotient.exact e, fun e => Quotient.sound e⟩

/-! ### Interactions and congruence -/

section Action

variable {A : Type u₁}

/-- An interaction respects behavioural equivalence. -/
def Congruent (act : A → H → H) : Prop :=
  ∀ a h h', S.BehEq h h' → S.BehEq (act a h) (act a h')

/-- **Automatic congruence.**  If for every action `a` and context `c` there is
a context `c'` such that observing `act a h` in `c` is observing `h` in `c'`
(contexts are closed under pre-composition with the interaction), then the
interaction is congruent.  For a deterministic sequential substrate whose
contexts are all continuations this always holds: congruence there is a
consequence of the definitions, not a discovery. -/
theorem congruent_of_closed (act : A → H → H)
    (hcl : ∀ a c, ∃ c', ∀ h, S.obs (act a h) c = S.obs h c') : S.Congruent act := by
  intro a h h' e c
  obtain ⟨c', hc'⟩ := hcl a c
  rw [hc' h, hc' h']
  exact e c'

/-- Descent of a congruent interaction to the quotient. -/
def actQ {act : A → H → H} (hc : S.Congruent act) (a : A) : S.Quot → S.Quot :=
  Quotient.lift (s := S.setoid) (fun h => S.typeOf (act a h))
    (fun h h' e => Quotient.sound (hc a h h' e))

@[simp] theorem actQ_typeOf {act : A → H → H} (hc : S.Congruent act) (a : A) (h : H) :
    S.actQ hc a (S.typeOf h) = S.typeOf (act a h) := rfl

/-- **Inheritance of laws.**  Any law of the form `act a (act b h) ~ act (m a b) h`
that holds *up to behavioural equivalence* on states holds *exactly* on types. -/
theorem actQ_law {act : A → H → H} (hc : S.Congruent act) (m : A → A → A)
    (law : ∀ a b h, S.BehEq (act a (act b h)) (act (m a b) h)) :
    ∀ a b x, S.actQ hc a (S.actQ hc b x) = S.actQ hc (m a b) x := by
  intro a b x
  induction x using Quotient.inductionOn with | _ h =>
  exact Quotient.sound (law a b h)

end Action

/-! ### Partial binary operations -/

section Op

def OptBehEq : Option H → Option H → Prop
  | none, none => True
  | some d, some d' => S.BehEq d d'
  | _, _ => False

theorem OptBehEq.map_typeOf_eq {o o' : Option H} (e : S.OptBehEq o o') :
    o.map S.typeOf = o'.map S.typeOf := by
  cases o <;> cases o' <;> simp_all [OptBehEq, Option.map]
  exact Quotient.sound e

/-- A partial binary interaction respecting `~` in each argument. -/
structure Congruent₂ (op : H → H → Option H) : Prop where
  left : ∀ a a' b, S.BehEq a a' → S.OptBehEq (op a b) (op a' b)
  right : ∀ a b b', S.BehEq b b' → S.OptBehEq (op a b) (op a b')

theorem OptBehEq.trans {o₁ o₂ o₃ : Option H} (e₁ : S.OptBehEq o₁ o₂) (e₂ : S.OptBehEq o₂ o₃) :
    S.OptBehEq o₁ o₃ := by
  cases o₁ <;> cases o₂ <;> cases o₃ <;> simp_all [OptBehEq]
  exact BehEq.trans S e₁ e₂

/-- Descent of a congruent partial operation. -/
def opQ {op : H → H → Option H} (hc : S.Congruent₂ op) : S.Quot → S.Quot → Option S.Quot :=
  Quotient.lift₂ (s₁ := S.setoid) (s₂ := S.setoid) (fun a b => (op a b).map S.typeOf)
    (fun a b a' b' ea eb =>
      OptBehEq.map_typeOf_eq S (OptBehEq.trans S (hc.left a a' b ea) (hc.right a' b b' eb)))

@[simp] theorem opQ_typeOf {op : H → H → Option H} (hc : S.Congruent₂ op) (a b : H) :
    S.opQ hc (S.typeOf a) (S.typeOf b) = (op a b).map S.typeOf := rfl

/-- **Closure** descends: a total operation on states is total on types. -/
theorem opQ_total {op : H → H → Option H} (hc : S.Congruent₂ op) (ht : ∀ a b, (op a b).isSome = true) :
    ∀ x y, (S.opQ hc x y).isSome = true := by
  intro x y
  induction x using Quotient.inductionOn with | _ a =>
  induction y using Quotient.inductionOn with | _ b =>
  show ((op a b).map S.typeOf).isSome = true
  have := ht a b
  revert this
  cases op a b <;> simp

theorem bind_map_typeOf {op : H → H → Option H} (hc : S.Congruent₂ op) (o : Option H) (c : H) :
    (o.map S.typeOf).bind (fun x => S.opQ hc x (S.typeOf c)) = (o.bind fun x => op x c).map S.typeOf := by
  cases o <;> rfl

theorem map_typeOf_bind {op : H → H → Option H} (hc : S.Congruent₂ op) (a : H) (o : Option H) :
    (o.map S.typeOf).bind (fun y => S.opQ hc (S.typeOf a) y) = (o.bind fun y => op a y).map S.typeOf := by
  cases o <;> rfl

/-- Weak associativity up to `~` (both bracketings defined ⇒ results equivalent)
becomes exact equality on types. -/
theorem opQ_assoc_of_some {op : H → H → Option H} (hc : S.Congruent₂ op)
    (wa : ∀ a b c d d', (op a b).bind (fun x => op x c) = some d →
                          (op b c).bind (fun y => op a y) = some d' → S.BehEq d d')
    (a b c : H) {w w' : S.Quot}
    (hl : (S.opQ hc (S.typeOf a) (S.typeOf b)).bind (fun x => S.opQ hc x (S.typeOf c)) = some w)
    (hr : (S.opQ hc (S.typeOf b) (S.typeOf c)).bind (fun y => S.opQ hc (S.typeOf a) y) = some w') :
    w = w' := by
  rw [opQ_typeOf, S.bind_map_typeOf hc] at hl
  rw [opQ_typeOf, S.map_typeOf_bind hc] at hr
  revert hl hr
  cases hL : (op a b).bind (fun x => op x c) with
  | none => intro hl; simp at hl
  | some d =>
    cases hR : (op b c).bind (fun y => op a y) with
    | none => intro _ hr; simp at hr
    | some d' =>
      intro hl hr
      simp only [Option.map, Option.some.injEq] at hl hr
      subst hl; subst hr
      exact Quotient.sound (wa a b c d d' hL hR)

end Op

/-! ### Structure-preserving maps and equivalence of quotients -/

section Hom

variable {H₁ : Type u₁} {C₁ : Type v₁} {H₂ : Type u₂} {C₂ : Type v₂}

/-- A map of states with a pull-back of contexts that preserves observations. -/
structure Hom (S₁ : BehSystem H₁ C₁ O) (S₂ : BehSystem H₂ C₂ O) where
  onState : H₁ → H₂
  onCtx : C₂ → C₁
  obs_eq : ∀ h c, S₂.obs (onState h) c = S₁.obs h (onCtx c)

variable {S₁ : BehSystem H₁ C₁ O} {S₂ : BehSystem H₂ C₂ O} (f : Hom S₁ S₂)

theorem Hom.preserves {h h' : H₁} (e : S₁.BehEq h h') : S₂.BehEq (f.onState h) (f.onState h') := by
  intro c
  rw [f.obs_eq, f.obs_eq]
  exact e (f.onCtx c)

/-- With a surjective pull-back of contexts, equivalence is also reflected. -/
theorem Hom.reflects (surj : ∀ c₁, ∃ c₂, f.onCtx c₂ = c₁) {h h' : H₁}
    (e : S₂.BehEq (f.onState h) (f.onState h')) : S₁.BehEq h h' := by
  intro c₁
  obtain ⟨c₂, rfl⟩ := surj c₁
  have := e c₂
  rwa [f.obs_eq, f.obs_eq] at this

/-- The induced map on behavioural quotients. -/
def Hom.onQuot : S₁.Quot → S₂.Quot :=
  Quotient.lift (s := S₁.setoid) (fun h => S₂.typeOf (f.onState h))
    (fun _ _ e => Quotient.sound (f.preserves e))

@[simp] theorem Hom.onQuot_typeOf (h : H₁) : f.onQuot (S₁.typeOf h) = S₂.typeOf (f.onState h) := rfl

theorem Hom.onQuot_injective (surj : ∀ c₁, ∃ c₂, f.onCtx c₂ = c₁) :
    ∀ x y, f.onQuot x = f.onQuot y → x = y := by
  intro x y
  induction x using Quotient.inductionOn with | _ a =>
  induction y using Quotient.inductionOn with | _ b =>
  intro e
  exact Quotient.sound (f.reflects surj (Quotient.exact e))

/-- The induced map commutes with descended congruent interactions that `f` intertwines. -/
theorem Hom.onQuot_act {A : Type u} {act₁ : A → H₁ → H₁} {act₂ : A → H₂ → H₂}
    (hc₁ : S₁.Congruent act₁) (hc₂ : S₂.Congruent act₂)
    (inter : ∀ a h, f.onState (act₁ a h) = act₂ a (f.onState h)) (a : A) (x : S₁.Quot) :
    f.onQuot (S₁.actQ hc₁ a x) = S₂.actQ hc₂ a (f.onQuot x) := by
  induction x using Quotient.inductionOn with | _ h =>
  show S₂.typeOf (f.onState (act₁ a h)) = S₂.typeOf (act₂ a (f.onState h))
  rw [inter]

/-- Two systems whose behavioural quotients are equivalent: the substrates may
differ arbitrarily, the types correspond one to one. -/
structure QuotEquiv (S₁ : BehSystem H₁ C₁ O) (S₂ : BehSystem H₂ C₂ O) where
  fwd : Hom S₁ S₂
  bwd : Hom S₂ S₁
  left_inv : ∀ x, bwd.onQuot (fwd.onQuot x) = x
  right_inv : ∀ y, fwd.onQuot (bwd.onQuot y) = y

end Hom

/-! ### The four notions: finite, approximate, totally bounded, exact algebra -/

/-- A relation is *finitely covered* when a finite list of states reaches every
state through it.  With `R = BehEq` this is **A**, a finite quotient. -/
def Covered (_S : BehSystem H C O) (R : H → H → Prop) : Prop := ∃ l : List H, ∀ h, ∃ s, s ∈ l ∧ R s h

/-- **A.** Finitely many exact types. -/
def FiniteQuotient : Prop := S.Covered S.BehEq

/-- **C.** Finitely many ε-types, for a closeness relation `Close` standing for
"`d_B ≤ ε`" at some resolution. -/
def ApproxCovered (Close : H → H → Prop) : Prop := S.Covered Close

/-- A finite exact quotient is finitely covered at every resolution that is
coarser than exact equivalence. -/
theorem finiteQuotient_approxCovered (Close : H → H → Prop)
    (coarser : ∀ h h', S.BehEq h h' → Close h h') (hf : S.FiniteQuotient) : S.ApproxCovered Close := by
  obtain ⟨l, hl⟩ := hf
  refine ⟨l, fun h => ?_⟩
  obtain ⟨s, hs, e⟩ := hl h
  exact ⟨s, hs, coarser s h e⟩

/-- **B.** Finitely covered at every resolution of a family `Close n` (finer as
`n` grows).  This is the *continuous behavioural object*: a quotient that need
not be finite but has finite complexity `N(ε)` at every ε.  `FiniteQuotient`
implies it whenever each `Close n` is coarser than exact equivalence. -/
def TotallyBounded (Close : Nat → H → H → Prop) : Prop := ∀ n, S.ApproxCovered (Close n)

theorem totallyBounded_of_finite (Close : Nat → H → H → Prop)
    (coarser : ∀ n h h', S.BehEq h h' → Close n h h') (hf : S.FiniteQuotient) :
    S.TotallyBounded Close :=
  fun n => S.finiteQuotient_approxCovered (Close n) (coarser n) hf

/-- **D.** An exact algebra on the quotient: a congruent, total, weakly
associative partial operation.  By `opQ_total` and `opQ_assoc_of_some` the
descended operation is total and exactly associative wherever defined. -/
structure ExactAlgebra (op : H → H → Option H) : Prop where
  congruent : S.Congruent₂ op
  total : ∀ a b, (op a b).isSome = true
  weak_assoc : ∀ a b c d d', (op a b).bind (fun x => op x c) = some d →
                 (op b c).bind (fun y => op a y) = some d' → S.BehEq d d'

end BehSystem

end Emergence
