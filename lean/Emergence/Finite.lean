import Emergence.Basic

/-!
# Finite interaction structures and decidability

A structure discovered by the Python pipeline is a table over `Fin k`.  All
the hypotheses of `Emergence.Basic` become decidable propositions, so a
concrete exported table can be certified by `decide`.

`Interaction.ofTable` builds an interaction from a `k × k` table of
`Option Nat` (an entry `some j` with `j < k` means the click is defined with
product `j`; anything else means undefined).
-/

namespace Emergence.Finite

open Interaction

/-- Decidable `∀` over `Fin n` (avoids depending on Mathlib). -/
instance decForallFin {n : Nat} (p : Fin n → Prop) [DecidablePred p] : Decidable (∀ i, p i) :=
  decidable_of_iff (∀ i : Nat, (h : i < n) → p ⟨i, h⟩)
    ⟨fun H i => H i.val i.isLt, fun H i h => H ⟨i, h⟩⟩

variable {n : Nat} (S : Interaction (Fin n))

instance decBehEq (a b : Fin n) : Decidable (S.BehEq a b) :=
  inferInstanceAs (Decidable (∀ _, _ ∧ _))

instance decOptBehEq (o o' : Option (Fin n)) : Decidable (S.OptBehEq o o') := by
  cases o <;> cases o' <;> simp only [OptBehEq] <;> infer_instance

instance decOptWeakEq (o o' : Option (Fin n)) : Decidable (S.OptWeakEq o o') := by
  cases o <;> cases o' <;> simp only [OptWeakEq] <;> infer_instance

instance decAssoc : Decidable S.AssocUpToBehEq :=
  inferInstanceAs (Decidable (∀ _ _ _, S.OptBehEq _ _))

instance decWeakAssoc : Decidable S.WeakAssocUpToBehEq :=
  inferInstanceAs (Decidable (∀ _ _ _, S.OptWeakEq _ _))

instance decRespects : Decidable S.Respects :=
  decidable_of_iff
    ((∀ a a' b, S.BehEq a a' → S.OptBehEq (S.click a b) (S.click a' b)) ∧
      (∀ a b b', S.BehEq b b' → S.OptBehEq (S.click a b) (S.click a b')))
    ⟨fun h => ⟨h.1, h.2⟩, fun h => ⟨h.left, h.right⟩⟩

/-- Build an interaction on `Fin k` from a nested list.  An entry `some j`
with `j < k` is a defined click with product `j`; anything else is undefined. -/
def ofTable (k : Nat) (t : List (List (Option Nat))) : Interaction (Fin k) where
  click a b :=
    match (t.getD a.val []).getD b.val none with
    | some j => if h : j < k then some ⟨j, h⟩ else none
    | none => none

/-- The interaction structure of the hand-designed landscape (Experiment A),
written by hand.  Things: 0=A1 1=A2 2=B 3=C 4=D1 5=D2 6=Dp 7=E1 8=E2 9=Ep1 10=Ep2 11=J.
The pipeline rediscovers this table; `emergence/export.py` writes the
discovered one to `Emergence/Discovered.lean` in the same format. -/
def designed : Interaction (Fin 12) :=
  ofTable 12 [
    -- A1: clicks with B -> D1, with Dp -> Ep1
    [none, none, some 4, none, none, none, some 9, none, none, none, none, none],
    -- A2: clicks with B -> D2, with Dp -> Ep2
    [none, none, some 5, none, none, none, some 10, none, none, none, none, none],
    -- B: clicks with A1 -> D1, A2 -> D2, C -> Dp
    [some 4, some 5, none, some 6, none, none, none, none, none, none, none, none],
    -- C: clicks with B -> Dp, D1 -> E1, D2 -> E2
    [none, none, some 6, none, some 7, some 8, none, none, none, none, none, none],
    -- D1: clicks with C -> E1
    [none, none, none, some 7, none, none, none, none, none, none, none, none],
    -- D2: clicks with C -> E2
    [none, none, none, some 8, none, none, none, none, none, none, none, none],
    -- Dp: clicks with A1 -> Ep1, A2 -> Ep2
    [some 9, some 10, none, none, none, none, none, none, none, none, none, none],
    -- E1, E2, Ep1, Ep2, J: terminal
    [none, none, none, none, none, none, none, none, none, none, none, none],
    [none, none, none, none, none, none, none, none, none, none, none, none],
    [none, none, none, none, none, none, none, none, none, none, none, none],
    [none, none, none, none, none, none, none, none, none, none, none, none],
    [none, none, none, none, none, none, none, none, none, none, none, none]]

/-- A1 and A2 are behaviourally the same thing. -/
example : designed.BehEq 0 1 := by decide

/-- E1 = (A1*B)*C and Ep1 = A1*(B*C) differ microscopically ... -/
example : (7 : Fin 12) ≠ 9 := by decide

/-- ... but are the same emergent type. -/
example : designed.typeOf 7 = designed.typeOf 9 := by
  rw [typeOf_eq_iff]; decide

theorem designed_respects : designed.Respects := by
  constructor <;> decide

/-- Whenever both bracketings are defined they agree behaviourally ... -/
theorem designed_weak_assoc : designed.WeakAssocUpToBehEq := by decide

/-- ... but definedness itself is not associative: `(B*C)*A1 = Ep1` is defined
while `B*(C*A1)` is not, because `C` and `A1` do not click.  So the designed
structure is a partial commutative magma, not a category. -/
example : ¬ designed.AssocUpToBehEq := by decide

/-- Hence emergent types of the designed landscape carry a partial operation
that is exactly associative wherever both sides are defined. -/
theorem designed_types_assoc (x y z : designed.EmergentType) {w w' : designed.EmergentType}
    (hl : (EmergentType.click designed designed_respects x y).bind
        (fun v => EmergentType.click designed designed_respects v z) = some w)
    (hr : (EmergentType.click designed designed_respects y z).bind
        (fun v => EmergentType.click designed designed_respects x v) = some w') :
    w = w' :=
  EmergentType.click_assoc_of_some designed designed_respects designed_weak_assoc x y z hl hr

end Emergence.Finite
