import Emergence.Basic

/-!
# Maps between interaction structures and task-level equivalence

Different seeds or architectures trained on the same task induce different
substrate-level structures `S₁ : Interaction B₁`, `S₂ : Interaction B₂`.
The question "do they realise the same mathematics?" is not whether `B₁` and
`B₂` look alike, but whether there is a **structure-preserving map** between
them *up to behavioural equivalence*, i.e. whether the quotients are
equivalent.  This file defines such maps and proves that they descend to
the emergent types.

Three layers:

    microscopic parameters   →   idiosyncratic emergent algebra   →   task-level class
    (B₁, B₂ differ)              (S₁ / ~, S₂ / ~)                    (quotients equivalent)
-/

universe u v

namespace Emergence.Interaction

variable {B₁ : Type u} {B₂ : Type v} (S₁ : Interaction B₁) (S₂ : Interaction B₂)

/-- A map of things that preserves clicks up to behavioural equivalence and
neither merges nor separates behaviourally distinct things. -/
structure Hom where
  toFun : B₁ → B₂
  /-- clicks are preserved: `f (a * b) ~ f a * f b` (including definedness) -/
  map_click : ∀ a b, S₂.OptBehEq ((S₁.click a b).map toFun) (S₂.click (toFun a) (toFun b))
  /-- behavioural equivalence is reflected and preserved -/
  behEq_iff : ∀ a b, S₁.BehEq a b ↔ S₂.BehEq (toFun a) (toFun b)

namespace Hom

variable {S₁ S₂} (f : Hom S₁ S₂)

/-- The induced map on emergent types. -/
def onTypes : S₁.EmergentType → S₂.EmergentType :=
  Quotient.lift (fun a => S₂.typeOf (f.toFun a))
    (fun a b hab => Quotient.sound ((f.behEq_iff a b).1 hab))

@[simp] theorem onTypes_typeOf (a : B₁) : f.onTypes (S₁.typeOf a) = S₂.typeOf (f.toFun a) := rfl

/-- The induced map on types is injective: idiosyncratic things that were
distinguishable stay distinguishable. -/
theorem onTypes_injective : ∀ x y, f.onTypes x = f.onTypes y → x = y := by
  intro x y
  induction x using Quotient.inductionOn with | _ a =>
  induction y using Quotient.inductionOn with | _ b =>
  intro h
  exact Quotient.sound ((f.behEq_iff a b).2 (Quotient.exact h))

/-- The induced map commutes with the descended click operations exactly. -/
theorem onTypes_click (h₁ : S₁.Respects) (h₂ : S₂.Respects) (x y : S₁.EmergentType) :
    (EmergentType.click S₁ h₁ x y).map f.onTypes =
      EmergentType.click S₂ h₂ (f.onTypes x) (f.onTypes y) := by
  induction x using Quotient.inductionOn with | _ a =>
  induction y using Quotient.inductionOn with | _ b =>
  show ((S₁.click a b).map S₁.typeOf).map f.onTypes = (S₂.click (f.toFun a) (f.toFun b)).map S₂.typeOf
  have key := OptBehEq.map_typeOf_eq S₂ (f.map_click a b)
  rw [← key]
  cases S₁.click a b <;> rfl

end Hom

/-- Two structures are **task-equivalent** when there are homs both ways
whose induced maps on types are mutually inverse: the quotients are the same
algebra even if the substrates are not. -/
structure TaskEquiv where
  fwd : Hom S₁ S₂
  bwd : Hom S₂ S₁
  left_inv : ∀ x, bwd.onTypes (fwd.onTypes x) = x
  right_inv : ∀ y, fwd.onTypes (bwd.onTypes y) = y

end Emergence.Interaction
