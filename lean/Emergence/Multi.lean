import Emergence.Basic

/-!
# Multiary interactions and substitution across contexts

`Emergence.Basic` treats binary clicks.  Here an interaction takes a *list*
of things of any arity (a tuple placed into the substrate together) and may
produce a thing.  Behavioural equivalence is defined exactly as in the
extraction procedure: `a ~ b` when substituting `a` for `b` in every sampled
*context* (a tuple with a hole) does not change whether the interaction is
defined.  The main result is again descent: an operation that respects `~`
in each slot is a well-defined operation on the quotient, and coherence laws
that hold up to `~` on the substrate hold exactly on the quotient.
-/

universe u

namespace Emergence

/-- A multiary interaction structure: a partial operation on tuples. -/
structure MultiInteraction (B : Type u) where
  op : List B → Option B

namespace MultiInteraction

variable {B : Type u} (S : MultiInteraction B)

/-- A context is a tuple with a hole: the things before and after it. -/
abbrev Context (B : Type u) := List B × List B

/-- Plug a thing into the hole. -/
def plug (c : Context B) (a : B) : List B := c.1 ++ a :: c.2

/-- `a` fits in context `c`. -/
def Fits (c : Context B) (a : B) : Prop := (S.op (plug c a)).isSome = true

instance (c : Context B) (a : B) : Decidable (S.Fits c a) := inferInstanceAs (Decidable (_ = true))

/-- Substitutability across all contexts. -/
def BehEq (a b : B) : Prop := ∀ c : Context B, S.Fits c a ↔ S.Fits c b

theorem BehEq.refl (a : B) : S.BehEq a a := fun _ => Iff.rfl
theorem BehEq.symm {a b : B} (h : S.BehEq a b) : S.BehEq b a := fun c => (h c).symm
theorem BehEq.trans {a b d : B} (h₁ : S.BehEq a b) (h₂ : S.BehEq b d) : S.BehEq a d :=
  fun c => (h₁ c).trans (h₂ c)

def behEqSetoid : Setoid B := ⟨S.BehEq, ⟨BehEq.refl S, fun h => BehEq.symm S h, fun h₁ h₂ => BehEq.trans S h₁ h₂⟩⟩

/-- Emergent types: things modulo substitutability. -/
def EmergentType : Type u := Quotient S.behEqSetoid

def typeOf (a : B) : S.EmergentType := Quotient.mk S.behEqSetoid a

/-- Results agree up to `~` (undefined matches undefined). -/
def OptBehEq : Option B → Option B → Prop
  | none, none => True
  | some d, some d' => S.BehEq d d'
  | _, _ => False

theorem OptBehEq.refl (o : Option B) : S.OptBehEq o o := by
  cases o <;> simp [OptBehEq, BehEq.refl]

theorem OptBehEq.trans {o₁ o₂ o₃ : Option B} (h₁ : S.OptBehEq o₁ o₂) (h₂ : S.OptBehEq o₂ o₃) :
    S.OptBehEq o₁ o₃ := by
  cases o₁ <;> cases o₂ <;> cases o₃ <;> simp_all [OptBehEq]
  exact BehEq.trans S h₁ h₂

theorem OptBehEq.map_typeOf_eq {o o' : Option B} (h : S.OptBehEq o o') :
    o.map S.typeOf = o'.map S.typeOf := by
  cases o <;> cases o' <;> simp_all [OptBehEq, Option.map]
  exact Quotient.sound h

/-- Pointwise equivalence of tuples. -/
def ListBehEq : List B → List B → Prop
  | [], [] => True
  | a :: l, b :: m => S.BehEq a b ∧ ListBehEq l m
  | _, _ => False

theorem ListBehEq.refl : ∀ l : List B, S.ListBehEq l l
  | [] => trivial
  | _ :: l => ⟨BehEq.refl S _, ListBehEq.refl l⟩

/-- Hypothesis (measured by `substitutability`): replacing one slot of a tuple
by an equivalent thing gives an equivalent result. -/
def Respects : Prop :=
  ∀ (c : Context B) (a b : B), S.BehEq a b → S.OptBehEq (S.op (plug c a)) (S.op (plug c b))

/-- One-slot substitution extends to all slots. -/
theorem Respects.list (h : S.Respects) :
    ∀ (pre : List B) (l m : List B), S.ListBehEq l m →
      S.OptBehEq (S.op (pre ++ l)) (S.op (pre ++ m))
  | pre, [], [], _ => OptBehEq.refl S _
  | pre, a :: l, b :: m, ⟨hab, hlm⟩ => by
    have step : S.OptBehEq (S.op (pre ++ a :: l)) (S.op (pre ++ b :: l)) := h (pre, l) a b hab
    have rest : S.OptBehEq (S.op ((pre ++ [b]) ++ l)) (S.op ((pre ++ [b]) ++ m)) :=
      Respects.list h (pre ++ [b]) l m hlm
    simp only [List.append_assoc, List.singleton_append] at rest
    exact OptBehEq.trans S step rest

/-- The operation on tuples of emergent types.  A tuple of quotients is
turned into a quotient of tuples, then `op` is lifted. -/
def listSetoid : Setoid (List B) :=
  ⟨S.ListBehEq, ⟨ListBehEq.refl S, fun {l m} h => by
      induction l generalizing m with
      | nil => cases m <;> simp_all [ListBehEq]
      | cons a l ih => cases m with
        | nil => simp_all [ListBehEq]
        | cons b m => exact ⟨BehEq.symm S h.1, ih h.2⟩,
    fun {l m n} h₁ h₂ => by
      induction l generalizing m n with
      | nil => cases m <;> cases n <;> simp_all [ListBehEq]
      | cons a l ih => cases m with
        | nil => simp_all [ListBehEq]
        | cons b m => cases n with
          | nil => simp_all [ListBehEq]
          | cons d n => exact ⟨BehEq.trans S h₁.1 h₂.1, ih h₁.2 h₂.2⟩⟩⟩

def tupleOfTypes : List S.EmergentType → Quotient S.listSetoid
  | [] => Quotient.mk _ []
  | q :: qs =>
    Quotient.lift₂ (s₁ := S.behEqSetoid) (s₂ := S.listSetoid)
      (fun a l => Quotient.mk S.listSetoid (a :: l))
      (fun _ _ _ _ ha hl => Quotient.sound (show S.ListBehEq _ _ from ⟨ha, hl⟩)) q (tupleOfTypes qs)

theorem tupleOfTypes_map (l : List B) :
    S.tupleOfTypes (l.map S.typeOf) = Quotient.mk S.listSetoid l := by
  induction l with
  | nil => rfl
  | cons a l ih => simp [tupleOfTypes, List.map, ih, typeOf]; rfl

/-- **Descent** for multiary operations. -/
def EmergentType.op (h : S.Respects) (qs : List S.EmergentType) : Option S.EmergentType :=
  Quotient.lift (s := S.listSetoid) (fun l => (S.op l).map S.typeOf)
    (fun l m hlm => OptBehEq.map_typeOf_eq S (Respects.list S h [] l m (show S.ListBehEq l m from hlm)))
    (S.tupleOfTypes qs)

theorem EmergentType.op_map (h : S.Respects) (l : List B) :
    EmergentType.op S h (l.map S.typeOf) = (S.op l).map S.typeOf := by
  unfold EmergentType.op
  rw [tupleOfTypes_map]
  rfl

/-- A coherence law is a pair of ways of composing the operation on a tuple
(e.g. two bracketings).  If they agree up to `~` on the substrate whenever
both are defined, they agree *exactly* on the quotient. -/
theorem EmergentType.law_of_some (lhs rhs : List B → Option B)
    (hlaw : ∀ l d d', lhs l = some d → rhs l = some d' → S.BehEq d d')
    (l : List B) {d d' : B} (h₁ : lhs l = some d) (h₂ : rhs l = some d') :
    S.typeOf d = S.typeOf d' :=
  Quotient.sound (hlaw l d d' h₁ h₂)

end MultiInteraction

end Emergence
