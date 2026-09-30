import Emergence.Behavioural

/-!
# Interfaces, multiway fits and higher-order interfaces

This file states the substrate-independent content of the "click" thesis:

* a **fit** is placing a piece into a context and observing the result;
* the **interface** of a piece is its equivalence class under all possible
  fits (`Interface = Quot`, `iface = typeOf`).  Nothing about the pieces is
  assumed — they may be states, activations, histories, programs — and the
  interface is not assumed finite, discrete, or of any particular form;
* a **multiway fit** `(A₁ … Aₙ) ⇝ B` is an *assembly* `f : (Fin n → H) → H`.
  When it respects interfaces (`CongruentN`), the interface of the assembled
  piece is a function of the interfaces of the parts alone (`assembleQ`,
  `assembleQ_iface`).  `congruentN_of_closed` gives the exact condition under
  which this is automatic: the contexts are closed under "fix the other parts
  and assemble".  This is what test C (context-general fit) and test E
  (composition) estimate on samples;
* **recursive generation**: from generators and a signature of assemblies,
  every term denotes a piece (`eval`) and its interface is computed
  recursively from the interfaces of the generators (`evalQ`, `evalQ_eq`),
  so new interfaces are produced by assembling old ones;
* **higher-order interfaces**: assemblies themselves are pieces of a system
  whose contexts are (parts, context) pairs (`opSystem`).  The interface of
  an assembly in that system is exactly its induced map on first-order
  interfaces (`opSystem_behEq_iff_assembleQ`) — the notion of interface
  applies to itself one level up, with the same definition.

Nothing here says which interfaces exist in a given substrate; that is the
empirical question.  What the file fixes is what "interface", "fit",
"composition" and "higher-order interface" mean, independently of the
substrate, so that the finite-sample tests are approximations of definite
exact statements.
-/

universe u v w u₁

namespace Emergence

namespace BehSystem

variable {H : Type u} {C : Type v} {O : Type w} (S : BehSystem H C O)

/-! ### Fits and interfaces -/

/-- A fit: place the piece `h` in the context `c` and observe. -/
abbrev fit (h : H) (c : C) : O := S.obs h c

/-- The interface of a piece: its class under all possible fits. -/
abbrev Interface : Type u := S.Quot

abbrev iface : H → S.Interface := S.typeOf

theorem iface_eq_iff {h h' : H} : S.iface h = S.iface h' ↔ ∀ c, S.fit h c = S.fit h' c :=
  S.typeOf_eq_iff

/-- Universal property: anything that depends on a piece only through its
fits is a function of its interface. -/
def throughIface {X : Sort u₁} (F : H → X) (hF : ∀ h h', S.BehEq h h' → F h = F h') :
    S.Interface → X :=
  Quotient.lift (s := S.setoid) F hF

@[simp] theorem throughIface_iface {X : Sort u₁} (F : H → X) (hF) (h : H) :
    S.throughIface F hF (S.iface h) = F h := rfl

/-- A representative of an interface (classical). -/
noncomputable def out (q : S.Interface) : H := Classical.choose (Quot.exists_rep q)

theorem out_spec (q : S.Interface) : S.iface (S.out q) = q :=
  Classical.choose_spec (Quot.exists_rep q)

theorem out_beh (h : H) : S.BehEq (S.out (S.iface h)) h :=
  Quotient.exact (S.out_spec (S.iface h))

/-! ### Multiway fits `(A₁ … Aₙ) ⇝ B` -/

/-- An `n`-ary assembly: `n` pieces put together give a piece. -/
abbrev Assembly (H : Type u) (n : Nat) : Type u := (Fin n → H) → H

/-- An assembly respects interfaces: parts with the same interfaces give
assembled pieces with the same interface. -/
def CongruentN {n : Nat} (f : Assembly H n) : Prop :=
  ∀ a a' : Fin n → H, (∀ i, S.BehEq (a i) (a' i)) → S.BehEq (f a) (f a')

/-- Replace the `i`-th part. -/
def upd {n : Nat} (a : Fin n → H) (i : Fin n) (h : H) : Fin n → H :=
  fun j => if j = i then h else a j

/-- The first `k` parts from `a'`, the rest from `a`. -/
def mix {n : Nat} (a a' : Fin n → H) (k : Nat) : Fin n → H :=
  fun j => if j.val < k then a' j else a j

theorem mix_zero {n : Nat} (a a' : Fin n → H) : mix a a' 0 = a :=
  funext fun j => by simp [mix]

theorem mix_all {n : Nat} (a a' : Fin n → H) : mix a a' n = a' :=
  funext fun j => by simp [mix, j.isLt]

theorem upd_mix_self {n : Nat} (a a' : Fin n → H) (k : Nat) (i : Fin n) (hi : i.val = k) :
    upd (mix a a' k) i (a i) = mix a a' k := by
  funext j
  by_cases hj : j = i
  · subst hj; simp [upd, mix, hi]
  · simp [upd, hj]

theorem upd_mix_succ {n : Nat} (a a' : Fin n → H) (k : Nat) (i : Fin n) (hi : i.val = k) :
    upd (mix a a' k) i (a' i) = mix a a' (k + 1) := by
  funext j
  by_cases hj : j = i
  · subst hj; simp [upd, mix, hi]
  · have hne : j.val ≠ k := fun e => hj (Fin.ext (e.trans hi.symm))
    simp only [upd, mix, if_neg hj]
    by_cases hlt : j.val < k
    · simp [hlt, Nat.lt_succ_of_lt hlt]
    · have : ¬ j.val < k + 1 := fun h => (Nat.lt_succ_iff_lt_or_eq.mp h).elim hlt hne
      simp [hlt, this]

/-- **Automatic congruence for multiway fits.**  If, for every assembly slot,
observing the assembled piece in some context is observing the part alone in
some other context (the contexts are closed under "fix the other parts and
assemble"), then the assembly respects interfaces.  Test C measures the
failure of this closure on sampled contexts. -/
theorem congruentN_of_closed {n : Nat} (f : Assembly H n)
    (hcl : ∀ (a : Fin n → H) (i : Fin n) (c : C), ∃ c', ∀ h, S.obs (f (upd a i h)) c = S.obs h c') :
    S.CongruentN f := by
  intro a a' e
  have key : ∀ k, k ≤ n → S.BehEq (f a) (f (mix a a' k)) := by
    intro k
    induction k with
    | zero => intro _; rw [mix_zero]; exact BehEq.refl S _
    | succ k ih =>
      intro hk
      have hkn : k < n := hk
      refine BehEq.trans S (ih (Nat.le_of_lt hkn)) ?_
      intro c
      obtain ⟨c', hc'⟩ := hcl (mix a a' k) ⟨k, hkn⟩ c
      have h1 := hc' (a ⟨k, hkn⟩)
      have h2 := hc' (a' ⟨k, hkn⟩)
      rw [upd_mix_self a a' k ⟨k, hkn⟩ rfl] at h1
      rw [upd_mix_succ a a' k ⟨k, hkn⟩ rfl] at h2
      rw [h1, h2]
      exact e ⟨k, hkn⟩ c'
  have := key n (Nat.le_refl n)
  rwa [mix_all] at this

/-- **Descent of a multiway fit to interfaces.**  The interface of the
assembled piece as a function of the interfaces of the parts. -/
noncomputable def assembleQ {n : Nat} {f : Assembly H n} (_hc : S.CongruentN f)
    (x : Fin n → S.Interface) : S.Interface :=
  S.iface (f (fun i => S.out (x i)))

theorem assembleQ_iface {n : Nat} {f : Assembly H n} (hc : S.CongruentN f) (a : Fin n → H) :
    S.assembleQ hc (fun i => S.iface (a i)) = S.iface (f a) :=
  Quotient.sound (hc _ _ (fun i => S.out_beh (a i)))

/-- A unary assembly is an action; a binary one a total operation.  The
descended maps agree with those of `Behavioural.lean`. -/
theorem assembleQ_unary {f : Assembly H 1} (hc : S.CongruentN f) (h : H) :
    S.assembleQ hc (fun _ => S.iface h) = S.iface (f (fun _ => h)) :=
  S.assembleQ_iface hc (fun _ => h)

/-! ### Recursive generation of interfaces -/

/-- A signature: a family of assemblies of given arities. -/
structure Signature (H : Type u) where
  Op : Type u₁
  arity : Op → Nat
  assemble : ∀ o, (Fin (arity o) → H) → H

/-- Terms over a signature and generators. -/
inductive Term {Op : Type u₁} (arity : Op → Nat) (G : Type u) : Type (max u u₁)
  | gen : G → Term arity G
  | node : (o : Op) → (Fin (arity o) → Term arity G) → Term arity G

variable {G : Type u}

/-- A term denotes a piece. -/
def Signature.eval (Sg : Signature.{u, u₁} H) (g : G → H) : Term Sg.arity G → H
  | .gen x => g x
  | .node o ts => Sg.assemble o (fun i => Sg.eval g (ts i))

/-- A signature all of whose assemblies respect interfaces. -/
def Signature.Congruent (Sg : Signature.{u, u₁} H) : Prop := ∀ o, S.CongruentN (Sg.assemble o)

/-- The interface of a term, computed recursively from the interfaces of the
generators through the descended assemblies. -/
noncomputable def evalQ {Sg : Signature.{u, u₁} H} (hc : Sg.Congruent S) (g : G → H) :
    Term Sg.arity G → S.Interface
  | .gen x => S.iface (g x)
  | .node o ts => S.assembleQ (hc o) (fun i => evalQ hc g (ts i))

/-- **Recursive interfaces.**  The interface of an assembled term is the
recursive assembly of the interfaces of its parts: new interfaces are
produced from old ones by the descended multiway fits. -/
theorem evalQ_eq {Sg : Signature.{u, u₁} H} (hc : Sg.Congruent S) (g : G → H) :
    ∀ t : Term Sg.arity G, S.evalQ hc g t = S.iface (Sg.eval g t)
  | .gen _ => rfl
  | .node o ts => by
    simp only [evalQ, Signature.eval]
    rw [show (fun i => S.evalQ hc g (ts i)) = fun i => S.iface (Sg.eval g (ts i)) from
      funext fun i => evalQ_eq hc g (ts i)]
    exact S.assembleQ_iface (hc o) _

/-- Two generator assignments with the same interfaces generate the same
interfaces everywhere. -/
theorem evalQ_congr {Sg : Signature.{u, u₁} H} (hc : Sg.Congruent S) (g g' : G → H)
    (e : ∀ x, S.BehEq (g x) (g' x)) : ∀ t : Term Sg.arity G, S.iface (Sg.eval g t) = S.iface (Sg.eval g' t) := by
  intro t
  rw [← S.evalQ_eq hc g, ← S.evalQ_eq hc g']
  induction t with
  | gen x => exact Quotient.sound (e x)
  | node o ts ih =>
    simp only [evalQ]
    rw [funext ih]

/-! ### Higher-order interfaces -/

/-- The system whose pieces are `n`-ary assemblies.  A fit of an assembly is a
choice of parts together with a context for the result. -/
def opSystem (n : Nat) : BehSystem (Assembly H n) ((Fin n → H) × C) O :=
  ⟨fun f ac => S.obs (f ac.1) ac.2⟩

theorem opSystem_behEq_iff {n : Nat} (f g : Assembly H n) :
    (S.opSystem n).BehEq f g ↔ ∀ a, S.BehEq (f a) (g a) :=
  ⟨fun e a c => e (a, c), fun e ac => e ac.1 ac.2⟩

/-- **The interface of an assembly is its action on interfaces.**  Two
congruent assemblies have the same second-order interface exactly when they
induce the same map on first-order interfaces.  So "interface" is one
notion applied at every order: a higher-order interface is the equivalence
class of a composition schema under all its fits, and it is determined by
what the schema does to lower-order interfaces. -/
theorem opSystem_behEq_iff_assembleQ {n : Nat} {f g : Assembly H n}
    (hf : S.CongruentN f) (hg : S.CongruentN g) :
    (S.opSystem n).BehEq f g ↔ ∀ x, S.assembleQ hf x = S.assembleQ hg x := by
  rw [opSystem_behEq_iff]
  constructor
  · intro e x
    have := e (fun i => S.out (x i))
    exact Quotient.sound this
  · intro e a
    have h1 := S.assembleQ_iface hf a
    have h2 := S.assembleQ_iface hg a
    rw [e] at h1
    exact Quotient.exact (h1.symm.trans h2)

/-- The second-order quotient: interfaces of `n`-ary assemblies. -/
abbrev Interface₂ (n : Nat) : Type u := (S.opSystem n).Interface

end BehSystem

end Emergence
