import numpy as np

from emergence.compose import ClickTable, associativity, exact_types, substitutability
from emergence.types import fingerprints, type_classes


def _table(k, clicks):
    C = np.zeros((k, k))
    P = np.full((k, k), -1, dtype=int)
    for (i, j), d in clicks.items():
        C[i, j] = C[j, i] = 1.0
        P[i, j] = P[j, i] = d
    return ClickTable(k=k, C=C, product=P, c_thresh=0.5)


def test_designed_structure_types_and_laws():
    # 0=A1 1=A2 2=B 3=C 4=D1 5=D2 6=Dp 7=E (terminal) 8=J (junk)
    t = _table(9, {(0, 2): 4, (1, 2): 5, (2, 3): 6, (4, 3): 7, (5, 3): 7, (0, 6): 7, (1, 6): 7})
    ex = exact_types(t)
    assert ex[0] == ex[1]  # A1 ~ A2
    assert ex[4] == ex[5]  # D1 ~ D2
    assert ex[7] == ex[8]  # terminals are one type
    assert len(set(ex)) == 6
    labels = type_classes(fingerprints(t.C), eps=0.5)
    assert (labels == ex).all() or len(set(labels)) == 6
    s = substitutability(t, ex)
    assert s["substitutability"] == 1.0 and not s["vacuous"]
    a = associativity(t, ex)
    assert a["n_both_defined"] > 0 and a["associativity"] == 1.0
    # (B*C)*A1 is defined but B*(C*A1) is not: one-sided cases are expected
    assert a["n_one_sided"] > 0


def test_substitutability_violation_detected():
    # 0=A1 1=A2 2=B 3=D1 4=D2 5=C 6=E.  A1 and A2 both click only with B, so
    # they are behaviourally equivalent (BehEq is one-step, not recursive) ...
    t = _table(7, {(0, 2): 3, (1, 2): 4, (3, 5): 6})
    ex = exact_types(t)
    assert ex[0] == ex[1]
    # ... but their products D1, D2 differ in type (D1*C is defined, D2*C is
    # not), so clicking does not respect equivalence: `Respects` fails.
    assert ex[3] != ex[4]
    s = substitutability(t, ex)
    assert s["definedness_respected"] == 1.0
    assert s["products_respected"] < 1.0
