import numpy as np
import pytest

from emergence.grok.compare import adjusted_rand_index, half_rise_step, invariants, isomorphic_by_generator


def test_ari_basic():
    a = np.array([0, 0, 1, 1, 2, 2])
    assert adjusted_rand_index(a, a) == 1.0
    b = np.array([5, 5, 7, 7, 9, 9])  # same partition, different labels
    assert adjusted_rand_index(a, b) == 1.0
    c = np.array([0, 1, 0, 1, 0, 1])
    assert adjusted_rand_index(a, c) < 0.2


def test_cyclic_tables_isomorphic_up_to_relabelling():
    p = 11
    add = (np.arange(p)[:, None] + np.arange(p)[None, :]) % p
    # relabel by multiplication by 3 (an automorphism), then by a random bijection
    rng = np.random.default_rng(0)
    phi = rng.permutation(p)
    t2 = np.empty_like(add)
    t2[phi[:, None], phi[None, :]] = phi[add]
    r = isomorphic_by_generator(add, t2)
    assert r["isomorphic"]
    inv = invariants(add)
    assert inv["associativity"] == 1.0 and inv["commutativity"] == 1.0 and inv["latin"] == 1.0
    # a memorised-looking table (random) is not
    bad = rng.integers(0, p, (p, p))
    assert invariants(bad)["associativity"] < 0.5


def test_half_rise():
    steps = [0, 100, 200, 300, 400]
    vals = [0.1, 0.1, 0.2, 0.8, 0.9]
    assert half_rise_step(steps, vals) == 300
    assert half_rise_step(steps, [0.5] * 5) is None
