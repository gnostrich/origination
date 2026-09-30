import numpy as np

from emergence.lang.model import InterfaceModel, Fit
from emergence.lang.discover import compose
from emergence.lang.evaluate import measure_complexity


def toy_model():
    K, A, o = 3, 2, 2
    leaders = np.zeros((K, 2, 1, o))
    em = np.zeros((K, A, o))
    em[:, :, 0] = 1.0
    em[2, 1] = [0.0, 1.0]
    m = InterfaceModel(eps=0.05, n_actions=A, n_obs=o, contexts=[(0,), (1,)], leaders=leaders,
                       realizations=[[0], [1], [2]], within=np.zeros(K), emissions=em)
    # a0 cycles 0->1->2->0 ; a1 fixes everything
    for k in range(K):
        m.fits[(k, (0,))] = Fit(k, (0,), (k + 1) % K, 1.0, 5)
        m.fits[(k, (1,))] = Fit(k, (1,), k, 1.0, 5)
    return m


def test_run_chains_fits_and_emits():
    m = toy_model()
    out = m.run(0, (0, 0, 1, 0))
    assert [int(np.argmax(p)) for p in out] == [0, 0, 1, 0]   # state 2 under a1 emits o1
    assert m.run_chunk_map(0, (0, 0, 0)) == 0


def test_compose_derives_second_order_structure():
    m = compose(toy_model(), max_len=2)
    assert m.fits[(0, (0, 0))].dst == 2 and not m.fits[(0, (0, 0))].observed
    assert len(m.chunk_maps) == 3            # identity-like, shift, shift twice
    w0 = m.chunk_types[(0,)]
    assert m.second_fits[(w0, w0)] == m.chunk_types[(0, 0)]


def test_complexity_counts_exceptions():
    m = toy_model()
    c1 = measure_complexity(m)
    m.fits[(0, (0,))] = Fit(0, (0,), 1, 0.6, 5)
    c2 = measure_complexity(m)
    assert c2["exceptions"] == 2 and c2["total_bits_argmax"] > c1["total_bits_argmax"]
