import numpy as np

from emergence.coupling import JointEnergy
from emergence.energy import MLPEnergy, WellEnergy, designed_landscape, superpose
from emergence.dynamics import quench


def _fd_check(energy, X, h=1e-5):
    E, g = energy.value_and_grad(X)
    num = np.zeros_like(X)
    for i in range(X.shape[1]):
        Xp = X.copy()
        Xp[:, i] += h
        Xm = X.copy()
        Xm[:, i] -= h
        num[:, i] = (energy.value(Xp) - energy.value(Xm)) / (2 * h)
    assert np.allclose(E, energy.value(X))
    assert np.allclose(g, num, atol=1e-5, rtol=1e-4), np.abs(g - num).max()


def test_mlp_gradient():
    e = MLPEnergy(N=8, hidden=16, depth=2, seed=1)
    X = np.random.default_rng(0).standard_normal((5, 8))
    _fd_check(e, X)


def test_well_gradient():
    e = WellEnergy(np.random.default_rng(0).standard_normal((4, 6)) * 3, sigma=0.7)
    X = np.random.default_rng(1).standard_normal((5, 6))
    _fd_check(e, X)


def test_joint_gradient():
    base = MLPEnergy(N=6, hidden=12, depth=2, seed=2)
    e = JointEnergy(base, k=3, g=1.5)
    X = np.random.default_rng(3).standard_normal((4, 18))
    _fd_check(e, X)


def test_designed_wells_are_minima():
    e, truth = designed_landscape(N=16)
    Xm, Em = quench(e, e.centers + 0.05, lr=0.2, max_steps=500)
    # E1/Ep1 (and E2/Ep2) are 1.2 apart and blend slightly under the softmin
    assert np.linalg.norm(Xm - e.centers, axis=1).max() < 0.2
    # product wells sit exactly at superpositions of their parts
    n = truth["names"]
    idx = {k: i for i, k in enumerate(n)}
    for (a, b), d in truth["pair_clicks"].items():
        assert np.allclose(superpose([e.centers[idx[a]], e.centers[idx[b]]]), e.centers[idx[d]])
