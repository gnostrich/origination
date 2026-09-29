import numpy as np

from emergence.cont.extract_cont import ContConfig, extract_cont, leader_cluster_cont, rel_dist
from emergence.cont.systems import RandomLandscape, integrate, make_system


def test_integrator_relaxes_double_well():
    sys = make_system("double_well")
    X0 = np.array([[0.3], [-0.3], [1.7]])
    U = np.zeros((3, 20, 1))
    X, Y = integrate(sys, X0, U, 0.5, 0.01, 0.0, np.random.default_rng(0), sample_times=np.array([9.9]))
    assert np.allclose(X[:, 0], [1, -1, 1], atol=1e-3)
    assert Y.shape == (3, 1, 1)


def test_leader_cluster_relative():
    B = np.array([[0.0, 0.0], [0.01, 0.0], [1.0, 1.0], [1.0, 1.02]])
    labels, leaders = leader_cluster_cont(B, scale=1.0, eps=0.1)
    assert labels.tolist() == [0, 0, 1, 1]


def test_single_well_one_object_double_well_two():
    cfg = ContConfig(n_states=60, n_pulses=12, n_suffixes=4, K=6, seed=0)
    r1 = extract_cont(make_system("single_well"), cfg)
    assert r1["n_stable_classes"] == 1
    r2 = extract_cont(make_system("double_well"), cfg)
    assert r2["n_stable_classes"] == 2
    reps = np.array(r2["rep_states"]).ravel()
    assert np.sign(reps[0]) != np.sign(reps[1])  # evaluation only: one per basin


def test_random_landscape_gradient():
    sys = RandomLandscape(n=3, n_wells=5, seed=1)
    X = np.random.default_rng(0).standard_normal((4, 3))
    h = 1e-5
    g = -sys.f(X)
    for i in range(3):
        Xp = X.copy(); Xp[:, i] += h
        Xm = X.copy(); Xm[:, i] -= h
        num = (sys.V(Xp) - sys.V(Xm)) / (2 * h)
        assert np.allclose(g[:, i], num, atol=1e-4)
