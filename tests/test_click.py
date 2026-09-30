import numpy as np
import torch

from emergence.click.task import Circuit, split
from emergence.click.model import MLP
from emergence.click import interfaces as I


def test_circuit_balanced_and_reused():
    c = Circuit(seed=0)
    X = c.all_inputs()
    Y = c.outputs(X)
    assert X.shape == (4096, 12) and Y.shape == (4096, 6)
    assert (Y.mean(0) >= 0.25).all() and (Y.mean(0) <= 0.75).all()
    assert min(c.reuse()) >= 1


def test_plateau_criterion_separates_clustered_from_spread():
    assert I.plateau([3, 3, 3], 1) and I.plateau([3, 4, 3], 1)
    assert not I.plateau([12, 8, 6], 1)


def test_leader_cluster_and_assign():
    P = np.array([[0.0, 0.0], [0.1, 0.0], [5.0, 5.0], [5.1, 5.0]])
    labels, leaders = I.leader_cluster(P, 0.5)
    assert labels.tolist() == [0, 0, 1, 1] and leaders.tolist() == [0, 2]
    assert I.assign(np.array([[4.9, 5.1]]), leaders, P).tolist() == [1]


def test_pipeline_runs_on_untrained_model():
    c = Circuit(seed=0)
    X = c.all_inputs()
    tr, te = split(len(X), 0.5, 0)
    m = MLP(seed=0).eval()
    cfg = dict(I.CFG, n_dirs=2, pool=8)
    r = I.analyse_checkpoint(m, X, tr, te, cfg)
    for s in ("h1", "h2"):
        sr = r["sites"][s]
        assert sr["n_candidates"] == 6
        assert set(I.flags(sr, cfg)) == set("ABCDE")


def test_bid_leader_cluster_matrix_matches_reference():
    from emergence.click import bid as B
    rng = np.random.default_rng(3)
    for _ in range(10):
        P = rng.normal(size=(40, 30)) * rng.uniform(0.1, 3)
        eps = rng.uniform(0.5, 3)
        l1, L1 = I.leader_cluster(P, eps)
        l2, L2 = B.leader_cluster_matrix(B.rms_matrix(P, P), eps)
        assert (l1 == l2).all() and (L1 == L2).all()


def test_bid_objective_is_basis_invariant():
    from emergence.click import bid as B
    from emergence.click.task import Circuit, split
    c = Circuit(seed=0)
    X = c.all_inputs()
    tr, te = split(len(X), 0.5, 0)
    disc, val, fin = B.make_pools(tr, te)
    assert not (set(disc["d_fit"]) & set(val["d_fit"])) and not (set(val["d_fit"]) & set(fin["d_ref"]))
    m = MLP(seed=0).eval()
    sub = I.Sub(m, X)
    rng = np.random.default_rng(0)
    U = B.random_subspace(rng, 4, 64)
    Q, _ = np.linalg.qr(rng.normal(size=(4, 4)))
    J1 = B.components(sub, "h1", U, disc)["J"]
    J2 = B.components(sub, "h1", Q @ U, disc)["J"]
    assert abs(J1 - J2) < 1e-6
