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
