"""v0 experiment: do continuous, untyped Langevin dynamics on a random energy
landscape generate metastable discrete states whose interaction behaviour
induces types, and whose stable multiway interactions compose?

Pipeline (see README.md):

    energy E_theta on R^N
        -> Langevin trajectories                     (dynamics.py)
        -> metastable basins B_1..B_k, R_B           (basins.py)
        -> joint dynamics of k basin representatives (coupling.py)
        -> compatibility tensor C(i, j, ...)         (compat.py)
        -> behavioural fingerprints -> types         (types.py)
        -> closure / associativity / substitution    (compose.py)
        -> finite structure exported for Lean        (export.py)
"""

from .energy import MLPEnergy, WellEnergy, designed_landscape
from .experiment import run_experiment, ExperimentConfig

__all__ = [
    "MLPEnergy",
    "WellEnergy",
    "designed_landscape",
    "run_experiment",
    "ExperimentConfig",
]
