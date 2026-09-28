"""Grokking as mathematical crystallisation -- v0.

One task (modular addition), several seeds and architectures.  At each
training checkpoint the candidate emergent algebra ``M_t = S_theta_t / ~_t``
is extracted **without labels** (``extract.py``) and its crystallisation
measures are compared against train/test performance (``compare.py``).

Nothing about modular arithmetic, Fourier features or group structure enters
the extraction: it sees only sites, substitution and behaviour.
"""
