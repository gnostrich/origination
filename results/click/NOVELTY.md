# Sealed novelty diagnostic (behavioural interface discovery)

Generating circuit: gate inputs [[0, 6, 10], [4, 5, 7], [0, 2, 8], [7, 8, 11], [1, 10, 11], [7, 9, 11]], output gate sets [[0, 1, 2], [0, 2, 3], [0, 3, 4], [1, 4, 5], [0, 1, 5], [1, 3, 4]], reuse [4, 4, 2, 3, 3, 2].

## Per accepted interface (final checkpoints and all analysed checkpoints)

| run | step | site | idx | rank | types | A(final) | frac U in gate span | max ‖U w_g‖² (gate) | ARI best gate | purity best gate | best pair ARI / purity | ARI best output | combines | splits | non-isomorphic & valid |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| wd=0.0 s=1 | 467 | h2 | 0 | 2 | 7 | 0.97 | 0.09 | 0.12 (g0) | 0.17 (g0) | 0.84 | 0.19 / 0.62 (g0,g4) | 0.05 | False | False | True |
| wd=0.0 s=2 | 925 | h2 | 0 | 1 | 7 | 0.94 | 0.05 | 0.01 (g1) | 0.24 (g5) | 0.84 | 0.19 / 0.56 (g1,g5) | 0.19 | False | False | True |
| wd=0.01 s=0 | 3626 | h1 | 0 | 1 | 7 | 0.96 | 0.06 | 0.02 (g2) | 0.05 (g1) | 0.72 | 0.07 / 0.55 (g1,g4) | 0.09 | False | False | True |
| wd=0.1 s=0 | 3626 | h2 | 0 | 1 | 8 | 0.97 | 0.08 | 0.05 (g0) | 0.10 (g0) | 0.80 | 0.10 / 0.55 (g0,g1) | 0.05 | False | False | True |

## Cross-seed correspondence (final checkpoints)

Mean canonical correlation over all 4096 inputs between the projections of accepted subspaces of two seeds (same wd, same site).

- wd=0.0 h1: fewer than two seeds with accepted interfaces
- wd=0.0 h2: fewer than two seeds with accepted interfaces
- wd=0.01 h1: fewer than two seeds with accepted interfaces
- wd=0.01 h2: fewer than two seeds with accepted interfaces
- wd=0.1 h1: fewer than two seeds with accepted interfaces
- wd=0.1 h2: fewer than two seeds with accepted interfaces
- wd=1.0 h1: fewer than two seeds with accepted interfaces
- wd=1.0 h2: fewer than two seeds with accepted interfaces

## Summary

- accepted interfaces examined: 4; passing A on the final pool: 4
- mean fraction of U inside the gate-probe span: 0.07 (a random rank-k subspace of R^64 has ≈ 6/64 = 0.09)
- types matching a single gate (purity ≥ 0.9): 0; matching a pair of gates better than any single gate: 0
- splitting a gate (purity ≥ 0.95 with > 2 types): 0
- valid (A passed) but not a function of any gate or gate pair: 4
