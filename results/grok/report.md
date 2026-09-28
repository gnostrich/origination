# Grokking as crystallisation: report

## Transition timing per run (steps)

| run | t_mem | t_gen(0.9) | t_gen_half | metastability half-rise (lag) | discreteness half-rise (lag) | closure half-rise (lag) | polarization half-rise (lag) | compression half-rise (lag) | op.associativity half-rise (lag) | op.commutativity half-rise (lag) | op.latin half-rise (lag) | crystallization half-rise (lag) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| transformer_s1 | 400 | 15000 | 14000 | 15500 (1500) | None (None) | 15500 (1500) | 2000 (-12000) | 15000 (1000) | 15000 (1000) | 14000 (0) | 15500 (1500) | 15000 (1000) |
| transformer_s2 | 400 | 16500 | 15500 | 17000 (1500) | None (None) | 17000 (1500) | 2000 (-13500) | 16500 (1000) | 16000 (500) | 14500 (-1000) | 17000 (1500) | 16000 (500) |

## Plateau (memorised, not generalised) vs after generalisation

| run | metastability: plateau -> after | discreteness: plateau -> after | closure: plateau -> after | polarization: plateau -> after | compression: plateau -> after | op.associativity: plateau -> after | op.commutativity: plateau -> after | op.latin: plateau -> after | crystallization: plateau -> after |
|---|---|---|---|---|---|---|---|---|---|
| transformer_s1 | 0.21 -> 0.62 (final 0.73) | 0.99 -> 1.00 (final 1.00) | 0.25 -> 0.84 (final 0.98) | 0.70 -> 0.79 (final 0.97) | 0.26 -> 0.92 (final 1.00) | 0.11 -> 0.98 (final 1.00) | 0.79 -> 1.00 (final 1.00) | 0.00 -> 0.84 (final 1.00) | 0.26 -> 0.83 (final 0.91) |
| transformer_s2 | 0.23 -> 0.53 (final 0.41) | 0.99 -> 1.00 (final 1.00) | 0.28 -> 0.72 (final 0.55) | 0.68 -> 0.68 (final 0.54) | 0.29 -> 0.90 (final 0.78) | 0.12 -> 0.97 (final 0.90) | 0.85 -> 1.00 (final 1.00) | 0.00 -> 0.72 (final 0.04) | 0.28 -> 0.76 (final 0.66) |

## Cross-run equivalence of the extracted quotients

| step | mean ARI (hidden partitions) | mean op-table agreement | ARI to true-sum partition (reference) |
|---|---|---|---|
| 0 | 1.000 | 0.000 | transformer_s1:0.00, transformer_s2:0.00 |
| 10 | 1.000 | 0.013 | transformer_s1:0.00, transformer_s2:0.00 |
| 25 | 0.026 | 0.010 | transformer_s1:-0.00, transformer_s2:-0.00 |
| 50 | 0.156 | 0.013 | transformer_s1:0.02, transformer_s2:0.02 |
| 100 | 0.215 | 0.156 | transformer_s1:0.01, transformer_s2:0.01 |
| 150 | 0.103 | 0.236 | transformer_s1:0.24, transformer_s2:0.27 |
| 200 | 0.106 | 0.237 | transformer_s1:0.25, transformer_s2:0.27 |
| 300 | 0.111 | 0.239 | transformer_s1:0.27, transformer_s2:0.28 |
| 400 | 0.115 | 0.245 | transformer_s1:0.28, transformer_s2:0.30 |
| 500 | 0.124 | 0.250 | transformer_s1:0.29, transformer_s2:0.32 |
| 750 | 0.125 | 0.257 | transformer_s1:0.30, transformer_s2:0.33 |
| 1000 | 0.119 | 0.259 | transformer_s1:0.31, transformer_s2:0.32 |
| 1500 | 0.108 | 0.259 | transformer_s1:0.30, transformer_s2:0.32 |
| 2000 | 0.096 | 0.252 | transformer_s1:0.30, transformer_s2:0.28 |
| 2500 | 0.149 | 0.260 | transformer_s1:0.32, transformer_s2:0.35 |
| 3000 | 0.119 | 0.260 | transformer_s1:0.31, transformer_s2:0.33 |
| 3500 | 0.108 | 0.260 | transformer_s1:0.30, transformer_s2:0.32 |
| 4000 | 0.150 | 0.260 | transformer_s1:0.32, transformer_s2:0.35 |
| 4500 | 0.120 | 0.260 | transformer_s1:0.31, transformer_s2:0.33 |
| 5000 | 0.118 | 0.260 | transformer_s1:0.31, transformer_s2:0.32 |
| 5500 | 0.145 | 0.261 | transformer_s1:0.32, transformer_s2:0.35 |
| 6000 | 0.121 | 0.261 | transformer_s1:0.31, transformer_s2:0.34 |
| 6500 | 0.124 | 0.261 | transformer_s1:0.32, transformer_s2:0.32 |
| 7000 | 0.142 | 0.261 | transformer_s1:0.32, transformer_s2:0.36 |
| 7500 | 0.096 | 0.250 | transformer_s1:0.25, transformer_s2:0.33 |
| 8000 | 0.121 | 0.263 | transformer_s1:0.33, transformer_s2:0.32 |
| 8500 | 0.133 | 0.263 | transformer_s1:0.31, transformer_s2:0.35 |
| 9000 | 0.127 | 0.263 | transformer_s1:0.31, transformer_s2:0.33 |
| 9500 | 0.159 | 0.264 | transformer_s1:0.33, transformer_s2:0.36 |
| 10000 | 0.130 | 0.264 | transformer_s1:0.32, transformer_s2:0.35 |
| 10500 | 0.125 | 0.266 | transformer_s1:0.32, transformer_s2:0.33 |
| 11000 | 0.153 | 0.268 | transformer_s1:0.33, transformer_s2:0.37 |
| 11500 | 0.127 | 0.273 | transformer_s1:0.32, transformer_s2:0.35 |
| 12000 | 0.137 | 0.276 | transformer_s1:0.34, transformer_s2:0.33 |
| 12500 | 0.154 | 0.282 | transformer_s1:0.34, transformer_s2:0.37 |
| 13000 | 0.136 | 0.293 | transformer_s1:0.34, transformer_s2:0.35 |
| 13500 | 0.148 | 0.307 | transformer_s1:0.39, transformer_s2:0.33 |
| 14000 | 0.210 | 0.383 | transformer_s1:0.48, transformer_s2:0.39 |
| 14500 | 0.278 | 0.482 | transformer_s1:0.65, transformer_s2:0.40 |
| 15000 | 0.409 | 0.612 | transformer_s1:0.89, transformer_s2:0.44 |
| 15500 | 0.519 | 0.716 | transformer_s1:0.98, transformer_s2:0.53 |
| 16000 | 0.767 | 0.889 | transformer_s1:0.97, transformer_s2:0.78 |
| 16500 | 0.884 | 0.972 | transformer_s1:1.00, transformer_s2:0.88 |
| 17000 | 0.991 | 0.997 | transformer_s1:1.00, transformer_s2:0.99 |
| 17500 | 0.995 | 0.999 | transformer_s1:1.00, transformer_s2:0.99 |
| 18000 | 0.998 | 1.000 | transformer_s1:1.00, transformer_s2:1.00 |
| 18500 | 0.999 | 1.000 | transformer_s1:1.00, transformer_s2:1.00 |
| 19000 | 0.997 | 1.000 | transformer_s1:1.00, transformer_s2:1.00 |

### Final structures

- transformer_s1 vs transformer_s2: ARI 0.932, op agreement 0.967, isomorphic up to relabelling: {'isomorphic': False, 'reason': 'no generator of t1 maps compatibly'}

- transformer_s1 invariants: {'associativity': 1.0, 'commutativity': 1.0, 'identity': 1.0, 'latin': 1.0, 'n_outputs': 97}
- transformer_s2 invariants: {'associativity': 0.89605, 'commutativity': 0.9968115633967478, 'identity': 0.9690721649484536, 'latin': 0.041237113402061855, 'n_outputs': 97}
