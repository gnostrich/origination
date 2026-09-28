# Grokking as crystallisation: report

## Transition timing per run (steps)

| run | t_mem | t_gen(0.9) | t_gen_half | metastability half-rise (lag) | discreteness half-rise (lag) | closure half-rise (lag) | polarization half-rise (lag) | compression half-rise (lag) | op.associativity half-rise (lag) | op.commutativity half-rise (lag) | op.latin half-rise (lag) | crystallization half-rise (lag) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mlp_s0 | 50 | 9000 | 8500 | 10500 (2000) | None (None) | 10500 (2000) | 10500 (2000) | 9500 (1000) | 9000 (500) | 8500 (0) | 10000 (1500) | 9000 (500) |
| mlp_s1 | 50 | 8500 | 8000 | 9500 (1500) | None (None) | 9500 (1500) | 10000 (2000) | 8500 (500) | 8000 (0) | 8000 (0) | 9000 (1000) | 8000 (0) |
| mlp_s2 | 50 | 9000 | 8500 | 10000 (1500) | None (None) | 9500 (1000) | 10000 (1500) | 9500 (1000) | 9000 (500) | 8500 (0) | 9500 (1000) | 9500 (1000) |
| transformer_s1 | 400 | 15000 | 14000 | 15500 (1500) | None (None) | 15500 (1500) | 2000 (-12000) | 15000 (1000) | 15000 (1000) | 14000 (0) | 15500 (1500) | 15000 (1000) |
| transformer_s2 | 400 | 16500 | 15500 | 17000 (1500) | None (None) | 17000 (1500) | 2000 (-13500) | 16500 (1000) | 16000 (500) | 14500 (-1000) | 17000 (1500) | 16000 (500) |

## Plateau (memorised, not generalised) vs after generalisation

| run | metastability: plateau -> after | discreteness: plateau -> after | closure: plateau -> after | polarization: plateau -> after | compression: plateau -> after | op.associativity: plateau -> after | op.commutativity: plateau -> after | op.latin: plateau -> after | crystallization: plateau -> after |
|---|---|---|---|---|---|---|---|---|---|
| mlp_s0 | 0.29 -> 0.53 (final 0.41) | 0.99 -> 1.00 (final 1.00) | 0.34 -> 0.68 (final 0.50) | 0.89 -> 0.72 (final 0.43) | 0.11 -> 0.88 (final 1.00) | 0.04 -> 0.97 (final 1.00) | 0.16 -> 0.99 (final 1.00) | 0.00 -> 0.80 (final 1.00) | 0.23 -> 0.74 (final 0.66) |
| mlp_s1 | 0.30 -> 0.54 (final 0.46) | 0.99 -> 1.00 (final 1.00) | 0.35 -> 0.69 (final 0.63) | 0.88 -> 0.75 (final 0.49) | 0.11 -> 0.93 (final 1.00) | 0.06 -> 1.00 (final 1.00) | 0.16 -> 1.00 (final 1.00) | 0.00 -> 0.92 (final 1.00) | 0.24 -> 0.75 (final 0.73) |
| mlp_s2 | 0.27 -> 0.57 (final 0.57) | 0.99 -> 1.00 (final 1.00) | 0.33 -> 0.78 (final 0.88) | 0.88 -> 0.78 (final 0.76) | 0.11 -> 0.92 (final 1.00) | 0.05 -> 0.97 (final 1.00) | 0.15 -> 0.99 (final 1.00) | 0.00 -> 0.88 (final 1.00) | 0.22 -> 0.79 (final 0.83) |
| transformer_s1 | 0.21 -> 0.62 (final 0.73) | 0.99 -> 1.00 (final 1.00) | 0.25 -> 0.84 (final 0.98) | 0.70 -> 0.79 (final 0.97) | 0.26 -> 0.92 (final 1.00) | 0.11 -> 0.98 (final 1.00) | 0.79 -> 1.00 (final 1.00) | 0.00 -> 0.84 (final 1.00) | 0.26 -> 0.83 (final 0.91) |
| transformer_s2 | 0.23 -> 0.53 (final 0.41) | 0.99 -> 1.00 (final 1.00) | 0.28 -> 0.72 (final 0.55) | 0.68 -> 0.68 (final 0.54) | 0.29 -> 0.90 (final 0.78) | 0.12 -> 0.97 (final 0.90) | 0.85 -> 1.00 (final 1.00) | 0.00 -> 0.72 (final 0.04) | 0.28 -> 0.76 (final 0.66) |

## Cross-run equivalence of the extracted quotients

| step | mean ARI (hidden partitions) | mean op-table agreement | ARI to true-sum partition (reference) |
|---|---|---|---|
| 0 | 1.000 | 0.007 | mlp_s0:0.00, mlp_s1:0.00, mlp_s2:0.00, transformer_s1:0.00, transformer_s2:0.00 |
| 10 | 1.000 | 0.024 | mlp_s0:0.00, mlp_s1:0.00, mlp_s2:0.00, transformer_s1:0.00, transformer_s2:0.00 |
| 25 | 0.003 | 0.023 | mlp_s0:-0.00, mlp_s1:-0.00, mlp_s2:-0.00, transformer_s1:-0.00, transformer_s2:-0.00 |
| 50 | 0.316 | 0.062 | mlp_s0:0.00, mlp_s1:0.00, mlp_s2:0.00, transformer_s1:0.02, transformer_s2:0.02 |
| 100 | 0.054 | 0.149 | mlp_s0:0.16, mlp_s1:0.16, mlp_s2:0.16, transformer_s1:0.01, transformer_s2:0.01 |
| 150 | 0.150 | 0.175 | mlp_s0:0.16, mlp_s1:0.16, mlp_s2:0.16, transformer_s1:0.24, transformer_s2:0.27 |
| 200 | 0.151 | 0.176 | mlp_s0:0.16, mlp_s1:0.16, mlp_s2:0.16, transformer_s1:0.25, transformer_s2:0.27 |
| 300 | 0.147 | 0.177 | mlp_s0:0.16, mlp_s1:0.16, mlp_s2:0.16, transformer_s1:0.27, transformer_s2:0.28 |
| 400 | 0.141 | 0.179 | mlp_s0:0.16, mlp_s1:0.16, mlp_s2:0.16, transformer_s1:0.28, transformer_s2:0.30 |
| 500 | 0.137 | 0.180 | mlp_s0:0.16, mlp_s1:0.16, mlp_s2:0.16, transformer_s1:0.29, transformer_s2:0.32 |
| 750 | 0.121 | 0.182 | mlp_s0:0.15, mlp_s1:0.15, mlp_s2:0.16, transformer_s1:0.30, transformer_s2:0.33 |
| 1000 | 0.105 | 0.182 | mlp_s0:0.15, mlp_s1:0.15, mlp_s2:0.15, transformer_s1:0.31, transformer_s2:0.32 |
| 1500 | 0.095 | 0.182 | mlp_s0:0.15, mlp_s1:0.15, mlp_s2:0.15, transformer_s1:0.30, transformer_s2:0.32 |
| 2000 | 0.091 | 0.180 | mlp_s0:0.15, mlp_s1:0.15, mlp_s2:0.15, transformer_s1:0.30, transformer_s2:0.28 |
| 2500 | 0.125 | 0.182 | mlp_s0:0.15, mlp_s1:0.15, mlp_s2:0.15, transformer_s1:0.32, transformer_s2:0.35 |
| 3000 | 0.102 | 0.181 | mlp_s0:0.15, mlp_s1:0.15, mlp_s2:0.15, transformer_s1:0.31, transformer_s2:0.33 |
| 3500 | 0.096 | 0.181 | mlp_s0:0.15, mlp_s1:0.15, mlp_s2:0.15, transformer_s1:0.30, transformer_s2:0.32 |
| 4000 | 0.125 | 0.181 | mlp_s0:0.15, mlp_s1:0.15, mlp_s2:0.15, transformer_s1:0.32, transformer_s2:0.35 |
| 4500 | 0.102 | 0.181 | mlp_s0:0.15, mlp_s1:0.15, mlp_s2:0.15, transformer_s1:0.31, transformer_s2:0.33 |
| 5000 | 0.111 | 0.181 | mlp_s0:0.15, mlp_s1:0.15, mlp_s2:0.15, transformer_s1:0.31, transformer_s2:0.32 |
| 5500 | 0.122 | 0.182 | mlp_s0:0.15, mlp_s1:0.15, mlp_s2:0.15, transformer_s1:0.32, transformer_s2:0.35 |
| 6000 | 0.103 | 0.184 | mlp_s0:0.15, mlp_s1:0.15, mlp_s2:0.15, transformer_s1:0.31, transformer_s2:0.34 |
| 6500 | 0.114 | 0.189 | mlp_s0:0.15, mlp_s1:0.16, mlp_s2:0.15, transformer_s1:0.32, transformer_s2:0.32 |
| 7000 | 0.123 | 0.207 | mlp_s0:0.16, mlp_s1:0.18, mlp_s2:0.16, transformer_s1:0.32, transformer_s2:0.36 |
| 7500 | 0.114 | 0.251 | mlp_s0:0.17, mlp_s1:0.28, mlp_s2:0.18, transformer_s1:0.25, transformer_s2:0.33 |
| 8000 | 0.175 | 0.359 | mlp_s0:0.24, mlp_s1:0.61, mlp_s2:0.25, transformer_s1:0.33, transformer_s2:0.32 |
| 8500 | 0.303 | 0.503 | mlp_s0:0.48, mlp_s1:0.95, mlp_s2:0.52, transformer_s1:0.31, transformer_s2:0.35 |
| 9000 | 0.422 | 0.597 | mlp_s0:0.81, mlp_s1:0.96, mlp_s2:0.81, transformer_s1:0.31, transformer_s2:0.33 |
| 9500 | 0.511 | 0.628 | mlp_s0:0.96, mlp_s1:1.00, mlp_s2:0.99, transformer_s1:0.33, transformer_s2:0.36 |
| 10000 | 0.499 | 0.631 | mlp_s0:0.95, mlp_s1:1.00, mlp_s2:1.00, transformer_s1:0.32, transformer_s2:0.35 |
| 10500 | 0.506 | 0.633 | mlp_s0:1.00, mlp_s1:1.00, mlp_s2:1.00, transformer_s1:0.32, transformer_s2:0.33 |
| 11000 | 0.525 | 0.634 | mlp_s0:1.00, mlp_s1:1.00, mlp_s2:1.00, transformer_s1:0.33, transformer_s2:0.37 |
| 11500 | 0.513 | 0.637 | mlp_s0:1.00, mlp_s1:1.00, mlp_s2:1.00, transformer_s1:0.32, transformer_s2:0.35 |
| 12000 | 0.517 | 0.640 | mlp_s0:1.00, mlp_s1:1.00, mlp_s2:1.00, transformer_s1:0.34, transformer_s2:0.33 |

### Final structures

- mlp_s0 vs mlp_s1: ARI 1.000, op agreement 1.000, isomorphic up to relabelling: {'isomorphic': True, 'generator_pair': [1, 1]}
- mlp_s0 vs mlp_s2: ARI 1.000, op agreement 1.000, isomorphic up to relabelling: {'isomorphic': True, 'generator_pair': [1, 1]}
- mlp_s0 vs transformer_s1: ARI 1.000, op agreement 1.000, isomorphic up to relabelling: {'isomorphic': True, 'generator_pair': [1, 1]}
- mlp_s0 vs transformer_s2: ARI 0.932, op agreement 0.967, isomorphic up to relabelling: {'isomorphic': False, 'reason': 'no generator of t1 maps compatibly'}
- mlp_s1 vs mlp_s2: ARI 1.000, op agreement 1.000, isomorphic up to relabelling: {'isomorphic': True, 'generator_pair': [1, 1]}
- mlp_s1 vs transformer_s1: ARI 1.000, op agreement 1.000, isomorphic up to relabelling: {'isomorphic': True, 'generator_pair': [1, 1]}
- mlp_s1 vs transformer_s2: ARI 0.932, op agreement 0.967, isomorphic up to relabelling: {'isomorphic': False, 'reason': 'no generator of t1 maps compatibly'}
- mlp_s2 vs transformer_s1: ARI 1.000, op agreement 1.000, isomorphic up to relabelling: {'isomorphic': True, 'generator_pair': [1, 1]}
- mlp_s2 vs transformer_s2: ARI 0.932, op agreement 0.967, isomorphic up to relabelling: {'isomorphic': False, 'reason': 'no generator of t1 maps compatibly'}
- transformer_s1 vs transformer_s2: ARI 0.932, op agreement 0.967, isomorphic up to relabelling: {'isomorphic': False, 'reason': 'no generator of t1 maps compatibly'}

- mlp_s0 invariants: {'associativity': 1.0, 'commutativity': 1.0, 'identity': 1.0, 'latin': 1.0, 'n_outputs': 97}
- mlp_s1 invariants: {'associativity': 1.0, 'commutativity': 1.0, 'identity': 1.0, 'latin': 1.0, 'n_outputs': 97}
- mlp_s2 invariants: {'associativity': 1.0, 'commutativity': 1.0, 'identity': 1.0, 'latin': 1.0, 'n_outputs': 97}
- transformer_s1 invariants: {'associativity': 1.0, 'commutativity': 1.0, 'identity': 1.0, 'latin': 1.0, 'n_outputs': 97}
- transformer_s2 invariants: {'associativity': 0.89605, 'commutativity': 0.9968115633967478, 'identity': 0.9690721649484536, 'latin': 0.041237113402061855, 'n_outputs': 97}
