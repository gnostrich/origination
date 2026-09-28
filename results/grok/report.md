# Grokking as crystallisation: report

## Transition timing per run (steps)

| run | t_mem | t_gen(0.9) | t_gen_half | metastability half-rise (lag) | discreteness half-rise (lag) | closure half-rise (lag) | polarization half-rise (lag) | compression half-rise (lag) | op.associativity half-rise (lag) | op.commutativity half-rise (lag) | op.latin half-rise (lag) | crystallization half-rise (lag) | crystallization_lawfree half-rise (lag) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mlp_s0 | 50 | 9000 | 8500 | 10500 (2000) | None (None) | 10500 (2000) | 10500 (2000) | 9500 (1000) | 9000 (500) | 8500 (0) | 10000 (1500) | 9000 (500) | 10500 (2000) |
| mlp_s1 | 50 | 8500 | 8000 | 9500 (1500) | None (None) | 9500 (1500) | 10000 (2000) | 8500 (500) | 8000 (0) | 8000 (0) | 9000 (1000) | 8000 (0) | 9500 (1500) |
| mlp_s2 | 50 | 9000 | 8500 | 10000 (1500) | None (None) | 9500 (1000) | 10000 (1500) | 9500 (1000) | 9000 (500) | 8500 (0) | 9500 (1000) | 9500 (1000) | 9500 (1000) |
| transformer_s0 | 300 | 20500 | 19000 | 21500 (2500) | None (None) | 21500 (2500) | 2000 (-17000) | 21000 (2000) | 20000 (1000) | 18000 (-1000) | 21500 (2500) | 20000 (1000) | 21500 (2500) |
| transformer_s0_nowd | 750 | None | 100 | None (None) | None (None) | 2000 (1900) | 6000 (5900) | 9500 (9400) | None (None) | None (None) | None (None) | None (None) | None (None) |
| transformer_s1 | 400 | 15000 | 14000 | 15500 (1500) | None (None) | 15500 (1500) | 2000 (-12000) | 15000 (1000) | 15000 (1000) | 14000 (0) | 15500 (1500) | 15000 (1000) | 15500 (1500) |
| transformer_s2 | 400 | 16500 | 15500 | 17000 (1500) | None (None) | 17000 (1500) | 2000 (-13500) | 16500 (1000) | 16000 (500) | 14500 (-1000) | 17000 (1500) | 16000 (500) | 17000 (1500) |

## Plateau (memorised, not generalised) vs after generalisation

| run | metastability: plateau -> after | discreteness: plateau -> after | closure: plateau -> after | polarization: plateau -> after | compression: plateau -> after | op.associativity: plateau -> after | op.commutativity: plateau -> after | op.latin: plateau -> after | crystallization: plateau -> after | crystallization_lawfree: plateau -> after |
|---|---|---|---|---|---|---|---|---|---|---|
| mlp_s0 | 0.29 -> 0.53 (final 0.41) | 0.99 -> 1.00 (final 1.00) | 0.34 -> 0.68 (final 0.50) | 0.89 -> 0.72 (final 0.43) | 0.11 -> 0.88 (final 1.00) | 0.04 -> 0.97 (final 1.00) | 0.16 -> 0.99 (final 1.00) | 0.00 -> 0.80 (final 1.00) | 0.23 -> 0.74 (final 0.66) | 0.31 -> 0.59 (final 0.45) |
| mlp_s1 | 0.30 -> 0.54 (final 0.46) | 0.99 -> 1.00 (final 1.00) | 0.35 -> 0.69 (final 0.63) | 0.88 -> 0.75 (final 0.49) | 0.11 -> 0.93 (final 1.00) | 0.06 -> 1.00 (final 1.00) | 0.16 -> 1.00 (final 1.00) | 0.00 -> 0.92 (final 1.00) | 0.24 -> 0.75 (final 0.73) | 0.32 -> 0.60 (final 0.53) |
| mlp_s2 | 0.27 -> 0.57 (final 0.57) | 0.99 -> 1.00 (final 1.00) | 0.33 -> 0.78 (final 0.88) | 0.88 -> 0.78 (final 0.76) | 0.11 -> 0.92 (final 1.00) | 0.05 -> 0.97 (final 1.00) | 0.15 -> 0.99 (final 1.00) | 0.00 -> 0.88 (final 1.00) | 0.22 -> 0.79 (final 0.83) | 0.30 -> 0.66 (final 0.70) |
| transformer_s0 | 0.22 -> 0.57 (final 0.55) | 0.99 -> 1.00 (final 1.00) | 0.26 -> 0.78 (final 0.79) | 0.69 -> 0.72 (final 0.63) | 0.27 -> 0.91 (final 1.00) | 0.11 -> 0.97 (final 1.00) | 0.82 -> 1.00 (final 1.00) | 0.00 -> 0.77 (final 1.00) | 0.26 -> 0.80 (final 0.80) | 0.24 -> 0.66 (final 0.65) |
| transformer_s0_nowd | n/a -> n/a (final 0.05) | n/a -> n/a (final 1.00) | n/a -> n/a (final 0.04) | n/a -> n/a (final 0.92) | n/a -> n/a (final 0.60) | n/a -> n/a (final 0.07) | n/a -> n/a (final 0.78) | n/a -> n/a (final 0.00) | n/a -> n/a (final 0.11) | n/a -> n/a (final 0.04) |
| transformer_s1 | 0.21 -> 0.62 (final 0.73) | 0.99 -> 1.00 (final 1.00) | 0.25 -> 0.84 (final 0.98) | 0.70 -> 0.79 (final 0.97) | 0.26 -> 0.92 (final 1.00) | 0.11 -> 0.98 (final 1.00) | 0.79 -> 1.00 (final 1.00) | 0.00 -> 0.84 (final 1.00) | 0.26 -> 0.83 (final 0.91) | 0.23 -> 0.71 (final 0.84) |
| transformer_s2 | 0.23 -> 0.53 (final 0.41) | 0.99 -> 1.00 (final 1.00) | 0.28 -> 0.72 (final 0.55) | 0.68 -> 0.68 (final 0.54) | 0.29 -> 0.90 (final 0.78) | 0.12 -> 0.97 (final 0.90) | 0.85 -> 1.00 (final 1.00) | 0.00 -> 0.72 (final 0.04) | 0.28 -> 0.76 (final 0.66) | 0.25 -> 0.61 (final 0.47) |

## Blindly recovered algebra at the last checkpoint (closed-loop operation table)

| run | task | final test acc | classes | structure | latin | assoc | comm | identity | isotope is group | isotope cyclic | isotope order spectrum |
|---|---|---|---|---|---|---|---|---|---|---|---|
| mlp_s0 | zmod:97 | 1.000 | 97 | group, cyclic | 1.00 | 1.00 | 1.00 | 1.00 | True | True | {1: 1, 97: 96} |
| mlp_s1 | zmod:97 | 1.000 | 97 | group, cyclic | 1.00 | 1.00 | 1.00 | 1.00 | True | True | {1: 1, 97: 96} |
| mlp_s2 | zmod:97 | 1.000 | 97 | group, cyclic | 1.00 | 1.00 | 1.00 | 1.00 | True | True | {1: 1, 97: 96} |
| transformer_s0 | zmod:97 | 1.000 | 97 | group, cyclic | 1.00 | 1.00 | 1.00 | 1.00 | True | True | {1: 1, 97: 96} |
| transformer_s0_nowd | zmod:97 | 0.265 | 612 | no coherent algebra | 0.00 | 0.08 | 0.78 | 0.44 | n/a | n/a | n/a |
| transformer_s1 | zmod:97 | 1.000 | 98 | group, cyclic | 1.00 | 1.00 | 1.00 | 1.00 | True | True | {1: 1, 97: 96} |
| transformer_s2 | zmod:97 | 0.964 | 263 | no coherent algebra | 0.04 | 0.90 | 1.00 | 0.97 | n/a | n/a | n/a |

## Cross-run equivalence of the extracted quotients: task zmod:97

| step | mean ARI (hidden partitions) | mean op-table agreement | ARI to reference partition (true outputs) |
|---|---|---|---|
| 0 | 1.000 | 0.008 | mlp_s0:0.00, mlp_s1:0.00, mlp_s2:0.00, transformer_s0:0.00, transformer_s1:0.00, transformer_s2:0.00 |
| 10 | 1.000 | 0.025 | mlp_s0:0.00, mlp_s1:0.00, mlp_s2:0.00, transformer_s0:0.00, transformer_s1:0.00, transformer_s2:0.00 |
| 25 | 0.006 | 0.024 | mlp_s0:-0.00, mlp_s1:-0.00, mlp_s2:-0.00, transformer_s0:-0.00, transformer_s1:-0.00, transformer_s2:-0.00 |
| 50 | 0.238 | 0.055 | mlp_s0:0.00, mlp_s1:0.00, mlp_s2:0.00, transformer_s0:0.02, transformer_s1:0.02, transformer_s2:0.02 |
| 100 | 0.068 | 0.159 | mlp_s0:0.16, mlp_s1:0.16, mlp_s2:0.16, transformer_s0:0.01, transformer_s1:0.01, transformer_s2:0.01 |
| 150 | 0.151 | 0.188 | mlp_s0:0.16, mlp_s1:0.16, mlp_s2:0.16, transformer_s0:0.25, transformer_s1:0.24, transformer_s2:0.27 |
| 200 | 0.153 | 0.189 | mlp_s0:0.16, mlp_s1:0.16, mlp_s2:0.16, transformer_s0:0.26, transformer_s1:0.25, transformer_s2:0.27 |
| 300 | 0.150 | 0.191 | mlp_s0:0.16, mlp_s1:0.16, mlp_s2:0.16, transformer_s0:0.28, transformer_s1:0.27, transformer_s2:0.28 |
| 400 | 0.145 | 0.193 | mlp_s0:0.16, mlp_s1:0.16, mlp_s2:0.16, transformer_s0:0.29, transformer_s1:0.28, transformer_s2:0.30 |
| 500 | 0.141 | 0.194 | mlp_s0:0.16, mlp_s1:0.16, mlp_s2:0.16, transformer_s0:0.30, transformer_s1:0.29, transformer_s2:0.32 |
| 750 | 0.125 | 0.196 | mlp_s0:0.15, mlp_s1:0.15, mlp_s2:0.16, transformer_s0:0.31, transformer_s1:0.30, transformer_s2:0.33 |
| 1000 | 0.110 | 0.196 | mlp_s0:0.15, mlp_s1:0.15, mlp_s2:0.15, transformer_s0:0.31, transformer_s1:0.31, transformer_s2:0.32 |
| 1500 | 0.101 | 0.196 | mlp_s0:0.15, mlp_s1:0.15, mlp_s2:0.15, transformer_s0:0.30, transformer_s1:0.30, transformer_s2:0.32 |
| 2000 | 0.108 | 0.195 | mlp_s0:0.15, mlp_s1:0.15, mlp_s2:0.15, transformer_s0:0.30, transformer_s1:0.30, transformer_s2:0.28 |
| 2500 | 0.131 | 0.197 | mlp_s0:0.15, mlp_s1:0.15, mlp_s2:0.15, transformer_s0:0.32, transformer_s1:0.32, transformer_s2:0.35 |
| 3000 | 0.107 | 0.197 | mlp_s0:0.15, mlp_s1:0.15, mlp_s2:0.15, transformer_s0:0.31, transformer_s1:0.31, transformer_s2:0.33 |
| 3500 | 0.101 | 0.195 | mlp_s0:0.15, mlp_s1:0.15, mlp_s2:0.15, transformer_s0:0.28, transformer_s1:0.30, transformer_s2:0.32 |
| 4000 | 0.133 | 0.196 | mlp_s0:0.15, mlp_s1:0.15, mlp_s2:0.15, transformer_s0:0.32, transformer_s1:0.32, transformer_s2:0.35 |
| 4500 | 0.108 | 0.196 | mlp_s0:0.15, mlp_s1:0.15, mlp_s2:0.15, transformer_s0:0.31, transformer_s1:0.31, transformer_s2:0.33 |
| 5000 | 0.116 | 0.196 | mlp_s0:0.15, mlp_s1:0.15, mlp_s2:0.15, transformer_s0:0.30, transformer_s1:0.31, transformer_s2:0.32 |
| 5500 | 0.130 | 0.197 | mlp_s0:0.15, mlp_s1:0.15, mlp_s2:0.15, transformer_s0:0.32, transformer_s1:0.32, transformer_s2:0.35 |
| 6000 | 0.108 | 0.198 | mlp_s0:0.15, mlp_s1:0.15, mlp_s2:0.15, transformer_s0:0.31, transformer_s1:0.31, transformer_s2:0.34 |
| 6500 | 0.127 | 0.203 | mlp_s0:0.15, mlp_s1:0.16, mlp_s2:0.15, transformer_s0:0.32, transformer_s1:0.32, transformer_s2:0.32 |
| 7000 | 0.129 | 0.219 | mlp_s0:0.16, mlp_s1:0.18, mlp_s2:0.16, transformer_s0:0.32, transformer_s1:0.32, transformer_s2:0.36 |
| 7500 | 0.119 | 0.257 | mlp_s0:0.17, mlp_s1:0.28, mlp_s2:0.18, transformer_s0:0.31, transformer_s1:0.25, transformer_s2:0.33 |
| 8000 | 0.185 | 0.346 | mlp_s0:0.24, mlp_s1:0.61, mlp_s2:0.25, transformer_s0:0.32, transformer_s1:0.33, transformer_s2:0.32 |
| 8500 | 0.274 | 0.459 | mlp_s0:0.48, mlp_s1:0.95, mlp_s2:0.52, transformer_s0:0.32, transformer_s1:0.31, transformer_s2:0.35 |
| 9000 | 0.355 | 0.529 | mlp_s0:0.81, mlp_s1:0.96, mlp_s2:0.81, transformer_s0:0.31, transformer_s1:0.31, transformer_s2:0.33 |
| 9500 | 0.427 | 0.553 | mlp_s0:0.96, mlp_s1:1.00, mlp_s2:0.99, transformer_s0:0.33, transformer_s1:0.33, transformer_s2:0.36 |
| 10000 | 0.413 | 0.555 | mlp_s0:0.95, mlp_s1:1.00, mlp_s2:1.00, transformer_s0:0.32, transformer_s1:0.32, transformer_s2:0.35 |
| 10500 | 0.405 | 0.552 | mlp_s0:1.00, mlp_s1:1.00, mlp_s2:1.00, transformer_s0:0.27, transformer_s1:0.32, transformer_s2:0.33 |
| 11000 | 0.436 | 0.558 | mlp_s0:1.00, mlp_s1:1.00, mlp_s2:1.00, transformer_s0:0.33, transformer_s1:0.33, transformer_s2:0.37 |
| 11500 | 0.421 | 0.560 | mlp_s0:1.00, mlp_s1:1.00, mlp_s2:1.00, transformer_s0:0.32, transformer_s1:0.32, transformer_s2:0.35 |
| 12000 | 0.420 | 0.561 | mlp_s0:1.00, mlp_s1:1.00, mlp_s2:1.00, transformer_s0:0.30, transformer_s1:0.34, transformer_s2:0.33 |

### Final structures

- mlp_s0 vs mlp_s1: ARI 1.000, op agreement 1.000, isomorphic up to relabelling: {'isomorphic': True, 'generator_pair': [1, 1]}
- mlp_s0 vs mlp_s2: ARI 1.000, op agreement 1.000, isomorphic up to relabelling: {'isomorphic': True, 'generator_pair': [1, 1]}
- mlp_s0 vs transformer_s0: ARI 1.000, op agreement 1.000, isomorphic up to relabelling: {'isomorphic': True, 'generator_pair': [1, 1]}
- mlp_s0 vs transformer_s1: ARI 1.000, op agreement 1.000, isomorphic up to relabelling: {'isomorphic': True, 'generator_pair': [1, 1]}
- mlp_s0 vs transformer_s2: ARI 0.932, op agreement 0.967, isomorphic up to relabelling: {'isomorphic': False, 'reason': 'no generator of t1 maps compatibly'}
- mlp_s1 vs mlp_s2: ARI 1.000, op agreement 1.000, isomorphic up to relabelling: {'isomorphic': True, 'generator_pair': [1, 1]}
- mlp_s1 vs transformer_s0: ARI 1.000, op agreement 1.000, isomorphic up to relabelling: {'isomorphic': True, 'generator_pair': [1, 1]}
- mlp_s1 vs transformer_s1: ARI 1.000, op agreement 1.000, isomorphic up to relabelling: {'isomorphic': True, 'generator_pair': [1, 1]}
- mlp_s1 vs transformer_s2: ARI 0.932, op agreement 0.967, isomorphic up to relabelling: {'isomorphic': False, 'reason': 'no generator of t1 maps compatibly'}
- mlp_s2 vs transformer_s0: ARI 1.000, op agreement 1.000, isomorphic up to relabelling: {'isomorphic': True, 'generator_pair': [1, 1]}
- mlp_s2 vs transformer_s1: ARI 1.000, op agreement 1.000, isomorphic up to relabelling: {'isomorphic': True, 'generator_pair': [1, 1]}
- mlp_s2 vs transformer_s2: ARI 0.932, op agreement 0.967, isomorphic up to relabelling: {'isomorphic': False, 'reason': 'no generator of t1 maps compatibly'}
- transformer_s0 vs transformer_s1: ARI 1.000, op agreement 1.000, isomorphic up to relabelling: {'isomorphic': True, 'generator_pair': [1, 1]}
- transformer_s0 vs transformer_s2: ARI 0.932, op agreement 0.967, isomorphic up to relabelling: {'isomorphic': False, 'reason': 'no generator of t1 maps compatibly'}
- transformer_s1 vs transformer_s2: ARI 0.932, op agreement 0.967, isomorphic up to relabelling: {'isomorphic': False, 'reason': 'no generator of t1 maps compatibly'}

