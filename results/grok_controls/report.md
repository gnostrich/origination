# Grokking as crystallisation: report

## Transition timing per run (steps)

| run | t_mem | t_gen(0.9) | t_gen_half | metastability half-rise (lag) | discreteness half-rise (lag) | closure half-rise (lag) | polarization half-rise (lag) | compression half-rise (lag) | op.associativity half-rise (lag) | op.commutativity half-rise (lag) | op.latin half-rise (lag) | crystallization half-rise (lag) | crystallization_lawfree half-rise (lag) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mlp_corrupt-0p05-zmod-97_s0 | 50 | 14000 | 12000 | 13000 (1000) | None (None) | 13000 (1000) | None (None) | 13500 (1500) | 13000 (1000) | 12500 (500) | 16000 (4000) | 13000 (1000) | 13000 (1000) |
| mlp_corrupt-0p15-zmod-97_s0 | 50 | None | None | 400 (None) | None (None) | 100 (None) | None (None) | 100 (None) | None (None) | None (None) | None (None) | None (None) | 200 (None) |
| mlp_random-97_s0 | 50 | None | None | 400 (None) | None (None) | 100 (None) | None (None) | 100 (None) | None (None) | None (None) | None (None) | None (None) | 150 (None) |
| mlp_scramble-zmod-97_s0 | 50 | 9000 | 8000 | 9500 (1500) | None (None) | 9000 (1000) | 9500 (1500) | 9000 (1000) | None (None) | 8500 (500) | 9000 (1000) | 9000 (1000) | 9000 (1000) |
| mlp_sub-97_s0 | 50 | 9500 | 9000 | 9500 (500) | None (None) | 9500 (500) | 11000 (2000) | 9500 (500) | None (None) | None (None) | 10000 (1000) | 9500 (500) | 9500 (500) |
| mlp_zmod-101_s0 | 50 | 9000 | 8500 | 9500 (1000) | None (None) | 9500 (1000) | None (None) | 9500 (1000) | 8500 (0) | 8500 (0) | 9500 (1000) | 9500 (1000) | 9500 (1000) |
| mlp_zmod-89_s0 | 50 | 12500 | 11500 | 12500 (1000) | None (None) | 12500 (1000) | 300 (-11200) | 12500 (1000) | 12000 (500) | 11500 (0) | 12500 (1000) | 12000 (500) | 12500 (1000) |
| mlp_zprod-8x8_s0 | 50 | 13500 | 12500 | 14000 (1500) | None (None) | 14000 (1500) | 14000 (1500) | 14000 (1500) | 13000 (500) | 13000 (500) | 14000 (1500) | 13500 (1000) | 14000 (1500) |

## Plateau (memorised, not generalised) vs after generalisation

| run | metastability: plateau -> after | discreteness: plateau -> after | closure: plateau -> after | polarization: plateau -> after | compression: plateau -> after | op.associativity: plateau -> after | op.commutativity: plateau -> after | op.latin: plateau -> after | crystallization: plateau -> after | crystallization_lawfree: plateau -> after |
|---|---|---|---|---|---|---|---|---|---|---|
| mlp_corrupt-0p05-zmod-97_s0 | 0.28 -> 0.48 (final 0.36) | 0.99 -> 1.00 (final 1.00) | 0.34 -> 0.63 (final 0.43) | 0.86 -> 0.66 (final 0.45) | 0.13 -> 0.68 (final 0.66) | 0.08 -> 0.89 (final 0.93) | 0.20 -> 0.94 (final 0.96) | 0.00 -> 0.15 (final 0.24) | 0.24 -> 0.70 (final 0.61) | 0.31 -> 0.54 (final 0.39) |
| mlp_corrupt-0p15-zmod-97_s0 | n/a -> n/a (final 0.20) | n/a -> n/a (final 0.99) | n/a -> n/a (final 0.27) | n/a -> n/a (final 0.80) | n/a -> n/a (final 0.14) | n/a -> n/a (final 0.03) | n/a -> n/a (final 0.12) | n/a -> n/a (final 0.00) | n/a -> n/a (final 0.19) | n/a -> n/a (final 0.23) |
| mlp_random-97_s0 | n/a -> n/a (final 0.25) | n/a -> n/a (final 0.99) | n/a -> n/a (final 0.34) | n/a -> n/a (final 0.86) | n/a -> n/a (final 0.14) | n/a -> n/a (final 0.01) | n/a -> n/a (final 0.02) | n/a -> n/a (final 0.00) | n/a -> n/a (final 0.18) | n/a -> n/a (final 0.29) |
| mlp_scramble-zmod-97_s0 | 0.26 -> 0.60 (final 0.42) | 0.99 -> 1.00 (final 1.00) | 0.31 -> 0.79 (final 0.52) | 0.88 -> 0.70 (final 0.39) | 0.11 -> 0.96 (final 1.00) | 0.01 -> 0.02 (final 0.02) | 0.17 -> 1.00 (final 1.00) | 0.00 -> 0.97 (final 1.00) | 0.17 -> 0.32 (final 0.26) | 0.28 -> 0.68 (final 0.46) |
| mlp_sub-97_s0 | 0.26 -> 0.67 (final 0.87) | 0.99 -> 1.00 (final 1.00) | 0.30 -> 0.88 (final 1.00) | 0.88 -> 0.88 (final 1.00) | 0.11 -> 0.96 (final 1.00) | 0.01 -> 0.01 (final 0.01) | 0.02 -> 0.01 (final 0.01) | 0.00 -> 0.91 (final 1.00) | 0.17 -> 0.28 (final 0.31) | 0.28 -> 0.76 (final 0.92) |
| mlp_zmod-101_s0 | 0.27 -> 0.56 (final 0.42) | 0.99 -> 1.00 (final 1.00) | 0.33 -> 0.74 (final 0.52) | 0.88 -> 0.75 (final 0.39) | 0.11 -> 0.91 (final 1.00) | 0.05 -> 0.99 (final 1.00) | 0.16 -> 1.00 (final 1.00) | 0.00 -> 0.89 (final 1.00) | 0.22 -> 0.77 (final 0.67) | 0.30 -> 0.63 (final 0.46) |
| mlp_zmod-89_s0 | 0.25 -> 0.65 (final 0.46) | 0.99 -> 1.00 (final 1.00) | 0.29 -> 0.81 (final 0.63) | 0.86 -> 0.73 (final 0.43) | 0.12 -> 0.97 (final 1.00) | 0.06 -> 1.00 (final 1.00) | 0.17 -> 1.00 (final 1.00) | 0.00 -> 0.96 (final 1.00) | 0.21 -> 0.84 (final 0.73) | 0.27 -> 0.72 (final 0.53) |
| mlp_zprod-8x8_s0 | 0.30 -> 0.73 (final 0.55) | 0.99 -> 1.00 (final 1.00) | 0.35 -> 0.89 (final 0.80) | 0.80 -> 0.86 (final 0.63) | 0.16 -> 0.94 (final 1.00) | 0.08 -> 0.99 (final 1.00) | 0.18 -> 1.00 (final 1.00) | 0.00 -> 0.92 (final 1.00) | 0.27 -> 0.88 (final 0.80) | 0.32 -> 0.79 (final 0.65) |

## Blindly recovered algebra at the last checkpoint (closed-loop operation table)

| run | task | final test acc | classes | structure | latin | assoc | comm | identity | isotope is group | isotope cyclic | isotope order spectrum |
|---|---|---|---|---|---|---|---|---|---|---|---|
| mlp_corrupt-0p05-zmod-97_s0 | corrupt:0.05:zmod:97 | 0.944 | 451 | no coherent algebra | 0.24 | 0.93 | 0.96 | 0.99 | n/a | n/a | n/a |
| mlp_corrupt-0p15-zmod-97_s0 | corrupt:0.15:zmod:97 | 0.038 | 5015 | no coherent algebra | 0.00 | 0.03 | 0.12 | 0.30 | n/a | n/a | n/a |
| mlp_random-97_s0 | random:97 | 0.010 | 5064 | no coherent algebra | 0.00 | 0.01 | 0.02 | 0.03 | n/a | n/a | n/a |
| mlp_scramble-zmod-97_s0 | scramble:zmod:97 | 1.000 | 97 | quasigroup, non-associative, isotopic to a cyclic group | 1.00 | 0.02 | 1.00 | 0.06 | True | True | {1: 1, 97: 96} |
| mlp_sub-97_s0 | sub:97 | 1.000 | 97 | quasigroup, non-associative, isotopic to a cyclic group | 1.00 | 0.01 | 0.01 | 0.51 | True | True | {1: 1, 97: 96} |
| mlp_zmod-101_s0 | zmod:101 | 1.000 | 101 | group, cyclic | 1.00 | 1.00 | 1.00 | 1.00 | True | True | {1: 1, 101: 100} |
| mlp_zmod-89_s0 | zmod:89 | 1.000 | 89 | group, cyclic | 1.00 | 1.00 | 1.00 | 1.00 | True | True | {1: 1, 89: 88} |
| mlp_zprod-8x8_s0 | zprod:8x8 | 1.000 | 64 | group, non-cyclic | 1.00 | 1.00 | 1.00 | 1.00 | True | False | {1: 1, 2: 3, 4: 12, 8: 48} |

