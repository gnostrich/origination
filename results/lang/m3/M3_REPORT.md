# Milestone 3: sealed-test results

Substrate: GRU world model of the kinematic cart; held-out next-bin accuracy 0.953 (length 16), 0.892 (length 24; steps 17–24: 0.773); 2.03e+06 bits.

Primary model (validation rule, frozen before test): **eps0.5_C1_s0**, K = 49, L(M) = 8865 bits (argmax) / 20821 (distribution), 520 exceptions over 196 first-order fits, undefined fits 0.041.

Prospective predictions file `predictions_20261001T035637Z.json` (sha256[:16] 5835c02d107b3ddd) was written at 20261001T035637Z, before the substrate was executed on the sealed test (see `protocol_log.txt`).

## Sealed test: primary model and baselines (argmax fidelity; abstention = error)

| model | K | bits | T1 interpolation | T2 unseen compositions | T3 withheld chunks | T4 longer (8–12) | T4 coverage | whole-string T1 / T4 |
|---|---|---|---|---|---|---|---|---|
| **primary interface model** | 49 | 8865 | **0.898** | **0.775** | **0.709** | **0.518** | 0.95 | 0.84 / 0.07 |
| random | 49 | 13386 | 0.556 | 0.399 | 0.342 | 0.257 | 1.00 | 0.45 / 0.01 |
| lookup | 49 | 4704 | 0.910 | 0.488 | 0.330 | 0.157 | 0.17 | 0.86 / 0.00 |
| kmeans | 49 | 12220 | 0.695 | 0.618 | 0.570 | 0.423 | 1.00 | 0.61 / 0.03 |
| kmeans_pca8 | 49 | 12339 | 0.747 | 0.655 | 0.602 | 0.459 | 1.00 | 0.67 / 0.04 |
| full substrate | – | 2.03e+06 | 1 | 1 | 1 | 1 | 1 | 1 / 1 |

JS fidelity (1 − JS distance) of the primary: T1 0.845, T2 0.733, T3 0.668, T4 0.496

Per-step fidelity of the primary on T4 (composition depth): 1: 0.96, 2: 0.80, 3: 0.68, 4: 0.59, 5: 0.51, 6: 0.43, 7: 0.37, 8: 0.35, 9: 0.29, 10: 0.29, 11: 0.27, 12: 0.23

Causal commutation (do(I = i) through up to 5 distinct realisations, 8-step T4 prefixes, 37 interfaces tested): realisations agree 0.637; agree with the abstract prediction 0.526.

Reuse: each interface is reached by 4.0 first-order fits on average; 172 of 196 first-order fits are exercised by the test queries, 295 times each on average.

## Frontier (all discovered models, seed 0, scored on the sealed test after freezing; not used for selection)

| model | K | bits | validation | T1 | T2 | T3 | T4 | T4 coverage |
|---|---|---|---|---|---|---|---|---|
| eps0.5_C1_s0 | 49 | 8865 | 0.577 | 0.898 | 0.775 | 0.709 | 0.518 | 0.95 |
| eps0.3_C1_s0 | 116 | 17126 | 0.554 | 0.891 | 0.767 | 0.696 | 0.488 | 0.84 |
| eps0.2_C1_s0 | 194 | 21959 | 0.447 | 0.803 | 0.642 | 0.575 | 0.348 | 0.52 |
| eps0.5_C2_s0 | 211 | 22690 | 0.531 | 0.843 | 0.727 | 0.669 | 0.447 | 0.55 |
| eps0.5_C3_s0 | 348 | 25825 | 0.297 | 0.572 | 0.417 | 0.371 | 0.201 | 0.21 |
| eps0.3_C2_s0 | 352 | 26803 | 0.262 | 0.517 | 0.369 | 0.326 | 0.174 | 0.18 |
| eps0.1_C1_s0 | 332 | 27252 | 0.261 | 0.596 | 0.432 | 0.358 | 0.189 | 0.23 |
| eps0.2_C2_s0 | 443 | 29418 | 0.134 | 0.386 | 0.251 | 0.214 | 0.107 | 0.11 |
| eps0.3_C3_s0 | 485 | 29666 | 0.141 | 0.328 | 0.211 | 0.177 | 0.089 | 0.09 |
| eps0.05_C1_s0 | 436 | 30821 | 0.151 | 0.401 | 0.266 | 0.209 | 0.106 | 0.12 |
| eps0.2_C3_s0 | 547 | 32679 | 0.077 | 0.196 | 0.127 | 0.107 | 0.052 | 0.05 |
| eps0.1_C2_s0 | 552 | 33489 | 0.076 | 0.173 | 0.120 | 0.102 | 0.051 | 0.05 |
| eps0.1_C3_s0 | 595 | 35854 | 0.045 | 0.085 | 0.057 | 0.049 | 0.024 | 0.02 |
| eps0.05_C2_s0 | 598 | 36419 | 0.043 | 0.108 | 0.071 | 0.060 | 0.030 | 0.03 |
| eps0.05_C3_s0 | 638 | 38583 | 0.023 | 0.048 | 0.030 | 0.024 | 0.011 | 0.01 |

Exploratory (excluded from the verdict; grid edge check): eps0.7_C1_s0: K=31, validation 0.580, bits 5176, test T1=0.894 T2=0.762 T3=0.694 T4=0.503; eps1.0_C1_s0: K=2, validation 0.160, bits 77, test T1=0.222 T2=0.174 T3=0.167 T4=0.146

## Nested context families at the primary ε (seed 0)

| family | K | bits | validation | T1 | T2 | T3 | T4 |
|---|---|---|---|---|---|---|---|
| C1 | 49 | 8865 | 0.577 | 0.898 | 0.775 | 0.709 | 0.518 |
| C2 | 211 | 22690 | 0.531 | 0.843 | 0.727 | 0.669 | 0.447 |
| C3 | 348 | 25825 | 0.297 | 0.572 | 0.417 | 0.371 | 0.201 |

Refinement: fraction of C2 interfaces inside a single C1 interface 0.74; of C3 inside a single C2 interface 0.90; ARI(C1, C2) 0.31, ARI(C2, C3) 0.66.

## Reproducibility at the primary (ε, C), seeds 0–2

Representational agreement (ARI of discovery partitions): eps0.5_C1_s0|eps0.5_C1_s1: 1.00, eps0.5_C1_s0|eps0.5_C1_s2: 1.00, eps0.5_C1_s1|eps0.5_C1_s2: 1.00

Behavioural distance (disagreement of sealed-test predictions): eps0.5_C1_s0|eps0.5_C1_s1: 0.189, eps0.5_C1_s0|eps0.5_C1_s2: 0.198, eps0.5_C1_s1|eps0.5_C1_s2: 0.205

## Interpretation (simulator unsealed after all of the above)

Variance of the sealed latent explained by the interface partition of the discovery pool: position R² = 0.971, velocity R² = 0.570. ARI with the 8 observation bins 0.26; with a position×velocity grid (8×6, 53 cells occupied) 0.26.

Largest interfaces (mean ± sd of the sealed position and velocity of their realisations):

- I18: n = 47, x = 0.52 ± 0.03, v = -0.003 ± 0.020
- I24: n = 45, x = 0.43 ± 0.03, v = -0.026 ± 0.031
- I15: n = 44, x = 0.58 ± 0.03, v = +0.021 ± 0.032
- I0: n = 36, x = 0.93 ± 0.04, v = +0.071 ± 0.051
- I13: n = 32, x = 0.66 ± 0.05, v = -0.011 ± 0.046
- I9: n = 30, x = 0.69 ± 0.03, v = +0.045 ± 0.035
- I7: n = 25, x = 0.79 ± 0.05, v = -0.003 ± 0.054
- I31: n = 24, x = 0.31 ± 0.04, v = -0.037 ± 0.043
- I20: n = 22, x = 0.46 ± 0.03, v = +0.009 ± 0.030
- I22: n = 22, x = 0.48 ± 0.06, v = -0.035 ± 0.042
- I45: n = 22, x = 0.06 ± 0.05, v = -0.064 ± 0.050
- I27: n = 21, x = 0.34 ± 0.05, v = +0.013 ± 0.054

Median within-interface spread: position 0.036 (track length 1, observation bin width 0.125), velocity 0.037 (range ±0.15).