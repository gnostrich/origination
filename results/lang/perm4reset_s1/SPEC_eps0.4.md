# Extracted interface model (ε = 0.4, 4 interfaces, 120 fits)

## Interfaces
- Interface I0: 170 substitutable realisations (e.g. (), (1,), (2,), (4,)); accepted contexts: all 30 discovery contexts; within-class disagreement 0.129; effect: a0→o1 (1.00); a1→o0 (1.00); a2→o0 (1.00); a3→o1 (1.00); a4→o0 (1.00)
- Interface I1: 102 substitutable realisations (e.g. (0,), (3,), (0, 1), (0, 2)); accepted contexts: all 30 discovery contexts; within-class disagreement 0.219; effect: a0→o0 (1.00); a1→o1 (1.00); a2→o1 (1.00); a3→o0 (1.00); a4→o0 (1.00)
- Interface I2: 47 substitutable realisations (e.g. (3, 3), (0, 1, 0), (0, 2, 1), (0, 3, 3)); accepted contexts: all 30 discovery contexts; within-class disagreement 0.180; effect: a0→o3 (1.00); a1→o2 (1.00); a2→o2 (1.00); a3→o3 (1.00); a4→o0 (1.00)
- Interface I3: 37 substitutable realisations (e.g. (2, 3, 3), (3, 1, 1), (3, 1, 3), (3, 3, 0)); accepted contexts: all 30 discovery contexts; within-class disagreement 0.206; effect: a0→o2 (1.00); a1→o3 (1.00); a2→o3 (1.00); a3→o2 (1.00); a4→o0 (1.00)

## First-order fits  (I, action) ⇝ I'
- (I0, a0) ⇝ I1   confidence 1.00 (n=12)
- (I1, a0) ⇝ I0   confidence 1.00 (n=12)
- (I2, a0) ⇝ I3   confidence 1.00 (n=12)
- (I3, a0) ⇝ I2   confidence 0.75 (n=12)
- (I0, a1) ⇝ I0   confidence 0.75 (n=12)
- (I1, a1) ⇝ I1   confidence 0.50 (n=12)
- (I2, a1) ⇝ I1   confidence 0.58 (n=12)
- (I3, a1) ⇝ I1   confidence 0.50 (n=12)
- (I0, a2) ⇝ I0   confidence 0.83 (n=12)
- (I1, a2) ⇝ I1   confidence 0.42 (n=12)
- (I2, a2) ⇝ I2   confidence 0.58 (n=12)
- (I3, a2) ⇝ I3   confidence 0.58 (n=12)
- (I0, a3) ⇝ I1   confidence 0.67 (n=12)
- (I1, a3) ⇝ I2   confidence 0.42 (n=12)
- (I2, a3) ⇝ I1   confidence 0.50 (n=12)
- (I3, a3) ⇝ I2   confidence 0.58 (n=12)
- (I0, a4) ⇝ I0   confidence 1.00 (n=12)
- (I1, a4) ⇝ I0   confidence 1.00 (n=12)
- (I2, a4) ⇝ I0   confidence 1.00 (n=12)
- (I3, a4) ⇝ I0   confidence 1.00 (n=12)

## Observed multi-step fits (76; the rest are derived by chaining)
- (I0, (0, 1)) ⇝ I3   confidence 0.58
- (I0, (0, 2)) ⇝ I1   confidence 0.83
- (I0, (0, 4)) ⇝ I0   confidence 1.00
- (I0, (1, 0)) ⇝ I1   confidence 0.83
- (I0, (1, 1)) ⇝ I0   confidence 0.92
- (I0, (1, 2)) ⇝ I0   confidence 0.83
- (I0, (1, 3)) ⇝ I0   confidence 0.67
- (I0, (2, 1)) ⇝ I2   confidence 0.50
- (I0, (2, 2)) ⇝ I0   confidence 1.00
- (I0, (2, 3)) ⇝ I1   confidence 0.83
- (I0, (2, 4)) ⇝ I0   confidence 1.00
- (I0, (3, 0)) ⇝ I0   confidence 0.67
- … 64 more

## Second-order interfaces (chunk types): 14
- W0: chunks [(0,), (0, 2), (2, 0), (2, 3)]; induced map I0→I1 I1→I0 I2→I3 I3→I2
- W1: chunks [(1,)]; induced map I0→I0 I1→I1 I2→I1 I3→I1
- W2: chunks [(2,), (0, 0), (1, 1), (2, 2)]; induced map I0→I0 I1→I1 I2→I2 I3→I3
- W3: chunks [(3,)]; induced map I0→I1 I1→I2 I2→I1 I3→I2
- W4: chunks [(4,), (0, 4), (1, 4), (2, 4), (3, 4), (4, 1)] …; induced map I0→I0 I1→I0 I2→I0 I3→I0
- W5: chunks [(0, 1), (3, 2)]; induced map I0→I3 I1→I2 I2→I1 I3→I2
- W6: chunks [(0, 3)]; induced map I0→I2 I1→I1 I2→I2 I3→I1
- W7: chunks [(1, 0)]; induced map I0→I1 I1→I0 I2→I0 I3→I0
- W8: chunks [(1, 2)]; induced map I0→I0 I1→I2 I2→I1 I3→I1
- W9: chunks [(1, 3)]; induced map I0→I0 I1→I3 I2→I0 I3→I3
- W10: chunks [(2, 1)]; induced map I0→I2 I1→I1 I2→I2 I3→I0
- W11: chunks [(3, 0)]; induced map I0→I0 I1→I1 I2→I0 I3→I0
- W12: chunks [(3, 1), (4, 0), (4, 3)]; induced map I0→I1 I1→I1 I2→I1 I3→I1
- W13: chunks [(3, 3)]; induced map I0→I2 I1→I3 I2→I0 I3→I0

## Second-order fits  (W, W') ⇝ W''  (114)
- (W0, W0) ⇝ W2
- (W0, W1) ⇝ W5
- (W0, W2) ⇝ W0
- (W0, W3) ⇝ W6
- (W0, W4) ⇝ W4
- (W0, W6) ⇝ W3
- (W0, W7) ⇝ W11
- (W0, W11) ⇝ W7
- (W0, W12) ⇝ W12
- (W1, W0) ⇝ W7
- (W1, W1) ⇝ W2
- (W1, W2) ⇝ W8
- (W1, W3) ⇝ W9
- (W1, W4) ⇝ W4
- (W1, W7) ⇝ W7
- (W1, W11) ⇝ W1
- (W1, W12) ⇝ W12
- (W2, W0) ⇝ W0
- (W2, W1) ⇝ W10
- (W2, W2) ⇝ W2
- (W2, W3) ⇝ W0
- (W2, W4) ⇝ W4
- (W2, W5) ⇝ W5
- (W2, W6) ⇝ W6
- (W2, W7) ⇝ W7
- (W2, W8) ⇝ W8
- (W2, W9) ⇝ W9
- (W2, W10) ⇝ W10
- (W2, W11) ⇝ W11
- (W2, W12) ⇝ W12
- (W2, W13) ⇝ W13
- (W3, W0) ⇝ W11
- (W3, W1) ⇝ W12
- (W3, W2) ⇝ W5
- (W3, W3) ⇝ W13
- (W3, W4) ⇝ W4
- (W3, W5) ⇝ W6
- (W3, W6) ⇝ W3
- (W3, W7) ⇝ W4
- (W3, W8) ⇝ W6
- … 74 more

Withheld from discovery (predicted only by chaining): [(0, 0), (0, 3), (1, 4), (2, 0), (3, 1), (4, 1)]