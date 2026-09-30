# Extracted interface model (ε = 0.4, 4 interfaces, 80 fits)

## Interfaces
- Interface I0: 176 substitutable realisations (e.g. (), (0,), (1,), (3,)); accepted contexts: all 20 discovery contexts; within-class disagreement 0.046; effect: a0→o0 (1.00); a1→o0 (1.00); a2→o2 (1.00); a3→o0 (1.00)
- Interface I1: 74 substitutable realisations (e.g. (2,), (0, 2), (1, 2), (2, 0)); accepted contexts: all 20 discovery contexts; within-class disagreement 0.061; effect: a0→o2 (1.00); a1→o2 (1.00); a2→o0 (1.00); a3→o0 (1.00)
- Interface I2: 23 substitutable realisations (e.g. (0, 0, 0), (1, 0, 0), (3, 0, 0), (np.int64(3), np.int64(2), np.int64(2), np.int64(1), np.int64(1), np.int64(0), np.int64(0), np.int64(0))); accepted contexts: all 20 discovery contexts; within-class disagreement 0.108; effect: a0→o1 (1.00); a1→o1 (1.00); a2→o3 (1.00); a3→o0 (1.00)
- Interface I3: 12 substitutable realisations (e.g. (np.int64(1), np.int64(0), np.int64(0), np.int64(0), np.int64(0), np.int64(2), np.int64(2), np.int64(2)), (np.int64(2), np.int64(0), np.int64(3), np.int64(2), np.int64(0), np.int64(2), np.int64(2), np.int64(0)), (np.int64(3), np.int64(3), np.int64(1), np.int64(0), np.int64(2), np.int64(0), np.int64(0), np.int64(0)), (np.int64(2), np.int64(2), np.int64(2), np.int64(3), np.int64(0), np.int64(0), np.int64(0), np.int64(2))); accepted contexts: all 20 discovery contexts; within-class disagreement 0.119; effect: a0→o3 (1.00); a1→o3 (1.00); a2→o1 (1.00); a3→o0 (1.00)

## First-order fits  (I, action) ⇝ I'
- (I0, a0) ⇝ I0   confidence 0.50 (n=12)
- (I1, a0) ⇝ I1   confidence 0.75 (n=12)
- (I2, a0) ⇝ I2   confidence 1.00 (n=12)
- (I3, a0) ⇝ I3   confidence 1.00 (n=12)
- (I0, a1) ⇝ I0   confidence 1.00 (n=12)
- (I1, a1) ⇝ I1   confidence 1.00 (n=12)
- (I2, a1) ⇝ I0   confidence 0.50 (n=12)
- (I3, a1) ⇝ I1   confidence 0.67 (n=12)
- (I0, a2) ⇝ I1   confidence 1.00 (n=12)
- (I1, a2) ⇝ I0   confidence 1.00 (n=12)
- (I2, a2) ⇝ I3   confidence 1.00 (n=12)
- (I3, a2) ⇝ I2   confidence 1.00 (n=12)
- (I0, a3) ⇝ I0   confidence 1.00 (n=12)
- (I1, a3) ⇝ I0   confidence 1.00 (n=12)
- (I2, a3) ⇝ I0   confidence 1.00 (n=12)
- (I3, a3) ⇝ I0   confidence 1.00 (n=12)

## Observed multi-step fits (48; the rest are derived by chaining)
- (I0, (0, 0)) ⇝ I2   confidence 1.00
- (I0, (0, 1)) ⇝ I0   confidence 1.00
- (I0, (0, 2)) ⇝ I1   confidence 0.50
- (I0, (0, 3)) ⇝ I0   confidence 1.00
- (I0, (1, 0)) ⇝ I0   confidence 1.00
- (I0, (1, 1)) ⇝ I0   confidence 1.00
- (I0, (2, 0)) ⇝ I1   confidence 0.50
- (I0, (2, 1)) ⇝ I1   confidence 1.00
- (I0, (2, 2)) ⇝ I0   confidence 1.00
- (I0, (3, 0)) ⇝ I0   confidence 1.00
- (I0, (3, 1)) ⇝ I0   confidence 1.00
- (I0, (3, 2)) ⇝ I1   confidence 1.00
- … 36 more

## Second-order interfaces (chunk types): 7
- W0: chunks [(0,), (0, 1), (1, 0), (2, 2)]; induced map I0→I0 I1→I1 I2→I2 I3→I3
- W1: chunks [(1,), (1, 1)]; induced map I0→I0 I1→I1 I2→I0 I3→I1
- W2: chunks [(2,), (0, 2), (2, 0)]; induced map I0→I1 I1→I0 I2→I3 I3→I2
- W3: chunks [(3,), (0, 3), (1, 3), (2, 3), (3, 0), (3, 1)] …; induced map I0→I0 I1→I0 I2→I0 I3→I0
- W4: chunks [(0, 0)]; induced map I0→I2 I1→I3 I2→I2 I3→I3
- W5: chunks [(1, 2), (2, 1)]; induced map I0→I1 I1→I0 I2→I1 I3→I0
- W6: chunks [(3, 2)]; induced map I0→I1 I1→I1 I2→I1 I3→I1

## Second-order fits  (W, W') ⇝ W''  (44)
- (W0, W0) ⇝ W4
- (W0, W1) ⇝ W0
- (W0, W2) ⇝ W2
- (W0, W3) ⇝ W3
- (W0, W4) ⇝ W4
- (W0, W5) ⇝ W5
- (W0, W6) ⇝ W6
- (W1, W0) ⇝ W0
- (W1, W1) ⇝ W1
- (W1, W2) ⇝ W5
- (W1, W3) ⇝ W3
- (W1, W4) ⇝ W4
- (W1, W5) ⇝ W5
- (W1, W6) ⇝ W6
- (W2, W0) ⇝ W2
- (W2, W1) ⇝ W5
- (W2, W2) ⇝ W0
- (W2, W3) ⇝ W3
- (W2, W5) ⇝ W1
- (W2, W6) ⇝ W6
- (W3, W0) ⇝ W3
- (W3, W1) ⇝ W3
- (W3, W2) ⇝ W6
- (W3, W3) ⇝ W3
- (W3, W5) ⇝ W6
- (W3, W6) ⇝ W6
- (W4, W0) ⇝ W4
- (W4, W1) ⇝ W1
- (W4, W3) ⇝ W3
- (W4, W4) ⇝ W4
- (W4, W5) ⇝ W5
- (W4, W6) ⇝ W6
- (W5, W0) ⇝ W5
- (W5, W1) ⇝ W5
- (W5, W2) ⇝ W1
- (W5, W3) ⇝ W3
- (W5, W5) ⇝ W1
- (W5, W6) ⇝ W6
- (W6, W0) ⇝ W6
- (W6, W1) ⇝ W6
- … 4 more

Withheld from discovery (predicted only by chaining): [(1, 2), (1, 3), (2, 3), (3, 3)]