# Click experiment: results

Circuit seed 0 (accepted after 2 draws), gate reuse [4, 4, 2, 3, 3, 2], train fraction 0.5, 20000 steps, lr 0.001.

Flags per site: A substitutability (>=1 accepted direction), B sufficiency, C context-generality, D reuse, E composition.  'click' = first checkpoint from which A,B,C,D all hold and keep holding.
wd 0 / 0.01 / 0.1 is the frozen sweep; wd 1.0 is the post-hoc extension of the same axis (everything else identical),
added because the frozen levels produced nearly identical weight norms and trajectories.

## wd=0.0 seed=0

train_acc>=0.99 at step 400; test_acc>=0.99 at step 467; final test_acc 1.000, test_all 1.000, ‖w‖ 35.6

| step | train | test | h1: acc/rand A B C D E | h2: acc/rand A B C D E |
|---|---|---|---|---|
| 0 | 0.477 | 0.472 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 1 | 0.496 | 0.491 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 2 | 0.519 | 0.516 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 3 | 0.546 | 0.545 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 4 | 0.573 | 0.572 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 6 | 0.621 | 0.619 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 8 | 0.642 | 0.648 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 11 | 0.653 | 0.662 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 15 | 0.657 | 0.667 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 22 | 0.661 | 0.668 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 30 | 0.668 | 0.674 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 43 | 0.697 | 0.695 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 60 | 0.737 | 0.732 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 85 | 0.760 | 0.752 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 119 | 0.792 | 0.774 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 168 | 0.842 | 0.816 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 236 | 0.923 | 0.898 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 332 | 0.981 | 0.964 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 467 | 0.999 | 0.991 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 658 | 1.000 | 0.999 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 925 | 1.000 | 0.999 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 1302 | 1.000 | 1.000 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 1832 | 1.000 | 1.000 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 2577 | 1.000 | 1.000 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 3626 | 1.000 | 1.000 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 5103 | 1.000 | 1.000 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 7180 | 1.000 | 1.000 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 10102 | 1.000 | 1.000 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 14214 | 1.000 | 1.000 | 1/0 A=0.96✓ B=0.60(r0.56)  C=0.97✓ D=6✓ E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 20000 | 1.000 | 1.000 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |

- h1: click at None; first pass A/B/C/D/E = 14214/None/14214/14214/None; persistent from None/None/None/None/None
- h2: click at None; first pass A/B/C/D/E = None/None/None/None/None; persistent from None/None/None/None/None

Novelty at the final checkpoint (per accepted direction: max |corr| of the projection with any gate / output / input; ARI of the donor typing with the gate / output / input partitions):

Threshold robustness (raw passing candidates before dedup, final checkpoint): h1: A_min 0.7/0.8/0.9 → 25/25/24 (rand 13/13/13); eps 0.2/0.3/0.4 → 1/0/31; h2: A_min 0.7/0.8/0.9 → 21/21/21 (rand 13/13/13); eps 0.2/0.3/0.4 → 0/0/22

## wd=0.0 seed=1

train_acc>=0.99 at step 400; test_acc>=0.99 at step 467; final test_acc 1.000, test_all 1.000, ‖w‖ 34.4

| step | train | test | h1: acc/rand A B C D E | h2: acc/rand A B C D E |
|---|---|---|---|---|
| 0 | 0.549 | 0.549 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 1 | 0.568 | 0.562 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 2 | 0.581 | 0.576 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 3 | 0.591 | 0.587 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 4 | 0.601 | 0.599 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 6 | 0.626 | 0.625 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 8 | 0.645 | 0.645 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 11 | 0.661 | 0.660 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 15 | 0.667 | 0.667 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 22 | 0.663 | 0.665 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 30 | 0.666 | 0.670 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 43 | 0.704 | 0.705 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 60 | 0.737 | 0.728 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 85 | 0.764 | 0.754 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 119 | 0.800 | 0.783 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 168 | 0.855 | 0.831 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/1 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 236 | 0.938 | 0.918 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 332 | 0.987 | 0.973 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 467 | 1.000 | 0.995 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 658 | 1.000 | 0.998 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 925 | 1.000 | 0.999 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 1302 | 1.000 | 0.999 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 1832 | 1.000 | 0.999 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 2577 | 1.000 | 1.000 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 3626 | 1.000 | 1.000 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 5103 | 1.000 | 1.000 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 7180 | 1.000 | 1.000 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 10102 | 1.000 | 1.000 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 14214 | 1.000 | 1.000 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 20000 | 1.000 | 1.000 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |

- h1: click at None; first pass A/B/C/D/E = None/None/None/None/None; persistent from None/None/None/None/None
- h2: click at None; first pass A/B/C/D/E = None/None/None/None/None; persistent from None/None/None/None/None

Novelty at the final checkpoint (per accepted direction: max |corr| of the projection with any gate / output / input; ARI of the donor typing with the gate / output / input partitions):

Threshold robustness (raw passing candidates before dedup, final checkpoint): h1: A_min 0.7/0.8/0.9 → 25/25/25 (rand 12/12/12); eps 0.2/0.3/0.4 → 2/0/32; h2: A_min 0.7/0.8/0.9 → 18/18/18 (rand 13/13/13); eps 0.2/0.3/0.4 → 2/0/22

## wd=0.0 seed=2

train_acc>=0.99 at step 400; test_acc>=0.99 at step 467; final test_acc 1.000, test_all 1.000, ‖w‖ 35.0

| step | train | test | h1: acc/rand A B C D E | h2: acc/rand A B C D E |
|---|---|---|---|---|
| 0 | 0.463 | 0.472 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 1 | 0.484 | 0.493 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 2 | 0.501 | 0.516 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 3 | 0.520 | 0.534 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 4 | 0.542 | 0.553 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 6 | 0.591 | 0.596 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 8 | 0.622 | 0.630 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 11 | 0.646 | 0.654 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 15 | 0.654 | 0.658 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 22 | 0.651 | 0.654 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 30 | 0.654 | 0.657 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 43 | 0.690 | 0.690 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 60 | 0.732 | 0.725 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 85 | 0.760 | 0.752 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 119 | 0.789 | 0.774 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 168 | 0.843 | 0.824 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 236 | 0.918 | 0.896 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 332 | 0.980 | 0.967 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 467 | 0.998 | 0.992 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 658 | 1.000 | 0.998 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 925 | 1.000 | 0.999 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 1/0 A=0.93✓ B=0.61(r0.56)  C=1.00✓ D=6✓ E=nan  |
| 1302 | 1.000 | 1.000 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 1832 | 1.000 | 1.000 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 2577 | 1.000 | 1.000 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 3626 | 1.000 | 1.000 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 5103 | 1.000 | 1.000 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 7180 | 1.000 | 1.000 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 10102 | 1.000 | 1.000 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 14214 | 1.000 | 1.000 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 20000 | 1.000 | 1.000 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |

- h1: click at None; first pass A/B/C/D/E = None/None/None/None/None; persistent from None/None/None/None/None
- h2: click at None; first pass A/B/C/D/E = 925/None/925/925/None; persistent from None/None/None/None/None

Novelty at the final checkpoint (per accepted direction: max |corr| of the projection with any gate / output / input; ARI of the donor typing with the gate / output / input partitions):

Threshold robustness (raw passing candidates before dedup, final checkpoint): h1: A_min 0.7/0.8/0.9 → 24/24/23 (rand 12/12/12); eps 0.2/0.3/0.4 → 0/0/32; h2: A_min 0.7/0.8/0.9 → 20/20/20 (rand 14/14/14); eps 0.2/0.3/0.4 → 1/0/22

## wd=0.01 seed=0

train_acc>=0.99 at step 400; test_acc>=0.99 at step 467; final test_acc 1.000, test_all 1.000, ‖w‖ 34.4

| step | train | test | h1: acc/rand A B C D E | h2: acc/rand A B C D E |
|---|---|---|---|---|
| 0 | 0.477 | 0.472 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 1 | 0.496 | 0.491 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 2 | 0.519 | 0.516 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 3 | 0.547 | 0.545 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 4 | 0.573 | 0.572 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 6 | 0.621 | 0.619 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 8 | 0.642 | 0.648 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 11 | 0.653 | 0.662 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 15 | 0.657 | 0.667 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 22 | 0.661 | 0.668 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 30 | 0.668 | 0.674 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 43 | 0.696 | 0.695 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 60 | 0.737 | 0.732 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 85 | 0.760 | 0.752 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 119 | 0.792 | 0.774 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 168 | 0.842 | 0.816 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 236 | 0.923 | 0.897 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 332 | 0.981 | 0.964 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 467 | 0.999 | 0.991 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 658 | 1.000 | 0.999 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 925 | 1.000 | 0.999 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 1302 | 1.000 | 1.000 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 1832 | 1.000 | 1.000 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 2577 | 1.000 | 1.000 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 3626 | 1.000 | 1.000 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 5103 | 1.000 | 1.000 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 7180 | 1.000 | 1.000 | 1/0 A=0.97✓ B=0.57(r0.56)  C=0.97✓ D=6✓ E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 10102 | 1.000 | 1.000 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 14214 | 1.000 | 1.000 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 20000 | 1.000 | 1.000 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |

- h1: click at None; first pass A/B/C/D/E = 7180/None/7180/7180/None; persistent from None/None/None/None/None
- h2: click at None; first pass A/B/C/D/E = None/None/None/None/None; persistent from None/None/None/None/None

Novelty at the final checkpoint (per accepted direction: max |corr| of the projection with any gate / output / input; ARI of the donor typing with the gate / output / input partitions):

Threshold robustness (raw passing candidates before dedup, final checkpoint): h1: A_min 0.7/0.8/0.9 → 29/29/29 (rand 14/14/14); eps 0.2/0.3/0.4 → 1/0/32; h2: A_min 0.7/0.8/0.9 → 18/18/17 (rand 14/14/14); eps 0.2/0.3/0.4 → 1/0/22

## wd=0.01 seed=1

train_acc>=0.99 at step 400; test_acc>=0.99 at step 467; final test_acc 1.000, test_all 1.000, ‖w‖ 33.1

| step | train | test | h1: acc/rand A B C D E | h2: acc/rand A B C D E |
|---|---|---|---|---|
| 0 | 0.549 | 0.549 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 1 | 0.568 | 0.562 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 2 | 0.581 | 0.576 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 3 | 0.591 | 0.587 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 4 | 0.601 | 0.599 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 6 | 0.626 | 0.625 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 8 | 0.645 | 0.645 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 11 | 0.661 | 0.660 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 15 | 0.667 | 0.667 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 22 | 0.663 | 0.665 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 30 | 0.666 | 0.670 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 43 | 0.704 | 0.705 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 60 | 0.737 | 0.728 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 85 | 0.764 | 0.754 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 119 | 0.800 | 0.783 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 168 | 0.854 | 0.831 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/1 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 236 | 0.938 | 0.918 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 332 | 0.987 | 0.972 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 467 | 1.000 | 0.994 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 658 | 1.000 | 0.998 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 925 | 1.000 | 0.999 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 1302 | 1.000 | 0.999 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 1832 | 1.000 | 1.000 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 2577 | 1.000 | 1.000 | 1/0 A=0.95✓ B=0.56(r0.56)  C=0.97✓ D=6✓ E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 3626 | 1.000 | 1.000 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 5103 | 1.000 | 1.000 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 7180 | 1.000 | 1.000 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 10102 | 1.000 | 1.000 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 14214 | 1.000 | 1.000 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 20000 | 1.000 | 1.000 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |

- h1: click at None; first pass A/B/C/D/E = 2577/None/2577/2577/None; persistent from None/None/None/None/None
- h2: click at None; first pass A/B/C/D/E = None/None/None/None/None; persistent from None/None/None/None/None

Novelty at the final checkpoint (per accepted direction: max |corr| of the projection with any gate / output / input; ARI of the donor typing with the gate / output / input partitions):

Threshold robustness (raw passing candidates before dedup, final checkpoint): h1: A_min 0.7/0.8/0.9 → 26/26/26 (rand 13/13/13); eps 0.2/0.3/0.4 → 2/0/32; h2: A_min 0.7/0.8/0.9 → 16/16/16 (rand 9/9/9); eps 0.2/0.3/0.4 → 0/0/22

## wd=0.01 seed=2

train_acc>=0.99 at step 400; test_acc>=0.99 at step 467; final test_acc 1.000, test_all 1.000, ‖w‖ 33.7

| step | train | test | h1: acc/rand A B C D E | h2: acc/rand A B C D E |
|---|---|---|---|---|
| 0 | 0.463 | 0.472 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 1 | 0.484 | 0.493 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 2 | 0.501 | 0.516 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 3 | 0.520 | 0.534 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 4 | 0.542 | 0.553 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 6 | 0.591 | 0.596 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 8 | 0.622 | 0.630 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 11 | 0.646 | 0.654 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 15 | 0.654 | 0.658 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 22 | 0.651 | 0.654 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 30 | 0.654 | 0.657 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 43 | 0.690 | 0.690 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 60 | 0.732 | 0.725 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 85 | 0.760 | 0.752 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 119 | 0.789 | 0.774 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 168 | 0.843 | 0.824 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 236 | 0.918 | 0.896 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 332 | 0.980 | 0.966 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 467 | 0.997 | 0.992 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 658 | 1.000 | 0.998 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 925 | 1.000 | 0.999 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 1302 | 1.000 | 1.000 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 1832 | 1.000 | 1.000 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 2577 | 1.000 | 1.000 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 3626 | 1.000 | 1.000 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 5103 | 1.000 | 1.000 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 7180 | 1.000 | 1.000 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 10102 | 1.000 | 1.000 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 14214 | 1.000 | 1.000 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 20000 | 1.000 | 1.000 | 1/0 A=0.98✓ B=0.58(r0.56)  C=0.92✓ D=6✓ E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |

- h1: click at None; first pass A/B/C/D/E = 20000/None/20000/20000/None; persistent from 20000/None/20000/20000/None
- h2: click at None; first pass A/B/C/D/E = None/None/None/None/None; persistent from None/None/None/None/None

Novelty at the final checkpoint (per accepted direction: max |corr| of the projection with any gate / output / input; ARI of the donor typing with the gate / output / input partitions):
- h1[0] (pca0, 7 types, A=0.98, moves 6 outputs): corr gate 0.85(g1) out 0.42(y4) in 0.57; ARI gate 0.34(g1) out 0.03 in 0.27

Threshold robustness (raw passing candidates before dedup, final checkpoint): h1: A_min 0.7/0.8/0.9 → 28/28/28 (rand 11/11/11); eps 0.2/0.3/0.4 → 1/1/32; h2: A_min 0.7/0.8/0.9 → 17/17/17 (rand 12/12/12); eps 0.2/0.3/0.4 → 0/0/22

## wd=0.1 seed=0

train_acc>=0.99 at step 400; test_acc>=0.99 at step 467; final test_acc 1.000, test_all 1.000, ‖w‖ 29.2

| step | train | test | h1: acc/rand A B C D E | h2: acc/rand A B C D E |
|---|---|---|---|---|
| 0 | 0.477 | 0.472 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 1 | 0.496 | 0.491 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 2 | 0.519 | 0.516 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 3 | 0.546 | 0.545 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 4 | 0.573 | 0.572 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 6 | 0.621 | 0.619 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 8 | 0.643 | 0.648 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 11 | 0.653 | 0.662 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 15 | 0.657 | 0.667 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 22 | 0.661 | 0.667 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 30 | 0.668 | 0.674 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 43 | 0.696 | 0.695 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 60 | 0.737 | 0.732 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 85 | 0.760 | 0.752 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 119 | 0.791 | 0.774 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 168 | 0.840 | 0.815 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 236 | 0.921 | 0.895 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 332 | 0.980 | 0.963 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 467 | 0.998 | 0.991 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 658 | 1.000 | 0.998 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 925 | 1.000 | 0.999 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 1302 | 1.000 | 1.000 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 1832 | 1.000 | 1.000 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 2577 | 1.000 | 1.000 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 3626 | 1.000 | 1.000 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 5103 | 1.000 | 1.000 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 7180 | 1.000 | 1.000 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 10102 | 1.000 | 1.000 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 14214 | 1.000 | 1.000 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 20000 | 1.000 | 1.000 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |

- h1: click at None; first pass A/B/C/D/E = None/None/None/None/None; persistent from None/None/None/None/None
- h2: click at None; first pass A/B/C/D/E = None/None/None/None/None; persistent from None/None/None/None/None

Novelty at the final checkpoint (per accepted direction: max |corr| of the projection with any gate / output / input; ARI of the donor typing with the gate / output / input partitions):

Threshold robustness (raw passing candidates before dedup, final checkpoint): h1: A_min 0.7/0.8/0.9 → 26/26/26 (rand 12/12/12); eps 0.2/0.3/0.4 → 2/0/32; h2: A_min 0.7/0.8/0.9 → 18/18/17 (rand 8/8/8); eps 0.2/0.3/0.4 → 1/0/22

## wd=0.1 seed=1

train_acc>=0.99 at step 400; test_acc>=0.99 at step 467; final test_acc 1.000, test_all 1.000, ‖w‖ 28.1

| step | train | test | h1: acc/rand A B C D E | h2: acc/rand A B C D E |
|---|---|---|---|---|
| 0 | 0.549 | 0.549 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 1 | 0.568 | 0.562 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 2 | 0.581 | 0.576 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 3 | 0.591 | 0.587 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 4 | 0.601 | 0.599 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 6 | 0.626 | 0.624 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 8 | 0.645 | 0.645 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 11 | 0.661 | 0.660 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 15 | 0.667 | 0.667 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 22 | 0.663 | 0.665 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 30 | 0.666 | 0.670 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 43 | 0.704 | 0.705 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 60 | 0.737 | 0.728 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 85 | 0.763 | 0.754 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 119 | 0.800 | 0.782 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 168 | 0.852 | 0.829 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/1 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 236 | 0.935 | 0.915 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 332 | 0.986 | 0.972 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 467 | 1.000 | 0.994 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 658 | 1.000 | 0.998 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 925 | 1.000 | 0.999 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 1302 | 1.000 | 0.999 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 1832 | 1.000 | 1.000 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 2577 | 1.000 | 1.000 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 3626 | 1.000 | 1.000 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 1/0 A=0.95✓ B=0.57(r0.56)  C=0.90✓ D=6✓ E=nan  |
| 5103 | 1.000 | 1.000 | 0/1 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 7180 | 1.000 | 1.000 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 10102 | 1.000 | 1.000 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 14214 | 1.000 | 1.000 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 20000 | 1.000 | 1.000 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |

- h1: click at None; first pass A/B/C/D/E = None/None/None/None/None; persistent from None/None/None/None/None
- h2: click at None; first pass A/B/C/D/E = 3626/None/3626/3626/None; persistent from None/None/None/None/None

Novelty at the final checkpoint (per accepted direction: max |corr| of the projection with any gate / output / input; ARI of the donor typing with the gate / output / input partitions):

Threshold robustness (raw passing candidates before dedup, final checkpoint): h1: A_min 0.7/0.8/0.9 → 27/27/27 (rand 14/14/14); eps 0.2/0.3/0.4 → 4/0/32; h2: A_min 0.7/0.8/0.9 → 22/22/21 (rand 8/8/7); eps 0.2/0.3/0.4 → 2/0/21

## wd=0.1 seed=2

train_acc>=0.99 at step 400; test_acc>=0.99 at step 467; final test_acc 1.000, test_all 1.000, ‖w‖ 28.6

| step | train | test | h1: acc/rand A B C D E | h2: acc/rand A B C D E |
|---|---|---|---|---|
| 0 | 0.463 | 0.472 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 1 | 0.484 | 0.493 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 2 | 0.501 | 0.516 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 3 | 0.520 | 0.534 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 4 | 0.542 | 0.553 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 6 | 0.591 | 0.596 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 8 | 0.622 | 0.630 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 11 | 0.646 | 0.654 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 15 | 0.654 | 0.658 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 22 | 0.651 | 0.654 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 30 | 0.654 | 0.657 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 43 | 0.690 | 0.689 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 60 | 0.732 | 0.724 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 85 | 0.760 | 0.751 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 119 | 0.789 | 0.773 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 168 | 0.842 | 0.822 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 236 | 0.916 | 0.893 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 332 | 0.978 | 0.966 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 467 | 0.997 | 0.992 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 658 | 1.000 | 0.998 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 925 | 1.000 | 0.999 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 1302 | 1.000 | 1.000 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 1832 | 1.000 | 1.000 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 2577 | 1.000 | 1.000 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 3626 | 1.000 | 1.000 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 5103 | 1.000 | 1.000 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 7180 | 1.000 | 1.000 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 10102 | 1.000 | 1.000 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 14214 | 1.000 | 1.000 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 20000 | 1.000 | 1.000 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |

- h1: click at None; first pass A/B/C/D/E = None/None/None/None/None; persistent from None/None/None/None/None
- h2: click at None; first pass A/B/C/D/E = None/None/None/None/None; persistent from None/None/None/None/None

Novelty at the final checkpoint (per accepted direction: max |corr| of the projection with any gate / output / input; ARI of the donor typing with the gate / output / input partitions):

Threshold robustness (raw passing candidates before dedup, final checkpoint): h1: A_min 0.7/0.8/0.9 → 27/27/26 (rand 10/10/10); eps 0.2/0.3/0.4 → 0/0/31; h2: A_min 0.7/0.8/0.9 → 20/19/18 (rand 13/13/13); eps 0.2/0.3/0.4 → 3/0/21

## wd=1.0 seed=0

train_acc>=0.99 at step 467; test_acc>=0.99 at step 600; final test_acc 1.000, test_all 1.000, ‖w‖ 27.5

| step | train | test | h1: acc/rand A B C D E | h2: acc/rand A B C D E |
|---|---|---|---|---|
| 0 | 0.477 | 0.472 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 1 | 0.497 | 0.491 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 2 | 0.519 | 0.516 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 3 | 0.546 | 0.545 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 4 | 0.574 | 0.572 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 6 | 0.621 | 0.620 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 8 | 0.643 | 0.648 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 11 | 0.652 | 0.662 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 15 | 0.657 | 0.668 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 22 | 0.661 | 0.666 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 30 | 0.667 | 0.672 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 43 | 0.691 | 0.693 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 60 | 0.734 | 0.730 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 85 | 0.757 | 0.750 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 119 | 0.783 | 0.770 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 168 | 0.828 | 0.804 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 236 | 0.900 | 0.873 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 332 | 0.970 | 0.949 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 467 | 0.993 | 0.985 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 658 | 1.000 | 0.997 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 925 | 1.000 | 0.999 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 1302 | 1.000 | 1.000 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 1832 | 1.000 | 1.000 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/1 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 2577 | 1.000 | 1.000 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 3626 | 1.000 | 1.000 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 5103 | 1.000 | 1.000 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 7180 | 1.000 | 1.000 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 10102 | 1.000 | 1.000 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 14214 | 1.000 | 1.000 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 20000 | 1.000 | 1.000 | 1/0 A=0.96✓ B=0.59(r0.56)  C=1.00✓ D=6✓ E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |

- h1: click at None; first pass A/B/C/D/E = 20000/None/20000/20000/None; persistent from 20000/None/20000/20000/None
- h2: click at None; first pass A/B/C/D/E = None/None/None/None/None; persistent from None/None/None/None/None

Novelty at the final checkpoint (per accepted direction: max |corr| of the projection with any gate / output / input; ARI of the donor typing with the gate / output / input partitions):
- h1[0] (pca8, 4 types, A=0.96, moves 6 outputs): corr gate 0.95(g3) out 0.57(y5) in 0.11; ARI gate 0.60(g3) out 0.26 in 0.07

Threshold robustness (raw passing candidates before dedup, final checkpoint): h1: A_min 0.7/0.8/0.9 → 25/25/25 (rand 12/12/11); eps 0.2/0.3/0.4 → 5/1/27; h2: A_min 0.7/0.8/0.9 → 18/18/18 (rand 14/14/14); eps 0.2/0.3/0.4 → 2/0/20

## wd=1.0 seed=1

train_acc>=0.99 at step 467; test_acc>=0.99 at step 600; final test_acc 1.000, test_all 1.000, ‖w‖ 27.8

| step | train | test | h1: acc/rand A B C D E | h2: acc/rand A B C D E |
|---|---|---|---|---|
| 0 | 0.549 | 0.549 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 1 | 0.568 | 0.562 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 2 | 0.581 | 0.576 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 3 | 0.591 | 0.587 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 4 | 0.600 | 0.599 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 6 | 0.626 | 0.623 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 8 | 0.645 | 0.645 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 11 | 0.661 | 0.661 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 15 | 0.668 | 0.667 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 22 | 0.663 | 0.664 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 30 | 0.665 | 0.668 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 43 | 0.702 | 0.704 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 60 | 0.733 | 0.726 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 85 | 0.758 | 0.749 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 119 | 0.792 | 0.775 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 168 | 0.838 | 0.815 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 236 | 0.916 | 0.895 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 332 | 0.973 | 0.958 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 467 | 0.997 | 0.990 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 658 | 1.000 | 0.997 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 925 | 1.000 | 0.999 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 1302 | 1.000 | 1.000 | 1/0 A=0.97✓ B=0.59(r0.56)  C=0.81✓ D=6✓ E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 1832 | 1.000 | 1.000 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 1/0 A=0.96✓ B=0.61(r0.56)  C=1.00✓ D=5✓ E=nan  |
| 2577 | 1.000 | 1.000 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 3626 | 1.000 | 1.000 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 5103 | 1.000 | 1.000 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 7180 | 1.000 | 1.000 | 1/0 A=0.97✓ B=0.56(r0.56)  C=0.95✓ D=6✓ E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 10102 | 1.000 | 1.000 | 1/0 A=0.98✓ B=0.59(r0.56)  C=1.00✓ D=6✓ E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 14214 | 1.000 | 1.000 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 20000 | 1.000 | 1.000 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |

- h1: click at None; first pass A/B/C/D/E = 1302/None/1302/1302/None; persistent from None/None/None/None/None
- h2: click at None; first pass A/B/C/D/E = 1832/None/1832/1832/None; persistent from None/None/None/None/None

Novelty at the final checkpoint (per accepted direction: max |corr| of the projection with any gate / output / input; ARI of the donor typing with the gate / output / input partitions):

Threshold robustness (raw passing candidates before dedup, final checkpoint): h1: A_min 0.7/0.8/0.9 → 26/26/26 (rand 13/13/13); eps 0.2/0.3/0.4 → 6/0/30; h2: A_min 0.7/0.8/0.9 → 19/19/19 (rand 14/14/14); eps 0.2/0.3/0.4 → 2/0/21

## wd=1.0 seed=2

train_acc>=0.99 at step 467; test_acc>=0.99 at step 600; final test_acc 1.000, test_all 1.000, ‖w‖ 28.8

| step | train | test | h1: acc/rand A B C D E | h2: acc/rand A B C D E |
|---|---|---|---|---|
| 0 | 0.463 | 0.472 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 1 | 0.484 | 0.493 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 2 | 0.501 | 0.516 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 3 | 0.520 | 0.533 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 4 | 0.542 | 0.553 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 6 | 0.591 | 0.595 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 8 | 0.623 | 0.630 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 11 | 0.647 | 0.654 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 15 | 0.654 | 0.659 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 22 | 0.651 | 0.653 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 30 | 0.654 | 0.656 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 43 | 0.684 | 0.684 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 60 | 0.727 | 0.721 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 85 | 0.756 | 0.746 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 119 | 0.782 | 0.769 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 168 | 0.828 | 0.809 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 236 | 0.896 | 0.873 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 332 | 0.965 | 0.947 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 467 | 0.994 | 0.986 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 658 | 1.000 | 0.997 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 925 | 1.000 | 1.000 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 1/0 A=0.96✓ B=0.63(r0.56)  C=0.92✓ D=6✓ E=nan  |
| 1302 | 1.000 | 1.000 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 1832 | 1.000 | 1.000 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 1/0 A=0.97✓ B=0.58(r0.56)  C=0.90✓ D=5✓ E=nan  |
| 2577 | 1.000 | 1.000 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 3626 | 1.000 | 1.000 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 5103 | 1.000 | 1.000 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 7180 | 1.000 | 1.000 | 0/1 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 1/0 A=0.97✓ B=0.62(r0.56)  C=1.00✓ D=6✓ E=nan  |
| 10102 | 1.000 | 1.000 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 14214 | 1.000 | 1.000 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |
| 20000 | 1.000 | 1.000 | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  | 0/0 A=nan  B=nan(rnan)  C=nan  D=0  E=nan  |

- h1: click at None; first pass A/B/C/D/E = None/None/None/None/None; persistent from None/None/None/None/None
- h2: click at None; first pass A/B/C/D/E = 925/None/925/925/None; persistent from None/None/None/None/None

Novelty at the final checkpoint (per accepted direction: max |corr| of the projection with any gate / output / input; ARI of the donor typing with the gate / output / input partitions):

Threshold robustness (raw passing candidates before dedup, final checkpoint): h1: A_min 0.7/0.8/0.9 → 27/27/26 (rand 12/12/11); eps 0.2/0.3/0.4 → 2/0/29; h2: A_min 0.7/0.8/0.9 → 19/19/18 (rand 11/11/11); eps 0.2/0.3/0.4 → 3/0/22

## Cross-seed correspondence (final checkpoint)

For each accepted direction of one seed, the max |corr| over all inputs of its projection with any accepted direction of another seed at the same site; mean over directions.

- wd=0.0 h1: n/a
- wd=0.0 h2: n/a
- wd=0.01 h1: n/a
- wd=0.01 h2: n/a
- wd=0.1 h1: n/a
- wd=0.1 h2: n/a
- wd=1.0 h1: n/a
- wd=1.0 h2: n/a
