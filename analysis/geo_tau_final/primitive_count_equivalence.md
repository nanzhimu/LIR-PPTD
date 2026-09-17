# Geometry-scale primitive-count equivalence

The final geometry integration changes only the public resolved consistency constant `tau_r`. It does not alter the secure loop structure, worker count, dimension, or iteration count.

A D=4 categorical Dog task was executed through the deterministic f=24 fixed-point backend in both explicit modes. Primitive traces are identical: **True**.

| Primitive | legacy D-scaling | geometry scaling |
|---|---:|---:|
| DivPositive | 100 | 100 |
| SecMulPositive | 100 | 100 |
| SecSqr | 400 | 400 |
| local_add | 100 | 100 |
| local_sub | 400 | 400 |

The calibration path is structurally identical as well: for each participating worker it evaluates the same D coordinate differences/squares, one public-positive division for `q_cal`, and the same bounded reputation transition. Only the public value supplied as `tau_r` changes. Consequently multiplication, addition/subtraction, division, comparison, and communication-round structure are unchanged by geometry scaling.

The real MP-SPDZ program has likewise been changed from a hard-coded `TAU=0.2` to a single injected resolved `tau` consumed by both ordinary and calibration branches; no secure branch is introduced by the scale resolver.
