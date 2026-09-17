# Dog near/exact-tie fixed-point and MPC release audit

## Scope

The 11 Dog rows previously identified by the independent continuous float/plaintext replay were traced to the single released task responsible for the one-task endpoint difference. The final geometry mode, `f=24`, and deterministic lowest-index tie rule were held fixed.

## Fixed-point release

- audited cases: **11**
- fixed-point ties after quantization: **11/11**
- fixed-point release matches the release used by the 80-digit formal benchmark pipeline: **11/11**
- fixed-point release differs from the unquantized/high-precision internal ordering in **4/11** cases; these are recorded explicitly as quantization/precision-boundary release differences rather than performance failures.

The exact printed score margins are zero to the retained 80 decimal digits in most cases; two cases show a margin of `1E-80`. These cases are therefore at the precision boundary and must not be used to claim label-permutation robustness. The secure release contract is the deterministic `f=24` quantized release with lowest-index tie breaking.

## MPC consistency

The current categorical MP-SPDZ benchmark harness is not a vector-input real-MPC harness, so the instruction's conditional real MP-SPDZ near-tie replay is not applicable without redesigning that benchmark. Instead, all **11/11** cases were replayed through the repository's Shamir MPC semantic backend, which executes the same f=24 integer primitive path under shares; all releases matched the deterministic fixed-point backend. The real MP-SPDZ numerical harness was separately repaired so that `tau` is no longer hard-coded and is injected from the final resolved geometry configuration for both ordinary and calibration branches.

This closes the active release semantics: paper classification endpoints use the deterministic fixed-point release convention; continuous high-precision scores remain diagnostic and can differ in a precision-boundary tie.
