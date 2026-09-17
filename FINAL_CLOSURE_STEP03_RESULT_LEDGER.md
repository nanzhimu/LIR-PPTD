# Final Closure Step 3 Result Ledger

Status: **COMPLETE**. The three P0 studies were executed under the frozen Step-2 protocols with no post-outcome retuning.

## P0-A - Same-information attribution
- 738 frozen stream conditions; 2,214 successful method runs; 0 failures.
- Frozen decision: **NO_SAME_INFO_SUPERIORITY**.
- Stationary A1 vs CAL-EMA-TL: LIR mean wins/losses = 18/2; Holm significant wins/losses = 17/1.
- Stationary A1 vs COWA: LIR mean wins/losses = 6/14; Holm significant wins/losses = 5/12.
- Complete A1-A3 cells are in `results/fc2_p0a/cell_summary.csv`.

## P0-B - NetEaseCrowd external baselines
- Six capabilities, five causal chronological chunks, no future leakage.
- Equal-capability macro Accuracy/MacroF1: LIR-PPTD 0.9115/0.8930; DS 0.8733/0.8543; LA-onepass 0.9103/0.8925; MV 0.8946/0.8718.
- LIR exceeds DS and LA-onepass by point estimate in 4/6 capabilities on each endpoint; capability-level bootstrap results are released.

## P0-C - Calibration budget
- Dog and Weather, persistent attack at rho_m=0.7; pi in 1%, 2%, 5%, 10%; 1,600 successful LIR runs; 0 failures.
- All budgets use the common complement of the 10% selector mask for scoring.
- Relative to 5%, 1% and 2% increase loss and 10% decreases loss on both datasets; all six planned contrasts have Holm p=0.01171875.
- Frozen interpretation: 5% is a justified evaluated operating point, not an optimum; more authenticated evidence can help.

The public artifact excludes private selector seeds and raw third-party source datasets; released masks/hashes, formal outcomes, runners, and provenance are retained.
