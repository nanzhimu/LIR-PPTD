# Final P0/P1 Evidence Map

This addendum maps the final manuscript-facing closure results to the released artifact. It contains only paper-facing evidence, deterministic reanalysis code, and derived outputs; internal review/audit memoranda and private selector seeds are excluded.

## P0 — equal-budget mechanism comparison

- `results/final_scientific_closure_p0/mechanism/selected_hyperparameters.json` — selected global parameters (`eta=0.4` for LIR/EMA; `w=3` for Window-COWA).
- `results/final_scientific_closure_p0/mechanism/equal_tuning_cell_comparisons.csv` — cell-level tuned-LIR contrasts and corrected significance.
- `results/final_scientific_closure_p0/mechanism/equal_tuning_summary.json` — manuscript W/T/L and Holm W/L summaries.
- `results/final_scientific_closure_p0/mechanism/*runs*.jsonl` — released run-level tuning/evaluation records.
- `analysis/final_scientific_closure_p0/reanalyze_equal_budget.py` — regenerates the two equal-budget summary families from released cell records.

The public artifact intentionally does not release active private selector seeds. The run-level results retain calibration-mask hashes and frozen pairing identifiers; the supplied reanalysis reproduces the manuscript statistics from those released records.

## P0 — categorical hard-label fast path

- `results/final_scientific_closure_p0/fastpath/fastpath_summary.json` — 570 categorical streams, 1,662,310 ordinary tasks, zero hard-label mismatches, and 384 exact ties.
- `results/final_scientific_closure_p0/fastpath/arithmetic_cost.csv` — full-path versus initialization-only nonlinear-call counts.
- `results/final_scientific_closure_p0/fastpath/kernel_timing_summary.csv` — matched-kernel timing reductions.
- `results/final_scientific_closure_p0/fastpath/stream_results.jsonl` — stream-level finite-precision validation.
- `analysis/final_scientific_closure_p0/fastpath_kernel_benchmark.py` — artifact-relative deterministic kernel benchmark; rerun outputs are written separately and do not overwrite released evidence.

## P1 — calibration-history/deployment boundary

- `results/final_scientific_closure_p1/deployment_boundary/deployment_boundary_summary.json` — worker-capability calibration coverage and task-level history completeness.
- `results/final_scientific_closure_p1/deployment_boundary/worker_calibration_coverage_by_capability.csv` — capability-level exposure distribution.
- `results/final_scientific_closure_p1/deployment_boundary/task_cold_start_by_capability.csv` — capability-level history-complete/cold-task distribution.
- `analysis/final_scientific_closure_p1/deployment_boundary_postprocess.py` — deterministic stdlib postprocessing from the already released `p1c_netease_longitudinal_geometry_v1` records.

## One-command sync validation

```bash
python verify_public_artifact.py
python validate_final_p0_p1_evidence.py
```
