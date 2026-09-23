# LIR-PPTD Reproducibility Artifact

This is the reviewer-facing reproducibility artifact for the LIR-PPTD paper. It contains the implementation, frozen experiment configurations, released paper-result records, statistical analyses, system-evaluation material, public dataset hash manifests, and validation tests needed to audit the paper's reported results.

The active algorithm uses geometry-aware consistency scaling,
`tau_r = lambda_tau * diam^2(X_r)`: numerical reports normalized to `[0,1]^D` use `diam^2=D`, while one-hot categorical reports use `diam^2=2`.

## Quick validation

Recommended Python: 3.11-3.13.

```bash
python -m pip install -e '.[test]'
python verify_public_artifact.py
python -m pytest -q
```

The target is **0 failed tests**. One external-data equivalence test is intentionally skipped when the non-redistributed Product/Duck raw CSVs are absent; it runs automatically when matching files are supplied under the expected data paths.

## Reviewer-oriented layout

- `src/` — LIR-PPTD implementation, fixed-point logic, MPC helpers, baselines, and experiment support code.
- `tests/` — public conformance, regression, and contract tests.
- `configs/` — frozen experiment and protocol configurations.
- `results/` — released experiment outputs used by the manuscript and supplementary material.
- `analysis/geo_tau_final/` — paper-facing derived analyses and claim/result bindings.
- `figures/` — released paper-facing figures generated from the retained evidence.
- `data/` — public dataset manifests and hash audits; restricted third-party raw data are not redistributed.
- `final_closure_step02_protocols/` — frozen protocols for the final same-information, NetEase, and calibration-budget studies.
- `round6_2/` — final multi-selector robustness design, runner, analyzer, and tests.
- `p1d_mpspdz/` and `P1_D_GATE_D_4VM_VPC_V1_8_ALIYUN_FORMAL_RESUMABLE/` — MP-SPDZ and network/system-evaluation material.
- `PUBLIC_ARTIFACT_MANIFEST.json` — SHA-256 manifest for every released file.
- `ENVIRONMENT.json` and `requirements-test-lock.txt` — environment and dependency records.

## Paper-evidence navigation

Start with:

- `analysis/geo_tau_final/paper_result_consistency.csv` — maps paper claims/figures/tables to released evidence files.
- `FINAL_CLOSURE_STEP03_RESULT_LEDGER.md` — human-readable summary of the final P0 studies.
- `FINAL_CLOSURE_STEP03_RESULT_LEDGER.json` — machine-readable form of the same final-study ledger.
- `analysis/geo_tau_final/final_pass_matrix.csv` — final geometry/selector/fixed-point evidence gates.
- `PUBLIC_ARTIFACT_MANIFEST.json` — exact released-file hashes.


## Final P0/P1 evidence integration

The submission-candidate artifact additionally includes the final equal-budget mechanism comparison, categorical hard-label fast-path validation, and calibration-history/deployment-boundary postprocessing used by the final manuscript and supplement. Start with `FINAL_P0_P1_EVIDENCE_MAP.md`; `validate_final_p0_p1_evidence.py` checks the manuscript-facing headline values and reruns the released P0/P1 postprocessing that does not require excluded third-party raw data or active private selector seeds.

## External data

Raw Product/Duck CSVs are not redistributed. Their frozen sizes and SHA-256 values are recorded in `data/phase_r1_dataset_manifest.json`.

Raw NetEaseCrowd partitions are also excluded. `data/neteasecrowd/dataset_audit.json` records the canonical upstream repository, the 15 expected partition hashes, and processed dataset statistics. See `DATA_POLICY.md` for redistribution and ethics boundaries.

## Submission-hygiene pruning

This reviewer-facing package intentionally omits files that do not contribute to reproducing or auditing the paper, including:

- generated caches and bytecode;
- obsolete build-provenance snapshots and packaging-repair notes;
- internal reviewer/audit memoranda and superseded status reports;
- superseded GEO-TAU preregistration-package copies already replaced by the final geometry evidence;
- historical cloud-runner version notes/tests superseded by the final resumable runner and current contract tests;
- repository/API probe scripts, private-seed freezing helpers, and other development-only orchestration;
- one-off execution wrappers whose required private selector seeds are intentionally not released; frozen protocols, result records, public commitments/hashes, and analysis code are retained instead;
- exact duplicate result/analysis copies when one canonical copy is retained.

Internal experiment identifiers such as `round6_2` and `fc2_p0*` are intentionally retained because released result ledgers and evidence bindings refer to those frozen identifiers. Renaming these internal IDs would reduce reproducibility.

## Data/privacy exclusions

The package excludes active private selector seeds, non-example cloud credentials, and raw third-party datasets whose redistribution is unnecessary or restricted. Released masks/hashes, formal outcomes, public configuration contracts, and all paper-facing evidence records are retained.
