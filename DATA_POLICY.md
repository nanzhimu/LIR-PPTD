# Data policy and provenance

## NetEaseCrowd source and license

The longitudinal study uses the public **NetEaseCrowd** dataset released by Fuxi AI Lab, NetEase. The artifact's dataset audit records the canonical upstream repository as:

`https://github.com/fuxiAIlab/NetEaseCrowd-Dataset`

The study runner obtains the 15 CSV partitions from the upstream repository path `data/NetEaseCrowd_part_{1..15}.csv`. The upstream project currently states that NetEaseCrowd is licensed under **CC-BY-SA-4.0** and provides an alternative Hugging Face distribution. This artifact does not grant additional rights beyond the upstream license and terms.

## What this artifact redistributes

Raw NetEaseCrowd source CSVs are **not** redistributed in this archive. Instead, the archive provides:

- `data/neteasecrowd/dataset_audit.json`, containing the upstream repository identifier, the expected dataset cardinalities, and SHA-256 digests for all 15 raw CSV partitions;
- the preprocessing/execution path in `src/lir_pptd/experiments/p1c_netease_longitudinal/runner.py`;
- frozen per-task prediction files and derived result summaries used by the manuscript;
- dependence-aware bootstrap and capability-126 sensitivity scripts/results derived from those frozen predictions.

A user can independently obtain the upstream dataset, verify the 15 raw-part SHA-256 values against `dataset_audit.json`, and then follow the included runner/preprocessing path. The artifact intentionally makes no additional redistribution claim for the source data.

## Privacy and ethics boundary

The upstream NetEaseCrowd repository states that sensitive identifiers/content are anonymized. The LIR-PPTD longitudinal analysis consumes the released annotation-record fields (task/taskset/worker identifiers, answer, completion timestamp, truth label, and capability) and does not attempt to re-identify workers. No new human-subject interaction or data collection was performed for this study. This artifact does **not** assert an institutional-review-board or ethics determination for the upstream data collection; such provenance remains with the source provider and applicable local requirements.

## Other exclusions

Active private selector-seed material, non-example cloud host credentials, and any source data whose redistribution is not required for verification are excluded. Retired selector commitments/reveals are included only after the associated experiments are complete. These exclusions do not change the hashes of the frozen result files reported by the manuscript.

## Final-closure external-data rule

The public artifact intentionally does not bundle the Product/Duck raw CSV files. Their frozen SHA-256 contracts are released in `data/phase_r1_dataset_manifest.json`. The G7 real-data equivalence test therefore skips with an explicit message when these external files are absent and runs automatically when matching files are populated. `data/neteasecrowd/dataset_audit.json` is included as the promised metadata/hash audit; raw NetEaseCrowd partitions remain excluded.

## Final P0 execution inputs

The Step-3 P0 studies were executed against externally supplied raw inputs whose SHA-256 contracts match the released dataset audits. Raw Product, Duck, Dog, Weather, and NetEaseCrowd source files are not redistributed in this public archive. Private selector seeds also remain excluded; the formal run ledgers retain calibration-mask hashes/commitments and all released outcomes needed to audit the claim-evidence integration.
