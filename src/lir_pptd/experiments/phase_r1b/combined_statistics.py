from __future__ import annotations

import argparse
import hashlib
import json
from fractions import Fraction
from pathlib import Path
from typing import Any

from lir_pptd.experiments.phase_r1.statistics import (
    ALPHA,
    BOOTSTRAP_RESAMPLES,
    PhaseR1StatisticsError,
    _fraction,
    _fraction_json,
    _median_fraction,
    _planned_pair_count,
    _read_jsonl,
    bootstrap_mean_ci,
    exact_two_sided_sign_flip,
    holm_adjust,
    paired_rank_biserial,
)
from .runner import (
    R1B_PLANNED_METHOD_RUNS,
    UPSTREAM_E1_RAW_SHA256,
    UPSTREAM_PHASE_R1_START_BINDING_SHA256,
)


class CombinedStatisticsError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_complete_records(root: Path) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any], str]:
    e1_summary_path = root / "results/summary/phase_r1_formal_summary.json"
    r1b_summary_path = root / "results/summary/phase_r1b_formal_summary.json"
    if not e1_summary_path.is_file() or not r1b_summary_path.is_file():
        raise CombinedStatisticsError("COMBINED_FORMAL_SUMMARY_MISSING")

    e1_summary = json.loads(e1_summary_path.read_text(encoding="utf-8"))
    r1b_summary = json.loads(r1b_summary_path.read_text(encoding="utf-8"))
    if e1_summary.get("start_binding_sha256") != UPSTREAM_PHASE_R1_START_BINDING_SHA256:
        raise CombinedStatisticsError("E1_BINDING_MISMATCH")
    if e1_summary.get("raw_sha256") != UPSTREAM_E1_RAW_SHA256:
        raise CombinedStatisticsError("E1_RAW_SHA_IN_SUMMARY_MISMATCH")
    if e1_summary.get("committed_method_runs") != 1686 or e1_summary.get("failure_method_runs") != 0:
        raise CombinedStatisticsError("E1_COMPLETION_CONTRACT_MISMATCH")
    if r1b_summary.get("status") != "complete":
        raise CombinedStatisticsError("R1B_NOT_COMPLETE")
    if int(r1b_summary.get("committed_method_runs", -1)) != R1B_PLANNED_METHOD_RUNS:
        raise CombinedStatisticsError("R1B_COMPLETION_COUNT_MISMATCH")

    e1_raw = root / str(e1_summary["raw_path"])
    r1b_raw = root / str(r1b_summary["raw_path"])
    if _sha256(e1_raw) != str(e1_summary["raw_sha256"]):
        raise CombinedStatisticsError("E1_RAW_HASH_MISMATCH")
    if _sha256(r1b_raw) != str(r1b_summary["raw_sha256"]):
        raise CombinedStatisticsError("R1B_RAW_HASH_MISMATCH")

    e1_records = _read_jsonl(e1_raw)
    r1b_records = _read_jsonl(r1b_raw)
    if len(e1_records) != 1686 or len(r1b_records) != R1B_PLANNED_METHOD_RUNS:
        raise CombinedStatisticsError("COMBINED_RAW_COUNT_MISMATCH")
    if any(r.get("stage") != "E1" for r in e1_records):
        raise CombinedStatisticsError("E1_STAGE_CONTAMINATION")
    if any(r.get("stage") not in {"E2", "E3", "E4"} for r in r1b_records):
        raise CombinedStatisticsError("R1B_STAGE_CONTAMINATION")

    run_ids = [str(r.get("run_id")) for r in e1_records + r1b_records]
    if len(set(run_ids)) != 3186:
        raise CombinedStatisticsError("COMBINED_RUN_ID_UNIQUENESS_FAILURE")

    r1b_binding = str(r1b_summary["r1b_start_binding_sha256"])
    if any(r.get("r1b_start_binding_sha256") != r1b_binding for r in r1b_records):
        raise CombinedStatisticsError("R1B_RECORD_BINDING_MISMATCH")

    seed_root = hashlib.sha256(
        (
            "phase-r1-combined-analysis-v1|"
            + UPSTREAM_PHASE_R1_START_BINDING_SHA256
            + "|"
            + UPSTREAM_E1_RAW_SHA256
            + "|"
            + r1b_binding
            + "|"
            + str(r1b_summary["raw_sha256"])
        ).encode("utf-8")
    ).hexdigest()
    return e1_records + r1b_records, e1_summary, r1b_summary, seed_root


def analyze(root: Path) -> dict[str, Any]:
    records, e1_summary, r1b_summary, seed_root = _load_complete_records(root)
    by_key = {(r["pairing_id"], r["method"]): r for r in records}
    pairings = sorted({r["pairing_id"] for r in records})
    contrasts: list[dict[str, Any]] = []

    comparison_specs = [
        ("E1", "lir_pptd_full", "crh", "external_primary"),
        ("E1", "lir_pptd_full", "mean_vote", "external_secondary"),
        ("E2", "lir_pptd_full", "crh", "external_secondary_on_off"),
        ("E2", "lir_pptd_full", "mean_vote", "external_secondary_on_off"),
        ("E3", "lir_pptd_full", "lir_pptd_nr", "ablation_secondary"),
        ("E3", "lir_pptd_full", "lir_pptd_ho", "ablation_secondary"),
        ("E4", "lir_pptd_full", "lir_pptd_nr", "order_sensitivity_secondary"),
        ("E4", "lir_pptd_full", "lir_pptd_ho", "order_sensitivity_secondary"),
    ]

    for stage, full_method, comparator, analysis_role in comparison_specs:
        relevant_pairings = [p for p in pairings if f"stage={stage}|" in p]
        groups: dict[tuple[str, str, str, str], list[str]] = {}
        for pairing in relevant_pairings:
            full = by_key.get((pairing, full_method))
            comp = by_key.get((pairing, comparator))
            sample = full or comp
            if not sample:
                continue
            key = (
                sample["dataset_id"],
                sample["attack_condition"],
                sample["rho"],
                str(sample.get("task_order_seed")),
            )
            groups.setdefault(key, []).append(pairing)

        for (dataset, attack_condition, rho, order_seed_text), members in sorted(groups.items()):
            if stage == "E4":
                if order_seed_text != "6101":
                    continue
                members = [
                    p
                    for p in relevant_pairings
                    if f"dataset={dataset}|" in p
                    and f"|attack={attack_condition}|rho={rho}|" in p
                ]
            for endpoint_field, endpoint_label in (
                ("primary_loss", "primary"),
                ("secondary_loss", "secondary"),
            ):
                deltas: list[Fraction] = []
                full_losses: list[Fraction] = []
                comp_losses: list[Fraction] = []
                failed_or_missing = 0
                for pairing in sorted(set(members)):
                    full = by_key.get((pairing, full_method))
                    comp = by_key.get((pairing, comparator))
                    if (
                        not full
                        or not comp
                        or full.get("status") != "success"
                        or comp.get("status") != "success"
                    ):
                        failed_or_missing += 1
                        continue
                    f = _fraction(full[endpoint_field])
                    c = _fraction(comp[endpoint_field])
                    full_losses.append(f)
                    comp_losses.append(c)
                    deltas.append(f - c)

                planned = _planned_pair_count(stage, dataset, rho)
                if stage == "E4":
                    planned = 5
                if len(deltas) + failed_or_missing > planned:
                    raise CombinedStatisticsError("PAIR_COUNT_EXCEEDS_PLAN")

                hypothesis_id = (
                    f"stage={stage}|dataset={dataset}|attack={attack_condition}|endpoint={endpoint_label}"
                    f"|comparator={comparator}|rho={rho}"
                )
                if not deltas:
                    contrasts.append(
                        {
                            "hypothesis_id": hypothesis_id,
                            "analysis_role": analysis_role,
                            "stage": stage,
                            "dataset_id": dataset,
                            "attack_condition": attack_condition,
                            "rho": rho,
                            "endpoint": endpoint_label,
                            "comparator": comparator,
                            "planned_n": planned,
                            "observed_n": 0,
                            "failure_or_missing_n": planned,
                            "inference_status": "no_complete_pairs",
                            "holm": None,
                        }
                    )
                    continue

                mean_delta = sum(deltas, Fraction(0, 1)) / len(deltas)
                med_delta = _median_fraction(deltas)
                ci_low, ci_high, boot_seed = bootstrap_mean_ci(
                    deltas,
                    seed_text=seed_root + "|" + hypothesis_id,
                )
                p_two = exact_two_sided_sign_flip(deltas) if len(deltas) >= 2 else Fraction(1, 1)
                rb = paired_rank_biserial(deltas)
                contrasts.append(
                    {
                        "hypothesis_id": hypothesis_id,
                        "analysis_role": analysis_role,
                        "stage": stage,
                        "dataset_id": dataset,
                        "attack_condition": attack_condition,
                        "rho": rho,
                        "endpoint": endpoint_label,
                        "comparator": comparator,
                        "planned_n": planned,
                        "observed_n": len(deltas),
                        "failure_or_missing_n": planned - len(deltas),
                        "inference_status": (
                            "complete" if len(deltas) == planned else "degraded_by_failures_or_missing_pairs"
                        ),
                        "mean_full_loss": _fraction_json(
                            sum(full_losses, Fraction(0, 1)) / len(full_losses)
                        ),
                        "mean_comparator_loss": _fraction_json(
                            sum(comp_losses, Fraction(0, 1)) / len(comp_losses)
                        ),
                        "mean_difference": _fraction_json(mean_delta),
                        "median_difference": _fraction_json(med_delta),
                        "bootstrap_95_ci": {
                            "low": ci_low,
                            "high": ci_high,
                            "resamples": BOOTSTRAP_RESAMPLES,
                            "seed": boot_seed,
                        },
                        "two_sided_exact_sign_flip_p": _fraction_json(p_two),
                        "paired_rank_biserial": _fraction_json(rb),
                        "holm": None,
                    }
                )

    families: dict[str, list[dict[str, Any]]] = {}
    for c in contrasts:
        if c["rho"] == "0" or c.get("observed_n", 0) == 0 or c["stage"] == "E4":
            continue
        family_id = (
            f"stage={c['stage']}|dataset={c['dataset_id']}|attack={c['attack_condition']}"
            f"|endpoint={c['endpoint']}|comparator={c['comparator']}"
        )
        families.setdefault(family_id, []).append(c)

    for family_id, items in families.items():
        raw = {
            c["hypothesis_id"]: Fraction(
                int(c["two_sided_exact_sign_flip_p"]["numerator"]),
                int(c["two_sided_exact_sign_flip_p"]["denominator"]),
            )
            for c in items
        }
        adj = holm_adjust(raw)
        for c in items:
            c["holm"] = {
                "family_id": family_id,
                "family_size": len(items),
                "adjusted_p": _fraction_json(adj[c["hypothesis_id"]]),
                "alpha": float(ALPHA),
                "reject": adj[c["hypothesis_id"]] <= ALPHA,
            }

    return {
        "schema_version": "1.0",
        "phase": "Phase-R1+R1B",
        "analysis_status": "post_phase11_extension_prebound_before_R1B_outcomes",
        "upstream_phase_r1_start_binding_sha256": UPSTREAM_PHASE_R1_START_BINDING_SHA256,
        "upstream_e1_raw_sha256": UPSTREAM_E1_RAW_SHA256,
        "r1b_start_binding_sha256": r1b_summary["r1b_start_binding_sha256"],
        "r1b_raw_sha256": r1b_summary["raw_sha256"],
        "e1_formal_summary_sha256": _sha256(root / "results/summary/phase_r1_formal_summary.json"),
        "r1b_formal_summary_sha256": _sha256(root / "results/summary/phase_r1b_formal_summary.json"),
        "analysis_seed_root": seed_root,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "test_direction": "two_sided",
        "delta_definition": "loss(lir_pptd_full)-loss(comparator)",
        "combined_method_run_count": len(records),
        "contrasts": contrasts,
        "holm_family_count": len(families),
        "reporting_rule": (
            "failures/missing pairs retained in counts; non-supporting outcomes are not automatically baseline wins"
        ),
    }


def validate_static_contract() -> dict[str, Any]:
    return {
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "test_direction": "two_sided",
        "delta_definition": "loss(lir_pptd_full)-loss(comparator)",
        "comparison_roles": [
            "E1 external_primary CRH",
            "E1 external_secondary Mean/Vote",
            "E2 external_secondary_on_off CRH/MeanVote",
            "E3 ablation_secondary Full-vs-NR/HO",
            "E4 order_sensitivity_secondary Full-vs-NR/HO",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase-R1 + R1B prebound combined statistical analyzer")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument(
        "--output",
        default="results/summary/phase_r1_combined_statistical_analysis.json",
    )
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.validate_only:
        result = validate_static_contract()
        print("R1_COMBINED_STATISTICAL_PLAN_VALIDATE=PASS")
        print(f"BOOTSTRAP_RESAMPLES={result['bootstrap_resamples']}")
        print("TEST_DIRECTION=two_sided")
        print("FORMAL_ANALYSIS_RUN=0")
        return 0

    root = Path(args.repo_root).resolve()
    result = analyze(root)
    out = root / args.output
    if out.exists():
        raise CombinedStatisticsError(f"REFUSING_TO_OVERWRITE:{out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("R1_COMBINED_STATISTICAL_ANALYSIS=PASS")
    print(f"COMBINED_METHOD_RUNS={result['combined_method_run_count']}")
    print(f"CONTRASTS={len(result['contrasts'])}")
    print(f"HOLM_FAMILIES={result['holm_family_count']}")
    print(f"OUTPUT={out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
