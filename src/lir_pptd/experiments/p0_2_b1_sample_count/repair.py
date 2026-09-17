from __future__ import annotations

import copy
import hashlib
import itertools
import math
from fractions import Fraction
from typing import Any, Iterable, Mapping

from lir_pptd.experiments.phase_r1.attacks import (
    malicious_count_half_up,
    select_malicious_workers,
)

DATASETS = ("product", "duck", "dog", "weather")
RHOS = ("1/10", "3/10", "1/2", "7/10", "9/10")
EXTREME_RHOS = ("1/10", "9/10")
EXPECTED_SOURCE_STREAMS = 370
EXPECTED_FINAL_STREAMS = 378
EXPECTED_STANDARD_N = 20
EXPECTED_WEATHER_EXTREME_SOURCE_N = 5
EXPECTED_WEATHER_EXTREME_FINAL_N = 9
COMPLETION_SEED_START = 120001
NAMESPACE = "phase-r1-malicious-subset-v1"


class P02SampleCountError(RuntimeError):
    pass


def malicious_count(worker_count: int, rho: str) -> int:
    return malicious_count_half_up(worker_count, rho)


def possible_unique_subsets(worker_count: int, rho: str) -> int:
    k = malicious_count(worker_count, rho)
    return math.comb(worker_count, k)


def _canon_workers(values: Iterable[Any]) -> tuple[str, ...]:
    return tuple(sorted(map(str, values)))


def _all_unique(worker_ids: Iterable[str], rho: str) -> set[tuple[str, ...]]:
    workers = tuple(sorted(set(map(str, worker_ids))))
    k = malicious_count(len(workers), rho)
    return {tuple(c) for c in itertools.combinations(workers, k)}


def _entry_subsets(entry: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = [dict(x) for x in entry.get("subsets", [])]
    seen: set[tuple[str, ...]] = set()
    for i, row in enumerate(rows, 1):
        workers = _canon_workers(row.get("workers", ()))
        if not workers:
            raise P02SampleCountError(f"EMPTY_SUBSET:{i}")
        if workers in seen:
            raise P02SampleCountError(f"DUPLICATE_SOURCE_SUBSET:{workers}")
        seen.add(workers)
        row["workers"] = list(workers)
    return rows


def complete_weather_extreme_subsets(
    worker_ids: Iterable[str],
    rho: str,
    existing_rows: Iterable[Mapping[str, Any]],
    *,
    dataset_id: str = "weather",
    seed_start: int = COMPLETION_SEED_START,
    namespace: str = NAMESPACE,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Preserve the observed rows and append every still-unseen unique subset.

    A real generator seed is found for every appended subset, so downstream
    code that records subset_seed remains source-compatible.  No duplicate
    subset is ever used to inflate n.
    """
    workers = tuple(sorted(set(map(str, worker_ids))))
    all_possible = _all_unique(workers, rho)
    existing = [dict(x) for x in existing_rows]
    existing_set = {_canon_workers(x["workers"]) for x in existing}
    if len(existing_set) != len(existing):
        raise P02SampleCountError(f"DUPLICATE_EXISTING:{dataset_id}:{rho}")
    if not existing_set <= all_possible:
        raise P02SampleCountError(f"INVALID_EXISTING_SUBSET:{dataset_id}:{rho}")

    missing = set(all_possible - existing_set)
    appended: list[dict[str, Any]] = []
    seed = int(seed_start)
    # We deliberately find a source-compatible deterministic seed for each
    # remaining subset rather than inventing duplicate pseudo-replicates.
    while missing and seed < seed_start + 1_000_000:
        subset = select_malicious_workers(
            workers,
            dataset_id=dataset_id,
            rho=rho,
            seed=seed,
            namespace=namespace,
        )
        if subset in missing:
            appended.append({
                "seed": seed,
                "workers": list(subset),
                "p0_2_origin": "exhaustive_unique_completion",
            })
            missing.remove(subset)
        seed += 1

    if missing:
        raise P02SampleCountError(
            f"UNABLE_TO_ENUMERATE_ALL_MISSING:{dataset_id}:{rho}:{len(missing)}"
        )

    final = existing + appended
    final_set = {_canon_workers(x["workers"]) for x in final}
    if final_set != all_possible:
        raise P02SampleCountError(
            f"NOT_EXHAUSTIVE:{dataset_id}:{rho}:{len(final_set)}:{len(all_possible)}"
        )
    return final, appended


def source_stream_count(manifest: Mapping[str, Any]) -> int:
    return sum(
        len(manifest["datasets"][ds][rho]["subsets"])
        for ds in DATASETS for rho in RHOS
    )


def repair_manifest(
    source_manifest: Mapping[str, Any],
    *,
    weather_worker_ids: Iterable[str],
    source_manifest_sha256: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if source_stream_count(source_manifest) != EXPECTED_SOURCE_STREAMS:
        raise P02SampleCountError(
            f"SOURCE_STREAM_COUNT:{source_stream_count(source_manifest)}:{EXPECTED_SOURCE_STREAMS}"
        )

    out = copy.deepcopy(dict(source_manifest))
    weather_workers = tuple(sorted(set(map(str, weather_worker_ids))))
    if len(weather_workers) != 9:
        raise P02SampleCountError(f"WEATHER_WORKER_COUNT:{len(weather_workers)}:9")

    additions: list[dict[str, Any]] = []
    for ds in DATASETS:
        for rho in RHOS:
            entry = out["datasets"][ds][rho]
            rows = _entry_subsets(entry)
            if ds == "weather" and rho in EXTREME_RHOS:
                if len(rows) != EXPECTED_WEATHER_EXTREME_SOURCE_N:
                    raise P02SampleCountError(
                        f"SOURCE_EXTREME_N:{ds}:{rho}:{len(rows)}:{EXPECTED_WEATHER_EXTREME_SOURCE_N}"
                    )
                possible = possible_unique_subsets(len(weather_workers), rho)
                if possible != EXPECTED_WEATHER_EXTREME_FINAL_N:
                    raise P02SampleCountError(f"EXTREME_COMBINATORIAL_CAP:{rho}:{possible}:9")
                final, added = complete_weather_extreme_subsets(
                    weather_workers, rho, rows, dataset_id=ds
                )
                entry["subsets"] = final
                entry["target"] = len(final)
                entry["p0_2_sample_count"] = {
                    "source_n": len(rows),
                    "final_n": len(final),
                    "possible_unique_subsets": possible,
                    "exhaustive": True,
                    "duplicate_subsets_used": False,
                }
                for idx, row in enumerate(added, start=len(rows)+1):
                    additions.append({
                        "dataset": ds,
                        "rho": rho,
                        "replicate": idx,
                        "seed": int(row["seed"]),
                        "workers": list(row["workers"]),
                    })
            else:
                if len(rows) != EXPECTED_STANDARD_N:
                    raise P02SampleCountError(
                        f"STANDARD_CELL_N:{ds}:{rho}:{len(rows)}:{EXPECTED_STANDARD_N}"
                    )
                entry["subsets"] = rows

    out["p0_2_sample_count_repair"] = {
        "schema_version": "1.0",
        "source_manifest_sha256": source_manifest_sha256,
        "rule": "preserve all existing paired subsets; exhaustively complete Weather extreme cells to all unique malicious-worker subsets; never duplicate a subset to inflate replicate count",
        "standard_cell_n": 20,
        "weather_extreme_cell_n": 9,
        "weather_extreme_possible_unique": 9,
        "added_unique_subsets": len(additions),
        "final_nonzero_streams": EXPECTED_FINAL_STREAMS,
    }
    validate_repaired_manifest(out, weather_worker_ids=weather_workers)
    return out, additions


def validate_repaired_manifest(
    manifest: Mapping[str, Any], *, weather_worker_ids: Iterable[str]
) -> None:
    weather_workers = tuple(sorted(set(map(str, weather_worker_ids))))
    total = 0
    for ds in DATASETS:
        for rho in RHOS:
            rows = manifest["datasets"][ds][rho]["subsets"]
            subsets = [_canon_workers(x["workers"]) for x in rows]
            if len(subsets) != len(set(subsets)):
                raise P02SampleCountError(f"DUPLICATE_FINAL_SUBSET:{ds}:{rho}")
            expected = 9 if ds == "weather" and rho in EXTREME_RHOS else 20
            if len(subsets) != expected:
                raise P02SampleCountError(f"FINAL_CELL_N:{ds}:{rho}:{len(subsets)}:{expected}")
            if ds == "weather" and rho in EXTREME_RHOS:
                if set(subsets) != _all_unique(weather_workers, rho):
                    raise P02SampleCountError(f"WEATHER_EXTREME_NOT_EXHAUSTIVE:{rho}")
            total += len(subsets)
    if total != EXPECTED_FINAL_STREAMS:
        raise P02SampleCountError(f"FINAL_STREAM_COUNT:{total}:{EXPECTED_FINAL_STREAMS}")


def subset_digest(workers: Iterable[str]) -> str:
    return hashlib.sha256("\n".join(_canon_workers(workers)).encode()).hexdigest()
