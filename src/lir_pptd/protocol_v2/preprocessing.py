from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .binding import build_preprocessing_bundle, build_preprocessing_manifest, build_preprocessing_record
from .model import OperationKind, PreprocessingBundle, PreprocessingManifest, PreprocessingRecord


@dataclass(frozen=True)
class PreprocessingArtifacts:
    manifest: PreprocessingManifest
    bundle: PreprocessingBundle
    records: tuple[PreprocessingRecord, ...]


def generate_preprocessing_artifacts(
    *,
    task_id,
    tx_id,
    attempt_id,
    cfg_hash: str,
    part_hash: str,
    generation: int,
    committee_hash: str,
    record_specs: Sequence[tuple[str, str, OperationKind]],
    bundle_id: str,
) -> PreprocessingArtifacts:
    placeholder_records = tuple(
        build_preprocessing_record(
            record_id=record_id,
            op_id=op_id,
            operation_kind=operation_kind,
            task_id=task_id,
            tx_id=tx_id,
            attempt_id=attempt_id,
            cfg_hash=cfg_hash,
            part_hash=part_hash,
            generation=generation,
            committee_hash=committee_hash,
            manifest_hash="pending",
            bundle_hash="pending",
        )
        for record_id, op_id, operation_kind in record_specs
    )
    manifest = build_preprocessing_manifest(
        task_id=task_id,
        tx_id=tx_id,
        attempt_id=attempt_id,
        cfg_hash=cfg_hash,
        part_hash=part_hash,
        generation=generation,
        committee_hash=committee_hash,
        records=placeholder_records,
    )
    bundle = build_preprocessing_bundle(bundle_id=bundle_id, manifest=manifest)
    manifest_hash = manifest.manifest_hash()
    bundle_hash = bundle.bundle_hash()
    records = tuple(
        record.model_copy(update={"manifest_hash": manifest_hash, "bundle_hash": bundle_hash})
        for record in placeholder_records
    )
    manifest = manifest.model_copy(update={"records": records})
    bundle = build_preprocessing_bundle(bundle_id=bundle_id, manifest=manifest)
    return PreprocessingArtifacts(manifest=manifest, bundle=bundle, records=records)


def activate_preprocessing(bundle: PreprocessingBundle) -> PreprocessingBundle:
    return bundle.model_copy(update={"activated": True})


def consume_preprocessing_record(record: PreprocessingRecord) -> PreprocessingRecord:
    return record.model_copy(update={"consumed": True})
