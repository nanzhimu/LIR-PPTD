from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Iterable

from .canonical import canonical_json_bytes
from .config import CanonicalModel, ExactParameterConfig
from .identifiers import BackendProfileHash, CodeHash, ConfigHash, DataHash, RunID


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def source_hash(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def data_hash(path: Path) -> str:
    return source_hash(path)


def semantic_hash(schema_version: str, semantic_object: Any) -> str:
    return sha256_bytes(
        canonical_json_bytes({"schema_version": schema_version, "semantic_object": semantic_object})
    )


def config_hash(config: ExactParameterConfig | CanonicalModel) -> ConfigHash:
    return ConfigHash(semantic_hash(config.schema_version, config.semantic_object()))


def backend_profile_hash(schema_version: str, profile: Any) -> BackendProfileHash:
    return BackendProfileHash(semantic_hash(schema_version, profile))


def code_hash(paths: Iterable[Path]) -> CodeHash:
    entries = []
    for path in sorted(paths, key=lambda item: item.as_posix()):
        entries.append({"path": path.as_posix(), "sha256": source_hash(path)})
    return CodeHash(semantic_hash("code-tree-v1", entries))


def run_id(
    *,
    schema_version: str,
    config_digest: ConfigHash | str,
    code_digest: CodeHash | str,
    data_digest: DataHash | str,
    backend_profile_digest: BackendProfileHash | str,
    master_seed: int,
    run_namespace: str,
) -> RunID:
    if isinstance(master_seed, bool) or master_seed < 0:
        raise ValueError("master_seed must be a nonnegative integer")
    if not run_namespace:
        raise ValueError("run_namespace must be nonempty")
    semantic_object = {
        "backend_profile_hash": BackendProfileHash(backend_profile_digest),
        "code_hash": CodeHash(code_digest),
        "config_hash": ConfigHash(config_digest),
        "data_hash": DataHash(data_digest),
        "master_seed": master_seed,
        "run_namespace": run_namespace,
        "schema_version": schema_version,
    }
    return RunID(semantic_hash("run-id-v2", semantic_object))
