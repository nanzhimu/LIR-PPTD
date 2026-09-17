from __future__ import annotations

import json
from typing import Any, ClassVar

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .canonical import CanonicalRational, canonical_round_trip, parse_rational
from .identifiers import TaskID, WorkerID


class ConfigurationLoadError(ValueError, TypeError):
    def __init__(self, reason_code: str, message: str) -> None:
        self.reason_code = reason_code
        super().__init__(message)


class _UniqueKeyLoader(yaml.SafeLoader):
    pass


def _construct_unique_mapping(loader: _UniqueKeyLoader, node: yaml.MappingNode, deep: bool = False) -> dict[Any, Any]:
    mapping: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise ConfigurationLoadError("DUPLICATE_KEY", f"duplicate YAML key: {key}")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeyLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_unique_mapping)


def _unique_json_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ConfigurationLoadError("DUPLICATE_KEY", f"duplicate JSON key: {key}")
        result[key] = value
    return result


class CanonicalModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: str = "1.0"
    original_literals: dict[str, str] = Field(default_factory=dict, exclude=True)

    @field_validator("original_literals")
    @classmethod
    def audit_literals_are_strings(cls, value: dict[str, str]) -> dict[str, str]:
        if any(type(item) is not str for item in value.values()):
            raise TypeError("original_literals are audit strings only")
        return value

    def semantic_object(self) -> dict[str, Any]:
        return self.model_dump(mode="python", exclude={"original_literals"})

    def validate_round_trip(self) -> None:
        canonical_round_trip(self.semantic_object())


class ExactParameterConfig(CanonicalModel):
    parameters: dict[str, CanonicalRational]

    @field_validator("parameters", mode="before")
    @classmethod
    def parse_parameters(cls, value: Any) -> dict[str, CanonicalRational]:
        if not isinstance(value, dict):
            raise TypeError("parameters must be an object")
        return {str(key): parse_rational(item) for key, item in value.items()}


class RationalFieldsModel(CanonicalModel):
    rational_fields: ClassVar[tuple[str, ...]] = ()


def _parse_exact(value: Any) -> CanonicalRational:
    return parse_rational(value)


class BackendProfile(RationalFieldsModel):
    profile_id: str
    backend_kind: str
    prime: str
    fractional_bits: int = Field(ge=1)
    scale: CanonicalRational
    error_bound: CanonicalRational
    range_bound: CanonicalRational

    _scale = field_validator("scale", mode="before")(_parse_exact)
    _error = field_validator("error_bound", mode="before")(_parse_exact)
    _range = field_validator("range_bound", mode="before")(_parse_exact)


class ProtocolConfig(RationalFieldsModel):
    threshold: int = Field(ge=2)
    server_count: int = Field(ge=2)
    timeout_seconds: CanonicalRational

    _timeout = field_validator("timeout_seconds", mode="before")(_parse_exact)


class ExperimentConfig(RationalFieldsModel):
    experiment_id: str
    master_seed: int = Field(ge=0)
    run_namespace: str
    parameters: dict[str, CanonicalRational]

    @field_validator("parameters", mode="before")
    @classmethod
    def exact_parameters(cls, value: Any) -> dict[str, CanonicalRational]:
        if not isinstance(value, dict):
            raise TypeError("parameters must be an object")
        return {str(key): parse_rational(item) for key, item in value.items()}


class DatasetManifest(RationalFieldsModel):
    dataset_id: str
    version: str
    normalization_lower: CanonicalRational
    normalization_upper: CanonicalRational

    _lower = field_validator("normalization_lower", mode="before")(_parse_exact)
    _upper = field_validator("normalization_upper", mode="before")(_parse_exact)


class AttackManifest(RationalFieldsModel):
    attack_id: str
    oracle: bool
    malicious_fraction: CanonicalRational

    _fraction = field_validator("malicious_fraction", mode="before")(_parse_exact)


class BaselineManifest(RationalFieldsModel):
    baseline_id: str
    implementation_source: str
    compute_budget_seconds: CanonicalRational

    _budget = field_validator("compute_budget_seconds", mode="before")(_parse_exact)


class StatisticsFamilyManifest(RationalFieldsModel):
    family_id: str
    endpoint: str
    alpha: CanonicalRational

    _alpha = field_validator("alpha", mode="before")(_parse_exact)


class EvaluationTaskManifest(CanonicalModel):
    manifest_id: str
    task_ids: tuple[TaskID, ...]
    worker_ids: tuple[WorkerID, ...] = ()


class Phase6TaskManifest(CanonicalModel):
    task_id: TaskID
    dataset_id: str
    modality: str
    split: str
    primary_endpoint: str
    primary_loss_direction: str
    attack_family: str | None = None
    fault_profile_id: str | None = None
    notes: str | None = None
    validation_formal_split_policy: str | None = None
    final_test_split_policy: str | None = None


class Phase6EvaluationTaskManifest(CanonicalModel):
    manifest_id: str
    frozen_against_phase5_commit: str
    implementation_commit: str
    scope: str
    data_policy: dict[str, Any]
    task_order_policy: str
    paired_seed_count: int = Field(ge=1)
    shared_across_all_methods: bool
    scored_task_ids: tuple[TaskID, ...]
    tasks: tuple[Phase6TaskManifest, ...]

    @property
    def task_ids(self) -> tuple[TaskID, ...]:
        return self.scored_task_ids

    @model_validator(mode="after")
    def validate_phase6_consistency(self) -> "Phase6EvaluationTaskManifest":
        task_ids = [task.task_id for task in self.tasks]
        if len(set(task_ids)) != len(task_ids):
            raise ValueError("phase6 task IDs must be unique")
        if tuple(task_ids) != self.scored_task_ids:
            raise ValueError("scored_task_ids must match task order")
        recovery = next((task for task in self.tasks if task.task_id == "synthetic_recovery"), None)
        if recovery is None:
            raise ValueError("synthetic_recovery task is required")
        if recovery.attack_family != "not_applicable":
            raise ValueError("synthetic_recovery attack_family must be not_applicable")
        if recovery.fault_profile_id != "phase5_e8a_frozen_fault_profile":
            raise ValueError("synthetic_recovery fault profile must be frozen")
        if recovery.validation_formal_split_policy != "validation_and_formal_candidate_only":
            raise ValueError("synthetic_recovery validation/formal split policy must be frozen")
        if recovery.final_test_split_policy != "blocked_not_active":
            raise ValueError("synthetic_recovery final test split policy must be blocked_not_active")
        return self


class Phase6FaultProfile(CanonicalModel):
    fault_profile_id: str
    scope: str
    source: str
    source_commit: str
    implementation_commit: str
    backend: str
    allowed_fault_boundary: tuple[str, ...]
    explicitly_not_extended_to: tuple[str, ...]
    status: str = "candidate_only"


class Phase6SeedManifest(CanonicalModel):
    seed_manifest_id: str
    generator_implementation_path: str
    generator_implementation_commit: str
    generator_config_path: str
    generator_config_sha256: str
    validation_seeds: tuple[int, ...]
    formal_candidate_seeds: tuple[int, ...]
    final_test_seeds: tuple[int, ...] = ()
    paired_seed_count: int = Field(ge=1)
    deterministic_ordering_policy: str
    raw_artifact_hash_policy: str
    processed_artifact_hash_policy: str
    non_determinism_prohibition: tuple[str, ...]
    validation_formal_disjoint: bool = True
    final_test_access: str = "not_created_and_inaccessible_in_phase6"
    status: str = "candidate_only"


def _reject_yaml_floats(value: Any, path: str = "root") -> None:
    if isinstance(value, float):
        raise ConfigurationLoadError("FLOAT_FORBIDDEN", f"YAML float forbidden at {path}")
    if isinstance(value, dict):
        for key, item in value.items():
            _reject_yaml_floats(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_yaml_floats(item, f"{path}[{index}]")


def load_exact_yaml(text: str) -> ExactParameterConfig:
    try:
        raw = yaml.load(text, Loader=_UniqueKeyLoader)
    except ConfigurationLoadError:
        raise
    except yaml.YAMLError as exc:
        raise ConfigurationLoadError("INVALID_YAML", "invalid YAML configuration") from exc
    if not isinstance(raw, dict):
        raise ConfigurationLoadError("INVALID_ROOT", "configuration root must be an object")
    _reject_yaml_floats(raw)
    return ExactParameterConfig.model_validate(raw)


def load_exact_json(text: str) -> ExactParameterConfig:
    try:
        raw = json.loads(text, object_pairs_hook=_unique_json_pairs, parse_constant=lambda value: (_ for _ in ()).throw(ConfigurationLoadError("NONFINITE_FORBIDDEN", value)))
    except ConfigurationLoadError:
        raise
    except json.JSONDecodeError as exc:
        raise ConfigurationLoadError("INVALID_JSON", "invalid JSON configuration") from exc
    if not isinstance(raw, dict):
        raise ConfigurationLoadError("INVALID_ROOT", "configuration root must be an object")
    _reject_yaml_floats(raw)
    return ExactParameterConfig.model_validate(raw)
