from __future__ import annotations

import re
from typing import Any

from pydantic_core import core_schema

_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}$")
_HASH_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class _ValidatedString(str):
    pattern = _ID_PATTERN
    description = "identifier"

    @classmethod
    def _validate(cls, value: Any) -> "_ValidatedString":
        if isinstance(value, _ValidatedString) and type(value) is not cls:
            raise ValueError(f"cannot substitute {type(value).__name__} for {cls.__name__}")
        if type(value) is not str and type(value) is not cls:
            raise ValueError(f"invalid {cls.description}")
        if not cls.pattern.fullmatch(str(value)):
            raise ValueError(f"invalid {cls.description}")
        return cls(value)

    @classmethod
    def __get_pydantic_core_schema__(cls, source_type: Any, handler: Any) -> core_schema.CoreSchema:
        def reject_cross_type(value: Any) -> Any:
            if isinstance(value, _ValidatedString) and type(value) is not cls:
                raise ValueError(f"cannot substitute {type(value).__name__} for {cls.__name__}")
            return value

        return core_schema.no_info_after_validator_function(
            cls._validate,
            core_schema.no_info_before_validator_function(
                reject_cross_type,
                core_schema.str_schema(strict=True),
            ),
        )


class WorkerID(_ValidatedString): description = "WorkerID"
class TaskID(_ValidatedString): description = "TaskID"
class AttemptID(_ValidatedString): description = "AttemptID"
class TransactionID(_ValidatedString): description = "TransactionID"
class UploadID(_ValidatedString): description = "UploadID"
class RunID(_ValidatedString): description = "RunID"


class _HashString(_ValidatedString):
    pattern = _HASH_PATTERN


class ConfigHash(_HashString): description = "ConfigHash"
class DataHash(_HashString): description = "DataHash"
class CodeHash(_HashString): description = "CodeHash"
class BackendProfileHash(_HashString): description = "BackendProfileHash"
class ArtifactHash(_HashString): description = "ArtifactHash"
