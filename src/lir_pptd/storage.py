from __future__ import annotations

import json
import os
import time
import uuid
from contextlib import contextmanager
from enum import StrEnum
from pathlib import Path
from typing import Iterator

from pydantic import BaseModel

from .identifiers import RunID


class AppendOnlyError(RuntimeError):
    pass


class RawRunState(StrEnum):
    INITIALIZING = "initializing"
    COMPLETE = "complete"
    FAILED = "failed"


class RawArtifactStore:
    def __init__(self, root: Path) -> None:
        self.root = root

    def write_once(self, relative_path: Path, content: bytes) -> Path:
        target = self.root / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            with target.open("xb") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
        except FileExistsError as exc:
            raise AppendOnlyError(f"raw artifact already exists: {target}") from exc
        return target


class RawRunDirectory:
    def __init__(self, raw_root: Path, experiment: str, run_id: RunID | str) -> None:
        if not experiment or any(char not in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789._-" for char in experiment):
            raise ValueError("invalid experiment identifier")
        self.raw_root = raw_root
        self.experiment = experiment
        self.run_id = RunID(run_id)
        self.final_path = raw_root / experiment / self.run_id
        self.initializing_path: Path | None = None

    def initialize(self) -> Path:
        self.final_path.parent.mkdir(parents=True, exist_ok=True)
        if self.final_path.exists() or any(self.final_path.parent.glob(f".{self.run_id}.initializing.*")):
            raise AppendOnlyError("raw run has already been initialized")
        candidate = self.final_path.parent / f".{self.run_id}.initializing.{uuid.uuid4().hex}"
        candidate.mkdir(exist_ok=False)
        (candidate / "RUN_STATE").write_text(RawRunState.INITIALIZING.value, encoding="ascii")
        self.initializing_path = candidate
        return candidate

    def write_once(self, relative_path: Path, content: bytes) -> Path:
        if self.initializing_path is None:
            raise AppendOnlyError("raw run is not initializing")
        return RawArtifactStore(self.initializing_path).write_once(relative_path, content)

    def complete(self, manifest: BaseModel | dict[str, object]) -> Path:
        if self.initializing_path is None or not self.initializing_path.exists():
            raise AppendOnlyError("raw run is not initializing")
        body = manifest.model_dump(mode="json") if isinstance(manifest, BaseModel) else manifest
        RawArtifactStore(self.initializing_path).write_once(
            Path("manifest.json"), json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
        )
        (self.initializing_path / "RUN_STATE").write_text(RawRunState.COMPLETE.value, encoding="ascii")
        os.replace(self.initializing_path, self.final_path)
        self.initializing_path = None
        return self.final_path

    def mark_failed(self, reason: str) -> Path:
        if self.initializing_path is None or not self.initializing_path.exists():
            raise AppendOnlyError("raw run is not initializing")
        RawArtifactStore(self.initializing_path).write_once(Path("failure.json"), json.dumps({"reason": reason}).encode())
        (self.initializing_path / "RUN_STATE").write_text(RawRunState.FAILED.value, encoding="ascii")
        failed_path = self.final_path.parent / f"{self.run_id}.failed"
        if failed_path.exists():
            raise AppendOnlyError("failed raw run already exists")
        os.replace(self.initializing_path, failed_path)
        self.initializing_path = None
        return failed_path

    @staticmethod
    def inspect(path: Path) -> RawRunState:
        state_path = path / "RUN_STATE"
        if not state_path.exists():
            return RawRunState.INITIALIZING
        state = state_path.read_text(encoding="ascii").strip()
        return RawRunState(state)


class JsonlAppendLog:
    def __init__(self, path: Path) -> None:
        self.path = path

    def append(self, record: BaseModel) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
        with self.path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(line + "\n")
            handle.flush()
            os.fsync(handle.fileno())


@contextmanager
def coordinator_timer() -> Iterator[dict[str, float]]:
    result: dict[str, float] = {}
    started = time.monotonic()
    try:
        yield result
    finally:
        result["duration_seconds"] = time.monotonic() - started
