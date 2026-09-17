from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

from .contract import get_dataset_contract

DatasetID = Literal["product", "duck", "dog", "weather"]


class RealDataError(ValueError):
    pass


@dataclass(frozen=True)
class RealTask:
    task_id: str
    modality: Literal["numerical", "categorical"]
    participant_ids: tuple[str, ...]
    reports: dict[str, tuple[str | int, ...]]
    truth_vector: tuple[str | int, ...]
    truth_class: int | None
    source_index: int


@dataclass(frozen=True)
class RealDataset:
    dataset_id: DatasetID
    modality: Literal["numerical", "categorical"]
    class_count: int | None
    worker_ids: tuple[str, ...]
    tasks: tuple[RealTask, ...]
    answer_sha256: str
    truth_sha256: str


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            block = f.read(1024 * 1024)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RealDataError(f"empty csv: {path}")
    return rows


def _one_hot(label: int, width: int) -> tuple[int, ...]:
    if not 0 <= label < width:
        raise RealDataError(f"categorical label {label} outside [0,{width - 1}]")
    return tuple(1 if i == label else 0 for i in range(width))


def load_real_dataset(
    data_root: Path,
    dataset_id: DatasetID,
    *,
    expected_answer_sha256: str | None = None,
    expected_truth_sha256: str | None = None,
) -> RealDataset:
    if dataset_id not in {"product", "duck", "dog", "weather"}:
        raise RealDataError(f"unsupported dataset: {dataset_id}")

    answer_path = data_root / dataset_id / "answer.csv"
    truth_path = data_root / dataset_id / "truth.csv"
    if not answer_path.is_file() or not truth_path.is_file():
        raise RealDataError(f"dataset files missing for {dataset_id}")

    answer_sha = _sha256(answer_path)
    truth_sha = _sha256(truth_path)
    if expected_answer_sha256 is not None and answer_sha != expected_answer_sha256:
        raise RealDataError(f"answer SHA256 mismatch for {dataset_id}")
    if expected_truth_sha256 is not None and truth_sha != expected_truth_sha256:
        raise RealDataError(f"truth SHA256 mismatch for {dataset_id}")

    answers = _read_csv(answer_path)
    truths = _read_csv(truth_path)
    required_answer_fields = {"question", "worker", "answer"}
    required_truth_fields = {"question", "truth"}
    if set(answers[0]) != required_answer_fields:
        raise RealDataError(f"unexpected answer schema for {dataset_id}: {sorted(answers[0])}")
    if set(truths[0]) != required_truth_fields:
        raise RealDataError(f"unexpected truth schema for {dataset_id}: {sorted(truths[0])}")

    truth_ids = [row["question"] for row in truths]
    if len(truth_ids) != len(set(truth_ids)):
        raise RealDataError(f"duplicate truth question in {dataset_id}")
    truth_id_set = set(truth_ids)

    seen_pairs: set[tuple[str, str]] = set()
    answers_by_question: dict[str, list[tuple[str, str]]] = {}
    all_workers: set[str] = set()
    for row in answers:
        q = row["question"]
        worker = row["worker"]
        if q not in truth_id_set:
            raise RealDataError(f"answer without truth in {dataset_id}: {q}")
        pair = (q, worker)
        if pair in seen_pairs:
            raise RealDataError(f"duplicate worker answer in {dataset_id}: {q}/{worker}")
        seen_pairs.add(pair)
        all_workers.add(worker)
        answers_by_question.setdefault(q, []).append((worker, row["answer"]))

    missing_answers = [q for q in truth_ids if q not in answers_by_question]
    if missing_answers:
        raise RealDataError(f"truth without answers in {dataset_id}: {missing_answers[0]}")

    contract = get_dataset_contract(dataset_id)
    modality: Literal["numerical", "categorical"] = contract.modality
    class_count: int | None = None
    if modality == "categorical":
        labels = []
        for row in truths:
            try:
                labels.append(int(row["truth"]))
            except ValueError as exc:
                raise RealDataError(f"non-integer truth label in {dataset_id}") from exc
        for row in answers:
            try:
                labels.append(int(row["answer"]))
            except ValueError as exc:
                raise RealDataError(f"non-integer answer label in {dataset_id}") from exc
        if min(labels) != 0:
            raise RealDataError(f"categorical labels must start at zero in {dataset_id}")
        class_count = max(labels) + 1
        if set(labels) != set(range(class_count)):
            raise RealDataError(f"categorical labels must be contiguous in {dataset_id}")
        if class_count != contract.class_count:
            raise RealDataError(
                f"dataset contract mismatch for {dataset_id}: expected {contract.class_count} classes, got {class_count}"
            )

    tasks: list[RealTask] = []
    for index, truth_row in enumerate(truths):
        question = truth_row["question"]
        raw_reports = answers_by_question[question]
        participant_ids = tuple(sorted(worker for worker, _ in raw_reports))
        raw_by_worker = {worker: value for worker, value in raw_reports}

        if modality == "categorical":
            assert class_count is not None
            truth_class = int(truth_row["truth"])
            truth_vector = _one_hot(truth_class, class_count)
            reports = {
                worker: _one_hot(int(raw_by_worker[worker]), class_count)
                for worker in participant_ids
            }
        else:
            truth_class = None
            truth_value = truth_row["truth"]
            truth_number = float(truth_value)
            if not 0.0 <= truth_number <= 1.0:
                raise RealDataError(f"weather truth outside [0,1]: {question}")
            truth_vector = (truth_value,)
            reports = {}
            for worker in participant_ids:
                value = raw_by_worker[worker]
                value_number = float(value)
                if not 0.0 <= value_number <= 1.0:
                    raise RealDataError(f"weather report outside [0,1]: {question}/{worker}")
                reports[worker] = (value,)

        tasks.append(
            RealTask(
                task_id=f"{dataset_id}:{question}",
                modality=modality,
                participant_ids=participant_ids,
                reports=reports,
                truth_vector=truth_vector,
                truth_class=truth_class,
                source_index=index,
            )
        )

    return RealDataset(
        dataset_id=dataset_id,
        modality=modality,
        class_count=class_count,
        worker_ids=tuple(sorted(all_workers)),
        tasks=tuple(tasks),
        answer_sha256=answer_sha,
        truth_sha256=truth_sha,
    )


def task_artifact(task: RealTask) -> dict[str, object]:
    return {
        "task_id": task.task_id,
        "modality": task.modality,
        "participant_ids": task.participant_ids,
        "reports": task.reports,
    }
