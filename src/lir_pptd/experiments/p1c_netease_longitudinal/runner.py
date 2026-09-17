from __future__ import annotations

from ...core.consistency_scale import DEFAULT_CONSISTENCY_SCALE_MODE, resolve_from_modality

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
import secrets
import sqlite3
import statistics
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

from lir_pptd.core.hidden_calibration import HiddenCalibrationSchedule
from lir_pptd.experiments.calibration_anchored import run_calibration_task
from lir_pptd.experiments.phase_r1.longitudinal_state import initial_global_state
from lir_pptd.experiments.phase_r1.methods import run_lir_task
from lir_pptd.experiments.phase_r1.real_data_loader import RealTask

STUDY_ID = "p1c_netease_longitudinal_v1"
DATASET_NAME = "NetEaseCrowd"
OFFICIAL_REPO = "https://github.com/fuxiAIlab/NetEaseCrowd-Dataset"
RAW_URL_TEMPLATE = (
    "https://raw.githubusercontent.com/fuxiAIlab/NetEaseCrowd-Dataset/"
    "main/data/NetEaseCrowd_part_{part}.csv"
)
RAW_PARTS = tuple(f"NetEaseCrowd_part_{i}.csv" for i in range(1, 16))

# Official public-dataset cardinalities from the repository/dataset card.
EXPECTED_ANNOTATIONS = 6_016_319
EXPECTED_TASKS = 999_799
EXPECTED_WORKERS = 2_413
EXPECTED_CAPABILITIES = 6
EXPECTED_LABELS = (0, 1, 2)

CONFIG = {
    "K": 10,
    "c0": "1/2",
    "epsilon_c": "1/1024",
    "lambda_tau": "1/5",
    "kappa": "2",
    "eta": "1/10",
    "mu": "1/5",
}
PERIOD = 20
REENTRY_GAP_DAYS = 30
REENTRY_GAP_MS = REENTRY_GAP_DAYS * 24 * 60 * 60 * 1000
TIME_BIN_DAYS = 30
TIME_BIN_MS = TIME_BIN_DAYS * 24 * 60 * 60 * 1000
METHODS = ("lir_pptd", "cowa", "no_hr", "mv", "lir_reset30d")

RAW_REL = Path("data/neteasecrowd/raw")
DB_REL = Path("data/neteasecrowd/neteasecrowd.sqlite")
AUDIT_REL = Path("data/neteasecrowd/dataset_audit.json")
SEEDS_REL = Path("configs/p1c_netease/private_selector_seeds.json")
OUT_REL = Path("results/p1c_netease_longitudinal_v1")


class P1CError(RuntimeError):
    pass


def _atomic_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _sha_obj(obj: Any) -> str:
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def _download(url: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file() and target.stat().st_size > 0:
        return
    tmp = target.with_suffix(target.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "LIR-PPTD-P1C/1.0"})
    with urllib.request.urlopen(req, timeout=120) as src, tmp.open("wb") as dst:
        while True:
            block = src.read(1024 * 1024)
            if not block:
                break
            dst.write(block)
    if tmp.stat().st_size == 0:
        raise P1CError(f"EMPTY_DOWNLOAD:{url}")
    os.replace(tmp, target)


def download_raw_parts(root: Path) -> None:
    raw_dir = root / RAW_REL
    for i, name in enumerate(RAW_PARTS, start=1):
        target = raw_dir / name
        print(f"DOWNLOAD={i}/15 {name}", flush=True)
        _download(RAW_URL_TEMPLATE.format(part=i), target)


def _connect(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA temp_store=MEMORY")
    return conn


def prepare_dataset(root: Path, *, force: bool = False) -> dict[str, Any]:
    raw_dir = root / RAW_REL
    missing = [name for name in RAW_PARTS if not (raw_dir / name).is_file()]
    if missing:
        raise P1CError(
            "RAW_PARTS_MISSING:" + ",".join(missing) + "; use --download or place the official files under " + str(raw_dir)
        )

    db = root / DB_REL
    if force and db.exists():
        db.unlink()
    db.parent.mkdir(parents=True, exist_ok=True)
    conn = _connect(db)
    try:
        conn.executescript(
            """
            DROP TABLE IF EXISTS annotations;
            DROP TABLE IF EXISTS tasks;
            CREATE TABLE annotations(
                task_id INTEGER NOT NULL,
                taskset_id INTEGER NOT NULL,
                worker_id INTEGER NOT NULL,
                answer INTEGER NOT NULL,
                complete_time INTEGER NOT NULL,
                truth INTEGER NOT NULL,
                capability INTEGER NOT NULL,
                PRIMARY KEY(task_id, worker_id)
            );
            """
        )
        insert_sql = (
            "INSERT INTO annotations(task_id,taskset_id,worker_id,answer,complete_time,truth,capability) "
            "VALUES(?,?,?,?,?,?,?)"
        )
        total = 0
        for part_index, name in enumerate(RAW_PARTS, start=1):
            path = raw_dir / name
            batch: list[tuple[int, int, int, int, int, int, int]] = []
            with path.open("r", encoding="utf-8-sig", newline="") as f:
                reader = csv.DictReader(f)
                required = {"taskId", "tasksetId", "workerId", "answer", "completeTime", "truth", "capability"}
                if reader.fieldnames is None or set(reader.fieldnames) != required:
                    raise P1CError(f"RAW_SCHEMA:{name}:{reader.fieldnames}")
                for row in reader:
                    item = (
                        int(row["taskId"]), int(row["tasksetId"]), int(row["workerId"]),
                        int(row["answer"]), int(row["completeTime"]), int(row["truth"]), int(row["capability"]),
                    )
                    if item[3] not in EXPECTED_LABELS or item[5] not in EXPECTED_LABELS:
                        raise P1CError(f"LABEL_OUT_OF_RANGE:{name}:{item[0]}")
                    batch.append(item)
                    if len(batch) >= 50_000:
                        conn.executemany(insert_sql, batch)
                        conn.commit()
                        total += len(batch)
                        batch.clear()
                if batch:
                    conn.executemany(insert_sql, batch)
                    conn.commit()
                    total += len(batch)
            print(f"INGEST={part_index}/15 ROWS={total}", flush=True)

        inconsistent = conn.execute(
            """
            SELECT COUNT(*) FROM (
              SELECT task_id
              FROM annotations
              GROUP BY task_id
              HAVING MIN(truth)<>MAX(truth)
                 OR MIN(capability)<>MAX(capability)
                 OR MIN(taskset_id)<>MAX(taskset_id)
            )
            """
        ).fetchone()[0]
        if inconsistent:
            raise P1CError(f"INCONSISTENT_TASK_METADATA:{inconsistent}")

        conn.executescript(
            """
            CREATE TABLE tasks AS
            SELECT task_id,
                   MIN(taskset_id) AS taskset_id,
                   MIN(truth) AS truth,
                   MIN(capability) AS capability,
                   MAX(complete_time) AS task_end,
                   COUNT(*) AS report_count
            FROM annotations
            GROUP BY task_id;
            CREATE UNIQUE INDEX idx_tasks_id ON tasks(task_id);
            CREATE INDEX idx_tasks_cap_time ON tasks(capability, task_end, task_id);
            CREATE INDEX idx_annotations_task ON annotations(task_id, worker_id);
            CREATE INDEX idx_annotations_worker ON annotations(worker_id, complete_time);
            """
        )
        conn.commit()

        ann = int(conn.execute("SELECT COUNT(*) FROM annotations").fetchone()[0])
        tasks = int(conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0])
        workers = int(conn.execute("SELECT COUNT(DISTINCT worker_id) FROM annotations").fetchone()[0])
        caps = [int(x[0]) for x in conn.execute("SELECT DISTINCT capability FROM tasks ORDER BY capability")]
        labels = [int(x[0]) for x in conn.execute("SELECT DISTINCT truth FROM tasks ORDER BY truth")]
        if ann != EXPECTED_ANNOTATIONS:
            raise P1CError(f"OFFICIAL_ANNOTATION_COUNT:{ann}:{EXPECTED_ANNOTATIONS}")
        if tasks != EXPECTED_TASKS:
            raise P1CError(f"OFFICIAL_TASK_COUNT:{tasks}:{EXPECTED_TASKS}")
        if workers != EXPECTED_WORKERS:
            raise P1CError(f"OFFICIAL_WORKER_COUNT:{workers}:{EXPECTED_WORKERS}")
        if len(caps) != EXPECTED_CAPABILITIES:
            raise P1CError(f"OFFICIAL_CAPABILITY_COUNT:{len(caps)}:{EXPECTED_CAPABILITIES}")
        if labels != list(EXPECTED_LABELS):
            raise P1CError(f"GLOBAL_LABEL_SET:{labels}")

        capability_stats = []
        for cap in caps:
            row = conn.execute(
                """
                SELECT COUNT(*), SUM(report_count), MIN(task_end), MAX(task_end),
                       MIN(report_count), MAX(report_count), COUNT(DISTINCT taskset_id)
                FROM tasks WHERE capability=?
                """, (cap,)
            ).fetchone()
            cap_workers = int(conn.execute(
                """SELECT COUNT(DISTINCT a.worker_id)
                   FROM annotations a JOIN tasks t ON a.task_id=t.task_id
                   WHERE t.capability=?""", (cap,)
            ).fetchone()[0])
            cap_labels = [int(x[0]) for x in conn.execute(
                "SELECT DISTINCT truth FROM tasks WHERE capability=? ORDER BY truth", (cap,)
            )]
            capability_stats.append({
                "capability": cap,
                "tasks": int(row[0]),
                "annotations": int(row[1]),
                "workers": cap_workers,
                "tasksets": int(row[6]),
                "start_ms": int(row[2]),
                "end_ms": int(row[3]),
                "span_days": (int(row[3]) - int(row[2])) / (24*60*60*1000),
                "min_reports_per_task": int(row[4]),
                "max_reports_per_task": int(row[5]),
                "truth_labels": cap_labels,
            })

        raw_hashes = {name: _sha256(raw_dir / name) for name in RAW_PARTS}
        audit = {
            "schema_version": "1.0",
            "study_id": STUDY_ID,
            "dataset": DATASET_NAME,
            "official_repository": OFFICIAL_REPO,
            "raw_parts": 15,
            "raw_sha256": raw_hashes,
            "annotations": ann,
            "tasks": tasks,
            "workers": workers,
            "capabilities": caps,
            "capability_count": len(caps),
            "labels": labels,
            "task_event_time": "max completeTime over the task's observed annotation batch",
            "capability_stats": capability_stats,
        }
        _atomic_json(root / AUDIT_REL, audit)
    finally:
        conn.close()

    # Generate selector seeds once, only after the observational data have already
    # been frozen on disk.  They are method-internal; no report generation depends on them.
    seeds_path = root / SEEDS_REL
    if not seeds_path.exists():
        seeds_path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_json(seeds_path, {
            "schema_version": "1.0",
            "study_id": STUDY_ID,
            "seeds": {str(cap): secrets.token_hex(32) for cap in audit["capabilities"]},
            "note": "Private during the run; safe to reveal after the observational study is retired.",
        })
    return audit


@dataclass(frozen=True)
class TaskRecord:
    task_id: int
    taskset_id: int
    capability: int
    truth: int
    task_end: int
    reports: dict[int, int]

    @property
    def canonical_id(self) -> str:
        return f"netease:{self.capability}:{self.task_id}"

    @property
    def participants(self) -> tuple[int, ...]:
        return tuple(sorted(self.reports))


def iter_capability_tasks(conn: sqlite3.Connection, capability: int) -> Iterator[TaskRecord]:
    cur = conn.execute(
        """
        SELECT t.task_id,t.taskset_id,t.capability,t.truth,t.task_end,a.worker_id,a.answer
        FROM tasks t JOIN annotations a ON a.task_id=t.task_id
        WHERE t.capability=?
        ORDER BY t.task_end,t.task_id,a.worker_id
        """, (capability,)
    )
    current_id = None
    meta = None
    reports: dict[int, int] = {}
    for task_id, taskset_id, cap, truth, task_end, worker_id, answer in cur:
        task_id = int(task_id)
        if current_id is not None and task_id != current_id:
            assert meta is not None
            yield TaskRecord(*meta, reports=dict(reports))
            reports.clear()
        if task_id != current_id:
            current_id = task_id
            meta = (int(task_id), int(taskset_id), int(cap), int(truth), int(task_end))
        reports[int(worker_id)] = int(answer)
    if current_id is not None:
        assert meta is not None
        yield TaskRecord(*meta, reports=dict(reports))


def _config_hash(audit: Mapping[str, Any], capability: int) -> str:
    cap_stat = next(x for x in audit["capability_stats"] if int(x["capability"]) == capability)
    return _sha_obj({
        "study_id": STUDY_ID,
        "dataset": DATASET_NAME,
        "capability": capability,
        "capability_stat": cap_stat,
        "algorithm_config": CONFIG,
        "nominal_period": PERIOD,
        "task_event_time": audit["task_event_time"],
        "capability_specific_state": True,
    })


def load_selectors(root: Path, audit: Mapping[str, Any]) -> dict[int, HiddenCalibrationSchedule]:
    seeds_obj = json.loads((root / SEEDS_REL).read_text(encoding="utf-8"))
    seeds = seeds_obj["seeds"]
    selectors = {}
    for cap in audit["capabilities"]:
        cap = int(cap)
        selectors[cap] = HiddenCalibrationSchedule(
            seed=bytes.fromhex(str(seeds[str(cap)])),
            session_id=f"LIR-PPTD/P1C/NetEaseCrowd/capability/{cap}",
            selector_epoch=0,
            config_hash=_config_hash(audit, cap),
            period=PERIOD,
        )
    return selectors


def _as_float(value: Any) -> float:
    text = str(value)
    if "/" in text:
        a, b = text.split("/", 1)
        return float(int(a) / int(b))
    return float(text)


def _one_hot(label: int) -> tuple[int, int, int]:
    return tuple(1 if i == label else 0 for i in range(3))  # type: ignore[return-value]


def _float_lir_predict(task: TaskRecord, reputations: Mapping[int, float] | None) -> tuple[tuple[float, float, float], int]:
    ids = task.participants
    eps = _as_float(CONFIG["epsilon_c"])
    c0 = _as_float(CONFIG["c0"])
    tau = float(resolve_from_modality(
        "categorical", 3, CONFIG["lambda_tau"],
        str(CONFIG.get("consistency_scale_mode", DEFAULT_CONSISTENCY_SCALE_MODE)),
    ).resolved_tau)
    reps = {w: (c0 if reputations is None else float(reputations.get(w, c0))) for w in ids}
    denom = sum(reps[w] + eps for w in ids)
    truth = [
        sum((reps[w] + eps) * (1.0 if task.reports[w] == h else 0.0) for w in ids) / denom
        for h in range(3)
    ]
    for _ in range(int(CONFIG["K"])):
        distances = {
            w: sum(((1.0 if task.reports[w] == h else 0.0) - truth[h]) ** 2 for h in range(3))
            for w in ids
        }
        q = {w: tau / (tau + distances[w]) for w in ids}
        a = {w: (reps[w] + eps) * q[w] for w in ids}
        denom = sum(a.values())
        truth = [
            sum(a[w] * (1.0 if task.reports[w] == h else 0.0) for w in ids) / denom
            for h in range(3)
        ]
    winner = min(i for i, x in enumerate(truth) if x == max(truth))
    return (float(truth[0]), float(truth[1]), float(truth[2])), winner


def _float_calibrate(task: TaskRecord, reputations: dict[int, float], counts: dict[int, int]) -> dict[int, float]:
    c0 = _as_float(CONFIG["c0"])
    tau = float(resolve_from_modality(
        "categorical", 3, CONFIG["lambda_tau"],
        str(CONFIG.get("consistency_scale_mode", DEFAULT_CONSISTENCY_SCALE_MODE)),
    ).resolved_tau)
    eta = _as_float(CONFIG["eta"])
    kappa = _as_float(CONFIG["kappa"])
    q_out: dict[int, float] = {}
    truth = _one_hot(task.truth)
    for w, label in task.reports.items():
        report = _one_hot(label)
        d = sum((float(report[h]) - float(truth[h])) ** 2 for h in range(3))
        q = tau / (tau + d)
        c = float(reputations.get(w, c0))
        next_c = (1.0 - eta * kappa) * c + eta * q + eta * (kappa - 1.0) * c * q
        reputations[w] = next_c
        counts[w] = int(counts.get(w, 0)) + 1
        q_out[w] = q
    return q_out


@dataclass
class COWAHistory:
    sums: dict[int, float]
    counts: dict[int, int]

    def weight(self, worker: int) -> float:
        n = int(self.counts.get(worker, 0))
        return _as_float(CONFIG["c0"]) if n == 0 else self.sums[worker] / n

    def observe(self, evidence: Mapping[int, float]) -> None:
        for w, q in evidence.items():
            self.sums[w] = float(self.sums.get(w, 0.0)) + float(q)
            self.counts[w] = int(self.counts.get(w, 0)) + 1


def _cowa_predict(task: TaskRecord, history: COWAHistory) -> int:
    scores = [0.0, 0.0, 0.0]
    denom = 0.0
    for w, label in task.reports.items():
        weight = history.weight(w)
        scores[label] += weight
        denom += weight
    if denom <= 0:
        raise P1CError("NONPOSITIVE_COWA_DENOMINATOR")
    values = [x / denom for x in scores]
    return min(i for i, x in enumerate(values) if x == max(values))


def _mv_predict(task: TaskRecord) -> int:
    counts = [0, 0, 0]
    for label in task.reports.values():
        counts[label] += 1
    return min(i for i, x in enumerate(counts) if x == max(counts))


def _real_task(task: TaskRecord) -> RealTask:
    return RealTask(
        task_id=task.canonical_id,
        modality="categorical",
        participant_ids=tuple(str(w) for w in task.participants),
        reports={str(w): _one_hot(label) for w, label in task.reports.items()},
        truth_vector=_one_hot(task.truth),
        truth_class=task.truth,
        source_index=0,
    )


def exact_equivalence_gate(
    conn: sqlite3.Connection,
    capability: int,
    selector: HiddenCalibrationSchedule,
    *,
    max_tasks: int = 80,
) -> dict[str, Any]:
    workers = [str(x[0]) for x in conn.execute(
        """SELECT DISTINCT a.worker_id FROM annotations a JOIN tasks t ON a.task_id=t.task_id
           WHERE t.capability=? ORDER BY a.worker_id""", (capability,)
    )]
    exact_state = initial_global_state(workers, str(CONFIG["c0"]))
    float_rep: dict[int, float] = {}
    float_count: dict[int, int] = {}
    ordinary_checked = 0
    calibration_checked = 0
    max_output_error = 0.0
    max_reputation_error = 0.0

    for index, task in enumerate(iter_capability_tasks(conn, capability), start=1):
        if index > max_tasks:
            break
        real = _real_task(task)
        if selector.is_calibration_task(task.canonical_id):
            _, exact_state = run_calibration_task(real, CONFIG, exact_state, dps=80)
            _float_calibrate(task, float_rep, float_count)
            calibration_checked += 1
            for w in task.participants:
                err = abs(float(exact_state.reputations[str(w)]) - float(float_rep[w]))
                max_reputation_error = max(max_reputation_error, err)
        else:
            exact = run_lir_task(real, CONFIG, exact_state, ablation="full", dps=80)
            fvec, fcls = _float_lir_predict(task, float_rep)
            if int(exact.prediction_class) != fcls:
                raise P1CError(f"FLOAT_EXACT_CLASS_MISMATCH:{capability}:{task.task_id}")
            err = max(abs(float(a) - float(b)) for a, b in zip(exact.prediction_vector, fvec))
            max_output_error = max(max_output_error, err)
            ordinary_checked += 1

    # If the first chronological slice happens to contain no calibration event,
    # compare one selected calibration task from later in the stream from c0.
    if calibration_checked == 0:
        for task in iter_capability_tasks(conn, capability):
            if selector.is_calibration_task(task.canonical_id):
                real = _real_task(task)
                local_workers = [str(w) for w in task.participants]
                state = initial_global_state(local_workers, str(CONFIG["c0"]))
                _, state = run_calibration_task(real, CONFIG, state, dps=80)
                rep: dict[int, float] = {}
                cnt: dict[int, int] = {}
                _float_calibrate(task, rep, cnt)
                for w in task.participants:
                    max_reputation_error = max(
                        max_reputation_error,
                        abs(float(state.reputations[str(w)]) - float(rep[w]))
                    )
                calibration_checked = 1
                break

    if ordinary_checked == 0 or calibration_checked == 0:
        raise P1CError(f"EQUIVALENCE_INSUFFICIENT_CASES:{capability}")
    if max_output_error > 1e-10 or max_reputation_error > 1e-10:
        raise P1CError(
            f"FLOAT_EXACT_TOLERANCE:{capability}:{max_output_error}:{max_reputation_error}"
        )
    return {
        "capability": capability,
        "ordinary_tasks_checked": ordinary_checked,
        "calibration_tasks_checked": calibration_checked,
        "max_abs_output_error": max_output_error,
        "max_abs_reputation_error": max_reputation_error,
        "tolerance": 1e-10,
        "status": "PASS",
    }


class Metric:
    def __init__(self) -> None:
        self.n = 0
        self.correct = 0
        self.conf = [[0 for _ in range(3)] for _ in range(3)]

    def add(self, truth: int, pred: int) -> None:
        self.n += 1
        self.correct += int(truth == pred)
        self.conf[truth][pred] += 1

    def result(self) -> dict[str, Any]:
        if self.n == 0:
            return {"n": 0, "accuracy": None, "macro_f1": None}
        f1s = []
        for c in range(3):
            tp = self.conf[c][c]
            fp = sum(self.conf[t][c] for t in range(3) if t != c)
            fn = sum(self.conf[c][p] for p in range(3) if p != c)
            denom = 2 * tp + fp + fn
            f1s.append(0.0 if denom == 0 else 2 * tp / denom)
        return {
            "n": self.n,
            "accuracy": self.correct / self.n,
            "macro_f1": statistics.fmean(f1s),
        }


def _cold_bin(cold_fraction: float) -> str:
    if cold_fraction == 0.0:
        return "history_all"
    if cold_fraction < 0.5:
        return "minority_cold"
    if cold_fraction < 1.0:
        return "majority_cold"
    return "all_cold"


def _new_metric_map() -> dict[str, Metric]:
    return {m: Metric() for m in METHODS}


def _metric_rows(group: Mapping[Any, Mapping[str, Metric]], *, capability: int, key_name: str) -> list[dict[str, Any]]:
    rows = []
    for key, methods in sorted(group.items(), key=lambda kv: str(kv[0])):
        for method, metric in methods.items():
            rows.append({"capability": capability, key_name: key, "method": method, **metric.result()})
    return rows


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    seen = set()
    for row in rows:
        for k in row:
            if k not in seen:
                fields.append(k); seen.add(k)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)


def run_capability(
    root: Path,
    conn: sqlite3.Connection,
    audit: Mapping[str, Any],
    capability: int,
    selector: HiddenCalibrationSchedule,
    *,
    limit_tasks: int | None = None,
) -> dict[str, Any]:
    out = root / OUT_REL
    out.mkdir(parents=True, exist_ok=True)
    cap_stat = next(x for x in audit["capability_stats"] if int(x["capability"]) == capability)
    cap_start = int(cap_stat["start_ms"])

    equivalence = exact_equivalence_gate(conn, capability, selector, max_tasks=80)

    full_rep: dict[int, float] = {}
    full_cal_count: dict[int, int] = {}
    reset_rep: dict[int, float] = {}
    reset_cal_count: dict[int, int] = {}
    cowa = COWAHistory(sums={}, counts={})
    last_seen: dict[int, int] = {}
    worker_first: dict[int, int] = {}
    worker_last: dict[int, int] = {}
    worker_tasks: dict[int, int] = {}
    worker_reentry30: dict[int, int] = {}

    overall = _new_metric_map()
    time_groups: dict[int, dict[str, Metric]] = {}
    cold_groups: dict[str, dict[str, Metric]] = {}
    reentry_groups: dict[str, dict[str, Metric]] = {}

    cal_count = 0
    scored_count = 0
    reset_events = 0
    reentry_task_count = 0
    tasks_seen = 0

    pred_path = out / f"predictions_cap_{capability}.csv.gz"
    tmp_path = pred_path.with_suffix(pred_path.suffix + ".tmp")
    with gzip.open(tmp_path, "wt", encoding="utf-8", newline="") as gz:
        writer = csv.DictWriter(gz, fieldnames=[
            "task_id","taskset_id","task_end","truth","time_bin","cold_fraction","cold_bin","reentry30",
            "lir_pptd","cowa","no_hr","mv","lir_reset30d"
        ])
        writer.writeheader()

        for task in iter_capability_tasks(conn, capability):
            tasks_seen += 1
            if limit_tasks is not None and tasks_seen > limit_tasks:
                break

            participants = task.participants
            cold_fraction = sum(int(full_cal_count.get(w, 0)) == 0 for w in participants) / len(participants)
            cold = _cold_bin(cold_fraction)
            reentry_workers = tuple(
                w for w in participants
                if w in last_seen and task.task_end - last_seen[w] >= REENTRY_GAP_MS
            )
            reentry = bool(reentry_workers)
            if reentry:
                reentry_task_count += 1
                for w in reentry_workers:
                    worker_reentry30[w] = worker_reentry30.get(w, 0) + 1
                    reset_rep[w] = _as_float(CONFIG["c0"])
                    reset_cal_count[w] = 0
                    reset_events += 1

            for w in participants:
                worker_first.setdefault(w, task.task_end)
                worker_last[w] = task.task_end
                worker_tasks[w] = worker_tasks.get(w, 0) + 1

            is_cal = selector.is_calibration_task(task.canonical_id)
            if is_cal:
                evidence = _float_calibrate(task, full_rep, full_cal_count)
                _float_calibrate(task, reset_rep, reset_cal_count)
                cowa.observe(evidence)
                cal_count += 1
            else:
                _, p_full = _float_lir_predict(task, full_rep)
                p_cowa = _cowa_predict(task, cowa)
                _, p_nohr = _float_lir_predict(task, None)
                p_mv = _mv_predict(task)
                _, p_reset = _float_lir_predict(task, reset_rep)
                preds = {
                    "lir_pptd": p_full,
                    "cowa": p_cowa,
                    "no_hr": p_nohr,
                    "mv": p_mv,
                    "lir_reset30d": p_reset,
                }
                scored_count += 1
                time_bin = int((task.task_end - cap_start) // TIME_BIN_MS)
                time_groups.setdefault(time_bin, _new_metric_map())
                cold_groups.setdefault(cold, _new_metric_map())
                reentry_key = "reentry30" if reentry else "non_reentry"
                reentry_groups.setdefault(reentry_key, _new_metric_map())
                for method, pred in preds.items():
                    overall[method].add(task.truth, pred)
                    time_groups[time_bin][method].add(task.truth, pred)
                    cold_groups[cold][method].add(task.truth, pred)
                    reentry_groups[reentry_key][method].add(task.truth, pred)
                writer.writerow({
                    "task_id": task.task_id,
                    "taskset_id": task.taskset_id,
                    "task_end": task.task_end,
                    "truth": task.truth,
                    "time_bin": time_bin,
                    "cold_fraction": cold_fraction,
                    "cold_bin": cold,
                    "reentry30": int(reentry),
                    **preds,
                })

            for w in participants:
                last_seen[w] = task.task_end

            if tasks_seen % 50_000 == 0:
                print(
                    f"CAPABILITY={capability} PROGRESS={tasks_seen}/{cap_stat['tasks']} "
                    f"CAL={cal_count} SCORED={scored_count}",
                    flush=True,
                )

    os.replace(tmp_path, pred_path)

    overall_rows = [
        {"capability": capability, "method": method, **metric.result()}
        for method, metric in overall.items()
    ]
    time_rows = _metric_rows(time_groups, capability=capability, key_name="time_bin_30d")
    cold_rows = _metric_rows(cold_groups, capability=capability, key_name="cold_bin")
    reentry_rows = _metric_rows(reentry_groups, capability=capability, key_name="reentry_group")
    lifecycle_rows = [
        {
            "capability": capability,
            "worker_id": w,
            "first_task_end_ms": worker_first[w],
            "last_task_end_ms": worker_last[w],
            "active_span_days": (worker_last[w] - worker_first[w]) / (24*60*60*1000),
            "task_count": worker_tasks[w],
            "hidden_calibration_exposures": full_cal_count.get(w, 0),
            "reentry30_count": worker_reentry30.get(w, 0),
            "final_reputation": full_rep.get(w, _as_float(CONFIG["c0"])),
        }
        for w in sorted(worker_tasks)
    ]

    _write_csv(out / f"capability_{capability}_overall.csv", overall_rows)
    _write_csv(out / f"capability_{capability}_time_windows.csv", time_rows)
    _write_csv(out / f"capability_{capability}_cold_start.csv", cold_rows)
    _write_csv(out / f"capability_{capability}_reentry.csv", reentry_rows)
    _write_csv(out / f"capability_{capability}_worker_lifecycle.csv", lifecycle_rows)

    summary = {
        "schema_version": "1.0",
        "study_id": STUDY_ID,
        "capability": capability,
        "status": "success",
        "tasks_seen": tasks_seen if limit_tasks is None else min(tasks_seen, limit_tasks),
        "hidden_calibration_tasks": cal_count,
        "scored_tasks": scored_count,
        "realized_calibration_rate": cal_count / max(1, (cal_count + scored_count)),
        "workers_seen": len(worker_tasks),
        "reentry30_tasks": reentry_task_count,
        "reentry30_worker_events": sum(worker_reentry30.values()),
        "reset30_events": reset_events,
        "equivalence_gate": equivalence,
        "overall": {r["method"]: {"n": r["n"], "accuracy": r["accuracy"], "macro_f1": r["macro_f1"]} for r in overall_rows},
        "prediction_artifact": str(pred_path.relative_to(root)),
        "limit_tasks": limit_tasks,
    }
    _atomic_json(out / f"capability_{capability}_summary.json", summary)
    return summary


def _merge_small_outputs(root: Path, capabilities: Sequence[int]) -> dict[str, Any]:
    out = root / OUT_REL
    overall: list[dict[str, Any]] = []
    time_rows: list[dict[str, Any]] = []
    cold_rows: list[dict[str, Any]] = []
    reentry_rows: list[dict[str, Any]] = []
    lifecycle: list[dict[str, Any]] = []
    summaries = []
    for cap in capabilities:
        summaries.append(json.loads((out / f"capability_{cap}_summary.json").read_text(encoding="utf-8")))
        for name, dest in [
            (f"capability_{cap}_overall.csv", overall),
            (f"capability_{cap}_time_windows.csv", time_rows),
            (f"capability_{cap}_cold_start.csv", cold_rows),
            (f"capability_{cap}_reentry.csv", reentry_rows),
            (f"capability_{cap}_worker_lifecycle.csv", lifecycle),
        ]:
            with (out / name).open("r", encoding="utf-8-sig", newline="") as f:
                dest.extend(dict(r) for r in csv.DictReader(f))
    _write_csv(out / "capability_metrics.csv", overall)
    _write_csv(out / "time_window_metrics.csv", time_rows)
    _write_csv(out / "cold_start_metrics.csv", cold_rows)
    _write_csv(out / "reentry_metrics.csv", reentry_rows)
    _write_csv(out / "worker_lifecycle.csv", lifecycle)

    # Macro-average capabilities equally, preserving the multiple-skill nature of the dataset.
    macro = {}
    for method in METHODS:
        rows = [r for r in overall if r["method"] == method]
        macro[method] = {
            "capabilities": len(rows),
            "macro_accuracy": statistics.fmean(float(r["accuracy"]) for r in rows),
            "macro_macro_f1": statistics.fmean(float(r["macro_f1"]) for r in rows),
        }
    return {"capability_summaries": summaries, "macro_across_capabilities": macro}


def validate(root: Path) -> dict[str, Any]:
    db = root / DB_REL
    audit_path = root / AUDIT_REL
    seeds = root / SEEDS_REL
    hidden = root / "src/lir_pptd/core/hidden_calibration.py"
    for path in (db, audit_path, seeds, hidden):
        if not path.is_file():
            raise P1CError(f"MISSING:{path}")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if int(audit["annotations"]) != EXPECTED_ANNOTATIONS:
        raise P1CError("AUDIT_ANNOTATION_COUNT")
    if int(audit["tasks"]) != EXPECTED_TASKS:
        raise P1CError("AUDIT_TASK_COUNT")
    if int(audit["workers"]) != EXPECTED_WORKERS:
        raise P1CError("AUDIT_WORKER_COUNT")
    if int(audit["capability_count"]) != EXPECTED_CAPABILITIES:
        raise P1CError("AUDIT_CAPABILITY_COUNT")
    if list(map(int, audit["labels"])) != list(EXPECTED_LABELS):
        raise P1CError("AUDIT_LABEL_SET")
    selectors = load_selectors(root, audit)
    commitments = {
        str(cap): {
            "seed_commitment_sha256": selector.seed_commitment_sha256,
            "config_hash": selector.config_hash,
            "nominal_probability": 1 / PERIOD,
        }
        for cap, selector in selectors.items()
    }
    print("P1C_NETEASE_VALIDATE=PASS")
    print(f"ANNOTATIONS={audit['annotations']}")
    print(f"TASKS={audit['tasks']}")
    print(f"WORKERS={audit['workers']}")
    print(f"CAPABILITIES={','.join(map(str,audit['capabilities']))}")
    print("TASK_EVENT_TIME=MAX_COMPLETE_TIME_OF_OBSERVED_BATCH")
    print("PRIMARY_STREAMS=CAPABILITY_SPECIFIC_CHRONOLOGICAL")
    print("PRIMARY_METHODS=lir_pptd,cowa,no_hr,mv,lir_reset30d")
    print("HIDDEN_CALIBRATION_PI=0.05")
    print("ORDINARY_PERSISTENT_UPDATE=DISABLED")
    print("CALIBRATION_PERSISTENT_UPDATE=ENABLED")
    print("COLD_START_AXIS=PRIOR_AUTHENTICATED_CALIBRATION_EXPOSURE")
    print(f"NATURAL_REENTRY_GAP_DAYS={REENTRY_GAP_DAYS}")
    print("CHURN_CONTROL=LIR_RESET_REPUTATION_AFTER_30D_INACTIVITY")
    print("ALGORITHM_RETUNING=NO")
    print("RESULT_DEPENDENT_SELECTION=NO")
    return {"audit": audit, "selectors": selectors, "commitments": commitments}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--repo-root", default=".")
    p.add_argument("--download", action="store_true")
    p.add_argument("--prepare-only", action="store_true")
    p.add_argument("--force-rebuild", action="store_true")
    p.add_argument("--validate-only", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--probe-only", action="store_true")
    p.add_argument("--probe-tasks-per-capability", type=int, default=5000)
    args = p.parse_args()
    root = Path(args.repo_root).resolve()

    if args.download:
        download_raw_parts(root)
    if args.prepare_only or args.force_rebuild:
        audit = prepare_dataset(root, force=args.force_rebuild)
        print("P1C_NETEASE_PREPARE=PASS")
        print(f"ANNOTATIONS={audit['annotations']}")
        print(f"TASKS={audit['tasks']}")
        print(f"WORKERS={audit['workers']}")
        print(f"CAPABILITY_COUNT={audit['capability_count']}")
        print(f"AUDIT={root / AUDIT_REL}")
        if args.prepare_only:
            return 0

    ctx = validate(root)
    if args.validate_only:
        print("FORMAL_RUNS_STARTED=NO")
        return 0

    out = root / OUT_REL
    out.mkdir(parents=True, exist_ok=True)
    capabilities = [int(x) for x in ctx["audit"]["capabilities"]]
    limit = args.probe_tasks_per_capability if args.probe_only else None
    started = time.time()
    summaries = []
    for cap in capabilities:
        cap_summary = out / f"capability_{cap}_summary.json"
        if args.resume and not args.probe_only and cap_summary.is_file():
            old = json.loads(cap_summary.read_text(encoding="utf-8"))
            if old.get("status") == "success" and old.get("limit_tasks") is None:
                print(f"CAPABILITY={cap} RESUME=SKIP_COMPLETE")
                summaries.append(old)
                continue
        conn = _connect(root / DB_REL)
        try:
            summary = run_capability(
                root, conn, ctx["audit"], cap, ctx["selectors"][cap], limit_tasks=limit
            )
        finally:
            conn.close()
        summaries.append(summary)
        print(
            f"CAPABILITY={cap} COMPLETE SCORED={summary['scored_tasks']} "
            f"CAL={summary['hidden_calibration_tasks']} REENTRY30={summary['reentry30_tasks']}",
            flush=True,
        )

    if args.probe_only:
        probe = {
            "schema_version": "1.0",
            "study_id": STUDY_ID,
            "probe_only": True,
            "tasks_per_capability_limit": limit,
            "capabilities": capabilities,
            "summaries": summaries,
            "elapsed_seconds": time.time() - started,
        }
        _atomic_json(out / "probe_summary.json", probe)
        print("P1C_NETEASE_PROBE=COMPLETE")
        print(f"PROBE_SUMMARY={out / 'probe_summary.json'}")
        return 0

    merged = _merge_small_outputs(root, capabilities)
    # Reveal selector seeds only after the observational formal study is complete.
    private_seeds = json.loads((root / SEEDS_REL).read_text(encoding="utf-8"))
    _atomic_json(out / "retired_selector_reveal.json", private_seeds)
    selector_audit = {
        "schema_version": "1.0",
        "study_id": STUDY_ID,
        "observational_reports_preexist_selector": True,
        "report_generation_depends_on_selector": False,
        "nominal_probability": 1 / PERIOD,
        "capabilities": ctx["commitments"],
        "retired_seed_reveal": "retired_selector_reveal.json",
    }
    _atomic_json(out / "selector_audit.json", selector_audit)
    study_audit = {
        "schema_version": "1.0",
        "study_id": STUDY_ID,
        "dataset": DATASET_NAME,
        "dataset_audit": str(AUDIT_REL),
        "official_cardinality_gate": {
            "annotations": EXPECTED_ANNOTATIONS,
            "tasks": EXPECTED_TASKS,
            "workers": EXPECTED_WORKERS,
            "capabilities": EXPECTED_CAPABILITIES,
        },
        "chronology": {
            "task_event_time": "max completeTime of all observed annotations for the task",
            "tie_break": "taskId ascending",
            "future_annotation_leakage": False,
        },
        "skill_handling": {
            "primary_streams": "one chronological stream per capability",
            "reason": "avoid conflating capability-specific expertise with a scalar global reliability state",
        },
        "hidden_calibration": {
            "period": PERIOD,
            "probability": 1 / PERIOD,
            "task_id_based": True,
            "ordinary_persistent_updates": False,
        },
        "cold_start": "fraction of current participants with zero prior authenticated calibration exposures",
        "natural_reentry": f"same worker ID reappears after >= {REENTRY_GAP_DAYS} days of inactivity",
        "churn_control": f"LIR-reset30d resets that returning worker's persistent state to c0 after >= {REENTRY_GAP_DAYS} days",
        "identity_replacement_claimed": False,
        "algorithm_retuning": False,
        "result_dependent_selection": False,
        "inference_role": "longitudinal realism/mechanism study; not a new adversarial superiority gate",
    }
    _atomic_json(out / "study_audit.json", study_audit)
    summary = {
        "schema_version": "1.0",
        "study_id": STUDY_ID,
        "status": "COMPLETE",
        "dataset": DATASET_NAME,
        "annotations": int(ctx["audit"]["annotations"]),
        "tasks": int(ctx["audit"]["tasks"]),
        "workers": int(ctx["audit"]["workers"]),
        "capabilities": capabilities,
        "capability_count": len(capabilities),
        "methods": list(METHODS),
        "capability_summaries": merged["capability_summaries"],
        "macro_across_capabilities": merged["macro_across_capabilities"],
        "elapsed_seconds": time.time() - started,
        "scientific_decision": "REVIEW_AFTER_RESULT",
        "algorithm_retuning": False,
    }
    _atomic_json(out / "formal_summary.json", summary)
    print("P1C_NETEASE_LONGITUDINAL=COMPLETE")
    print(f"CAPABILITY_STREAMS={len(capabilities)}")
    print(f"TOTAL_TASKS={summary['tasks']}")
    print(f"TOTAL_ANNOTATIONS={summary['annotations']}")
    print("ALGORITHM_RETUNING=NO")
    print("P1C_SCIENTIFIC_DECISION=REVIEW_AFTER_RESULT")
    print(f"SUMMARY={out / 'formal_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
