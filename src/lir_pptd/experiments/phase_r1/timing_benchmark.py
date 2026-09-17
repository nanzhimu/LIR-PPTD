from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

from .attacks import apply_fixed_target, select_malicious_workers
from .contract import DATASET_CONTRACTS
from .longitudinal_state import initial_global_state
from .methods import run_lir_task, run_stateless_baseline
from .real_data_loader import RealDataset, RealTask, load_real_dataset

DATASETS = ('product', 'duck', 'dog', 'weather')
METHODS = ('mean_vote', 'crh', 'lir_pptd_full')
RHO = '1/2'
MALICIOUS_SEED = 5001
CONFIG_ID = 'lir_cfg_05_lambda020'
PILOT_TASK_LIMIT = 100
CORE_STREAM_COUNTS = {'product': 151, 'duck': 151, 'dog': 151, 'weather': 109}


class TimingBenchmarkError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def _load_config(path: Path) -> dict[str, Any]:
    obj = json.loads(path.read_text(encoding='utf-8-sig'))
    for config in obj.get('candidate_configs', []):
        if config.get('config_id') == CONFIG_ID:
            return dict(config)
    raise TimingBenchmarkError(f'config not found: {CONFIG_ID}')


def _attacked_task(task: RealTask, malicious_workers: tuple[str, ...]) -> RealTask:
    reports = apply_fixed_target(task.reports, malicious_workers, modality=task.modality)
    return replace(task, reports=reports)


def _assert_dataset_contract(dataset: RealDataset) -> None:
    contract = DATASET_CONTRACTS[dataset.dataset_id]
    if dataset.modality != contract.modality:
        raise TimingBenchmarkError(
            f'{dataset.dataset_id} modality mismatch: expected {contract.modality}, got {dataset.modality}'
        )
    if dataset.class_count != contract.class_count:
        raise TimingBenchmarkError(
            f'{dataset.dataset_id} class_count mismatch: expected {contract.class_count}, got {dataset.class_count}'
        )


def _run_method(
    dataset: RealDataset,
    attacked_tasks: tuple[RealTask, ...],
    method: str,
    candidate_config: Mapping[str, Any],
) -> dict[str, Any]:
    started = time.perf_counter()
    prediction_count = 0

    if method in {'mean_vote', 'crh'}:
        for task in attacked_tasks:
            run_stateless_baseline(task, method)
            prediction_count += 1
    elif method == 'lir_pptd_full':
        state = initial_global_state(dataset.worker_ids, candidate_config['c0'])
        for task in attacked_tasks:
            result = run_lir_task(task, candidate_config, state, ablation='full', dps=80)
            state = result.next_state
            prediction_count += 1
    else:
        raise TimingBenchmarkError(f'unsupported timing method: {method}')

    elapsed = time.perf_counter() - started
    return {
        'method': method,
        'task_count': prediction_count,
        'elapsed_seconds': elapsed,
        'seconds_per_task': elapsed / prediction_count if prediction_count else None,
    }


def benchmark(
    repo_root: Path,
    *,
    datasets: tuple[str, ...] = DATASETS,
    methods: tuple[str, ...] = METHODS,
) -> dict[str, Any]:
    data_root = repo_root / 'data'
    config_path = repo_root / 'docs' / 'PHASE6_R2_METHOD_CONFIG_DECISIONS.json'
    if not config_path.is_file():
        raise TimingBenchmarkError(f'missing config artifact: {config_path}')
    candidate_config = _load_config(config_path)

    records: list[dict[str, Any]] = []
    dataset_records: list[dict[str, Any]] = []
    wall_started = time.perf_counter()

    for dataset_id in datasets:
        dataset = load_real_dataset(data_root, dataset_id)  # hashes checked by future frozen manifest gate
        _assert_dataset_contract(dataset)
        malicious = select_malicious_workers(
            dataset.worker_ids,
            dataset_id=dataset_id,
            rho=RHO,
            seed=MALICIOUS_SEED,
        )
        all_attacked_tasks = tuple(_attacked_task(task, malicious) for task in dataset.tasks)
        attacked_tasks = all_attacked_tasks[: min(PILOT_TASK_LIMIT, len(all_attacked_tasks))]
        dataset_records.append({
            'dataset_id': dataset_id,
            'modality': dataset.modality,
            'class_count': dataset.class_count,
            'task_count': len(dataset.tasks),
            'pilot_task_count': len(attacked_tasks),
            'planned_core_stream_count': CORE_STREAM_COUNTS[dataset_id],
            'worker_count': len(dataset.worker_ids),
            'malicious_worker_count': len(malicious),
            'rho': RHO,
            'malicious_seed': MALICIOUS_SEED,
        })
        for method in methods:
            rec = _run_method(dataset, attacked_tasks, method, candidate_config)
            rec['dataset_id'] = dataset_id
            rec['modality'] = dataset.modality
            rec['full_dataset_task_count'] = len(dataset.tasks)
            rec['projected_seconds_per_full_stream'] = rec['seconds_per_task'] * len(dataset.tasks)
            rec['planned_core_stream_count'] = CORE_STREAM_COUNTS[dataset_id]
            rec['projected_core_seconds_for_method_dataset'] = rec['projected_seconds_per_full_stream'] * CORE_STREAM_COUNTS[dataset_id]
            records.append(rec)

    wall_elapsed = time.perf_counter() - wall_started
    return {
        'schema_version': '1.0-draft',
        'phase': 'Phase-R1',
        'gate': 'R1-G5',
        'benchmark_type': 'timing_only_non_inferential',
        'formal_experiment': False,
        'quality_metrics_computed': False,
        'dataset_contract': {
            'product': 'categorical_binary',
            'duck': 'categorical_binary',
            'dog': 'categorical_4class',
            'weather': 'numerical',
        },
        'rho': RHO,
        'malicious_seed': MALICIOUS_SEED,
        'config_id': CONFIG_ID,
        'config_artifact_sha256': _sha256(config_path),
        'datasets': dataset_records,
        'timings': records,
        'pilot_task_limit': PILOT_TASK_LIMIT,
        'wall_elapsed_seconds': wall_elapsed,
        'projected_core_total_seconds': sum(r['projected_core_seconds_for_method_dataset'] for r in records),
        'projected_core_total_hours': sum(r['projected_core_seconds_for_method_dataset'] for r in records) / 3600.0,
        'prohibitions': [
            'do_not_use_timing_run_for_method-quality claims',
            'do_not_change endpoints or methods based on hidden prediction quality',
            'do_not treat this benchmark as Phase-R1 empirical evidence',
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description='Phase-R1 G5 timing-only benchmark')
    parser.add_argument('--repo-root', default='.')
    parser.add_argument('--output', default='results/summary/phase_r1_g5_timing_benchmark.json')
    args = parser.parse_args()
    root = Path(args.repo_root).resolve()
    result = benchmark(root)
    out = root / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        raise TimingBenchmarkError(f'refusing to overwrite existing timing artifact: {out}')
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'R1_G5_TIMING_BENCHMARK=PASS')
    print(f'OUTPUT={out}')
    print(f'WALL_SECONDS={result["wall_elapsed_seconds"]:.6f}')
    for rec in result['timings']:
        print(
            f"TIMING dataset={rec['dataset_id']} modality={rec['modality']} method={rec['method']} "
            f"tasks={rec['task_count']} seconds={rec['elapsed_seconds']:.6f} "
            f"seconds_per_task={rec['seconds_per_task']:.9f}"
        )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
