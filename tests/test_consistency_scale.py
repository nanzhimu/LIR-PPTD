from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import pytest

from lir_pptd.core.consistency_scale import (
    DEFAULT_CONSISTENCY_SCALE_MODE,
    GEOMETRY_DIAMETER_SCALING,
    LEGACY_D_SCALING,
    resolve_consistency_scale,
    squared_diameter,
)
from lir_pptd.core.exact import ExactTaskFailure, run_exact_once
from lir_pptd.experiments.phase6_exact_adapter import LongitudinalState, build_exact_task_input
from lir_pptd.experiments.phase6_generator import generate_stream
from lir_pptd.experiments.phase_r1.real_data_loader import load_real_dataset, task_artifact
from lir_pptd.fixedpoint import FixedTaskFailure, default_profile, run_fixed

ROOT = Path(__file__).parents[1]
LAMBDA = Fraction(1, 5)


def _cfg(mode: str) -> dict[str, object]:
    return {
        "K": 10,
        "c0": "1/2",
        "epsilon_c": "1/1024",
        "lambda_tau": "1/5",
        "kappa": "2",
        "eta": "1/10",
        "consistency_scale_mode": mode,
    }


def _state_for(task: dict[str, object]) -> LongitudinalState:
    ids = tuple(task["participant_ids"])
    return LongitudinalState(
        reputations={w: "1/2" for w in ids},
        epochs={w: 0 for w in ids},
    )



def test_default_mode_is_geometry() -> None:
    assert DEFAULT_CONSISTENCY_SCALE_MODE == GEOMETRY_DIAMETER_SCALING
    resolved = resolve_consistency_scale("categorical_one_hot", 4, LAMBDA)
    assert resolved.resolved_tau == Fraction(2, 5)


def test_G1_numerical_tau_old_equals_geo() -> None:
    for D in (1, 2, 5, 10, 16, 32):
        old = resolve_consistency_scale("numerical_box", D, LAMBDA, LEGACY_D_SCALING)
        geo = resolve_consistency_scale("numerical_box", D, LAMBDA, GEOMETRY_DIAMETER_SCALING)
        assert old.resolved_tau == geo.resolved_tau == LAMBDA * D


def test_G2_binary_one_hot_backward_compatible() -> None:
    old = resolve_consistency_scale("categorical_one_hot", 2, LAMBDA, LEGACY_D_SCALING)
    geo = resolve_consistency_scale("categorical_one_hot", 2, LAMBDA, GEOMETRY_DIAMETER_SCALING)
    assert old.resolved_tau == geo.resolved_tau == Fraction(2, 5)


def test_G3_one_hot_geometry_tau_is_two_lambda() -> None:
    for D in (2, 4, 8, 16, 32, 64):
        geo = resolve_consistency_scale("categorical_one_hot", D, LAMBDA, GEOMETRY_DIAMETER_SCALING)
        assert geo.resolved_tau == 2 * LAMBDA
        assert geo.squared_diameter == 2


def test_G4_one_hot_vertex_squared_distance_is_two() -> None:
    for D in (2, 4, 8, 16, 32, 64):
        for a, b in ((0, 1), (0, D - 1)):
            ea = [1 if i == a else 0 for i in range(D)]
            eb = [1 if i == b else 0 for i in range(D)]
            d = sum((x - y) ** 2 for x, y in zip(ea, eb))
            assert d == 2
        assert squared_diameter("categorical_one_hot", D) == 2


def test_G5_geometry_q_wrong_is_dimension_invariant() -> None:
    expected = Fraction(1, 6)
    values = []
    for D in (2, 4, 8, 16, 32, 64):
        tau = resolve_consistency_scale(
            "categorical_one_hot", D, LAMBDA, GEOMETRY_DIAMETER_SCALING
        ).resolved_tau
        q_wrong = tau / (tau + 2)
        values.append(q_wrong)
        assert q_wrong == expected
    assert len(set(values)) == 1


def test_G6_numerical_exact_and_fixed_are_mode_equivalent() -> None:
    profile = default_profile()
    for seed in (1001, 1002, 1003, 1004, 1005):
        stream = generate_stream(ROOT, "validation", seed, "synthetic_num_small")
        task = stream["tasks"][0]
        state = _state_for(task)
        old_input = build_exact_task_input(task, _cfg(LEGACY_D_SCALING), state)
        geo_input = build_exact_task_input(task, _cfg(GEOMETRY_DIAMETER_SCALING), state)
        assert old_input.tau == geo_input.tau
        old = run_exact_once(old_input, dps=80)
        geo = run_exact_once(geo_input, dps=80)
        assert not isinstance(old, ExactTaskFailure)
        assert not isinstance(geo, ExactTaskFailure)
        assert old.final_output == geo.final_output
        assert old.next_reputation == geo.next_reputation
        assert tuple((x.worker_id, x.evidence_out) for x in old.output_evidence) == tuple(
            (x.worker_id, x.evidence_out) for x in geo.output_evidence
        )
        old_f = run_fixed(old_input, profile)
        geo_f = run_fixed(geo_input, profile)
        assert not isinstance(old_f, FixedTaskFailure)
        assert not isinstance(geo_f, FixedTaskFailure)
        assert old_f.final_output_integer == geo_f.final_output_integer
        assert old_f.reputation_transitions == geo_f.reputation_transitions


def test_G7_binary_real_data_exact_mode_equivalence() -> None:
    required = [ROOT / "data" / ds / name for ds in ("product", "duck") for name in ("answer.csv", "truth.csv")]
    if not all(p.is_file() for p in required):
        pytest.skip("Product/Duck raw CSVs are intentionally not redistributed; populate data/product and data/duck to run G7. Frozen SHA-256 values are in data/phase_r1_dataset_manifest.json.")
    for dataset_id in ("product", "duck"):
        dataset = load_real_dataset(ROOT / "data", dataset_id)
        for task in dataset.tasks[:5]:
            artifact = task_artifact(task)
            state = LongitudinalState(
                reputations={w: "1/2" for w in dataset.worker_ids},
                epochs={w: 0 for w in dataset.worker_ids},
            )
            old_input = build_exact_task_input(artifact, _cfg(LEGACY_D_SCALING), state)
            geo_input = build_exact_task_input(artifact, _cfg(GEOMETRY_DIAMETER_SCALING), state)
            assert old_input.tau == geo_input.tau == "2/5"
            old = run_exact_once(old_input, dps=80)
            geo = run_exact_once(geo_input, dps=80)
            assert not isinstance(old, ExactTaskFailure)
            assert not isinstance(geo, ExactTaskFailure)
            assert old.final_output == geo.final_output
            assert old.released_class_index == geo.released_class_index
            assert old.next_reputation == geo.next_reputation
