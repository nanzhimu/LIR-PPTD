from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]

def test_design_pins_version_and_threshold_mapping():
    d = json.loads((ROOT / "configs/p1d/p1d_e2e_design.json").read_text(encoding="utf-8"))
    assert d["runtime"]["version"] == "v0.4.3"
    assert d["runtime"]["paper_default_committee"] == {
        "N": 10, "reconstruction_threshold": 4}
    assert d["runtime"]["mpspdz_default_runtime"] == {
        "N": 10, "max_corrupted_T": 3}

def test_real_e2e_does_not_use_simulated_backend():
    t = (ROOT / "P1D_E2E_FROZEN_DESIGN.md").read_text(encoding="utf-8")
    assert "SimulatedShamirBackend" in t
    assert "cannot support a claim of real networked MPC execution" in t

def test_mpc_program_has_ordinary_and_calibration_branches():
    t = (ROOT / "p1d_mpspdz/Programs/Source/lir_pptd_e2e.mpc").read_text(encoding="utf-8")
    assert "mode == 0" in t and "mode == 1" in t
    assert "q_out" in t and "qcal" in t
    assert "sfix.receive_from_client" in t
    assert "sfix.reveal_to_clients" in t

def test_ordinary_has_no_persistent_reputation_transition():
    t = (ROOT / "p1d_mpspdz/Programs/Source/lir_pptd_e2e.mpc").read_text(encoding="utf-8")
    ordinary = t.split("if mode == 0:", 1)[1].split("elif mode == 1:", 1)[0]
    assert "candidates" not in ordinary
    assert "nxt =" not in ordinary

def test_fixed_precision_is_f24():
    t = (ROOT / "p1d_mpspdz/Programs/Source/lir_pptd_e2e.mpc").read_text(encoding="utf-8")
    assert "sfix.set_precision(24, 56)" in t
    assert "sfix.round_nearest = True" in t


def test_claim_boundaries_present():
    t = (ROOT / "P1D_E2E_FROZEN_DESIGN.md").read_text(encoding="utf-8")
    for x in (
        "production-ready MPC",
        "Byzantine fault tolerance",
        "bit-exact equivalence",
        "faster than FPTD"
    ):
        assert x in t

def test_workload_reference_non_degenerate():
    sys.path.insert(0, str(ROOT / "src"))
    from lir_pptd.experiments.p1d_e2e.reference import deterministic_workload
    r = deterministic_workload(20, 0)
    assert 0 < min(r["reports"]) < max(r["reports"]) < 1
    assert 0 < min(r["reputations"]) < max(r["reputations"]) < 1
    assert 0 < r["expected_truth"] < 1
    assert 0 < r["expected_calibration_mean_reputation"] < 1

def test_externalio_private_inputs_match_mpc_receive_boundaries():
    t = (ROOT / "p1d_mpspdz/ExternalIO/p1d_lir_client.py").read_text(
        encoding="utf-8")
    assert "c.send_private_inputs(report_payload)" in t
    assert "c.send_private_inputs(reputation_payload)" in t
    assert "c.send_private_inputs(reference_payload)" in t
    assert "private_input_batches" in t
