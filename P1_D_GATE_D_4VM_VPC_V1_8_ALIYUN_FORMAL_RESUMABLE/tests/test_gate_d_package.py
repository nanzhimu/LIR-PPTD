from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def test_topology_is_4_3_3_plus_requester():
    t=(ROOT/"GATE_D_SCIENTIFIC_PROTOCOL.md").read_text(encoding="utf-8")
    assert "4+3+3" in t
    assert "one separate requester VM" in t

def test_batch_size_frozen():
    e=(ROOT/"config/gate_d_hosts.example.env").read_text(encoding="utf-8")
    p=(ROOT/"GATE_D_SCIENTIFIC_PROTOCOL.md").read_text(encoding="utf-8")
    assert "P1D_BATCH_SIZE=101" in e
    assert "batch size 101" in p

def test_no_ten_physical_host_overclaim():
    t=(ROOT/"README_GATE_D_CLOUD.md").read_text(encoding="utf-8")
    assert "Do not call" in t
    assert "ten physical fog servers" in t

def test_private_key_isolation_script():
    t=(ROOT/"scripts/distribute_formal_ssl.sh").read_text(encoding="utf-8")
    assert "P0-P3" in t
    assert "P4-P6" in t
    assert "P7-P9" in t
    assert "C0.key" in t

def test_formal_matrix_is_200_total_160_measured():
    t=(ROOT/"README_GATE_D_CLOUD.md").read_text(encoding="utf-8")
    assert "200 total executions, 160 measured" in t

def test_smoke_rejects_unused_triples_warning():
    t=(ROOT/"orchestrator/run_gate_d.py").read_text(encoding="utf-8")
    assert '"unused_triples_warning"' in t
    assert "P1D_GATE_D_SMOKE=REVIEW_REQUIRED" in t
