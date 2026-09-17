from pathlib import Path
import importlib.util, json, tempfile

ROOT=Path(__file__).resolve().parents[1]
RUNNER=ROOT/"orchestrator/run_gate_d.py"

def load():
    spec=importlib.util.spec_from_file_location("v18",RUNNER)
    mod=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def text():
    return RUNNER.read_text(encoding="utf-8")

def test_import():
    load()

def test_final_batch101_guard():
    t=text()
    assert "configured_batch != 101" in t
    assert "expected=101" in t

def test_hygiene_proof_required_for_formal():
    t=text()
    assert "--hygiene-proof" in t
    assert "P1D_HYGIENE_PROOF_MISSING" in t
    assert "P1D_HYGIENE_PROOF_INVALID" in t

def test_resume_mode_exists():
    t=text()
    assert "--resume-formal" in t
    assert "P1D_FORMAL_RESUME_VALID_COMPLETED" in t
    assert "skip_completed" in t

def test_resume_validates_provenance_and_hygiene():
    t=text()
    for token in [
        "batch_size",
        "party_placement",
        "mpspdz_commit",
        "mpspdz_source_sha256",
        "correctness_pass",
        "unused_triples_warning",
    ]:
        assert token in t

def test_formal_manifest_prevents_config_mix():
    t=text()
    assert "formal_manifest.json" in t
    assert "P1D_FORMAL_MANIFEST_MISMATCH" in t

def test_canonical_200_run_matrix():
    mod=load()
    keys=mod.expected_formal_keys()
    assert len(keys)==200
    assert keys[0]==(20,"ordinary",0)
    assert keys[-1]==(200,"calibration",24)

def test_measured_count_is_160():
    mod=load()
    keys=mod.expected_formal_keys()
    assert sum(rep >= 5 for _,_,rep in keys)==160
