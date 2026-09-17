from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def test_exact_version_pinned():
    t=(ROOT/"scripts/setup_gate_d_aliyun_offline.sh").read_text(encoding="utf-8")
    assert 'EXPECTED_TAG="v0.4.3"' in t
    assert 'EXPECTED_COMMIT="26a605368e40fed3a7e9cee78c9a3f4390b85eb5"' in t

def test_no_remote_github_clone():
    t=(ROOT/"scripts/setup_gate_d_aliyun_offline.sh").read_text(encoding="utf-8")
    assert "github.com" not in t
    assert "git clone" not in t

def test_same_tarball_hash_audited():
    t=(ROOT/"scripts/setup_gate_d_aliyun_offline.sh").read_text(encoding="utf-8")
    assert "SOURCE_SHA" in t
    assert "source_tarball_sha256" in t

def test_builds_eight_schedules():
    t=(ROOT/"scripts/setup_gate_d_aliyun_offline.sh").read_text(encoding="utf-8")
    assert "for m in 20 50 100 200" in t
    assert '"$m" 0' in t and '"$m" 1' in t

def test_credentials_excluded():
    t=(ROOT/"scripts/setup_gate_d_aliyun_offline.sh").read_text(encoding="utf-8")
    assert "--exclude='Player-Data'" in t

def test_preflight_checks_private_ip():
    t=(ROOT/"scripts/aliyun_preflight.sh").read_text(encoding="utf-8")
    assert "PRIVATE_IP=PASS" in t
    assert "P1D_ALIYUN_PREFLIGHT=PASS" in t
