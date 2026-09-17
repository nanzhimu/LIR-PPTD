from pathlib import Path
import hashlib, json, sys

ROOT = Path(__file__).resolve().parent
MANIFEST_PATH = ROOT / "PUBLIC_ARTIFACT_MANIFEST.json"
M = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
errors = []

# 1) Every manifest entry must exist and match exact bytes/hash.
manifest_paths = set()
for e in M["files"]:
    rel = e["path"]
    manifest_paths.add(rel)
    p = ROOT / rel
    if not p.is_file():
        errors.append(f"MISSING {rel}")
        continue
    if p.stat().st_size != e["bytes"]:
        errors.append(f"SIZE {rel}")
        continue
    h = hashlib.sha256(p.read_bytes()).hexdigest()
    if h != e["sha256"]:
        errors.append(f"SHA {rel}")

# 2) The manifest must cover every released regular file except itself.
def released_file_set():
    out = set()
    for p in ROOT.rglob("*"):
        if not p.is_file() or p == MANIFEST_PATH:
            continue
        if any(part in {"__pycache__", ".pytest_cache", ".git"} for part in p.parts):
            continue
        out.add(p.relative_to(ROOT).as_posix())
    return out

released = released_file_set()
for rel in sorted(released - manifest_paths):
    errors.append(f"UNLISTED {rel}")
for rel in sorted(manifest_paths - released):
    errors.append(f"STALE_MANIFEST {rel}")
if M.get("file_count") != len(M.get("files", [])):
    errors.append("FILE_COUNT_FIELD")

# 3) Required reviewer-facing scientific/reproducibility files.
required = [
    "README.md",
    "DATA_POLICY.md",
    "LICENSE.md",
    "ENVIRONMENT.json",
    "requirements-test-lock.txt",
    "src/lir_pptd/core/consistency_scale.py",
    "src/lir_pptd/fixedpoint/range_analysis.py",
    "src/lir_pptd/mpc/secure_algorithm.py",
    "src/lir_pptd/cli.py",
    "analysis/geo_tau_final/paper_result_consistency.csv",
    "analysis/geo_tau_final/selector_corrective_summary.json",
    "analysis/geo_tau_final/selector_audit_geometry_merged/multi_selector_formal_summary.json",
    "analysis/geo_tau_final/selector_audit_geometry_merged/formal_runs_geometry_merged.jsonl",
    "analysis/geo_tau_final/final_pass_matrix.csv",
    "results/round6_2_multi_selector_geometry_final/dog_geometry_formal_runs_01_20.jsonl",
    "results/geo_tau_final_secondary/parameter_sensitivity/formal_summary.json",
    "results/geo_tau_final_secondary/calibration_quality/formal_summary.json",
    "results/geo_tau_final_secondary/task_order/formal_summary.json",
    "data/neteasecrowd/dataset_audit.json",
    "data/phase_r1_dataset_manifest.json",
    "docs/PHASE6_R2_GENERATOR_DECISIONS.json",
    "tests/golden/spec_examples.json",
    "tests/golden/fixed_primitive_vectors.json",
    "P1D_E2E_FROZEN_DESIGN.md",
    "P1_D_GATE_D_4VM_VPC_V1_8_ALIYUN_FORMAL_RESUMABLE/config/gate_d_hosts.example.env",
    "FINAL_CLOSURE_STEP03_RESULT_LEDGER.md",
    "FINAL_CLOSURE_STEP03_RESULT_LEDGER.json",
    "results/fc2_p0a/formal_summary.json",
    "results/fc2_p0a/cell_summary.csv",
    "results/fc2_p0b/formal_summary.json",
    "results/fc2_p0b/capability_metrics.csv",
    "results/fc2_p0c/formal_summary.json",
    "results/fc2_p0c/budget_summary.csv",
    "results/fc2_p0c/contrasts_vs_5pct.csv",
]
for r in required:
    if not (ROOT / r).is_file():
        errors.append(f"REQUIRED {r}")

# 4) Policy closure: raw restricted datasets and active private material must remain excluded.
for p in ROOT.rglob("*"):
    if not p.is_file():
        continue
    rel = p.relative_to(ROOT).as_posix()
    low = rel.lower()
    if "private_selector_seeds" in low:
        errors.append(f"PRIVATE_SELECTOR_MATERIAL {rel}")
    if low.endswith("/gate_d_hosts.env"):
        errors.append(f"PRIVATE_CLOUD_CONFIG {rel}")
    if "data/neteasecrowd/raw/neteasecrowd_part_" in low:
        errors.append(f"RAW_NETEASE {rel}")
    if low in {"data/product/answer.csv", "data/product/truth.csv", "data/duck/answer.csv", "data/duck/truth.csv"}:
        errors.append(f"RAW_PRODUCT_DUCK {rel}")

# 5) Metadata/document promises.
try:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for token in ("0 failed tests", "data/phase_r1_dataset_manifest.json", "data/neteasecrowd/dataset_audit.json", "paper_result_consistency.csv"):
        if token not in readme:
            errors.append(f"README_PROMISE {token}")
    dp = (ROOT / "DATA_POLICY.md").read_text(encoding="utf-8")
    for token in ("data/neteasecrowd/dataset_audit.json", "data/phase_r1_dataset_manifest.json"):
        if token not in dp:
            errors.append(f"DATA_POLICY_PROMISE {token}")
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    for token in ("pydantic", "PyYAML", "mpmath", "pytest"):
        if token not in pyproject:
            errors.append(f"DEPENDENCY_DECLARATION {token}")
except Exception as exc:
    errors.append(f"METADATA_CHECK {exc}")

# 6) Semantic selector checks.
try:
    s = json.loads((ROOT / "analysis/geo_tau_final/selector_corrective_summary.json").read_text())
    if s.get("final_rows") != 30240 or s.get("dog_geometry_rows_replaced") != 2000 or s.get("status") != "COMPLETE":
        errors.append("SELECTOR_SUMMARY")
    d = json.loads((ROOT / "results/round6_2_multi_selector_geometry_final/dog_geometry_execution_summary_01_20.json").read_text())
    if d.get("rows") != 2000 or d.get("failure") != 0 or d.get("mask_provenance_mismatches") != 0 or not d.get("complete"):
        errors.append("DOG_SELECTOR_EXECUTION")
except Exception as exc:
    errors.append(f"SELECTOR_CHECK {exc}")

# 7) Dataset-audit sanity checks.
try:
    nd = json.loads((ROOT / "data/neteasecrowd/dataset_audit.json").read_text())
    if nd.get("raw_parts") != 15 or nd.get("tasks") != 999799 or len(nd.get("raw_sha256", {})) != 15:
        errors.append("NETEASE_DATASET_AUDIT")
    rd = json.loads((ROOT / "data/phase_r1_dataset_manifest.json").read_text())
    if rd.get("raw_redistribution_policy") != "do_not_bundle_raw_csv_in_public_artifacts":
        errors.append("PHASE_R1_DATA_POLICY")
    for ds in ("product", "duck"):
        if len(rd.get("datasets", {}).get(ds, {}).get("answer_sha256", "")) != 64:
            errors.append(f"PHASE_R1_HASH {ds}")
except Exception as exc:
    errors.append(f"DATASET_AUDIT_CHECK {exc}")


# 8) Submission hygiene: reviewer-facing Python scripts must not contain hard-coded /mnt/data paths.
for p in ROOT.rglob("*.py"):
    if p.resolve() == Path(__file__).resolve():
        continue
    if any(part in {"__pycache__", ".pytest_cache"} for part in p.parts):
        continue
    try:
        txt = p.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        continue
    if "/mnt/data/" in txt:
        errors.append(f"HARDCODED_LOCAL_PATH {p.relative_to(ROOT).as_posix()}")

print(f"ARTIFACT_ID={M.get('artifact_id')}")
print(f"MANIFEST_FILE_COUNT={M['file_count']}")
print(f"CHECKED={len(M['files'])}")
print(f"RELEASED_FILE_COUNT_EXCLUDING_MANIFEST={len(released)}")
print(f"ERRORS={len(errors)}")
for e in errors:
    print(e)
sys.exit(1 if errors else 0)
