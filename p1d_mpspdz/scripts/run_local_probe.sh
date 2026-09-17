#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
MPSPDZ_ROOT="${MPSPDZ_ROOT:-$HOME/MP-SPDZ}"
OUT="$REPO_ROOT/results/p1d_mpspdz_real_e2e_v1/local_probe"
mkdir -p "$OUT"

cd "$REPO_ROOT"
python3 run_p1d_mpspdz_e2e.py \
  --repo-root . \
  --generate-probe-workloads >/dev/null

cd "$MPSPDZ_ROOT"
HOSTS="localhost,localhost,localhost,localhost,localhost,localhost,localhost,localhost,localhost,localhost"

run_one () {
  local m="$1"
  local mode_num="$2"
  local mode_name="$3"
  local rep="$4"
  local sched="lir_pptd_e2e-${m}-${mode_num}"
  local r2
  r2="$(printf '%02d' "$rep")"
  local workload="$REPO_ROOT/results/p1d_mpspdz_real_e2e_v1/workloads/workload_m${m}_r${r2}.json"
  local log="$OUT/server_${mode_name}_m${m}_r${r2}.log"
  local client_out="$OUT/client_${mode_name}_m${m}_r${r2}.json"

  PLAYERS=10 Scripts/shamir.sh "$sched" -T 3 >"$log" 2>&1 &
  local server_pid=$!
  sleep 2

  set +e
  python3 ExternalIO/p1d_lir_client.py \
    --workload "$workload" \
    --hosts "$HOSTS" \
    --mode "$mode_name" \
    --out "$client_out"
  local client_rc=$?
  set -e

  if [[ "$client_rc" -ne 0 ]]; then
    kill "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
    return "$client_rc"
  fi

  wait "$server_pid"

  python3 "$REPO_ROOT/run_p1d_mpspdz_e2e.py" \
    --repo-root "$REPO_ROOT" \
    --parse-log "$log" \
    > "$OUT/parsed_${mode_name}_m${m}_r${r2}.txt"

  python3 - "$client_out" "$mode_name" <<'PY'
import json, sys
p, mode = sys.argv[1:]
r = json.load(open(p))
tol = 2e-4 if mode == "ordinary" else 5e-4
if r["abs_error"] > tol:
    raise SystemExit(
        f"CORRECTNESS_GATE_FAIL:{mode}:{r['abs_error']} > {tol}")
print(
    f"CORRECTNESS_GATE_PASS mode={mode} "
    f"m={r['m']} error={r['abs_error']:.3e}")
PY
}

# rep0 warm-up; rep1..3 measured probe trials
for m in 20 50; do
  for rep in 0 1 2 3; do
    run_one "$m" 0 ordinary "$rep"
    run_one "$m" 1 calibration "$rep"
  done
done

echo "P1D_LOCAL_REAL_MPC_PROBE=COMPLETE"
echo "P1D_LOCAL_MULTI_PROCESS_ONLY=YES"
echo "MULTI_HOST_CLAIM=NO"
echo "OUTPUT_DIR=$OUT"
