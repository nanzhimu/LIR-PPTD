#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${1:-/mnt/d/LIR_PPTD}"
HERE="$(cd "$(dirname "$0")/.." && pwd)"
ENV_FILE="$HERE/config/gate_d_hosts.env"

[[ -f "$ENV_FILE" ]] || {
  echo "Missing $ENV_FILE; copy gate_d_hosts.example.env and fill addresses."
  exit 2
}
# shellcheck disable=SC1090
source "$ENV_FILE"

SSH=(ssh -o StrictHostKeyChecking=accept-new -i "$SSH_KEY")
SCP=(scp -o StrictHostKeyChecking=accept-new -i "$SSH_KEY")

for HOST in "$COMPUTE_A_PUBLIC" "$COMPUTE_B_PUBLIC" "$COMPUTE_C_PUBLIC"; do
  "${SCP[@]}" "$HERE/scripts/bootstrap_compute.sh" \
    "$SSH_USER@$HOST:/tmp/p1d_bootstrap_compute.sh"
  "${SSH[@]}" "$SSH_USER@$HOST" \
    "bash /tmp/p1d_bootstrap_compute.sh"
done

"${SCP[@]}" "$HERE/scripts/bootstrap_client.sh" \
  "$SSH_USER@$CLIENT_PUBLIC:/tmp/p1d_bootstrap_client.sh"
"${SSH[@]}" "$SSH_USER@$CLIENT_PUBLIC" \
  "bash /tmp/p1d_bootstrap_client.sh"

# Install P1-D MPC source and compile the exact eight schedules on each compute host.
for HOST in "$COMPUTE_A_PUBLIC" "$COMPUTE_B_PUBLIC" "$COMPUTE_C_PUBLIC"; do
  "${SCP[@]}" \
    "$REPO_ROOT/p1d_mpspdz/Programs/Source/lir_pptd_e2e.mpc" \
    "$SSH_USER@$HOST:~/MP-SPDZ/Programs/Source/lir_pptd_e2e.mpc"
  "${SSH[@]}" "$SSH_USER@$HOST" 'bash -s' <<'EOS'
set -euo pipefail
cd "$HOME/MP-SPDZ"
for m in 20 50 100 200; do
  ./compile.py -F 192 lir_pptd_e2e "$m" 0
  ./compile.py -F 192 lir_pptd_e2e "$m" 1
done
echo "P1D_REMOTE_COMPILE=PASS HOST=$(hostname)"
EOS
done

# Install client program on requester VM.
"${SCP[@]}" "$HERE/orchestrator/p1d_lir_client.py" \
  "$SSH_USER@$CLIENT_PUBLIC:~/MP-SPDZ/ExternalIO/p1d_lir_client.py"

# Generate deterministic formal workloads locally and copy them to requester.
export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"
python3 "$REPO_ROOT/run_p1d_mpspdz_e2e.py" \
  --repo-root "$REPO_ROOT" \
  --generate-formal-workloads >/dev/null

"${SCP[@]}" -r \
  "$REPO_ROOT/results/p1d_mpspdz_real_e2e_v1/workloads/." \
  "$SSH_USER@$CLIENT_PUBLIC:~/p1d_gate_d_workloads/"

# Party IP file: one line per logical party.
IPFILE="$(mktemp)"
trap 'rm -f "$IPFILE"' EXIT
{
  printf '%s\n' "$COMPUTE_A_PRIVATE" "$COMPUTE_A_PRIVATE" \
                 "$COMPUTE_A_PRIVATE" "$COMPUTE_A_PRIVATE"
  printf '%s\n' "$COMPUTE_B_PRIVATE" "$COMPUTE_B_PRIVATE" \
                 "$COMPUTE_B_PRIVATE"
  printf '%s\n' "$COMPUTE_C_PRIVATE" "$COMPUTE_C_PRIVATE" \
                 "$COMPUTE_C_PRIVATE"
} > "$IPFILE"

for HOST in "$COMPUTE_A_PUBLIC" "$COMPUTE_B_PUBLIC" "$COMPUTE_C_PUBLIC"; do
  "${SCP[@]}" "$IPFILE" \
    "$SSH_USER@$HOST:~/MP-SPDZ/Player-Data/p1d_hosts10.txt"
done

echo "P1D_REMOTE_SOFTWARE_AND_WORKLOADS=PASS"
echo "NEXT=distribute formal SSL material with scripts/distribute_formal_ssl.sh"
