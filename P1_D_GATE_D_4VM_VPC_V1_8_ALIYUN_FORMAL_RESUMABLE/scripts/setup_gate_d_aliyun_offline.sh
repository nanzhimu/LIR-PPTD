#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${1:-/mnt/d/LIR_PPTD}"
HERE="$(cd "$(dirname "$0")/.." && pwd)"
ENV_FILE="$HERE/config/gate_d_hosts.env"
LOCAL_MP="${LOCAL_MPSPDZ_ROOT:-$HOME/MP-SPDZ}"

EXPECTED_TAG="v0.4.3"
EXPECTED_COMMIT="26a605368e40fed3a7e9cee78c9a3f4390b85eb5"

[[ -f "$ENV_FILE" ]] || { echo "Missing $ENV_FILE"; exit 2; }
source "$ENV_FILE"

[[ -d "$LOCAL_MP/.git" ]] || { echo "P1D_OFFLINE_FAIL local repo missing"; exit 2; }
ACTUAL_TAG="$(git -C "$LOCAL_MP" describe --tags --exact-match)"
ACTUAL_COMMIT="$(git -C "$LOCAL_MP" rev-parse HEAD)"

[[ "$ACTUAL_TAG" == "$EXPECTED_TAG" ]] || { echo "P1D_OFFLINE_FAIL tag=$ACTUAL_TAG"; exit 2; }
[[ "$ACTUAL_COMMIT" == "$EXPECTED_COMMIT" ]] || { echo "P1D_OFFLINE_FAIL commit=$ACTUAL_COMMIT"; exit 2; }

echo "P1D_LOCAL_SOURCE_VERIFY=PASS"
echo "MPSPDZ_TAG=$ACTUAL_TAG"
echo "MPSPDZ_COMMIT=$ACTUAL_COMMIT"

SSH=(ssh -o StrictHostKeyChecking=accept-new -i "$SSH_KEY")
SCP=(scp -o StrictHostKeyChecking=accept-new -i "$SSH_KEY")
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

TARBALL="$TMP/mpspdz_v043_26a6053_source.tar.gz"
PROVENANCE="$TMP/MP-SPDZ_PROVENANCE.json"

cat > "$PROVENANCE" <<EOF
{"framework":"MP-SPDZ","tag":"$ACTUAL_TAG","commit":"$ACTUAL_COMMIT","distribution":"offline_from_verified_wsl_source"}
EOF

tar \
  --exclude='.git' \
  --exclude='*/.git' \
  --exclude='Player-Data' \
  --exclude='Programs/Bytecode' \
  --exclude='Programs/Schedules' \
  --exclude='*.o' \
  --exclude='*.a' \
  --exclude='*.so' \
  --exclude='*.x' \
  --exclude='__pycache__' \
  -C "$LOCAL_MP" -czf "$TARBALL" .

SOURCE_SHA="$(sha256sum "$TARBALL" | awk '{print $1}')"
echo "MPSPDZ_SOURCE_TARBALL_SHA256=$SOURCE_SHA"

bootstrap_host () {
  local PUBLIC_IP="$1"
  local ROLE="$2"
  echo "=== Bootstrap $ROLE @ $PUBLIC_IP ==="

  "${SSH[@]}" "$SSH_USER@$PUBLIC_IP" 'bash -s' <<'EOS'
set -euo pipefail
sudo apt-get update
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y \
  automake build-essential clang cmake \
  libboost-dev libboost-filesystem-dev libboost-iostreams-dev \
  libboost-thread-dev libgmp-dev libntl-dev libsodium-dev \
  libssl-dev libtool openssl python3 python3-pip python3-gmpy2 \
  rsync ca-certificates
rm -rf "$HOME/MP-SPDZ"
mkdir -p "$HOME/MP-SPDZ"
EOS

  "${SCP[@]}" "$TARBALL" "$SSH_USER@$PUBLIC_IP:/tmp/p1d_mpspdz_source.tar.gz"
  "${SCP[@]}" "$PROVENANCE" "$SSH_USER@$PUBLIC_IP:/tmp/MP-SPDZ_PROVENANCE.json"

  "${SSH[@]}" "$SSH_USER@$PUBLIC_IP" 'bash -s' <<'EOS'
set -euo pipefail
cd "$HOME/MP-SPDZ"
tar -xzf /tmp/p1d_mpspdz_source.tar.gz
cp /tmp/MP-SPDZ_PROVENANCE.json "$HOME/MP-SPDZ/MP-SPDZ_PROVENANCE.json"
rm -f /tmp/p1d_mpspdz_source.tar.gz /tmp/MP-SPDZ_PROVENANCE.json
echo "P1D_OFFLINE_SOURCE_INSTALL=PASS HOST=$(hostname)"
EOS
}

bootstrap_host "$COMPUTE_A_PUBLIC" "COMPUTE_A"
bootstrap_host "$COMPUTE_B_PUBLIC" "COMPUTE_B"
bootstrap_host "$COMPUTE_C_PUBLIC" "COMPUTE_C"
bootstrap_host "$CLIENT_PUBLIC" "REQUESTER"

for HOST in "$COMPUTE_A_PUBLIC" "$COMPUTE_B_PUBLIC" "$COMPUTE_C_PUBLIC"; do
  "${SCP[@]}" \
    "$REPO_ROOT/p1d_mpspdz/Programs/Source/lir_pptd_e2e.mpc" \
    "$SSH_USER@$HOST:~/MP-SPDZ/Programs/Source/lir_pptd_e2e.mpc"

  "${SSH[@]}" "$SSH_USER@$HOST" 'bash -s' <<'EOS'
set -euo pipefail
cd "$HOME/MP-SPDZ"
make -j"$(nproc)" shamir-party.x
test -x shamir-party.x

for m in 20 50 100 200; do
  ./compile.py -F 192 lir_pptd_e2e "$m" 0
  ./compile.py -F 192 lir_pptd_e2e "$m" 1
done

for f in \
  lir_pptd_e2e-20-0.sch lir_pptd_e2e-20-1.sch \
  lir_pptd_e2e-50-0.sch lir_pptd_e2e-50-1.sch \
  lir_pptd_e2e-100-0.sch lir_pptd_e2e-100-1.sch \
  lir_pptd_e2e-200-0.sch lir_pptd_e2e-200-1.sch
do
  test -f "Programs/Schedules/$f"
done

mkdir -p "$HOME/p1d_gate_d_logs"
echo "P1D_REMOTE_OFFLINE_BUILD=PASS HOST=$(hostname)"
EOS
done

"${SCP[@]}" "$HERE/orchestrator/p1d_lir_client.py" \
  "$SSH_USER@$CLIENT_PUBLIC:~/MP-SPDZ/ExternalIO/p1d_lir_client.py"

"${SSH[@]}" "$SSH_USER@$CLIENT_PUBLIC" \
  'mkdir -p "$HOME/p1d_gate_d_workloads" "$HOME/p1d_gate_d_client_results"'

export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"
python3 "$REPO_ROOT/run_p1d_mpspdz_e2e.py" \
  --repo-root "$REPO_ROOT" --generate-formal-workloads >/dev/null

"${SCP[@]}" -r \
  "$REPO_ROOT/results/p1d_mpspdz_real_e2e_v1/workloads/." \
  "$SSH_USER@$CLIENT_PUBLIC:~/p1d_gate_d_workloads/"

IPFILE="$TMP/p1d_hosts10.txt"
{
  printf '%s\n' "$COMPUTE_A_PRIVATE" "$COMPUTE_A_PRIVATE" "$COMPUTE_A_PRIVATE" "$COMPUTE_A_PRIVATE"
  printf '%s\n' "$COMPUTE_B_PRIVATE" "$COMPUTE_B_PRIVATE" "$COMPUTE_B_PRIVATE"
  printf '%s\n' "$COMPUTE_C_PRIVATE" "$COMPUTE_C_PRIVATE" "$COMPUTE_C_PRIVATE"
} > "$IPFILE"

for HOST in "$COMPUTE_A_PUBLIC" "$COMPUTE_B_PUBLIC" "$COMPUTE_C_PUBLIC"; do
  mkdir -p "$REPO_ROOT/results/p1d_gate_d_4vm_v1/deployment_audit"
  "${SSH[@]}" "$SSH_USER@$HOST" 'mkdir -p "$HOME/MP-SPDZ/Player-Data"'
  "${SCP[@]}" "$IPFILE" "$SSH_USER@$HOST:~/MP-SPDZ/Player-Data/p1d_hosts10.txt"
done

AUDIT_DIR="$REPO_ROOT/results/p1d_gate_d_4vm_v1/deployment_audit"
mkdir -p "$AUDIT_DIR"

collect_audit () {
  local PUBLIC_IP="$1"
  local NAME="$2"
  "${SSH[@]}" "$SSH_USER@$PUBLIC_IP" 'bash -s' > "$AUDIT_DIR/${NAME}.txt" <<'EOS'
set -euo pipefail
echo "HOSTNAME=$(hostname)"
echo "KERNEL=$(uname -srmo)"
echo "CPU_MODEL=$(lscpu | awk -F: '/Model name/{gsub(/^[ \t]+/,"",$2); print $2; exit}')"
echo "VCPU=$(nproc)"
grep MemTotal /proc/meminfo
cat "$HOME/MP-SPDZ/MP-SPDZ_PROVENANCE.json"
EOS
}

collect_audit "$COMPUTE_A_PUBLIC" "compute_a"
collect_audit "$COMPUTE_B_PUBLIC" "compute_b"
collect_audit "$COMPUTE_C_PUBLIC" "compute_c"
collect_audit "$CLIENT_PUBLIC" "requester"

cat > "$AUDIT_DIR/source_provenance.json" <<EOF
{
  "mpspdz_tag":"$ACTUAL_TAG",
  "mpspdz_commit":"$ACTUAL_COMMIT",
  "source_tarball_sha256":"$SOURCE_SHA",
  "distribution":"offline_from_verified_wsl_source",
  "algorithm_retuning":false
}
EOF

echo "P1D_ALIYUN_OFFLINE_DEPLOYMENT=PASS"
echo "MPSPDZ_TAG=$ACTUAL_TAG"
echo "MPSPDZ_COMMIT=$ACTUAL_COMMIT"
echo "MPSPDZ_SOURCE_SHA256=$SOURCE_SHA"
echo "COMPUTE_PLACEMENT=A:P0-P3,B:P4-P6,C:P7-P9"
echo "NEXT=distribute_formal_ssl.sh"
