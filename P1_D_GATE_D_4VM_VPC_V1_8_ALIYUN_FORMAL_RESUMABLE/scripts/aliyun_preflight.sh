#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
source "$HERE/config/gate_d_hosts.env"
SSH=(ssh -o BatchMode=yes -o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new -i "$SSH_KEY")

check () {
  local NAME="$1"; local PUB="$2"; local PRIV="$3"
  echo "=== $NAME ==="
  "${SSH[@]}" "$SSH_USER@$PUB" \
    "echo SSH=PASS; hostname; hostname -I; ip -4 addr show | grep -F '$PRIV' >/dev/null && echo PRIVATE_IP=PASS"
}

check COMPUTE_A "$COMPUTE_A_PUBLIC" "$COMPUTE_A_PRIVATE"
check COMPUTE_B "$COMPUTE_B_PUBLIC" "$COMPUTE_B_PRIVATE"
check COMPUTE_C "$COMPUTE_C_PUBLIC" "$COMPUTE_C_PRIVATE"
check REQUESTER "$CLIENT_PUBLIC" "$CLIENT_PRIVATE"

echo "P1D_ALIYUN_PREFLIGHT=PASS"
