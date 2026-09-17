#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "$0")/.." && pwd)"
ENV_FILE="$HERE/config/gate_d_hosts.env"
source "$ENV_FILE"

LOCAL_MP="${MPSPDZ_ROOT:-$HOME/MP-SPDZ}"
STAGE="$HERE/.formal_ssl_stage"
rm -rf "$STAGE"
mkdir -p "$STAGE"

# Regenerate formal benchmark credentials locally, then selectively distribute
# private keys according to logical-party placement.
cd "$LOCAL_MP"
Scripts/setup-ssl.sh 10
Scripts/setup-clients.sh 1

cp Player-Data/P*.pem "$STAGE"/
cp Player-Data/C*.pem "$STAGE"/ 2>/dev/null || true

SSH=(ssh -o StrictHostKeyChecking=accept-new -i "$SSH_KEY")
SCP=(scp -o StrictHostKeyChecking=accept-new -i "$SSH_KEY")

# Everyone gets public certificates.
for HOST in "$COMPUTE_A_PUBLIC" "$COMPUTE_B_PUBLIC" "$COMPUTE_C_PUBLIC" "$CLIENT_PUBLIC"; do
  "${SCP[@]}" "$STAGE"/*.pem "$SSH_USER@$HOST:~/MP-SPDZ/Player-Data/"
done

# Only the host actually executing a logical party gets that party private key.
for p in 0 1 2 3; do
  "${SCP[@]}" "$LOCAL_MP/Player-Data/P${p}.key" \
    "$SSH_USER@$COMPUTE_A_PUBLIC:~/MP-SPDZ/Player-Data/"
done
for p in 4 5 6; do
  "${SCP[@]}" "$LOCAL_MP/Player-Data/P${p}.key" \
    "$SSH_USER@$COMPUTE_B_PUBLIC:~/MP-SPDZ/Player-Data/"
done
for p in 7 8 9; do
  "${SCP[@]}" "$LOCAL_MP/Player-Data/P${p}.key" \
    "$SSH_USER@$COMPUTE_C_PUBLIC:~/MP-SPDZ/Player-Data/"
done

# Requester gets only its own client private key.
"${SCP[@]}" "$LOCAL_MP/Player-Data/C0.key" \
  "$SSH_USER@$CLIENT_PUBLIC:~/MP-SPDZ/Player-Data/"

# All hosts rebuild certificate hash links.
for HOST in "$COMPUTE_A_PUBLIC" "$COMPUTE_B_PUBLIC" "$COMPUTE_C_PUBLIC" "$CLIENT_PUBLIC"; do
  "${SSH[@]}" "$SSH_USER@$HOST" \
    'cd "$HOME/MP-SPDZ" && c_rehash Player-Data >/dev/null'
done

echo "P1D_FORMAL_SSL_DISTRIBUTION=PASS"
echo "PRIVATE_KEY_ISOLATION=A:P0-P3,B:P4-P6,C:P7-P9,CLIENT:C0"
