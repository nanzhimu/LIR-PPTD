#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
MPSPDZ_ROOT="${MPSPDZ_ROOT:-$HOME/MP-SPDZ}"

cp "$REPO_ROOT/p1d_mpspdz/Programs/Source/lir_pptd_e2e.mpc" \
   "$MPSPDZ_ROOT/Programs/Source/lir_pptd_e2e.mpc"
cp "$REPO_ROOT/p1d_mpspdz/ExternalIO/p1d_lir_client.py" \
   "$MPSPDZ_ROOT/ExternalIO/p1d_lir_client.py"

cd "$MPSPDZ_ROOT"
for m in 20 50 100 200; do
  ./compile.py -F 192 lir_pptd_e2e "$m" 0
  ./compile.py -F 192 lir_pptd_e2e "$m" 1
done

echo "P1D_MPSPDZ_COMPILE=PASS"
