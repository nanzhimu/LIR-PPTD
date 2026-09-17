#!/usr/bin/env bash
set -euo pipefail

TAG="${MPSPDZ_TAG:-v0.4.3}"
ROOT="${MPSPDZ_ROOT:-$HOME/MP-SPDZ}"

sudo apt-get update
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y \
  git openssl python3 python3-pip python3-gmpy2 rsync

if [[ ! -d "$ROOT/.git" ]]; then
  git clone --recursive --branch "$TAG" \
    https://github.com/data61/MP-SPDZ.git "$ROOT"
else
  git -C "$ROOT" fetch --tags
  git -C "$ROOT" checkout "$TAG"
  git -C "$ROOT" submodule update --init --recursive
fi

cd "$ROOT"
ACTUAL="$(git describe --tags --exact-match)"
[[ "$ACTUAL" == "$TAG" ]] || {
  echo "P1D_CLIENT_BOOTSTRAP_FAIL wrong tag=$ACTUAL"
  exit 2
}

mkdir -p "$HOME/p1d_gate_d_workloads" "$HOME/p1d_gate_d_client_results"

echo "P1D_CLIENT_BOOTSTRAP=PASS"
echo "MPSPDZ_TAG=$ACTUAL"
echo "HOST=$(hostname)"
