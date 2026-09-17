#!/usr/bin/env bash
set -euo pipefail
MPSPDZ_ROOT="${MPSPDZ_ROOT:-$HOME/MP-SPDZ}"
TAG="${MPSPDZ_TAG:-v0.4.3}"

if [[ ! -d "$MPSPDZ_ROOT/.git" ]]; then
  git clone --recursive --branch "$TAG" \
    https://github.com/data61/MP-SPDZ.git "$MPSPDZ_ROOT"
else
  git -C "$MPSPDZ_ROOT" fetch --tags
  git -C "$MPSPDZ_ROOT" checkout "$TAG"
  git -C "$MPSPDZ_ROOT" submodule update --init --recursive
fi

cd "$MPSPDZ_ROOT"
ACTUAL="$(git describe --tags --exact-match)"
[[ "$ACTUAL" == "$TAG" ]] || {
  echo "Wrong MP-SPDZ tag: $ACTUAL"
  exit 2
}

make -j"$(nproc)" shamir-party.x
Scripts/setup-ssl.sh 10
Scripts/setup-clients.sh 1

echo "P1D_MPSPDZ_SETUP=PASS"
echo "MPSPDZ_ROOT=$MPSPDZ_ROOT"
echo "MPSPDZ_TAG=$ACTUAL"
