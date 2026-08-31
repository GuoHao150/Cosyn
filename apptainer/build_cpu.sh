#!/usr/bin/env bash
# Build a CPU-only Apptainer image for cosyn.
#
# Usage:
#   ./apptainer/build_cpu.sh
#
# Environment variables:
#   LIBTORCH_URL   - Override the libtorch CPU download URL.
#   APPTAINER      - Path to apptainer binary (default: apptainer).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CACHE_DIR="${SCRIPT_DIR}/cache"

DEFAULT_LIBTORCH_URL="https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.5.1%2Bcpu.zip"
LIBTORCH_URL="${LIBTORCH_URL:-${DEFAULT_LIBTORCH_URL}}"
APPTAINER="${APPTAINER:-apptainer}"

mkdir -p "${CACHE_DIR}/cpu"

echo "[build_cpu] preparing libtorch..."
if [ ! -d "${CACHE_DIR}/cpu/libtorch" ]; then
    if [ ! -f "${CACHE_DIR}/cpu/libtorch-cpu.zip" ]; then
        echo "[build_cpu] downloading libtorch CPU from ${LIBTORCH_URL}"
        curl -L -o "${CACHE_DIR}/cpu/libtorch-cpu.zip" "${LIBTORCH_URL}"
    fi
    echo "[build_cpu] extracting libtorch..."
    unzip -q "${CACHE_DIR}/cpu/libtorch-cpu.zip" -d "${CACHE_DIR}/cpu"
    rm -f "${CACHE_DIR}/cpu/libtorch-cpu.zip"
else
    echo "[build_cpu] using cached ${CACHE_DIR}/cpu/libtorch"
fi

echo "[build_cpu] staging source..."
rm -rf "${CACHE_DIR}/src"
mkdir -p "${CACHE_DIR}/src"
cp -r \
    Cargo.toml Cargo.lock build.rs Makefile \
    src codon_eval include linearfold_rust \
    "${CACHE_DIR}/src/"

# Use Tsinghua crates.io mirror to speed up Rust dependency downloads.
mkdir -p "${CACHE_DIR}/src/.cargo"
cat > "${CACHE_DIR}/src/.cargo/config.toml" <<'EOF'
[registries.crates-io]
index = "sparse+https://mirrors.tuna.tsinghua.edu.cn/crates.io-index/"

[net]
git-fetch-with-cli = true
EOF

echo "[build_cpu] building SIF (this may take several minutes)..."
cd "${SCRIPT_DIR}"
${APPTAINER} build --fakeroot --force cosyn_cpu.sif cosyn_cpu.def

echo "[build_cpu] done: ${SCRIPT_DIR}/cosyn_cpu.sif"
