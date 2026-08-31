#!/usr/bin/env bash
# Build a self-contained GPU-enabled Apptainer image for cosyn (CUDA 11.8).
#
# The resulting image embeds CUDA 11.8 runtime, cuDNN 8, NCCL, and libtorch cu118,
# so the host only needs a working NVIDIA driver (>= 450.80.02 / R450+) and
# Apptainer with the --nv option. No host CUDA toolkit, cuDNN, or NCCL is required.
#
# Usage:
#   ./apptainer/build_gpu.sh
#
# Environment variables:
#   LIBTORCH_URL   - Override the libtorch CUDA 11.8 download URL.
#   APPTAINER      - Path to apptainer binary (default: apptainer).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CACHE_DIR="${SCRIPT_DIR}/cache"

DEFAULT_LIBTORCH_URL="https://download.pytorch.org/libtorch/cu118/libtorch-cxx11-abi-shared-with-deps-2.5.1%2Bcu118.zip"
LIBTORCH_URL="${LIBTORCH_URL:-${DEFAULT_LIBTORCH_URL}}"
APPTAINER="${APPTAINER:-apptainer}"

mkdir -p "${CACHE_DIR}/gpu"

echo "[build_gpu] preparing libtorch..."
if [ ! -d "${CACHE_DIR}/gpu/libtorch" ]; then
    if [ ! -f "${CACHE_DIR}/gpu/libtorch-cu118.zip" ]; then
        echo "[build_gpu] downloading libtorch CUDA 11.8 from ${LIBTORCH_URL}"
        curl -L -o "${CACHE_DIR}/gpu/libtorch-cu118.zip" "${LIBTORCH_URL}"
    fi
    echo "[build_gpu] extracting libtorch..."
    unzip -q "${CACHE_DIR}/gpu/libtorch-cu118.zip" -d "${CACHE_DIR}/gpu"
    rm -f "${CACHE_DIR}/gpu/libtorch-cu118.zip"
else
    echo "[build_gpu] using cached ${CACHE_DIR}/gpu/libtorch"
fi

echo "[build_gpu] staging source..."
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

echo "[build_gpu] building SIF (this may take several minutes)..."
cd "${SCRIPT_DIR}"
${APPTAINER} build --fakeroot --force cosyn_gpu.sif cosyn_gpu.def

echo "[build_gpu] done: ${SCRIPT_DIR}/cosyn_gpu.sif"
