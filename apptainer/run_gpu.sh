#!/usr/bin/env bash
# Run the self-contained GPU-enabled cosyn Apptainer image.
#
# The image already contains CUDA 11.8 runtime, cuDNN 8, NCCL, and libtorch cu118.
# The host only needs a working NVIDIA driver and Apptainer; no host CUDA toolkit
# is required.
#
# Usage:
#   ./apptainer/run_gpu.sh pll -f example/pll_test/test_seqs_valid.fa \
#       -j /path/to/model.pt -o example/pll_test -p cosyn

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SIF="${SCRIPT_DIR}/cosyn_gpu.sif"

if [ ! -f "${SIF}" ]; then
    echo "Error: ${SIF} not found. Run ./apptainer/build_gpu.sh first."
    exit 1
fi

# Parse -j/--model_path/--model-path and bind its directory so absolute paths
# continue to work inside the container.
BINDS=("--bind" "$(pwd):/workspace")
MODEL_DIR=""
args=("$@")
i=0
while [ $i -lt ${#args[@]} ]; do
    arg="${args[$i]}"
    case "$arg" in
        -j|--model_path|--model-path)
            next_idx=$((i + 1))
            if [ $next_idx -lt ${#args[@]} ]; then
                model_path="${args[$next_idx]}"
                if [[ "$model_path" = /* ]]; then
                    MODEL_DIR="$(dirname "$model_path")"
                fi
                i=$next_idx
            fi
            ;;
    esac
    i=$((i + 1))
done

if [ -n "$MODEL_DIR" ]; then
    BINDS+=("--bind" "$MODEL_DIR:$MODEL_DIR")
fi

apptainer run \
    --nv \
    "${BINDS[@]}" \
    --pwd /workspace \
    "${SIF}" "$@"
