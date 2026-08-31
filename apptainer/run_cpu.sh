#!/usr/bin/env bash
# Run the CPU-only cosyn Apptainer image.
#
# Usage:
#   ./apptainer/run_cpu.sh pll -f example/pll_test/test_seqs_valid.fa \
#       -j /path/to/model.pt -o example/pll_test -p cosyn

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SIF="${SCRIPT_DIR}/cosyn_cpu.sif"

if [ ! -f "${SIF}" ]; then
    echo "Error: ${SIF} not found. Run ./apptainer/build_cpu.sh first."
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
    "${BINDS[@]}" \
    --pwd /workspace \
    "${SIF}" "$@"
