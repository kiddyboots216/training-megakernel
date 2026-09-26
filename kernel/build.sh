#!/usr/bin/env bash
# Build the resident training megakernel into OUTPUT, a new directory outside the
# repository.  Extra arguments go to kernel/build.py (see --help).
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 OUTPUT [build.py options]" >&2
  exit 2
fi
KERNEL_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
OUTPUT=$1
shift

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export OMP_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
unset PYTORCH_CUDA_ALLOC_CONF
exec "${TMK_PYTHON:-python3}" "${KERNEL_ROOT}/build.py" --out "${OUTPUT}" "$@"
