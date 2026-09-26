#!/usr/bin/env bash
set -Eeuo pipefail

if [[ $# -ne 5 ]]; then
  echo "usage: $0 checkpoint|resume BUNDLE RUN_DIR CHECKPOINT_ROOT DCLM_SHARDS" >&2
  exit 2
fi
: "${TMK_STEPS:?set TMK_STEPS to the number of steps to run}"

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "${HERE}/../.." && pwd)
ARM=$1
BUNDLE=$(realpath "$2")
RUN_DIR=$(realpath -m "$3")
CHECKPOINT_ROOT=$(realpath -m "$4")
WORKLOAD=$(realpath "$5")
# Fails if the run directory already exists.
mkdir -- "${RUN_DIR}"

export PYTHONPATH="${ROOT}/src:${HERE}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_NVLS_ENABLE=1 TORCH_NCCL_ASYNC_ERROR_HANDLING=1 OMP_NUM_THREADS=2
unset CUDA_VISIBLE_DEVICES TMK_SEQUENCE CUTE_DSL_LINEINFO CUTE_DSL_PTXAS_PATH QUACK_ARCH

OPTIONAL_ARGS=()
if [[ -n "${TMK_LEARNING_RATE:-}" ]]; then
  OPTIONAL_ARGS+=(--learning-rate "${TMK_LEARNING_RATE}")
fi
if [[ -n "${TMK_WEIGHT_DECAY:-}" ]]; then
  OPTIONAL_ARGS+=(--weight-decay "${TMK_WEIGHT_DECAY}")
fi
if [[ -n "${TMK_HF_MODEL_SNAPSHOT:-}" ]]; then
  OPTIONAL_ARGS+=(--hf-model-snapshot "$(realpath "${TMK_HF_MODEL_SNAPSHOT}")")
fi
if [[ -n "${TMK_CHECKPOINT_STEPS:-}" ]]; then
  OPTIONAL_ARGS+=(--checkpoint-steps "${TMK_CHECKPOINT_STEPS}")
fi
if [[ -n "${TMK_WANDB_PROJECT:-}" ]]; then
  OPTIONAL_ARGS+=(--wandb-project "${TMK_WANDB_PROJECT}")
fi

# torchrun stops every rank when one fails; the timeout bounds a run that hangs.
timeout --signal=TERM --kill-after=30s 24h \
  "${TMK_PYTHON:-python3}" -m torch.distributed.run \
    --standalone --nproc-per-node=8 \
    --log-dir "${RUN_DIR}/torchrun" --redirects 3 --tee 3 \
    -m dclm_example.runtime \
    --arm "${ARM}" --bundle "${BUNDLE}" --output "${RUN_DIR}" \
    --checkpoint-root "${CHECKPOINT_ROOT}" \
    --workload "${WORKLOAD}" \
    --steps "${TMK_STEPS}" \
    --run-id "${TMK_RUN_ID:-$(basename -- "${RUN_DIR}")}" \
    --wandb-mode "${TMK_WANDB_MODE:-disabled}" \
    "${OPTIONAL_ARGS[@]}" \
    2>&1 | tee "${RUN_DIR}/run.log"
