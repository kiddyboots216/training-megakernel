#!/usr/bin/env bash
set -Eeuo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
if [[ $# -ne 4 ]]; then
  echo "usage: $0 BUNDLE RUN_DIR CHECKPOINT_ROOT DCLM_SHARDS" >&2
  exit 2
fi
exec bash "${HERE}/run_world8.sh" checkpoint "$@"
