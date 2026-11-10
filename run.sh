#!/usr/bin/env bash
set -euo pipefail
# Activate your matched environment and source CANN before invoking this script.
: "${ASCEND_RT_VISIBLE_DEVICES:?Set ASCEND_RT_VISIBLE_DEVICES to your allocated device}"
export TORCHINDUCTOR_COMPILE_THREADS="${TORCHINDUCTOR_COMPILE_THREADS:-1}"
export TORCHNPU_PRECOMPILE_THREADS="${TORCHNPU_PRECOMPILE_THREADS:-2}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export PYTHONSAFEPATH=1
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# Empty PYTHONPATH entries can accidentally import this repository's stale extensions.
PYTHONPATH="$(python -P -c 'import os; print(":".join(p for p in os.environ.get("PYTHONPATH", "").split(":") if p))')"
export PYTHONPATH
exec python -P "$script_dir/run.py" "$@"
