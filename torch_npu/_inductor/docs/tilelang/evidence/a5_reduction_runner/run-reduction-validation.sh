set -eo pipefail
source /home/yja/activate-tilelang-950.sh
root=/home/yja/experiments/tilelang-backend-a5-20261106
export PATH="$root/env/bin:$PATH"
export PYTHONPATH="$root/tilelang"
unset TVM_IMPORT_PYTHON_PATH TVM_LIBRARY_PATH
export TORCH_DEVICE_BACKEND_AUTOLOAD=0 TORCH_WARM_POOL=0 TORCHINDUCTOR_COMPILE_THREADS=1
evidence_dir="$root/${A5_REDUCTION_EVIDENCE:-evidence-reduction}"
export TORCHINDUCTOR_CACHE_DIR="$evidence_dir/inductor-cache"
export TILELANG_CACHE_DIR="$evidence_dir/tilelang-cache"
export TILELANG_KERNEL_CACHE_USE_LIB_STAMP=1
export TILELANG_CLEANUP_TEMP_FILES=0
export INDUCTOR_ASCEND_TILELANG_AUTOTUNE=0 TORCH_COMPILE_DEBUG=1
mkdir -p "$evidence_dir"
cd "$root"
test -s pytorch-wheel.sha256
test -s library.sha256
npu-guard status --json | tee "$evidence_dir/guard-before.json" || test "${A5_ALLOW_GUARD_QUEUE:-0}" = 1
python - <<'PY'
import json
import os
s=json.load(open(os.path.join(os.environ.get('A5_REDUCTION_EVIDENCE', 'evidence-reduction'), 'guard-before.json')))
assert not s['errors'], s
assert s['state']=='idle' or (os.environ.get('A5_ALLOW_GUARD_QUEUE')=='1' and s['state']=='busy'), s
PY
npu-guard run --owner "$USER/tilelang-backend-a5-reduction" --wait 3600 --timeout 3600 -- python "$root/run-reduction-validation.py" 2>&1 | tee "$evidence_dir/validation.log"
