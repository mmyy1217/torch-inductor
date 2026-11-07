set -eo pipefail
root=/home/yja/experiments/tilelang-backend-a5-20261106
unset A5_REDUCTION_SMOKE
export A5_ALLOW_GUARD_QUEUE=1 A5_REDUCTION_EVIDENCE=evidence-reduction-02
bash "$root/run-reduction-validation.sh"
npu-guard status --json > "$root/$A5_REDUCTION_EVIDENCE/guard-after.json" || true
