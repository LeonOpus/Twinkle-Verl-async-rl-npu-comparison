#!/usr/bin/env bash

ROOT=/mnt/lv_model/npu60005420a
WT=/workspace/twinkle_verl_cmp_b079_npu
PIN=b0798fd7260ce5c62771053c28a4ee731932d868

EXP=qy_owner_twinkle_full125_noeval_final_npu23
OUT=$ROOT/qy_owner_experiments_20260730/$EXP
RUNDIR=$OUT/run_cwd
CFG=$ROOT/qy_runtime_configs/${EXP}.yaml

MODEL=$ROOT/models/Qwen3-4B
TRAIN=$ROOT/data/gsm8k_owner_aligned_20260729/twinkle_train.parquet
TEST=$ROOT/data/gsm8k_owner_aligned_20260729/twinkle_test_official.parquet

RUNLOG=$ROOT/qy_logs/${EXP}.log
HBMLOG=$ROOT/qy_logs/${EXP}_hbm.log
COMMAND_RECORD=$OUT/container_training_command.txt

source /usr/local/Ascend/ascend-toolkit/set_env.sh || {
    echo 'ERROR: failed to source Ascend toolkit environment'
    exit 1
}

if test -f /usr/local/Ascend/cann-9.0.0/share/info/ascendnpu-ir/bin/set_env.sh; then
    source /usr/local/Ascend/cann-9.0.0/share/info/ascendnpu-ir/bin/set_env.sh || {
        echo 'ERROR: failed to source ascendnpu-ir environment'
        exit 1
    }
fi

if test -f /usr/local/Ascend/nnal/atb/set_env.sh; then
    source /usr/local/Ascend/nnal/atb/set_env.sh || {
        echo 'ERROR: failed to source ATB environment'
        exit 1
    }
fi

set -euo pipefail

export ASCEND_RT_VISIBLE_DEVICES=2,3
unset PYTORCH_NPU_ALLOC_CONF
export TASK_QUEUE_ENABLE=1
export HCCL_CONNECT_TIMEOUT=1500
export HCCL_HOST_SOCKET_PORT_RANGE=62000-62050
export HCCL_NPU_SOCKET_PORT_RANGE=63000-63050
export MODEL_ID="$MODEL"
export GSM8K_TRAIN_DATASET_ID="$TRAIN"
export GSM8K_TEST_DATASET_ID="$TEST"
export RAY_DEDUP_LOGS=0
export PYTHONPATH="$WT/src:$WT:${PYTHONPATH:-}"

for path in "$WT" "$MODEL" "$TRAIN" "$TEST" "$CFG" "$RUNDIR"; do
    test -e "$path"
done

test "$(git -C "$WT" rev-parse HEAD)" = "$PIN"

grep -q "if 'group_id' in source:" \
  "$WT/src/twinkle_agentic/async_rl/vllm_sampler_tq.py"

cd "$RUNDIR"
ray stop --force >/dev/null 2>&1 || true

cat > "$COMMAND_RECORD" <<EOF
container=qy_twinkle_verl_cmp_20260724
container_workdir=$RUNDIR
ASCEND_RT_VISIBLE_DEVICES=2,3
python3 -u $WT/cookbook/rl/async_multi_lora_grpo.py --config $CFG
EOF

echo '=== FORMAL IDENTITY ==='
echo "twinkle_commit=$(git -C "$WT" rev-parse HEAD)"
echo "config=$CFG"
sha256sum "$CFG"
echo "model=$MODEL"
echo "train=$TRAIN"
echo "test=$TEST"
echo "ASCEND_RT_VISIBLE_DEVICES=$ASCEND_RT_VISIBLE_DEVICES"
echo "PYTORCH_NPU_ALLOC_CONF=${PYTORCH_NPU_ALLOC_CONF-}"
echo "RESULT: TWINKLE_FULL125_CONTAINER_PREFLIGHT_PASS"

(
    while true; do
        echo "timestamp=$(date '+%F %T')"
        /usr/local/bin/npu-smi info
        sleep 10
    done
) >"$HBMLOG" 2>&1 &

MONITOR_PID=$!

cleanup() {
    kill "$MONITOR_PID" >/dev/null 2>&1 || true
    wait "$MONITOR_PID" >/dev/null 2>&1 || true
}
trap cleanup EXIT

START=$(date +%s)

set +e
python3 -u \
  "$WT/cookbook/rl/async_multi_lora_grpo.py" \
  --config "$CFG" 2>&1 | tee "$RUNLOG"
RUN_RC=${PIPESTATUS[0]}
set -e

END=$(date +%s)
ELAPSED=$((END - START))

cleanup
trap - EXIT

echo "QY_TWINKLE_RUN_RC=$RUN_RC" | tee -a "$RUNLOG"
echo "QY_TWINKLE_RUN_ELAPSED_SECONDS=$ELAPSED" | tee -a "$RUNLOG"

if test "$RUN_RC" = 0; then
    echo 'RESULT: OWNER_TWINKLE_FULL125_FINAL_RUN_PASS' | tee -a "$RUNLOG"
else
    echo "RESULT: OWNER_TWINKLE_FULL125_FINAL_RUN_FAIL_RC=$RUN_RC" | tee -a "$RUNLOG"
fi

exit "$RUN_RC"
