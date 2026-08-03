#!/usr/bin/env bash

ROOT=/mnt/lv_model/npu60005420a
WT=/workspace/verl_official_main_17ea153d95ee_20260729
PIN=17ea153d95eeb461815f547ceee3cfbac6ff6d11

EXP=qy_owner_verl_full125_noeval_final_npu23
CONFIG_NAME=$EXP
CFG=$ROOT/qy_runtime_configs/${EXP}.yaml
INSTALLED_CFG=$WT/verl/trainer/config/${EXP}.yaml

OUT=$ROOT/qy_owner_experiments_20260730/$EXP
RUNDIR=$OUT/run_cwd
RUNLOG=$ROOT/qy_logs/${EXP}.log
HBMLOG=$ROOT/qy_logs/${EXP}_hbm.log
COMMAND_RECORD=$OUT/container_training_command.txt

MODEL=$ROOT/models/Qwen3-4B
TRAIN=$ROOT/data/gsm8k_owner_aligned_20260729/verl_train_twinkle_aligned.parquet
TEST=$ROOT/data/gsm8k_owner_aligned_20260729/verl_test_official_twinkle_aligned.parquet
REWARD=$ROOT/packages/twinkle_aligned_verl_grpo_npu_2cards_owner_20260729/gsm8k_accuracy_reward.py

source /usr/local/Ascend/ascend-toolkit/set_env.sh || {
    echo 'ERROR: failed to source Ascend toolkit environment'
    exit 1
}

if test -f /usr/local/Ascend/cann-9.0.0/share/info/ascendnpu-ir/bin/set_env.sh; then
    source /usr/local/Ascend/cann-9.0.0/share/info/ascendnpu-ir/bin/set_env.sh || exit 1
fi

if test -f /usr/local/Ascend/nnal/atb/set_env.sh; then
    source /usr/local/Ascend/nnal/atb/set_env.sh || exit 1
fi

set -euo pipefail

export ASCEND_RT_VISIBLE_DEVICES=2,3
unset PYTORCH_NPU_ALLOC_CONF
unset RAY_ADDRESS

export HCCL_CONNECT_TIMEOUT=1500
export HCCL_HOST_SOCKET_PORT_RANGE=60000-60050
export HCCL_NPU_SOCKET_PORT_RANGE=61000-61050
export VLLM_USE_V1=1
export TASK_QUEUE_ENABLE=1
export CPU_AFFINITY_CONF=1
export TOKENIZERS_PARALLELISM=false
export VERL_LOGGING_LEVEL=INFO
export WANDB_MODE=disabled
export RAY_DEDUP_LOGS=0

export VERL_ROOT="$WT"
export PYTHONPATH="$WT:${PYTHONPATH:-}"
export RAY_DATA_HOME="$OUT/ray_data"
export MODEL_PATH="$MODEL"
export TRAIN_FILE="$TRAIN"
export TEST_FILE="$TEST"
export OUTPUT_DIR="$OUT"
export CKPTS_DIR="$OUT/checkpoints"
export GSM8K_REWARD_PATH="$REWARD"

for TARGET in \
  "$WT" "$MODEL" "$TRAIN" "$TEST" "$REWARD" \
  "$CFG" "$INSTALLED_CFG" "$RUNDIR"
do
    test -e "$TARGET"
done

test "$(git -C "$WT" rev-parse HEAD)" = "$PIN"
cmp -s "$CFG" "$INSTALLED_CFG"

cd "$RUNDIR"
ray stop --force >/dev/null 2>&1 || true

python3 - <<'PY'
import verl
print(f"verl_module={verl.__file__}")
PY

python3 -c "import torch_npu, transfer_queue, vllm"

python3 - "$TRAIN" "$TEST" <<'PY'
import sys
import pyarrow.parquet as pq

system_prompt = (
    "You are a helpful math assistant. Solve the problem step by step "
    "and put your final answer within \\boxed{}."
)

for path in sys.argv[1:]:
    rows = pq.read_table(path, columns=["prompt", "reward_model"]).to_pylist()
    if not rows:
        raise ValueError(f"{path} is empty")
    for index, row in enumerate(rows):
        prompt = row["prompt"]
        ground_truth = row["reward_model"]["ground_truth"]
        if (
            len(prompt) != 2
            or prompt[0] != {"role": "system", "content": system_prompt}
            or prompt[1]["role"] != "user"
        ):
            raise ValueError(f"{path}:{index} prompt contract mismatch")
        float(ground_truth)

print("Twinkle-aligned VERL prompt and ground-truth contract validated")
PY

cat >"$COMMAND_RECORD" <<EOF
container=qy_verl_cmp_20260727
container_workdir=$RUNDIR
ASCEND_RT_VISIBLE_DEVICES=2,3
PYTORCH_NPU_ALLOC_CONF=unset
python3 -u -m verl.trainer.main_ppo --config-name=$CONFIG_NAME
EOF

echo '=== FORMAL VERL IDENTITY ==='
echo "verl_commit=$(git -C "$WT" rev-parse HEAD)"
echo "config=$CFG"
sha256sum "$CFG"
echo "installed_config=$INSTALLED_CFG"
sha256sum "$INSTALLED_CFG"
echo "model=$MODEL"
echo "train=$TRAIN"
sha256sum "$TRAIN"
echo "test=$TEST"
sha256sum "$TEST"
echo "reward=$REWARD"
sha256sum "$REWARD"
echo "ASCEND_RT_VISIBLE_DEVICES=$ASCEND_RT_VISIBLE_DEVICES"
echo "PYTORCH_NPU_ALLOC_CONF=${PYTORCH_NPU_ALLOC_CONF-}"
echo 'RESULT: VERL_FULL125_NOEVAL_CONTAINER_PREFLIGHT_PASS'

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
python3 -u -m verl.trainer.main_ppo \
  --config-name="$CONFIG_NAME" \
  2>&1 | tee "$RUNLOG"

RUN_RC=${PIPESTATUS[0]}
set -e

END=$(date +%s)
ELAPSED=$((END - START))

cleanup
trap - EXIT
ray stop --force >/dev/null 2>&1 || true

echo "QY_VERL_RUN_RC=$RUN_RC" | tee -a "$RUNLOG"
echo "QY_VERL_RUN_ELAPSED_SECONDS=$ELAPSED" | tee -a "$RUNLOG"

if test "$RUN_RC" = 0; then
    echo 'RESULT: OWNER_VERL_FULL125_NOEVAL_RUN_PASS' | tee -a "$RUNLOG"
else
    echo "RESULT: OWNER_VERL_FULL125_NOEVAL_RUN_FAIL_RC=$RUN_RC" | tee -a "$RUNLOG"
fi

exit "$RUN_RC"
