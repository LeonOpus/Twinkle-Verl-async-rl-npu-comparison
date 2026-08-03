#!/usr/bin/env bash
set -euo pipefail

ROOT=/mnt/lv_model/npu60005420a
WT=/workspace/twinkle_verl_cmp_b079_npu

EXP=qy_owner_twinkle_v125_manual_eval_official1319_npu3
BASE_EVAL=$WT/cookbook/rl/eval_gsm8k_areal.py
EVAL=$ROOT/qy_container_scripts/qy_eval_gsm8k_owner_protocol.py
DIFF=$ROOT/qy_container_scripts/qy_eval_gsm8k_owner_protocol.diff
RUNNER=$ROOT/qy_container_scripts/${EXP}.sh

OUT=$ROOT/qy_owner_experiments_20260730/$EXP
LAUNCHER=$ROOT/qy_logs/${EXP}_launcher.log
PIDFILE=$ROOT/qy_logs/${EXP}.pid

for TARGET in "$EVAL" "$DIFF" "$RUNNER" "$OUT" "$LAUNCHER" "$PIDFILE"; do
    if test -e "$TARGET"; then
        echo "ERROR: target already exists: $TARGET"
        exit 1
    fi
done

export BASE_EVAL EVAL

python3 - <<'PY'
from pathlib import Path
import hashlib
import os

source = Path(os.environ["BASE_EVAL"])
target = Path(os.environ["EVAL"])
text = source.read_text(encoding="utf-8")

replacements = [
    (
        '"""Evaluate a Qwen3 base model or LoRA with the AReaL GSM8K protocol."""',
        '"""Evaluate a Qwen3 LoRA with the owner-aligned Twinkle GSM8K protocol."""',
    ),
    (
        "from twinkle.preprocessor import AReaLGSM8KProcessor",
        "from twinkle.preprocessor import GSM8KProcessor",
    ),
    (
        "from twinkle.reward import MathVerifyAccuracyReward",
        "from twinkle.reward import GSM8KAccuracyReward",
    ),
    (
        "parser.add_argument('--adapter-name', default='gsm8k_areal_pk_lora')",
        "parser.add_argument('--adapter-name', default='gsm8k_accuracy_lora')",
    ),
    (
        "processor = AReaLGSM8KProcessor()",
        "processor = GSM8KProcessor()",
    ),
    (
        "reward_fn = MathVerifyAccuracyReward()",
        "reward_fn = GSM8KAccuracyReward()",
    ),
    (
        "'GSM8K AReaL eval: prompts=%s/%s samples=%s accuracy=%.4f'",
        "'GSM8K owner eval: prompts=%s/%s samples=%s accuracy=%.4f'",
    ),
    (
        "'processor': 'AReaLGSM8KProcessor'",
        "'processor': 'GSM8KProcessor'",
    ),
    (
        "'reward': 'MathVerifyAccuracyReward'",
        "'reward': 'GSM8KAccuracyReward'",
    ),
]

for old, new in replacements:
    assert text.count(old) == 1, old
    text = text.replace(old, new, 1)

target.write_text(text, encoding="utf-8", newline="\n")

print(f"base_eval_sha256={hashlib.sha256(source.read_bytes()).hexdigest()}")
print(f"owner_eval_sha256={hashlib.sha256(target.read_bytes()).hexdigest()}")
print("RESULT: OWNER_PROTOCOL_EVALUATOR_CREATED")
PY

diff -u "$BASE_EVAL" "$EVAL" >"$DIFF" || true

cat >"$RUNNER" <<'EVAL_RUNNER'
#!/usr/bin/env bash

ROOT=/mnt/lv_model/npu60005420a
WT=/workspace/twinkle_verl_cmp_b079_npu
PIN=b0798fd7260ce5c62771053c28a4ee731932d868

EXP=qy_owner_twinkle_v125_manual_eval_official1319_npu3
TRAIN_EXP=qy_owner_twinkle_full125_noeval_final_npu23

EVAL=$ROOT/qy_container_scripts/qy_eval_gsm8k_owner_protocol.py
MODEL=$ROOT/models/Qwen3-4B
TEST=$ROOT/data/gsm8k_owner_aligned_20260729/twinkle_test_official.parquet
ADAPTER=$ROOT/qy_owner_experiments_20260730/$TRAIN_EXP/run_cwd/output/async_single_lora_gsm8k_accuracy/async-gsm8k_accuracy_lora-v125

OUT=$ROOT/qy_owner_experiments_20260730/$EXP
RUNLOG=$ROOT/qy_logs/${EXP}.log
HBMLOG=$ROOT/qy_logs/${EXP}_hbm.log
COMMAND_RECORD=$OUT/evaluation_command.txt

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

export ASCEND_RT_VISIBLE_DEVICES=3
unset PYTORCH_NPU_ALLOC_CONF
export TASK_QUEUE_ENABLE=1
export HCCL_CONNECT_TIMEOUT=1500
export HCCL_HOST_SOCKET_PORT_RANGE=64000-64050
export HCCL_NPU_SOCKET_PORT_RANGE=65000-65050
export RAY_DEDUP_LOGS=0
export PYTHONPATH="$WT/src:$WT:${PYTHONPATH:-}"

for TARGET in "$WT" "$EVAL" "$MODEL" "$TEST" "$ADAPTER"; do
    test -e "$TARGET"
done

test "$(git -C "$WT" rev-parse HEAD)" = "$PIN"

mkdir -p "$OUT"
ray stop --force >/dev/null 2>&1 || true

cat >"$COMMAND_RECORD" <<EOF
ASCEND_RT_VISIBLE_DEVICES=3
model=$MODEL
dataset=$TEST
adapter=$ADAPTER
batch_size=16
max_tokens=1024
temperature=0.6
top_p=1.0
num_samples=1
max_model_len=2048
max_num_seqs=64
gpu_memory_utilization=0.7
seed=1
EOF

echo '=== FORMAL MANUAL EVAL IDENTITY ==='
echo "twinkle_commit=$(git -C "$WT" rev-parse HEAD)"
echo "evaluator=$EVAL"
sha256sum "$EVAL"
echo "model=$MODEL"
echo "test=$TEST"
sha256sum "$TEST"
echo "adapter=$ADAPTER"
sha256sum "$ADAPTER/adapter_model.safetensors"
echo 'protocol=GSM8KProcessor+GSM8KAccuracyReward'
echo 'sample_count=1319'
echo 'RESULT: TWINKLE_V125_MANUAL_EVAL_PREFLIGHT_PASS'

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
python3 -u "$EVAL" \
  --model-id "$MODEL" \
  --dataset-id "$TEST" \
  --subset-name main \
  --split test \
  --data-num 0 \
  --adapter-path "$ADAPTER" \
  --adapter-name gsm8k_accuracy_lora \
  --max-lora-rank 16 \
  --batch-size 16 \
  --max-tokens 1024 \
  --temperature 0.6 \
  --top-p 1.0 \
  --num-samples 1 \
  --max-model-len 2048 \
  --max-length 2048 \
  --max-num-seqs 64 \
  --enable-thinking \
  --no-enforce-eager \
  --mode ray \
  --gpus 1 \
  --tp-size 1 \
  --gpu-memory-utilization 0.7 \
  --seed 1 \
  --output-dir "$OUT" \
  2>&1 | tee "$RUNLOG"

RUN_RC=${PIPESTATUS[0]}
set -e

END=$(date +%s)
ELAPSED=$((END - START))

cleanup
trap - EXIT
ray stop --force >/dev/null 2>&1 || true

echo "QY_TWINKLE_MANUAL_EVAL_RC=$RUN_RC" | tee -a "$RUNLOG"
echo "QY_TWINKLE_MANUAL_EVAL_ELAPSED_SECONDS=$ELAPSED" | tee -a "$RUNLOG"

if test "$RUN_RC" = 0; then
    echo 'RESULT: OWNER_TWINKLE_V125_MANUAL_EVAL_PASS' | tee -a "$RUNLOG"
else
    echo "RESULT: OWNER_TWINKLE_V125_MANUAL_EVAL_FAIL_RC=$RUN_RC" | tee -a "$RUNLOG"
fi

exit "$RUN_RC"
EVAL_RUNNER

chmod 700 "$RUNNER"
mkdir -p "$OUT"

echo '=== EVALUATOR DIFF ==='
cat "$DIFF"

nohup bash "$RUNNER" >"$LAUNCHER" 2>&1 </dev/null &
PID=$!
echo "$PID" >"$PIDFILE"
disown "$PID" 2>/dev/null || true

for _ in $(seq 1 30); do
    if grep -q \
      'RESULT: TWINKLE_V125_MANUAL_EVAL_PREFLIGHT_PASS' \
      "$LAUNCHER" 2>/dev/null
    then
        break
    fi
    if ! kill -0 "$PID" 2>/dev/null; then
        break
    fi
    sleep 1
done

echo "eval_runner_pid=$PID"
ps -p "$PID" -o pid,ppid,stat,etime,cmd || true
tail -n 80 "$LAUNCHER" 2>/dev/null || true

echo 'RESULT: TWINKLE_V125_MANUAL_EVAL_LAUNCHED'
