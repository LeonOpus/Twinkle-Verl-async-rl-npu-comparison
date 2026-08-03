#!/usr/bin/env bash
set -euo pipefail

ROOT=/mnt/lv_model/npu60005420a
WT=/workspace/twinkle_verl_cmp_b079_npu

OLD_EXP=qy_owner_twinkle_full125_final_npu23
NEW_EXP=qy_owner_twinkle_full125_noeval_final_npu23

OWNER=$ROOT/packages/twinkle_aligned_verl_grpo_npu_2cards_owner_20260729/twinkle_async_single_lora_gsm8k_accuracy_npu_2cards.yaml

OLD_OUT=$ROOT/qy_owner_experiments_20260730/$OLD_EXP
OLD_CFG=$ROOT/qy_runtime_configs/${OLD_EXP}.yaml
OLD_RUNNER=$ROOT/qy_container_scripts/${OLD_EXP}.sh

NEW_OUT=$ROOT/qy_owner_experiments_20260730/$NEW_EXP
NEW_CFG=$ROOT/qy_runtime_configs/${NEW_EXP}.yaml
NEW_RUNNER=$ROOT/qy_container_scripts/${NEW_EXP}.sh
NEW_DIFF=$ROOT/qy_runtime_configs/${NEW_EXP}.authorized.diff
NEW_NOTE=$ROOT/qy_runtime_configs/${NEW_EXP}.authorization.txt
NEW_LAUNCHER=$ROOT/qy_logs/${NEW_EXP}_launcher.log
NEW_PIDFILE=$ROOT/qy_logs/${NEW_EXP}.pid

EXPECTED_OLD_OUT=$ROOT/qy_owner_experiments_20260730/qy_owner_twinkle_full125_final_npu23
RESOLVED_OLD_OUT=$(readlink -m "$OLD_OUT")

if test "$RESOLVED_OLD_OUT" != "$EXPECTED_OLD_OUT"; then
    echo "ERROR: unsafe old output path: $RESOLVED_OLD_OUT"
    exit 1
fi

echo '=== STOP INCOMPLETE RUN ==='

PIDS=""

if test -f "$ROOT/qy_logs/${OLD_EXP}.pid"; then
    PID=$(tr -d '[:space:]' <"$ROOT/qy_logs/${OLD_EXP}.pid")
    if kill -0 "$PID" 2>/dev/null; then
        PIDS="$PIDS $PID"
    fi
fi

for PIDFILE in \
  "$ROOT/qy_logs/${OLD_EXP}_hbm_explicit_bin.pid"
do
    if test -f "$PIDFILE"; then
        PID=$(tr -d '[:space:]' <"$PIDFILE")
        if kill -0 "$PID" 2>/dev/null; then
            PIDS="$PIDS $PID"
        fi
    fi
done

PY_PIDS=$(pgrep -f \
  "async_multi_lora_grpo.py --config $OLD_CFG" || true)
PIDS="$PIDS $PY_PIDS"

if test -n "$(echo "$PIDS" | xargs)"; then
    echo "stopping_pids=$(echo "$PIDS" | xargs)"
    kill -TERM $PIDS 2>/dev/null || true
fi

for _ in $(seq 1 20); do
    if ! pgrep -f \
      "async_multi_lora_grpo.py --config $OLD_CFG" >/dev/null
    then
        break
    fi
    sleep 1
done

ray stop --force >/dev/null 2>&1 || true

REMAINING=$(pgrep -f \
  "async_multi_lora_grpo.py --config $OLD_CFG" || true)
if test -n "$REMAINING"; then
    kill -KILL $REMAINING 2>/dev/null || true
fi

if pgrep -f \
  "async_multi_lora_grpo.py --config $OLD_CFG" >/dev/null
then
    echo 'ERROR: old Python process still exists'
    exit 1
fi

echo 'old_training_processes_stopped'

for TARGET in "$NEW_OUT" "$NEW_CFG" "$NEW_RUNNER"; do
    if test -e "$TARGET"; then
        echo "ERROR: new target already exists: $TARGET"
        exit 1
    fi
done

echo
echo '=== CREATE AUTHORIZED NO-EVAL CONFIG AND RUNNER ==='

export OWNER OLD_RUNNER NEW_CFG NEW_RUNNER OLD_EXP NEW_EXP NEW_OUT

python3 - <<'PY'
from pathlib import Path
import hashlib
import os

owner = Path(os.environ["OWNER"])
old_runner = Path(os.environ["OLD_RUNNER"])
new_cfg = Path(os.environ["NEW_CFG"])
new_runner = Path(os.environ["NEW_RUNNER"])
old_exp = os.environ["OLD_EXP"]
new_exp = os.environ["NEW_EXP"]
new_out = os.environ["NEW_OUT"]

raw = owner.read_bytes()
owner_sha = hashlib.sha256(raw).hexdigest()
expected = "06fb6643548a609b8a82dc3a29d9a3506e66b762c4d43867ada9d5103b981a0d"
assert owner_sha == expected, (owner_sha, expected)

text = raw.decode("utf-8")

metrics_old = (
    "  output_dir: output/async_single_lora_gsm8k_accuracy\n"
)
metrics_new = (
    "  output_dir: output/async_single_lora_gsm8k_accuracy\n"
    f"  metrics_path: {new_out}/metrics.jsonl\n"
)

eval_switch_old = (
    "evaluation:\n"
    "  enabled: true\n"
)
eval_switch_new = (
    "evaluation:\n"
    "  enabled: false\n"
)

eval_dataset_old = (
    "    eval_dataset:\n"
    "      name: gsm8k/test\n"
)
eval_dataset_new = (
    "    eval_dataset:\n"
    "      reward_type: gsm8k_accuracy\n"
    "      name: gsm8k/test\n"
)

assert text.count(metrics_old) == 1
assert text.count(eval_switch_old) == 1
assert text.count(eval_dataset_old) == 1

text = text.replace(metrics_old, metrics_new, 1)
text = text.replace(eval_switch_old, eval_switch_new, 1)
text = text.replace(eval_dataset_old, eval_dataset_new, 1)
new_cfg.write_text(text, encoding="utf-8", newline="\n")

runner = old_runner.read_text(encoding="utf-8")
old_exp_line = f"EXP={old_exp}\n"
new_exp_line = f"EXP={new_exp}\n"
assert runner.count(old_exp_line) == 1
runner = runner.replace(old_exp_line, new_exp_line, 1)

bare_npu_smi = "        npu-smi info\n"
explicit_npu_smi = "        /usr/local/bin/npu-smi info\n"
if bare_npu_smi in runner:
    runner = runner.replace(bare_npu_smi, explicit_npu_smi, 1)
assert explicit_npu_smi in runner
assert "ERROR: failed to source Ascend toolkit environment" in runner

new_runner.write_text(runner, encoding="utf-8", newline="\n")
new_runner.chmod(0o700)

print(f"owner_sha256={owner_sha}")
print(f"runtime_sha256={hashlib.sha256(new_cfg.read_bytes()).hexdigest()}")
print(f"runner_sha256={hashlib.sha256(new_runner.read_bytes()).hexdigest()}")
print("RESULT: AUTHORIZED_NOEVAL_RUNTIME_CREATED")
PY

echo
echo '=== DELETE INCOMPLETE OLD ARTIFACTS ==='

rm -rf -- "$OLD_OUT"
rm -f -- \
  "$OLD_CFG" \
  "$OLD_RUNNER" \
  "$ROOT/qy_logs/${OLD_EXP}.pid" \
  "$ROOT/qy_logs/${OLD_EXP}.log" \
  "$ROOT/qy_logs/${OLD_EXP}_launcher.log" \
  "$ROOT/qy_logs/${OLD_EXP}_hbm.log" \
  "$ROOT/qy_logs/${OLD_EXP}_hbm_explicit_bin.log" \
  "$ROOT/qy_logs/${OLD_EXP}_hbm_explicit_bin.pid" \
  "$ROOT/qy_logs/${OLD_EXP}_eta10m_report.log"

echo "removed_incomplete_output=$OLD_OUT"

mkdir -p "$NEW_OUT/run_cwd"

diff -u "$OWNER" "$NEW_CFG" >"$NEW_DIFF" || true

cat >"$NEW_NOTE" <<EOF
authorization_time=$(date '+%F %T %z')
authorized_change=evaluation.enabled true -> false
reason=负责人明确同意训练中关闭评测，125-step 完成后单独手动评测官方 GSM8K test 1319 题
unchanged=max_steps,data_num,batch_size,num_generations,learning_rate,lora,loss,reward,sampling
owner_yaml_sha256=06fb6643548a609b8a82dc3a29d9a3506e66b762c4d43867ada9d5103b981a0d
EOF

echo
echo '=== AUTHORIZED CONFIG DIFF ==='
cat "$NEW_DIFF"

echo
echo '=== START NEW NO-EVAL FORMAL RUN ==='

nohup bash "$NEW_RUNNER" >"$NEW_LAUNCHER" 2>&1 </dev/null &
NEW_PID=$!
echo "$NEW_PID" >"$NEW_PIDFILE"
disown "$NEW_PID" 2>/dev/null || true

for _ in $(seq 1 30); do
    if grep -q \
      'RESULT: TWINKLE_FULL125_CONTAINER_PREFLIGHT_PASS' \
      "$NEW_LAUNCHER" 2>/dev/null
    then
        break
    fi
    if ! kill -0 "$NEW_PID" 2>/dev/null; then
        break
    fi
    sleep 1
done

echo "new_runner_pid=$NEW_PID"
ps -p "$NEW_PID" -o pid,ppid,stat,etime,cmd || true
tail -n 60 "$NEW_LAUNCHER" 2>/dev/null || true

echo 'RESULT: AUTHORIZED_NOEVAL_FORMAL_RUN_LAUNCHED'
