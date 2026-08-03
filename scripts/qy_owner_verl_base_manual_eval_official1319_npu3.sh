#!/usr/bin/env bash

ROOT=/mnt/lv_model/npu60005420a
WT=/workspace/verl_official_main_17ea153d95ee_20260729

TRAIN_EXP=qy_owner_verl_full125_noeval_final_npu23
EVAL_EXP=qy_owner_verl_base_manual_eval_official1319_npu3

MODEL=$ROOT/models/Qwen3-4B
ADAPTER=
TEST=$ROOT/data/gsm8k_owner_aligned_20260729/verl_test_official_twinkle_aligned.parquet
REWARD=$ROOT/packages/twinkle_aligned_verl_grpo_npu_2cards_owner_20260729/gsm8k_accuracy_reward.py

OUT=$ROOT/qy_owner_experiments_20260730/$EVAL_EXP
GENERATED=$OUT/generated.parquet
PREDICTIONS=$OUT/predictions.jsonl
SUMMARY=$OUT/summary.json

source /usr/local/Ascend/ascend-toolkit/set_env.sh || exit 1

if test -f /usr/local/Ascend/cann-9.0.0/share/info/ascendnpu-ir/bin/set_env.sh; then
    source /usr/local/Ascend/cann-9.0.0/share/info/ascendnpu-ir/bin/set_env.sh || exit 1
fi

if test -f /usr/local/Ascend/nnal/atb/set_env.sh; then
    source /usr/local/Ascend/nnal/atb/set_env.sh || exit 1
fi

export PYTHONPATH=$WT:${PYTHONPATH:-}
export ASCEND_RT_VISIBLE_DEVICES=3
export TASK_QUEUE_ENABLE=1
export VLLM_USE_V1=1
export WANDB_MODE=disabled
export TOKENIZERS_PARALLELISM=true

export MODEL_PATH=$MODEL
export TRAIN_FILE=$TEST
export TEST_FILE=$TEST
export OUTPUT_DIR=$OUT
export GSM8K_REWARD_PATH=$REWARD
export CKPTS_DIR=$OUT/checkpoints_unused

unset PYTORCH_NPU_ALLOC_CONF
unset RAY_ADDRESS

START_SECONDS=$(date +%s)

cd "$OUT"

python3 -u -m verl.trainer.main_generation_server \
  --config-name="$EVAL_EXP"

GEN_RC=$?

if test "$GEN_RC" != 0; then
    echo "QY_VERL_MANUAL_EVAL_GENERATION_RC=$GEN_RC"
    echo "RESULT: VERL_BASE_MANUAL_EVAL_GENERATION_FAIL_RC=$GEN_RC"
    exit "$GEN_RC"
fi

python3 - "$GENERATED" "$PREDICTIONS" "$SUMMARY" "$REWARD" "$MODEL" "$ADAPTER" <<'PY_SCORE'
import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd

generated = Path(sys.argv[1])
predictions = Path(sys.argv[2])
summary_path = Path(sys.argv[3])
reward_path = Path(sys.argv[4])
model_path = sys.argv[5]
adapter_path = sys.argv[6]

spec = importlib.util.spec_from_file_location(
    "owner_gsm8k_accuracy_reward",
    reward_path,
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
compute_score = module.compute_score

frame = pd.read_parquet(generated)

if len(frame) != 1319:
    raise RuntimeError(f"expected 1319 rows, got {len(frame)}")

rows = []
correct_total = 0.0

for index, row in frame.iterrows():
    responses = row["responses"]

    if len(responses) != 1:
        raise RuntimeError(
            f"row {index}: expected one response, got {len(responses)}"
        )

    response = str(responses[0])
    reward_data = row["reward_model"]
    ground_truth = str(reward_data["ground_truth"])

    score = float(
        compute_score(
            row["data_source"],
            response,
            ground_truth,
        )
    )

    correct_total += score

    rows.append(
        {
            "index": int(index),
            "response": response,
            "ground_truth": ground_truth,
            "correct": score,
            "model_path": model_path,
            "adapter_path": adapter_path,
        }
    )

with predictions.open("w", encoding="utf-8") as file:
    for row in rows:
        file.write(json.dumps(row, ensure_ascii=False) + "\n")

accuracy = correct_total / len(rows)

summary = {
    "eval/count": len(rows),
    "eval/correct": int(correct_total),
    "eval/incorrect": int(len(rows) - correct_total),
    "eval/accuracy": accuracy,
    "model_path": model_path,
    "adapter_path": adapter_path,
    "protocol": {
        "dataset": "official_gsm8k_test_1319",
        "reward": "owner_gsm8k_accuracy_reward.compute_score",
        "temperature": 0.6,
        "top_p": 1.0,
        "top_k": -1,
        "n": 1,
        "max_tokens": 1024,
        "enable_thinking": True,
    },
}

summary_path.write_text(
    json.dumps(summary, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

print(json.dumps(summary, ensure_ascii=False, indent=2))
print("RESULT: VERL_BASE_MANUAL_EVAL_SCORING_PASS")
PY_SCORE

SCORE_RC=$?
END_SECONDS=$(date +%s)
ELAPSED_SECONDS=$((END_SECONDS - START_SECONDS))

echo "QY_VERL_MANUAL_EVAL_GENERATION_RC=$GEN_RC"
echo "QY_VERL_MANUAL_EVAL_SCORING_RC=$SCORE_RC"
echo "QY_VERL_MANUAL_EVAL_ELAPSED_SECONDS=$ELAPSED_SECONDS"

if test "$SCORE_RC" = 0; then
    echo "RESULT: OWNER_VERL_BASE_MANUAL_EVAL_PASS"
else
    echo "RESULT: OWNER_VERL_BASE_MANUAL_EVAL_FAIL_RC=$SCORE_RC"
fi

exit "$SCORE_RC"
