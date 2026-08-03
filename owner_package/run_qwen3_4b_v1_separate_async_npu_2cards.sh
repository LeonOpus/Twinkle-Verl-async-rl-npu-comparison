#!/usr/bin/env bash
set -xeuo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=${VERL_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}
cd "${ROOT_DIR}"
export VERL_ROOT="${ROOT_DIR}"
export GSM8K_REWARD_PATH=${GSM8K_REWARD_PATH:-"${SCRIPT_DIR}/gsm8k_accuracy_reward.py"}

export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0,1}
export HCCL_CONNECT_TIMEOUT=${HCCL_CONNECT_TIMEOUT:-1500}
export HCCL_HOST_SOCKET_PORT_RANGE=${HCCL_HOST_SOCKET_PORT_RANGE:-60000-60050}
export HCCL_NPU_SOCKET_PORT_RANGE=${HCCL_NPU_SOCKET_PORT_RANGE:-61000-61050}
export VLLM_USE_V1=${VLLM_USE_V1:-1}
export TASK_QUEUE_ENABLE=${TASK_QUEUE_ENABLE:-2}
export CPU_AFFINITY_CONF=${CPU_AFFINITY_CONF:-1}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export VERL_LOGGING_LEVEL=${VERL_LOGGING_LEVEL:-INFO}

export RAY_DATA_HOME=${RAY_DATA_HOME:-"${HOME}/verl"}
export MODEL_PATH=${MODEL_PATH:-/nas/disk1/Qwen3-4B}
export TRAIN_FILE=${TRAIN_FILE:-/mnt/lv_model/npu60005420a/data/gsm8k_async_cmp2000/verl_train_twinkle_aligned.parquet}
export TEST_FILE=${TEST_FILE:-/mnt/lv_model/npu60005420a/data/gsm8k_async_cmp2000/verl_test_official_twinkle_aligned.parquet}
export OUTPUT_DIR=${OUTPUT_DIR:-/mnt/lv_model/npu60005420a/qy_outputs/verl_reviewed_cmp}
export CKPTS_DIR=${CKPTS_DIR:-"${OUTPUT_DIR}"}

python3 -c "import torch_npu, transfer_queue, vllm"
test -f "${GSM8K_REWARD_PATH}"
test -f "${TRAIN_FILE}"
test -f "${TEST_FILE}"
mkdir -p "${CKPTS_DIR}"
mkdir -p "${OUTPUT_DIR}"

python3 - "${TRAIN_FILE}" "${TEST_FILE}" <<'PY'
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
            raise ValueError(f"{path}:{index} does not use the Twinkle GSM8KProcessor prompt contract")
        try:
            float(ground_truth)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{path}:{index} does not contain a numeric GSM8K ground truth") from error
print("Twinkle GSM8KProcessor prompt and ground-truth contract validated")
PY

python3 -m verl.trainer.main_ppo \
  --config-name=v1_separate_async_grpo_npu_2cards \
  "$@"



# cd /Users/linjiajia/project/verl

# ray stop --force
# unset RAY_ADDRESS

# export ASCEND_RT_VISIBLE_DEVICES=2,3

# bash examples/ascend_extras/grpo_trainer/run_qwen3_4b_v1_separate_async_npu_2cards.sh
