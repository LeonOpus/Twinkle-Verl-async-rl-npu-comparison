#!/usr/bin/env bash
set -Eeuo pipefail

export LANG=C.UTF-8
export LC_ALL=C.UTF-8
export PYTHONUTF8=1
export PYTHONIOENCODING=UTF-8

set +u
source /usr/local/Ascend/ascend-toolkit/set_env.sh || exit 31
if test -f /usr/local/Ascend/cann-9.0.0/share/info/ascendnpu-ir/bin/set_env.sh; then
  source /usr/local/Ascend/cann-9.0.0/share/info/ascendnpu-ir/bin/set_env.sh || true
fi
if test -f /usr/local/Ascend/nnal/atb/set_env.sh; then
  source /usr/local/Ascend/nnal/atb/set_env.sh || true
fi
set -u

ROOT=/mnt/lv_model/npu60005420a
PROJECT="${ROOT}/qy/20260806_Twinkle_Verl"

EXP=verl_full125_rollout_utf8_20260806
VERL_SRC="${ROOT}/qy/qy_stage2_sources_20260803/verl_17ea153_stage2"
CONFIG_FILE="${PROJECT}/configs/${EXP}.yaml"
CONFIG_DIR="${PROJECT}/configs"

OUTPUT_DIR="${PROJECT}/outputs/${EXP}"
CKPTS_DIR="${OUTPUT_DIR}/checkpoints"
RUNLOG="${PROJECT}/logs/${EXP}.log"

MODEL_PATH="${ROOT}/models/Qwen3-4B"
TRAIN_FILE="${ROOT}/datasets/qy/data/gsm8k_owner_aligned_20260729/verl_train_twinkle_aligned.parquet"
TEST_FILE="${ROOT}/datasets/qy/data/gsm8k_owner_aligned_20260729/verl_test_official_twinkle_aligned.parquet"
GSM8K_REWARD_PATH="${PROJECT}/owner_package/gsm8k_accuracy_reward.py"

export ASCEND_RT_VISIBLE_DEVICES=2,3
export TASK_QUEUE_ENABLE=1
export TRANSFER_QUEUE_ENABLE=1
export TOKENIZERS_PARALLELISM=false
export HYDRA_FULL_ERROR=1
export WANDB_MODE=disabled
export RAY_DEDUP_LOGS=0
export RAY_TMPDIR=/mnt/lv_model/npu60005420a/qy/r/f6

export PYTHONPATH="${VERL_SRC}:${PROJECT}/runtime/vllm:${PROJECT}/runtime/vllm-ascend${PYTHONPATH:+:${PYTHONPATH}}"
export VERL_ROOT="${VERL_SRC}"

export MODEL_PATH
export TRAIN_FILE
export TEST_FILE
export OUTPUT_DIR
export CKPTS_DIR
export GSM8K_REWARD_PATH

unset PYTORCH_NPU_ALLOC_CONF

mkdir -p "${OUTPUT_DIR}" "${CKPTS_DIR}" "${RAY_TMPDIR}"

python3 - "${CONFIG_FILE}" "${OUTPUT_DIR}" <<'PY'
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1]))
assert cfg["trainer"]["total_training_steps"] == 125
assert cfg["trainer"]["save_freq"] == 125
assert cfg["trainer"]["test_freq"] == -1
assert cfg["trainer"]["rollout_data_dir"] == "${oc.env:OUTPUT_DIR}/rollouts"
print("PRELAUNCH_CONFIG_IS_FULL125")
print("CONFIG_FILE", sys.argv[1])
print("OUTPUT_DIR", sys.argv[2])
PY

cd "${VERL_SRC}"

python3 -u -m verl.trainer.main_ppo \
  --config-dir="${CONFIG_DIR}" \
  --config-name="${EXP}" \
  2>&1 | tee "${RUNLOG}"

RC=${PIPESTATUS[0]}
echo "VERL_FULL125_RC=${RC}" | tee -a "${RUNLOG}"
exit "${RC}"
