#!/usr/bin/env bash
set -Eeuo pipefail
export LC_ALL=C
umask 077

ROOT=/mnt/lv_model/npu60005420a
PROJECT="${ROOT}/qy/20260806_Twinkle_Verl"
CONTAINER=qy_verl_stage2_clean_20260804

EXP=verl_full125_rollout_utf8_20260806
CONFIG="${PROJECT}/configs/${EXP}.yaml"
RUNNER="${PROJECT}/scripts/run_${EXP}.sh"
RUNLOG="${PROJECT}/logs/${EXP}.log"
RUNPID="${PROJECT}/logs/${EXP}.pid"
OUT="${PROJECT}/outputs/${EXP}"

SMOKE_CONFIG="${PROJECT}/configs/verl_smoke5_rollout_utf8.yaml"

trap 'rc=$?; printf "FULL125_LAUNCH_FAILED rc=%s line=%s\n" "${rc}" "${BASH_LINENO[0]}"; exit "${rc}"' ERR

printf '\n[1/7] Preconditions\n'
test -f "${SMOKE_CONFIG}"
test -f "${PROJECT}/owner_package/gsm8k_accuracy_reward.py"
test -d "${PROJECT}/runtime/vllm"
test -d "${PROJECT}/runtime/vllm-ascend"
test -d "${ROOT}/qy/qy_stage2_sources_20260803/verl_17ea153_stage2"
test -d "${ROOT}/models/Qwen3-4B"
test -f "${ROOT}/datasets/qy/data/gsm8k_owner_aligned_20260729/verl_train_twinkle_aligned.parquet"
test -f "${ROOT}/datasets/qy/data/gsm8k_owner_aligned_20260729/verl_test_official_twinkle_aligned.parquet"

test ! -e "${CONFIG}"
test ! -e "${RUNNER}"
test ! -e "${RUNLOG}"
test ! -e "${RUNPID}"
test ! -e "${OUT}"

printf '\n[2/7] Check NPU 2/3 idle\n'
NPU_STATE=$(npu-smi info)
for npu in 2 3; do
  if grep -F "No running processes found in NPU ${npu}" <<< "${NPU_STATE}" >/dev/null; then
    printf 'NPU_%s_IDLE=yes\n' "${npu}"
  else
    printf 'NPU_%s_NOT_IDLE\n' "${npu}"
    printf '%s\n' "${NPU_STATE}"
    exit 20
  fi
done

printf '\n[3/7] Build full125 config\n'
cp -a "${SMOKE_CONFIG}" "${CONFIG}"

sed -i \
  -e 's/^  save_freq: .*/  save_freq: 125/' \
  -e 's/^  total_training_steps: .*/  total_training_steps: 125/' \
  -e 's/^  test_freq: .*/  test_freq: -1/' \
  "${CONFIG}"

docker exec -w / "${CONTAINER}" python3 - "${CONFIG}" <<'PY'
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1]))
print("total_training_steps", cfg["trainer"].get("total_training_steps"))
print("save_freq", cfg["trainer"].get("save_freq"))
print("test_freq", cfg["trainer"].get("test_freq"))
print("rollout_data_dir", cfg["trainer"].get("rollout_data_dir"))
print("max_response_length", cfg["data"].get("max_response_length"))
print("temperature", cfg["actor_rollout_ref"]["rollout"].get("temperature"))
assert cfg["trainer"].get("total_training_steps") == 125
assert cfg["trainer"].get("save_freq") == 125
assert cfg["trainer"].get("test_freq") == -1
assert cfg["trainer"].get("rollout_data_dir") == "${oc.env:OUTPUT_DIR}/rollouts"
assert cfg["data"].get("max_response_length") == 1024
assert float(cfg["actor_rollout_ref"]["rollout"].get("temperature")) == 1.0
print("FULL125_CONFIG_VALID")
PY

printf '\n[4/7] Build clean full125 runner\n'
cat > "${RUNNER}" <<'RUNNER'
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
RUNNER

chmod 0755 "${RUNNER}"
bash -n "${RUNNER}"

printf '\n[5/7] Runner audit\n'
grep -nE 'EXP=|RAY_TMPDIR|CONFIG_FILE|CONFIG_DIR|OUTPUT_DIR|TRAIN_FILE|TEST_FILE|GSM8K_REWARD_PATH|PYTHONPATH|PRELAUNCH|config-dir|config-name' "${RUNNER}"

printf '\n[6/7] Runtime import\n'
docker exec -w / "${CONTAINER}" bash -lc '
  set +e
  set +u
  source /usr/local/Ascend/ascend-toolkit/set_env.sh || exit 31
  export LANG=C.UTF-8
  export LC_ALL=C.UTF-8
  export PYTHONUTF8=1
  export PYTHONIOENCODING=UTF-8
  export PYTHONPATH=/mnt/lv_model/npu60005420a/qy/qy_stage2_sources_20260803/verl_17ea153_stage2:/mnt/lv_model/npu60005420a/qy/20260806_Twinkle_Verl/runtime/vllm:/mnt/lv_model/npu60005420a/qy/20260806_Twinkle_Verl/runtime/vllm-ascend
  python3 -c "import torch, torch_npu, verl, vllm, vllm_ascend; assert torch.npu.device_count()==2; print(\"FULL125_RUNTIME_VALID\", torch.npu.device_count())"
'

printf '\n[7/7] Launch full125 nohup inside container\n'
docker exec -w / "${CONTAINER}" bash -lc "
  unset PYTORCH_NPU_ALLOC_CONF
  nohup bash '${RUNNER}' > '${RUNLOG}' 2>&1 < /dev/null &
  echo \$! > '${RUNPID}'
"

printf 'FULL125_STARTED\n'
printf 'RUNLOG=%s\n' "${RUNLOG}"
printf 'RUNPID=%s\n' "${RUNPID}"
printf 'OUT=%s\n' "${OUT}"
printf 'ROLLOUT_DIR=%s\n' "${OUT}/rollouts"
