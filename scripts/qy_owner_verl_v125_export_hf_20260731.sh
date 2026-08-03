#!/usr/bin/env bash

ROOT=/mnt/lv_model/npu60005420a
WT=/workspace/verl_official_main_17ea153d95ee_20260729
TRAIN_EXP=qy_owner_verl_full125_noeval_final_npu23

ACTOR=$ROOT/qy_owner_experiments_20260730/$TRAIN_EXP/checkpoints/global_step_125/actor
TARGET=$ROOT/qy_owner_experiments_20260730/$TRAIN_EXP/exported_hf_v125

export PYTHONPATH=$WT:${PYTHONPATH:-}
export ASCEND_RT_VISIBLE_DEVICES=2,3
export TASK_QUEUE_ENABLE=1
unset PYTORCH_NPU_ALLOC_CONF
unset RAY_ADDRESS

START_SECONDS=$(date +%s)

cd "$WT"

python3 -u -m verl.model_merger merge \
  --backend fsdp \
  --local_dir "$ACTOR" \
  --target_dir "$TARGET" \
  --use_cpu_initialization

RUN_RC=$?
END_SECONDS=$(date +%s)
ELAPSED_SECONDS=$((END_SECONDS - START_SECONDS))

echo "QY_VERL_V125_EXPORT_RC=$RUN_RC"
echo "QY_VERL_V125_EXPORT_ELAPSED_SECONDS=$ELAPSED_SECONDS"

if test "$RUN_RC" = 0; then
    echo "RESULT: VERL_V125_HF_EXPORT_PASS"
else
    echo "RESULT: VERL_V125_HF_EXPORT_FAIL_RC=$RUN_RC"
fi

exit "$RUN_RC"
