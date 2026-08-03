#!/usr/bin/env bash

ROOT=/mnt/lv_model/npu60005420a
TRAIN_EXP=qy_owner_verl_full125_noeval_final_npu23

BASE=$ROOT/qy_owner_experiments_20260730/$TRAIN_EXP/exported_hf_v125
ADAPTER=$BASE/lora_adapter
MERGED=$ROOT/qy_owner_experiments_20260730/$TRAIN_EXP/merged_hf_v125_for_eval

export ASCEND_RT_VISIBLE_DEVICES=3
export TASK_QUEUE_ENABLE=1
unset PYTORCH_NPU_ALLOC_CONF
unset RAY_ADDRESS

START_SECONDS=$(date +%s)

python3 -u - "$BASE" "$ADAPTER" "$MERGED" <<'PY'
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
)

base_path = Path(sys.argv[1])
adapter_path = Path(sys.argv[2])
merged_path = Path(sys.argv[3])

print(f"base={base_path}")
print(f"adapter={adapter_path}")
print(f"merged={merged_path}")

base_model = AutoModelForCausalLM.from_pretrained(
    base_path,
    torch_dtype=torch.bfloat16,
    device_map={"": "cpu"},
    low_cpu_mem_usage=True,
)

peft_model = PeftModel.from_pretrained(
    base_model,
    adapter_path,
    is_trainable=False,
)

merged_model = peft_model.merge_and_unload(
    safe_merge=True,
)

merged_model.save_pretrained(
    merged_path,
    safe_serialization=True,
    max_shard_size="10GB",
)

tokenizer = AutoTokenizer.from_pretrained(base_path)
tokenizer.save_pretrained(merged_path)

generation_config = GenerationConfig.from_pretrained(base_path)
generation_config.save_pretrained(merged_path)

print("RESULT: VERL_V125_LORA_MERGE_MODEL_SAVED")
PY

RUN_RC=$?
END_SECONDS=$(date +%s)
ELAPSED_SECONDS=$((END_SECONDS - START_SECONDS))

echo "QY_VERL_V125_MERGE_RC=$RUN_RC"
echo "QY_VERL_V125_MERGE_ELAPSED_SECONDS=$ELAPSED_SECONDS"

if test "$RUN_RC" = 0; then
    echo "RESULT: VERL_V125_MERGE_FOR_EVAL_PASS"
else
    echo "RESULT: VERL_V125_MERGE_FOR_EVAL_FAIL_RC=$RUN_RC"
fi

exit "$RUN_RC"
