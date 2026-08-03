# 评测链路与脚本变化对照

## Twinkle

实际入口为 `scripts/qy_eval_gsm8k_owner_protocol.py`，由源码快照中的 `source/twinkle/cookbook/rl/eval_gsm8k_areal.py` 适配而来，完整差异保存在 `scripts/qy_eval_gsm8k_owner_protocol.diff`。

主要适配包括：读取本地 parquet/json 数据；固定使用 `GSM8KProcessor` 和 `GSM8KAccuracyReward`；支持 Base 或 LoRA 两种模式；保存逐题 `predictions.jsonl` 及字段完整的 `summary.json`。

最终 v125 runner 为 `scripts/qy_owner_twinkle_v125_manual_eval_official1319_npu3.sh`。实际参数为 `temperature=0.6`、`top_p=1.0`、`max_tokens=1024`、`n=1`、thinking 开启、seed 1。

## VERL

VERL 没有调用 Twinkle 的 `qy_eval_gsm8k_owner_protocol.py`。实际链路为：

1. `source/verl/verl/trainer/main_generation_server.py` 生成回答；
2. `scripts/qy_owner_verl_v125_merged_manual_eval_official1319_npu3.sh` 启动生成和后处理；
3. `owner_package/gsm8k_accuracy_reward.py` 提取 `\boxed{}` 或 `####` 答案并计分；
4. runner 写出 `generated.parquet`、`predictions.jsonl` 和 `summary.json`。

最终使用合并后的 v125 模型，参数同样为 `temperature=0.6`、`top_p=1.0`、`top_k=-1`、`max_tokens=1024`、`n=1`、thinking 开启。

## Summary 路径

- Twinkle Base：`results/twinkle/qy_owner_twinkle_base_manual_eval_official1319_npu3/summary.json`
- Twinkle v125：`results/twinkle/qy_owner_twinkle_v125_manual_eval_official1319_npu3/summary.json`
- VERL Base：`results/verl/qy_owner_verl_base_manual_eval_official1319_npu3/summary.json`
- VERL v125 merged：`results/verl/qy_owner_verl_v125_merged_manual_eval_official1319_npu3/summary.json`

两边 `summary.json` 字段数量不同，是因为写出脚本不同，不表示评测文件缺失。最终准确率均由对应目录的 1,319 条 `predictions.jsonl` 独立复算确认。
