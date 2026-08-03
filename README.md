# Twinkle vs VERL Async RL on Ascend NPU

本仓库保存一次 Twinkle 与 VERL 异步 LoRA GRPO 对比实验的源码、配置、容器启动信息、训练日志、rollout、完整评测回答和审计清单。这是单次实验结果，不能直接扩大为两个框架的普遍能力结论。

## 主要结果

| 项目 | Twinkle | VERL |
|---|---:|---:|
| 基础模型 | Qwen3-4B | Qwen3-4B |
| 训练设备 | Ascend NPU 2、3 | Ascend NPU 2、3 |
| 训练问题数 | 2,000 | 2,000 |
| 每题 generation | 4 | 4 |
| 总 trajectory | 8,000 | 8,000 |
| optimizer step | 125 | 125 |
| 训练耗时 | 3:10:31 | 5:28:08 |
| Base 准确率 | 481/1319（36.4670%） | 466/1319（35.3298%） |
| v125 准确率 | 1169/1319（88.6277%） | 532/1319（40.3336%） |

相同训练工作量下，本次 VERL 训练耗时约为 Twinkle 的 1.72 倍。

## 评测协议

最终评测使用 GSM8K official test 1,319 题：`temperature=0.6`、`top_p=1.0`、`max_tokens=1024`、`n=1`、开启 Qwen thinking，并要求最终答案写入 `\boxed{}`。

Twinkle 使用 `GSM8KProcessor + GSM8KAccuracyReward`；VERL 通过 `main_generation_server.py` 生成，再调用负责人提供的 `gsm8k_accuracy_reward.py` 计分。

## Reward 与输出长度

- Twinkle 前/后 25 步平均 reward：0.376875 / 0.904375；后 25 步截断率约 9.06%。
- VERL 前/后 25 步平均 reward：0.350000 / 0.401875；后 25 步截断率约 69.13%。

本次 Twinkle 学会了更快结束 thinking 并输出最终答案；VERL 没有出现同等幅度的长度和截断率改善。

## 代码版本

- Twinkle：`b0798fd7260ce5c62771053c28a4ee731932d868`
- VERL：`17ea153d95eeb461815f547ceee3cfbac6ff6d11`

## 仓库结构

- `source/`：两套完整源码快照及实际兼容修改
- `owner_package/`：负责人提供的配置、预处理和 reward
- `runtime_configs/`：正式配置、diff 和授权说明
- `scripts/`：训练、导出、合并和评测脚本
- `container_reproduction/`：完整 docker run 与 inspect JSON
- `logs/`：训练、评测和 HBM 日志
- `results/`：rollout、metrics 和完整评测回答
- `manifests/`：commit、安全扫描和 SHA256
- `docs/`：实验技术摘要

## Docker 复现

完整命令位于：

- `container_reproduction/qy_twinkle_verl_cmp_20260724_docker_run.sh`
- `container_reproduction/qy_verl_cmp_20260727_docker_run.sh`

## 大文件与完整性

普通 Git 不保存基础模型、checkpoint、optimizer、HF 导出模型、LoRA 权重和 Docker 镜像层。其路径、大小及 SHA256 位于：

- `manifests/excluded_large_artifacts.tsv`
- `manifests/excluded_large_artifacts.sha256`

上传文件 SHA256 位于 `manifests/included_files.sha256`；敏感信息扫描位于 `manifests/security_scan.txt`。

四组评测的完整推理回答、ground truth 和计分结果保存在各实验目录的 `predictions.jsonl` 中。

## License

Twinkle 和 VERL 源码快照均保留原始 `LICENSE` 和 `Notice.txt`。

## 负责人复核入口

- 评测链路及脚本变化：`docs/evaluation_protocol.md`
- 实际训练与测试数据：`data/README.md`
- Twinkle/VERL reward 对比曲线：`analysis/reward_curves.svg`
- 曲线底层分段均值：`analysis/reward_block_means.csv`
- 两个容器运行时版本：`manifests/runtime/`
- Qwen3-4B 文件身份：`manifests/base_model.sha256`
- 负责人原始 ZIP 身份：`manifests/owner_zip.sha256`
