# 实验技术摘要

## 工作量与耗时

两边均使用 Qwen3-4B、2,000 个 GSM8K 训练问题、每题 4 条 generation、共 8,000 条 trajectory，并完成 125 个 outer step 和 125 次 optimizer update。LoRA rank 均为 16。

| 框架 | 训练耗时 | 最终 Accuracy |
|---|---:|---:|
| Twinkle | 11,431 秒（3:10:31） | 1169/1319（88.6277%） |
| VERL | 19,688 秒（5:28:08） | 532/1319（40.3336%） |

对应 Base 经两套入口分别为 481/1319（36.4670%）和 466/1319（35.3298%）。最终评测实现不同，因此评测耗时不用于解释训练框架吞吐差异。

## 已完成审计

- 2,000 行训练数据均符合 VERL chat template，最大 prompt 长度 229 token。
- 每个训练问题恰好生成 4 次；125 个 rollout 文件合计 8,000 行。
- reward 全量复算与训练记录一致。
- VERL optimizer 的 504 个状态 step 均为 125。
- 两边 LoRA 均为 rank 16、504 个 tensor、33,030,144 个参数。
- 四组最终评测均保留 1,319 条完整回答。

## 兼容性调整

负责人授权两边关闭训练中的周期性测试集评测，训练完成后单独手动评测。该测试集评测不参与梯度，不改变异步 RL 的 125 step / 8,000 trajectory 工作量。

Twinkle 增加 metrics 路径、补充 eval reward type，并修复可选 group/generation 标识传递。VERL 将 `test_freq` 改为 `-1`、checkpoint bucket 从 256MiB 调为 1536MiB、使用 `TASK_QUEUE_ENABLE=1`，最终 LoRA 导出并与基础模型合并后评测。

## 结论边界

本次 Twinkle 训练更快，reward 明显上升，输出截断率明显下降。本次 VERL reward 主要持续震荡，最终只比对应 Base 提升约 5 个百分点。数据格式、reward、LoRA 装配和 optimizer 更新均未发现简单错误。

当前只有单 seed，是否为稳定现象仍需多 seed 或带额外 instrumentation 的复现实验确认。
