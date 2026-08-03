# 实际实验数据

本目录保存负责人预处理脚本实际生成并被正式训练/评测使用的四份 parquet。

| 框架 | 训练文件 | 官方测试文件 |
|---|---|---|
| Twinkle | `twinkle_train.parquet` | `twinkle_test_official.parquet` |
| VERL | `verl_train_twinkle_aligned.parquet` | `verl_test_official_twinkle_aligned.parquet` |

训练集包含相同的 2,000 个 GSM8K 问题；测试集为 GSM8K official test 的 1,319 题。Twinkle 文件保留 `question/answer`，运行时由 `GSM8KProcessor` 处理；VERL 文件已经转换为 `prompt/reward_model` 格式。

文件 SHA256 位于 `../manifests/datasets.sha256`，生成脚本为 `../owner_package/preprocess_gsm8k_twinkle_aligned.py`。
