# Twinkle 与 verl 两卡 NPU GSM8K GRPO 对比实验

本文只说明两件事：

1. 两个框架的数据怎么准备。
2. 两个框架分别怎么启动。

假定 Twinkle 和 verl 分别运行在两个独立 Pod 中，每个 Pod 都有两张可用 NPU，并且对应框架的运行环境已经安装完成。两个实验可以同时启动。下面以 Pod 内设备 `2,3` 为例。

## 1. 设置路径

根据服务器实际目录修改：

```bash
export PACKAGE_DIR=/path/to/unzipped/twinkle_aligned_verl_grpo_npu_2cards
export VERL_ROOT=/path/to/verl
export TWINKLE_ROOT=/path/to/twinkle
export MODEL_PATH=/path/to/Qwen3-4B
export DATA_DIR=/path/to/gsm8k_async_cmp2000
export EXP_ROOT=/path/to/gsm8k_framework_comparison

mkdir -p "${DATA_DIR}" "${EXP_ROOT}"
```

例如：

```bash
export PACKAGE_DIR=/mnt/lv_model/npu60005420a/packages/twinkle_aligned_verl_grpo_npu_2cards
export VERL_ROOT=/model/ljl/project/verl
export TWINKLE_ROOT=/model/ljl/project/twinkle
export MODEL_PATH=/nas/disk1/Qwen3-4B
export DATA_DIR=/mnt/lv_model/npu60005420a/data/gsm8k_async_cmp2000
export EXP_ROOT=/mnt/lv_model/npu60005420a/qy_outputs/gsm8k_comparison
```

在两个 Pod 中分别解压，或者解压到两个 Pod 都能访问的共享目录：

```bash
mkdir -p "${PACKAGE_DIR}"
unzip -o /path/to/twinkle_aligned_verl_grpo_npu_2cards.zip -d "${PACKAGE_DIR}"
```

ZIP 解压后的 `${PACKAGE_DIR}` 中应直接包含：

```text
README_TWINKLE_VERL_GSM8K.md
preprocess_gsm8k_twinkle_aligned.py
twinkle_async_single_lora_gsm8k_accuracy_npu_2cards.yaml
v1_separate_async_grpo_npu_2cards.yaml
run_qwen3_4b_v1_separate_async_npu_2cards.sh
gsm8k_accuracy_reward.py
```

## 2. 准备数据

在 verl 环境中运行：

```bash
python3 "${PACKAGE_DIR}/preprocess_gsm8k_twinkle_aligned.py" \
  --dataset openai/gsm8k \
  --subset main \
  --output-dir "${DATA_DIR}" \
  --train-samples 2000
```

该命令会下载 GSM8K，并生成四个文件：

```text
${DATA_DIR}/twinkle_train.parquet
${DATA_DIR}/twinkle_test_official.parquet
${DATA_DIR}/verl_train_twinkle_aligned.parquet
${DATA_DIR}/verl_test_official_twinkle_aligned.parquet
```

对应关系如下：

| 框架 | 训练文件 | 验证文件 |
| --- | --- | --- |
| Twinkle | `twinkle_train.parquet` | `twinkle_test_official.parquet` |
| verl | `verl_train_twinkle_aligned.parquet` | `verl_test_official_twinkle_aligned.parquet` |

不要混用：

- Twinkle 文件保存原始 `question` 和 `answer`，由 `GSM8KProcessor` 在运行时处理。
- verl 文件已经转换为 `prompt` 和 `reward_model` 格式。

两边使用相同的 system prompt、纯数值 ground truth 和 `GSM8KAccuracyReward`。

如果两个 Pod 挂载了相同的共享存储，只需要生成一次数据，并让两个 Pod 的 `DATA_DIR` 指向该目录。如果存储不共享，需要把对应 parquet 复制到各自 Pod，并分别设置 `DATA_DIR`。

确认四个文件存在：

```bash
test -f "${DATA_DIR}/twinkle_train.parquet"
test -f "${DATA_DIR}/twinkle_test_official.parquet"
test -f "${DATA_DIR}/verl_train_twinkle_aligned.parquet"
test -f "${DATA_DIR}/verl_test_official_twinkle_aligned.parquet"
```

## 3. 启动 Twinkle

在 Twinkle Pod 中，进入已经安装好 Twinkle、torch_npu、vLLM-Ascend 和 TransferQueue 的 Python 环境。

然后执行：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh

ray stop --force || true
unset RAY_ADDRESS

export ASCEND_RT_VISIBLE_DEVICES=2,3
export MODEL_ID="${MODEL_PATH}"
export GSM8K_TRAIN_DATASET_ID="${DATA_DIR}/twinkle_train.parquet"
export GSM8K_TEST_DATASET_ID="${DATA_DIR}/twinkle_test_official.parquet"

cd "${TWINKLE_ROOT}"

python3 cookbook/rl/async_multi_lora_grpo.py \
  --config "${PACKAGE_DIR}/twinkle_async_single_lora_gsm8k_accuracy_npu_2cards.yaml" \
  2>&1 | tee "${EXP_ROOT}/twinkle.log"
```

Twinkle 配置文件是：

```text
${PACKAGE_DIR}/twinkle_async_single_lora_gsm8k_accuracy_npu_2cards.yaml
```

该配置使用：

```text
NPU 0: trainer
NPU 1: rollout
训练数据: 2000 个 prompt
训练步数: 125
每个 prompt 生成: 4 条 trajectory
LoRA: rank=16, alpha=16
```

这里的 NPU 0 和 1 是 `ASCEND_RT_VISIBLE_DEVICES=2,3` 映射后的逻辑设备。

## 4. 启动 verl

在 verl Pod 中，进入已经安装好 verl NPU 依赖的 Python 环境。该实验可以和 Twinkle 同时启动。

然后执行：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh

ray stop --force || true
unset RAY_ADDRESS

export ASCEND_RT_VISIBLE_DEVICES=2,3
export MODEL_PATH="${MODEL_PATH}"
export TRAIN_FILE="${DATA_DIR}/verl_train_twinkle_aligned.parquet"
export TEST_FILE="${DATA_DIR}/verl_test_official_twinkle_aligned.parquet"
export OUTPUT_DIR="${EXP_ROOT}/verl"
export CKPTS_DIR="${OUTPUT_DIR}/checkpoints"
export VERL_ROOT="${VERL_ROOT}"
export GSM8K_REWARD_PATH="${PACKAGE_DIR}/gsm8k_accuracy_reward.py"

# Hydra 要求主配置位于 verl 的配置目录，启动前安装一次。
install -m 0644 \
  "${PACKAGE_DIR}/v1_separate_async_grpo_npu_2cards.yaml" \
  "${VERL_ROOT}/verl/trainer/config/v1_separate_async_grpo_npu_2cards.yaml"

bash "${PACKAGE_DIR}/run_qwen3_4b_v1_separate_async_npu_2cards.sh" \
  2>&1 | tee "${EXP_ROOT}/verl.log"
```

verl 启动脚本会自动：

1. 检查两份 parquet。
2. 检查 prompt 和 ground truth 格式。
3. 创建本地 Ray。
4. 使用 V1 `separate_async + TransferQueue + GRPO` 开始训练。

不需要提前执行 `ray start --head`。

## 5. 最终命令顺序

完整实验顺序是：

```text
1. 解压 ZIP 并设置 PACKAGE_DIR
2. 执行 ${PACKAGE_DIR}/preprocess_gsm8k_twinkle_aligned.py，生成四份 parquet
3. Twinkle Pod 使用 twinkle_* parquet 启动 Twinkle
4. verl Pod 使用 verl_* parquet 启动 verl
5. 两个实验并行运行，并分别写入自己的输出目录
```

两个 Pod 的 Ray、NPU 和输出目录必须彼此隔离。每个 Pod 中的 `ray stop --force` 只清理本 Pod 的 Ray，不会影响另一个 Pod。
