# VERL full125 rollout 排查结论（20260806）

本目录保存了 VERL full125 复跑的 rollout、配置、脚本、日志和分析结果。

本次复跑已设置 `trainer.rollout_data_dir`，因此 125 个训练 step 的 rollout 都已保存到 `rollouts/`。

## 结论

1. reward 函数没有接错。  
   对 8000 条 rollout 重新调用 owner `compute_score` 后，`reward_mismatches = 0`。

2. VERL reward 低的主要原因不是 reward 函数错误。  
   全量 `reward_mean = 0.388625`，后 25 步 `last25_reward_mean = 0.391875`，与训练日志里的 reward 曲线一致。

3. 输出过长和截断问题明显。  
   全量 `budget_hit_rate_chars_ge3500 = 0.30475`，后 25 步 `last25_budget_hit_rate = 0.3`。第 125 步日志里 `response_length/clip_ratio = 0.75`，说明大量输出接近 1024 response token 上限。

4. 答案位置不稳定。  
   `answer_anywhere_rate = 0.492875`，但 `answer_last500_rate = 0.397875`。说明部分样本完整输出里有答案痕迹，但最后 500 字符内没有可提取答案，因此会被 reward 判 0。

5. 乱码存在，但比例不高。  
   全量 `garbled_rate = 0.007375`，后 25 步 `last25_garbled_rate = 0.003125`。乱码更像是输出发散的一部分表现，不是唯一主因。

## 文件说明

- `rollouts/`：125 个 step 的原始 rollout jsonl。
- `analysis/summary.json`：汇总统计。
- `analysis/step_diagnosis.csv`：逐 step 统计。
- `analysis/row_diagnosis.csv`：逐样本诊断。
- `configs/`：复跑配置。
- `scripts/`：启动脚本和 runner。
- `logs/`：训练日志和启动日志。

## 后续建议

建议继续做两个对照实验：

1. 提高 `max_response_length`，验证 reward 是否因截断缓解而上升。
2. 约束输出格式，让最终答案尽早稳定写入 `\boxed{}`。
