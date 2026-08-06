# VERL full125 rollout diagnosis - 20260806

This directory contains the VERL full125 rerun with `trainer.rollout_data_dir` enabled.

## Key conclusion

The reward function is not wired incorrectly. Recomputing the saved rollout data with the owner `compute_score` gives `reward_mismatches=0`.

Main findings:

- total rollout rows: 8000
- steps: 125
- reward_mean: 0.388625
- last25_reward_mean: 0.391875
- budget_hit_rate_chars_ge3500: 0.30475
- last25_budget_hit_rate: 0.3
- answer_anywhere_rate: 0.492875
- answer_last500_rate: 0.397875
- garbled_rate: 0.007375
- last25_garbled_rate: 0.003125

Interpretation: VERL reward is low mainly because rollout outputs are too long and often approach the 1024 response-token budget. A subset of samples has answer-like content somewhere in the full output, but not in the final 500-character extraction window used by the reward function. Garbled output exists but its measured ratio is low, so it is better treated as part of the output-divergence symptoms rather than the sole cause.

## Files

- `rollouts/`: raw saved rollout jsonl files, one file per training step.
- `analysis/summary.json`: aggregate diagnosis.
- `analysis/step_diagnosis.csv`: step-level reward / length / answer-window / garbled statistics.
- `analysis/row_diagnosis.csv`: row-level diagnosis.
- `configs/`: exact full125 config.
- `scripts/`: exact runner and launcher scripts.
- `logs/`: full training and launch logs.
