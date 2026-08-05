# Stage-2 GSM8K Reward Equivalence

This directory preserves the reward code used to check Twinkle and VERL reward semantics for the 2026-08-04 stage-2 run.

Conclusion:

- Twinkle training uses `GSM8KAccuracyReward` from `twinkle_gsm8k_reward.py`.
- VERL training and the final unified evaluator use `compute_score` from `owner_gsm8k_accuracy_reward.py`.
- The scoring semantics are equivalent:
  - inspect the final 500 characters of the completion;
  - prefer the last valid `\boxed{...}` answer;
  - otherwise parse `#### <number>`;
  - remove commas and spaces;
  - compare numerically with tolerance `1e-5`, with string fallback;
  - return `1.0` for correct and `0.0` otherwise.

The implementations differ only in framework input interface:
Twinkle extracts completion and ground truth from a trajectory object, while VERL receives `solution_str` and `ground_truth` as reward function arguments.
