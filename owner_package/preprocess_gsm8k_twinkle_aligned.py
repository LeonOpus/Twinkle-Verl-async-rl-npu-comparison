#!/usr/bin/env python3
import argparse
import re
from pathlib import Path

import datasets


DATA_SOURCE = "openai/gsm8k"
SYSTEM_PROMPT = (
    "You are a helpful math assistant. Solve the problem step by step "
    "and put your final answer within \\boxed{}."
)


def extract_ground_truth(answer: str) -> str:
    match = re.search(r"####\s*([\-\d,.]+)", answer)
    if match:
        return match.group(1).replace(",", "").strip()
    return ""


def convert_row(example: dict, index: int, split: str) -> dict:
    question = str(example["question"])
    answer = str(example["answer"])
    ground_truth = extract_ground_truth(answer)
    if not ground_truth:
        raise ValueError(f"GSM8K row {split}:{index} has no #### ground truth")
    return {
        "data_source": DATA_SOURCE,
        "prompt": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": question,
            }
        ],
        "ability": "math",
        "reward_model": {
            "style": "gsm8k_accuracy",
            "ground_truth": ground_truth,
        },
        "extra_info": {
            "split": split,
            "index": index,
            "answer": answer,
            "question": question,
        },
    }


def convert_split(dataset, split: str, limit: int | None = None):
    if limit is not None:
        dataset = dataset.select(range(min(limit, len(dataset))))
    return dataset.map(
        lambda example, index: convert_row(example, index, split),
        with_indices=True,
        remove_columns=dataset.column_names,
        desc=f"Converting GSM8K {split}",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=DATA_SOURCE)
    parser.add_argument("--subset", default="main")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-samples", type=int, default=2000)
    args = parser.parse_args()

    source = datasets.load_dataset(args.dataset, args.subset)
    raw_train = source["train"].select(range(min(args.train_samples, len(source["train"]))))
    raw_test = source["test"]
    train = convert_split(raw_train, "train")
    test = convert_split(raw_test, "test")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    twinkle_train_path = output_dir / "twinkle_train.parquet"
    twinkle_test_path = output_dir / "twinkle_test_official.parquet"
    train_path = output_dir / "verl_train_twinkle_aligned.parquet"
    test_path = output_dir / "verl_test_official_twinkle_aligned.parquet"
    raw_train.to_parquet(twinkle_train_path)
    raw_test.to_parquet(twinkle_test_path)
    train.to_parquet(train_path)
    test.to_parquet(test_path)

    print(f"Wrote {len(raw_train)} rows to {twinkle_train_path}")
    print(f"Wrote {len(raw_test)} rows to {twinkle_test_path}")
    print(f"Wrote {len(train)} rows to {train_path}")
    print(f"Wrote {len(test)} rows to {test_path}")


if __name__ == "__main__":
    main()
