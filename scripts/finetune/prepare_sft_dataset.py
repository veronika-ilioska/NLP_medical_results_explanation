import argparse
import json
from pathlib import Path

import pandas as pd


SYSTEM_PROMPT = (
    "You produce concise, patient-friendly explanations of laboratory results. "
    "Use cautious wording, avoid diagnosis and do not recommend treatment."
)


def build_record(input_text, target_text):
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": str(input_text).strip()},
            {"role": "assistant", "content": str(target_text).strip()},
        ]
    }


def write_jsonl(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Convert lab explanation CSV rows into chat SFT JSONL files."
    )
    parser.add_argument("--input", default="data/medgemma_1000_outputs.csv")
    parser.add_argument("--output-dir", default="data/finetune")
    parser.add_argument("--prompt-column", default="input_text")
    parser.add_argument("--target-column", default="medgemma_output")
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    required = [args.prompt_column, args.target_column]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.dropna(subset=required).copy()
    df[args.prompt_column] = df[args.prompt_column].astype(str).str.strip()
    df[args.target_column] = df[args.target_column].astype(str).str.strip()
    df = df[(df[args.prompt_column] != "") & (df[args.target_column] != "")]

    if len(df) < 2:
        raise ValueError("Need at least 2 usable rows after filtering.")

    df = df.sample(frac=1, random_state=args.seed).reset_index(drop=True)
    val_size = max(1, int(len(df) * args.val_ratio))

    val_df = df.iloc[:val_size]
    train_df = df.iloc[val_size:]

    train_records = [
        build_record(row[args.prompt_column], row[args.target_column])
        for _, row in train_df.iterrows()
    ]
    val_records = [
        build_record(row[args.prompt_column], row[args.target_column])
        for _, row in val_df.iterrows()
    ]

    output_dir = Path(args.output_dir)
    write_jsonl(output_dir / "train.jsonl", train_records)
    write_jsonl(output_dir / "validation.jsonl", val_records)

    print(f"Train records: {len(train_records)}")
    print(f"Validation records: {len(val_records)}")
    print(f"Wrote: {output_dir / 'train.jsonl'}")
    print(f"Wrote: {output_dir / 'validation.jsonl'}")


if __name__ == "__main__":
    main()
