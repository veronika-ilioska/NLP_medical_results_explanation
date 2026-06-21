import argparse
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.testing.evaluate_tablellm_cv import (
    TABLELLM_MODEL_ID,
    build_prompt_from_row,
    generate_text,
    has_value,
    load_tablellm,
)


def main():
    parser = argparse.ArgumentParser(
        description="Generate a TableLLM output_text column for lab-result CSV rows."
    )
    parser.add_argument("--input", default="data/llama_tabular_outputs_with_targets.csv")
    parser.add_argument("--output", default="data/llama_tabular_outputs_with_tablellm.csv")
    parser.add_argument("--prompt-column", default="input_text")
    parser.add_argument("--output-column", default="output_text")
    parser.add_argument("--model-id", default=TABLELLM_MODEL_ID)
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=220)
    parser.add_argument("--max-rows", type=int, help="Optional quick-run row limit.")
    parser.add_argument("--start-row", type=int, default=0)
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Keep non-empty output_text values and only generate missing rows.",
    )
    parser.add_argument(
        "--tablellm-prompt",
        choices=["text", "code", "original"],
        default="original",
        help="Prompt format to send to TableLLM.",
    )
    parser.add_argument("--prompt-table-rows", type=int, default=20)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    if args.prompt_column not in df.columns:
        raise ValueError(f"Missing prompt column: {args.prompt_column}")
    if args.output_column not in df.columns:
        df[args.output_column] = ""
    df[args.output_column] = df[args.output_column].astype("object")

    if args.start_row < 0 or args.start_row >= len(df):
        raise IndexError(f"--start-row {args.start_row} is out of range for {len(df)} rows.")

    stop_row = len(df)
    if args.max_rows is not None:
        if args.max_rows < 1:
            raise ValueError("--max-rows must be at least 1")
        stop_row = min(stop_row, args.start_row + args.max_rows)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model, tokenizer = load_tablellm(args)
    processed = 0
    skipped = 0

    for row_index in range(args.start_row, stop_row):
        current_value = df.at[row_index, args.output_column]
        if args.skip_existing and has_value(current_value):
            skipped += 1
            continue

        row = df.iloc[row_index]
        prompt = build_prompt_from_row(row, args)
        df.at[row_index, args.output_column] = generate_text(
            model,
            tokenizer,
            prompt,
            args.max_new_tokens,
        )
        processed += 1
        df.to_csv(output_path, index=False)
        print(f"Processed row {row_index} ({processed} generated, {skipped} skipped)")

    df.to_csv(output_path, index=False)
    print(f"\nGenerated rows: {processed}")
    print(f"Skipped rows: {skipped}")
    print(f"Wrote: {output_path}")


if __name__ == "__main__":
    main()
