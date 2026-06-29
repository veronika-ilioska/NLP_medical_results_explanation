import argparse
from pathlib import Path

import pandas as pd


KEY_COLUMNS = [
    "SUBJECT_ID",
    "HADM_ID",
    "ITEMID",
    "CHARTTIME",
    "lab_name",
    "VALUE",
    "VALUEUOM",
]


def existing_columns(df, columns):
    return [column for column in columns if column in df.columns]


def copy_targets(source_path, output_paths, target_column):
    source_df = pd.read_csv(source_path)
    if target_column not in source_df.columns:
        raise ValueError(f"Missing target column in source: {target_column}")

    key_columns = existing_columns(source_df, KEY_COLUMNS)
    if not key_columns:
        raise ValueError("No shared key columns found in source CSV.")

    target_map = source_df[key_columns + [target_column]].copy()
    target_map = target_map.drop_duplicates(subset=key_columns, keep="first")

    for output_path in output_paths:
        if not output_path.exists():
            print(f"Skipping missing file: {output_path}")
            continue

        df = pd.read_csv(output_path)
        merge_keys = [column for column in key_columns if column in df.columns]
        if not merge_keys:
            print(f"Skipping {output_path}: no matching key columns.")
            continue

        target_lookup = target_map[merge_keys + [target_column]].copy()
        if target_column in df.columns:
            df = df.drop(columns=[target_column])

        df = df.merge(target_lookup, on=merge_keys, how="left")
        df.to_csv(output_path, index=False)

        filled_count = int(df[target_column].notna().sum())
        print(f"Updated {output_path}: filled {filled_count}/{len(df)} {target_column} values.")


def main():
    parser = argparse.ArgumentParser(
        description="Copy target_text values from existing Llama target CSV into tabular output CSVs."
    )
    parser.add_argument(
        "--source",
        default="llama/outputs/text_only_rompt/llama_tabular_outputs_with_targets.csv",
        help="CSV that already contains target_text values.",
    )
    parser.add_argument(
        "--outputs",
        nargs="+",
        default=[
            "llama/outputs/tabular_prompt_approach/llama_tabular_base_outputs.csv",
            "llama/outputs/tabular_prompt_approach/llama_tabular_finetuned_outputs.csv",
        ],
        help="One or more output CSVs to update.",
    )
    parser.add_argument("--target-column", default="target_text")
    args = parser.parse_args()

    copy_targets(
        Path(args.source),
        [Path(output) for output in args.outputs],
        args.target_column,
    )


if __name__ == "__main__":
    main()
