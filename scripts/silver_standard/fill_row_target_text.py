import argparse
from pathlib import Path

import pandas as pd


NORMAL_EXPLANATIONS = {
    "blood gas": "appears generally within the expected range for acid-base or oxygen balance",
    "chemistry": "appears generally within the expected range for body chemistry balance",
    "hematology": "appears generally within the expected range for blood cell function",
}

ABNORMAL_EXPLANATIONS = {
    "ph": "may reflect a change in the body's acid-base balance",
    "po2": "may reflect a change in oxygen levels in the blood",
    "pco2": "may reflect a change in breathing-related gas balance",
    "bicarbonate": "may reflect a change in the body's acid-base balance",
    "anion gap": "may suggest a change in acid-base balance",
    "amylase": "may reflect pancreatic or digestive enzyme activity",
    "creatinine": "may reflect a change in kidney filtering function",
    "urea nitrogen": "may reflect a change in kidney function or hydration",
    "glucose": "may suggest a change in blood sugar balance",
    "sodium": "may reflect a change in fluid or salt balance",
    "potassium": "may reflect a change in heart, muscle, or salt balance",
    "chloride": "may reflect a change in fluid or acid-base balance",
    "calcium": "may suggest a change in calcium balance in the blood",
    "magnesium": "may reflect a change in nerve or muscle mineral balance",
    "phosphate": "may suggest a change in phosphate balance in the blood",
    "hemoglobin": "may reflect a change in oxygen-carrying protein in the blood",
    "hematocrit": "may reflect a change in the amount of red blood cells",
    "red blood cells": "may reflect a change in red blood cell levels",
    "white blood cells": "can suggest a change in immune system activity",
    "platelet count": "may reflect a change in blood clotting cell levels",
    "mcv": "may reflect a change in red blood cell size",
    "mch": "may reflect a change in hemoglobin amount inside red blood cells",
    "mchc": "may reflect a change in hemoglobin concentration inside red blood cells",
    "rdw": "may reflect a change in red blood cell size variation",
    "lymphocytes": "can suggest a change in immune cell balance",
    "neutrophils": "can suggest a change in immune response",
    "monocytes": "can suggest a change in immune cell activity",
    "bands": "can suggest a change in early immune cell activity",
}


def has_value(value):
    return not pd.isna(value) and str(value).strip() and str(value).strip().lower() != "nan"


def value_from(row, *columns, default=""):
    for column in columns:
        if column in row and has_value(row[column]):
            return str(row[column]).strip()
    return default


def clean_flag(row):
    flag = value_from(row, "FLAG", "flag", default="normal").lower()
    return "abnormal" if flag in {"abnormal", "delta"} else "normal"


def explanation_for(row):
    lab_name = value_from(row, "lab_name", "LAB_NAME", "test_name", "label", default="Laboratory test")
    category = value_from(row, "category", "CATEGORY", default="").lower()
    flag = clean_flag(row)
    lab_key = lab_name.lower()

    if flag == "normal":
        for category_key, explanation in NORMAL_EXPLANATIONS.items():
            if category_key in category:
                return explanation
        return "appears generally within the expected range"

    for known_lab, explanation in ABNORMAL_EXPLANATIONS.items():
        if known_lab in lab_key:
            return explanation
    return "may suggest a change that should be interpreted with the full clinical context"


def build_target_text(row):
    lab_name = value_from(row, "lab_name", "LAB_NAME", "test_name", "label", default="Laboratory test")
    value = value_from(row, "VALUE", "value", default="unknown")
    unit = value_from(row, "VALUEUOM", "unit", "units", default="")
    measured = f"{value} {unit}".strip()
    explanation = explanation_for(row)

    return f"- {lab_name}: {measured} - {explanation.capitalize()}."


def main():
    parser = argparse.ArgumentParser(
        description="Fill empty row-level target_text values with cautious silver-standard lab explanations."
    )
    parser.add_argument("--input", default="data/llama_tabular_outputs.csv")
    parser.add_argument("--output", default="data/llama_tabular_outputs_with_targets.csv")
    parser.add_argument("--target-column", default="target_text")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing target_text values instead of filling only empty values.",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    if args.target_column not in df.columns:
        df[args.target_column] = ""
    df[args.target_column] = df[args.target_column].astype("object")

    generated = df.apply(build_target_text, axis=1)
    empty_mask = df[args.target_column].isna() | (df[args.target_column].astype(str).str.strip() == "")
    if args.overwrite:
        df[args.target_column] = generated
        filled_count = len(df)
    else:
        df.loc[empty_mask, args.target_column] = generated[empty_mask]
        filled_count = int(empty_mask.sum())

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"Rows: {len(df)}")
    print(f"Filled target_text rows: {filled_count}")
    print(f"Wrote: {output_path}")
    print(df[["lab_name", "VALUE", "VALUEUOM", "FLAG", args.target_column]].head().to_string(index=False))


if __name__ == "__main__":
    main()
