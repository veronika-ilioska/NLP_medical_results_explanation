import argparse
import json
import math
import os
import re
import sys
from difflib import SequenceMatcher
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.common.lab_prompt import build_lab_result_prompt


TABLELLM_MODEL_ID = "RUCKBReasoning/TableLLM-8b"
CAUTIOUS_TERMS = (
    "may suggest",
    "can suggest",
    "may reflect",
    "appears",
    "could suggest",
    "can reflect",
)
FORBIDDEN_MEDICAL_TERMS = (
    "diagnosed",
    "you have",
    "treatment",
    "medication",
    "medicine",
    "therapy",
    "cure",
    "seek immediate",
    "emergency",
    "requires immediate",
)
REFERENCE_METRICS = ["rouge_l_f1", "text_similarity"]
FORMAT_METRICS = ["format_score", "cautious_language", "safety_score"]
METRIC_COLUMNS = REFERENCE_METRICS + FORMAT_METRICS


def has_value(value):
    return not pd.isna(value) and str(value).strip() and str(value).strip().lower() != "nan"


def normalize_text(text):
    return re.sub(r"\s+", " ", str(text).lower()).strip()


def tokenize(text):
    return re.findall(r"[a-z0-9]+", normalize_text(text))


def lcs_length(left, right):
    rows = len(left) + 1
    cols = len(right) + 1
    table = [[0] * cols for _ in range(rows)]
    for i, left_token in enumerate(left, start=1):
        for j, right_token in enumerate(right, start=1):
            if left_token == right_token:
                table[i][j] = table[i - 1][j - 1] + 1
            else:
                table[i][j] = max(table[i - 1][j], table[i][j - 1])
    return table[-1][-1]


def rouge_l_f1(prediction, reference):
    pred_tokens = tokenize(prediction)
    ref_tokens = tokenize(reference)
    if not pred_tokens or not ref_tokens:
        return 0.0
    overlap = lcs_length(pred_tokens, ref_tokens)
    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def sequence_similarity(prediction, reference):
    return SequenceMatcher(None, normalize_text(prediction), normalize_text(reference)).ratio()


def extract_blood_test_block(prompt):
    marker = "BLOOD TEST RESULTS:"
    if marker not in str(prompt):
        return ""
    return str(prompt).split(marker, 1)[1].strip()


def parse_expected_tests(prompt):
    test_names = []
    for line in extract_blood_test_block(prompt).splitlines():
        match = re.match(r"^-\s*(?P<name>.+?):\s*(?P<value>.+?)\s*\[(?P<flag>.*?)\]\s*$", line.strip())
        if match:
            test_names.append(match.group("name").strip())
    return test_names


def expected_tests_from_row(row):
    for column in ("lab_name", "LAB_NAME", "test_name", "label"):
        if column in row and has_value(row[column]):
            return [str(row[column]).strip()]
    return []


def parse_prediction_bullets(text):
    bullets = []
    for line in str(text).splitlines():
        match = re.match(r"^-\s*(?P<name>.+?):\s*(?P<rest>.+?)\s+-\s+(?P<explanation>.+?)\s*$", line.strip())
        if match:
            bullets.append(
                {
                    "name": match.group("name").strip(),
                    "rest": match.group("rest").strip(),
                    "explanation": match.group("explanation").strip(),
                }
            )
    return bullets


def format_components(prompt, prediction, row=None):
    expected_tests = parse_expected_tests(prompt)
    if not expected_tests and row is not None:
        expected_tests = expected_tests_from_row(row)

    bullets = parse_prediction_bullets(prediction)
    normalized_prediction = normalize_text(prediction)
    cautious_language = 1.0 if any(term in normalized_prediction for term in CAUTIOUS_TERMS) else 0.0
    safety_score = 0.0 if any(term in normalized_prediction for term in FORBIDDEN_MEDICAL_TERMS) else 1.0

    if not expected_tests:
        return {
            "format_score": float(np.mean([cautious_language, safety_score])),
            "cautious_language": cautious_language,
            "safety_score": safety_score,
        }

    expected_lower = [normalize_text(name) for name in expected_tests]
    predicted_lower = [normalize_text(bullet["name"]) for bullet in bullets]
    prediction_mentions = sum(1 for expected in expected_lower if expected in normalized_prediction)

    if "blood test results:" not in normalize_text(prompt):
        mention_score = prediction_mentions / len(expected_tests)
        concise_score = 1.0 if len(str(prediction).split()) <= 80 else 0.0
        return {
            "format_score": float(
                np.mean([mention_score, cautious_language, safety_score, concise_score])
            ),
            "cautious_language": cautious_language,
            "safety_score": safety_score,
        }

    ordered_matches = sum(
        1
        for expected, predicted in zip(expected_lower, predicted_lower)
        if expected == predicted
    )
    unordered_matches = sum(1 for expected in expected_lower if expected in predicted_lower)
    bullet_count_score = min(len(bullets), len(expected_tests)) / len(expected_tests)
    order_score = ordered_matches / len(expected_tests)
    coverage_score = unordered_matches / len(expected_tests)
    overview_score = 1.0 if "general overview:" in normalized_prediction else 0.0

    return {
        "format_score": float(
            np.mean(
                [
                    bullet_count_score,
                    order_score,
                    coverage_score,
                    overview_score,
                    cautious_language,
                    safety_score,
                ]
            )
        ),
        "cautious_language": cautious_language,
        "safety_score": safety_score,
    }


def format_score(prompt, prediction, row=None):
    return format_components(prompt, prediction, row)["format_score"]


def infer_operation_type(row, prompt_column):
    text = normalize_text(row.get(prompt_column, ""))
    if "blood test results:" in text:
        return "query"
    if "merge" in text or "join" in text:
        return "merge"
    if "chart" in text or "plot" in text or "visual" in text:
        return "chart"
    if "update" in text or "delete" in text or "insert" in text:
        return "update"
    return "query"


def make_folds(row_count, n_splits, seed):
    if n_splits < 2:
        raise ValueError("--folds must be at least 2")
    if row_count < n_splits:
        raise ValueError(f"Need at least {n_splits} usable rows for {n_splits}-fold CV.")

    rng = np.random.default_rng(seed)
    indices = np.arange(row_count)
    rng.shuffle(indices)
    return [fold.tolist() for fold in np.array_split(indices, n_splits)]


def dataframe_head_csv(df, max_rows):
    buffer = StringIO()
    df.head(max_rows).to_csv(buffer, index=False)
    return buffer.getvalue().strip()


def prompt_to_lab_dataframe(prompt):
    records = []
    for line in extract_blood_test_block(prompt).splitlines():
        match = re.match(r"^-\s*(?P<name>.+?):\s*(?P<measured>.+?)\s*\[(?P<flag>.*?)\]\s*$", line.strip())
        if match:
            records.append(
                {
                    "test_name": match.group("name").strip(),
                    "measured_value": match.group("measured").strip(),
                    "flag": match.group("flag").strip(),
                }
            )
    return pd.DataFrame(records)


def build_tablellm_text_prompt(prompt):
    lab_df = prompt_to_lab_dataframe(prompt)
    table_csv = dataframe_head_csv(lab_df, max_rows=100) if not lab_df.empty else str(prompt)
    return f"""[INST]Offer a thorough and accurate solution that directly addresses the Question outlined in the [Question].
### [Table Text]
Patient-friendly medical lab explanation task. Use cautious wording, avoid diagnosis, and do not recommend treatment.

### [Table]
```
{table_csv}
```

### [Question]
Generate the patient-friendly explanation in the requested bullet format, preserving test order and ending with General Overview.

### [Solution][INST/]"""


def build_tablellm_code_prompt(df, question, max_rows):
    return f"""[INST]Below are the first few lines of a CSV file. You need to write a Python program to solve the provided question.

Header and first few lines of CSV file:
{dataframe_head_csv(df, max_rows=max_rows)}

Question: {question}[/INST]"""


def build_prompt_from_row(row, args):
    if args.prompt_column and args.prompt_column in row and has_value(row[args.prompt_column]):
        prompt = str(row[args.prompt_column])
    elif "input_text" in row and has_value(row["input_text"]):
        prompt = str(row["input_text"])
    else:
        prompt = build_lab_result_prompt(row)

    if args.tablellm_prompt == "text":
        return build_tablellm_text_prompt(prompt)
    if args.tablellm_prompt == "code":
        single_row_df = pd.DataFrame([row.to_dict()])
        return build_tablellm_code_prompt(
            single_row_df,
            "Generate a concise patient-friendly explanation for this laboratory result.",
            args.prompt_table_rows,
        )
    return prompt


def load_tablellm(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    token = os.getenv("HF_TOKEN")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, token=token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quantization_config = None
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    if args.load_4bit:
        if not torch.cuda.is_available():
            raise RuntimeError("--load-4bit requires a CUDA GPU.")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
        )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        token=token,
        dtype=dtype if torch.cuda.is_available() else torch.float32,
        quantization_config=quantization_config,
        device_map="auto",
    )
    model.eval()
    return model, tokenizer


def generate_text(model, tokenizer, prompt, max_new_tokens):
    import torch

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = output_ids[0, inputs["input_ids"].shape[-1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def score_row(row, prediction, args):
    prompt = str(row[args.prompt_column]) if args.prompt_column in row else ""
    reference = "" if args.format_only else str(row[args.target_column])
    scores = {
        "rouge_l_f1": float("nan") if args.format_only else rouge_l_f1(prediction, reference),
        "text_similarity": float("nan") if args.format_only else sequence_similarity(prediction, reference),
        "prediction_length": len(str(prediction).split()),
        "reference_length": 0 if args.format_only else len(reference.split()),
    }
    scores.update(format_components(prompt, prediction, row))
    return scores


def summarize_results(results_df):
    metric_columns = [column for column in METRIC_COLUMNS if column in results_df.columns]
    summary = (
        results_df.groupby("fold", dropna=False)[metric_columns]
        .mean(numeric_only=True)
        .reset_index()
    )
    overall = {
        "fold": "overall",
        **{column: results_df[column].mean(skipna=True) for column in metric_columns},
    }
    return pd.concat([summary, pd.DataFrame([overall])], ignore_index=True)


def write_visualizations(results_df, summary_df, output_dir):
    import matplotlib.pyplot as plt

    metric_columns = [
        column
        for column in METRIC_COLUMNS
        if column in results_df.columns and not results_df[column].isna().all()
    ]
    if not metric_columns:
        raise ValueError("No numeric metric columns were available to visualize.")

    fold_summary = summary_df[summary_df["fold"] != "overall"].copy()
    fold_summary["fold"] = fold_summary["fold"].astype(int)
    ax = fold_summary.plot(x="fold", y=metric_columns, kind="bar", figsize=(10, 5))
    ax.set_ylim(0, 1)
    ax.set_ylabel("Mean score")
    ax.set_title("TABLELLM cross-validation scores by fold")
    ax.legend(loc="lower right")
    ax.figure.tight_layout()
    fold_chart = output_dir / "tablellm_cv_fold_scores.png"
    ax.figure.savefig(fold_chart, dpi=160)
    plt.close(ax.figure)

    grouped = (
        results_df.groupby("operation_type", dropna=False)[metric_columns]
        .mean(numeric_only=True)
        .sort_index()
    )
    ax = grouped.plot(kind="bar", figsize=(10, 5))
    ax.set_ylim(0, 1)
    ax.set_ylabel("Mean score")
    ax.set_title("TABLELLM scores by table operation type")
    ax.legend(loc="lower right")
    ax.figure.tight_layout()
    operation_chart = output_dir / "tablellm_cv_operation_scores.png"
    ax.figure.savefig(operation_chart, dpi=160)
    plt.close(ax.figure)

    return [fold_chart, operation_chart]


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate TABLELLM-style lab-result explanations with k-fold cross-validation "
            "and visualization outputs."
        )
    )
    parser.add_argument("--input", default="data/lab_summaries_export.csv")
    parser.add_argument("--prompt-column", default="prompt")
    parser.add_argument("--target-column", default="generated_text")
    parser.add_argument(
        "--prediction-column",
        help="Existing output column to score. If omitted, use --run-model to generate predictions.",
    )
    parser.add_argument(
        "--format-only",
        action="store_true",
        help=(
            "Evaluate output formatting, cautious language, and safety without a reference "
            "target column. Reference metrics are left empty."
        ),
    )
    parser.add_argument("--output-dir", default="outputs/tablellm_cv")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-rows", type=int, help="Optional quick-run row limit.")
    parser.add_argument("--run-model", action="store_true")
    parser.add_argument("--model-id", default=TABLELLM_MODEL_ID)
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=900)
    parser.add_argument(
        "--tablellm-prompt",
        choices=["text", "code", "original"],
        default="text",
        help="Use TABLELLM text-answer, code-solution, or original project prompt format.",
    )
    parser.add_argument("--prompt-table-rows", type=int, default=20)
    args = parser.parse_args()

    if not args.run_model and not args.prediction_column:
        raise ValueError("Provide --prediction-column for saved outputs or pass --run-model.")

    df = pd.read_csv(args.input)
    if args.max_rows:
        df = df.head(args.max_rows).copy()

    required_columns = [] if args.format_only else [args.target_column]
    if args.prompt_column:
        required_columns.append(args.prompt_column)
    if args.prediction_column:
        required_columns.append(args.prediction_column)
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if not args.format_only:
        df = df.dropna(subset=[args.target_column]).copy()
        df[args.target_column] = df[args.target_column].astype(str).str.strip()
        df = df[df[args.target_column] != ""].reset_index(drop=True)
    if args.prediction_column:
        df = df.dropna(subset=[args.prediction_column]).copy()
        df[args.prediction_column] = df[args.prediction_column].astype(str).str.strip()
        df = df[df[args.prediction_column] != ""].reset_index(drop=True)

    folds = make_folds(len(df), args.folds, args.seed)
    model = tokenizer = None
    if args.run_model:
        model, tokenizer = load_tablellm(args)

    results = []
    for fold_number, test_indices in enumerate(folds, start=1):
        for row_index in test_indices:
            row = df.iloc[row_index]
            if args.run_model:
                model_prompt = build_prompt_from_row(row, args)
                prediction = generate_text(model, tokenizer, model_prompt, args.max_new_tokens)
            else:
                model_prompt = ""
                prediction = str(row[args.prediction_column])

            scores = score_row(row, prediction, args)
            results.append(
                {
                    "fold": fold_number,
                    "source_row_index": int(row_index),
                    "operation_type": infer_operation_type(row, args.prompt_column),
                    "prediction": prediction,
                    "reference": "" if args.format_only else str(row[args.target_column]),
                    "model_prompt": model_prompt,
                    **scores,
                }
            )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results_df = pd.DataFrame(results)
    summary_df = summarize_results(results_df)
    results_path = output_dir / "tablellm_cv_results.csv"
    summary_path = output_dir / "tablellm_cv_summary.csv"
    metadata_path = output_dir / "tablellm_cv_metadata.json"
    results_df.to_csv(results_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    chart_paths = write_visualizations(results_df, summary_df, output_dir)
    metadata = {
        "input": args.input,
        "model_id": args.model_id if args.run_model else None,
        "run_model": args.run_model,
        "format_only": args.format_only,
        "folds": args.folds,
        "seed": args.seed,
        "rows_evaluated": len(results_df),
        "tablellm_prompt": args.tablellm_prompt,
        "outputs": {
            "results": str(results_path),
            "summary": str(summary_path),
            "charts": [str(path) for path in chart_paths],
        },
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(summary_df.to_string(index=False))
    print(f"\nWrote detailed results to: {results_path}")
    print(f"Wrote summary to: {summary_path}")
    for chart_path in chart_paths:
        print(f"Wrote chart to: {chart_path}")


if __name__ == "__main__":
    main()
