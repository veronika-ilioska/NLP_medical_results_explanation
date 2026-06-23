# NLP medical results explanation

Utilities for generating and fine-tuning patient-friendly explanations of medical laboratory results.

The repo supports two related workflows:

1. Generate explanation data from lab-result inputs.
2. Fine-tune an instruct model with LoRA using the generated explanations.

The current fine-tuning scripts support both:

- MedGemma, using `input_text -> medgemma_output`
- Llama, using `prompt -> generated_text`

## Repository Layout

```text
data/
  lab_summaries_export.csv          Llama-generated panel explanations
  lab_summaries_export_10.csv       Small panel-explanation sample
  llama_tabular_outputs.csv         Row-level Llama adapter test outputs
  medgemma_1000_outputs.csv         MedGemma row-level outputs
  medgemma_20_outputs.csv           Small MedGemma test output
  mimic_labs_20_for_testing.csv     Small lab input sample
  mimic_labs_for_generation.csv     Larger lab input file
  finetune/                         MedGemma JSONL fine-tuning data
  finetune_llama/                   Llama JSONL fine-tuning data
  finetune_llama_10/                Small Llama JSONL fine-tuning sample

scripts/
  common/
    lab_prompt.py                   Shared lab prompt/message helpers
  db/
    create_tables.py
    export_lab_summaries_csv.py
  finetune/
    prepare_sft_dataset.py
    train_medgemma_lora.py
    train_llama_lora.py
  generate_input_csv/
    generate_input.py
  silver_standard/
    fill_row_target_text.py         Fill row-level target_text silver references
    generate_silver_standard.py
    generate_silver_standard_llama.py
    NLP_medgemma_base.ipynb
  testing/
    check_dataset.py
    create_20_sample.py
    evaluate_tablellm_cv.py         Cross-validation metrics and charts
    generate_tablellm_output_text.py Generate output_text with TableLLM
    patient_info_from_csv
    test_medgemma_one.py
    test_llama_lora.py

outputs/
  llama/                            Local/generated Llama outputs
  trainllm_cv/                      Local/generated TableLLM evaluation outputs
```

## Data Files

### MedGemma data

`data/medgemma_1000_outputs.csv` contains row-level lab examples.

Important columns:

- `input_text`: the prompt describing one patient/lab result
- `medgemma_output`: the generated explanation used as the fine-tuning target
- `target_text`: currently empty in this file

For MedGemma fine-tuning, use:

```text
input_text -> medgemma_output
```

### Llama data

`data/lab_summaries_export.csv` contains full lab-panel examples generated with Llama.

Important columns:

- `prompt`: the full instruction prompt with many lab results
- `generated_text`: the Llama answer
- `model_used`: the source model, currently `meta/llama-3.1-70b-instruct`

For Llama fine-tuning, use:

```text
prompt -> generated_text
```

## Setup

From the repo root:

```powershell
cd C:\Users\ilios\NLP_medical_results_explanation
```

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
```

Install dependencies:

```powershell
pip install -U pandas datasets torch peft trl numexpr
pip install -U "transformers==4.56.2" "accelerate==1.13.0"
```

For this text-only project, `torchvision` is not needed. If it causes an import error such as `operator torchvision::nms does not exist`, remove it:

```powershell
python -m pip uninstall -y torchvision
```

Set your Hugging Face token:

```powershell
$env:HF_TOKEN = "hf_your_token_here"
```

You need access to the gated model pages before training:

- MedGemma: https://huggingface.co/google/medgemma-4b-it
- Llama: https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct

GPU is strongly recommended. CPU fine-tuning will be very slow.

## Fine-Tune Llama

Use this workflow when you want to fine-tune a Llama model using the better Llama-generated results in `data/lab_summaries_export.csv`.

### 1. Prepare the Llama SFT dataset

```powershell
python scripts\finetune\prepare_sft_dataset.py `
  --input data\lab_summaries_export.csv `
  --output-dir data\finetune_llama `
  --prompt-column prompt `
  --target-column generated_text
```

This creates:

```text
data/finetune_llama/train.jsonl
data/finetune_llama/validation.jsonl
```

The current export creates 90 training examples and 10 validation examples.

### 2. Train the Llama LoRA adapter

```powershell
python scripts\finetune\train_llama_lora.py
```

By default this trains:

```text
meta-llama/Llama-3.1-8B-Instruct
```

and saves the adapter to:

```text
outputs/llama-lab-lora
```

To use a different Llama checkpoint:

```powershell
python scripts\finetune\train_llama_lora.py `
  --model-id meta-llama/Llama-3.1-8B-Instruct
```

Useful lower-memory options:

```powershell
python scripts\finetune\train_llama_lora.py `
  --batch-size 1 `
  --grad-accum 8 `
  --max-seq-length 2048
```

### 3. Test the Llama adapter

Run one validation example:

```powershell
python scripts\testing\test_llama_lora.py
```

Compare base Llama against the fine-tuned adapter:

```powershell
python scripts\testing\test_llama_lora.py --compare-base
```

Test a different validation example:

```powershell
python scripts\testing\test_llama_lora.py --compare-base --example-index 3
```

Test directly on a tabular CSV row:

```powershell
python scripts\testing\test_llama_lora.py `
  --input-csv data\mimic_labs_20_for_testing.csv `
  --row-index 0 `
  --show-prompt
```

For CSV input, the test script uses `input_text` or `prompt` if either column exists.
If neither exists, it builds a lab-result prompt from tabular columns such as
`GENDER`, `ADMISSION_TYPE`, `DIAGNOSIS`, `lab_name`, `fluid`, `category`,
`VALUE`, `VALUEUOM`, and `FLAG`.

You can also force specific CSV columns:

```powershell
python scripts\testing\test_llama_lora.py `
  --input-csv data\lab_summaries_export.csv `
  --prompt-column prompt `
  --target-column generated_text
```

Generate outputs for 10 tabular rows and save them to a CSV:

```powershell
python scripts\testing\test_llama_lora.py `
  --adapter-dir outputs\llama-lab-lora-10 `
  --input-csv data\mimic_labs_20_for_testing.csv `
  --output-csv data\llama_tabular_outputs.csv `
  --max-rows 10 `
  --max-new-tokens 200
```

Remove `--max-rows 10` to process every row. The output file keeps the original
input columns and adds `source_row_index`, `model_prompt`, and
`fine_tuned_output`. Add `--compare-base` to also save `base_model_output`.
The output CSV is updated after every processed row so partial progress is kept
if a long Colab run stops early.

Good signs:

- The output keeps the exact bullet format.
- Tests stay in the same order as the input.
- Each bullet is short and patient-friendly.
- The answer ends with `General Overview:`.
- The fine-tuned output is closer to the validation target than the base model output.

## Fine-Tune MedGemma

Use this workflow when you want to fine-tune MedGemma using `data/medgemma_1000_outputs.csv`.

### 1. Prepare the MedGemma SFT dataset

```powershell
python scripts\finetune\prepare_sft_dataset.py `
  --input data\medgemma_1000_outputs.csv `
  --output-dir data\finetune `
  --prompt-column input_text `
  --target-column medgemma_output
```

This creates:

```text
data/finetune/train.jsonl
data/finetune/validation.jsonl
```

The current file creates 900 training examples and 100 validation examples.

### 2. Train the MedGemma LoRA adapter

```powershell
python scripts\finetune\train_medgemma_lora.py
```

By default this trains:

```text
google/medgemma-4b-it
```

and saves the adapter to:

```text
outputs/medgemma-lab-lora
```

Useful lower-memory options:

```powershell
python scripts\finetune\train_medgemma_lora.py `
  --batch-size 1 `
  --grad-accum 8 `
  --max-seq-length 768
```

## What the Fine-Tuning Scripts Do

### `prepare_sft_dataset.py`

Converts a CSV into chat-style JSONL for supervised fine-tuning.

Each output row looks like:

```json
{
  "messages": [
    {"role": "system", "content": "You produce concise, patient-friendly explanations..."},
    {"role": "user", "content": "The prompt goes here"},
    {"role": "assistant", "content": "The target answer goes here"}
  ]
}
```

The script:

1. Reads the CSV.
2. Filters rows with missing prompts or targets.
3. Shuffles the rows.
4. Splits them into train and validation sets.
5. Writes `train.jsonl` and `validation.jsonl`.

### `train_llama_lora.py`

Loads a Llama instruct model and trains a LoRA adapter on `data/finetune_llama`.

Default input:

```text
data/finetune_llama/train.jsonl
data/finetune_llama/validation.jsonl
```

Default output:

```text
outputs/llama-lab-lora
```

### `train_medgemma_lora.py`

Loads MedGemma and trains a LoRA adapter on `data/finetune`.

Default input:

```text
data/finetune/train.jsonl
data/finetune/validation.jsonl
```

Default output:

```text
outputs/medgemma-lab-lora
```

### `test_llama_lora.py`

Loads the base Llama model plus the trained adapter and generates an answer for one validation prompt.

It can also print the base model output for comparison:

```powershell
python scripts\testing\test_llama_lora.py --compare-base
```

### `evaluate_tablellm_cv.py`

Evaluates TABLELLM-style tabular lab explanations with k-fold cross-validation,
TABLELLM prompt templates, text-similarity metrics, format checks, and charts.

Score saved model outputs:

```powershell
python scripts\testing\evaluate_tablellm_cv.py `
  --input data\lab_summaries_export.csv `
  --prompt-column prompt `
  --target-column generated_text `
  --prediction-column generated_text `
  --folds 5 `
  --output-dir outputs\tablellm_cv
```

Run the TableLLM checkpoint directly on held-out folds:

```powershell
python scripts\testing\evaluate_tablellm_cv.py `
  --run-model `
  --model-id RUCKBReasoning/TableLLM-8b `
  --load-4bit `
  --folds 5 `
  --output-dir outputs\tablellm_cv
```

The script writes detailed row scores, fold summaries, metadata, and PNG charts.

When a CSV has predictions but no reference answers, use format-only mode. This
is useful for `data/llama_tabular_outputs.csv`, where `target_text` can be
empty:

```powershell
python scripts\testing\evaluate_tablellm_cv.py `
  --input data\llama_tabular_outputs.csv `
  --prompt-column input_text `
  --prediction-column fine_tuned_output `
  --folds 5 `
  --format-only `
  --output-dir outputs\tablellm_format_eval
```

To create row-level silver reference targets first, fill `target_text` into a
new CSV and then run reference-based evaluation:

```powershell
python scripts\silver_standard\fill_row_target_text.py `
  --input data\llama_tabular_outputs.csv `
  --output data\llama_tabular_outputs_with_targets.csv

python scripts\testing\evaluate_tablellm_cv.py `
  --input data\llama_tabular_outputs_with_targets.csv `
  --prompt-column input_text `
  --target-column target_text `
  --prediction-column fine_tuned_output `
  --folds 5 `
  --output-dir outputs\tablellm_cv_real
```

To generate a separate `output_text` column with TableLLM before evaluating,
run this on a GPU machine or Colab GPU runtime:

```powershell
python scripts\testing\generate_tablellm_output_text.py `
  --input data\llama_tabular_outputs_with_targets.csv `
  --output data\llama_tabular_outputs_with_tablellm.csv `
  --prompt-column input_text `
  --output-column output_text `
  --model-id RUCKBReasoning/TableLLM-8b `
  --load-4bit `
  --max-rows 10

python scripts\testing\evaluate_tablellm_cv.py `
  --input data\llama_tabular_outputs_with_tablellm.csv `
  --prompt-column input_text `
  --target-column target_text `
  --prediction-column output_text `
  --folds 5 `
  --output-dir outputs\tablellm_output_text_eval
```

For Colab, clone or pull the repository, install `requirements-colab.txt`, and
run the same commands with shell cells. Use a GPU runtime only for direct
`RUCKBReasoning/TableLLM-8b` inference:

```python
%cd /content/NLP_medical_results_explanation
!pip uninstall -y torchvision torchao
!pip install -q -r requirements-colab.txt
!python scripts/testing/evaluate_tablellm_cv.py \
  --input data/lab_summaries_export.csv \
  --prompt-column prompt \
  --target-column generated_text \
  --prediction-column generated_text \
  --folds 5 \
  --output-dir outputs/tablellm_cv
```

## Generate More Llama Training Data

The current Llama fine-tuning file has only 100 examples. That is enough to test the pipeline, but too small for a strong final model.

To generate more Llama outputs, use:

```powershell
python scripts\silver_standard\generate_silver_standard_llama.py 1000
```

The optional number limits how many patients/panels are processed. If omitted, the script processes all available panels:

```powershell
python scripts\silver_standard\generate_silver_standard_llama.py
```

After generating more rows in the database, export them:

```powershell
python scripts\db\export_lab_summaries_csv.py
```

Then rebuild the fine-tuning JSONL:

```powershell
python scripts\finetune\prepare_sft_dataset.py `
  --input data\lab_summaries_export.csv `
  --output-dir data\finetune_llama `
  --prompt-column prompt `
  --target-column generated_text
```

## Environment Variables for Generation Scripts

Some generation and database scripts use environment variables from `.env` or PowerShell.

Database variables:

```powershell
$env:DB_NAME = "your_db"
$env:DB_USER = "your_user"
$env:DB_PASSWORD = "your_password"
$env:DB_HOST = "localhost"
$env:DB_PORT = "5432"
$env:DB_SCHEMA = "your_schema"
```

NVIDIA/OpenAI-compatible Llama endpoint variables:

```powershell
$env:NVIDIA_API_KEY = "your_key"
$env:NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
$env:NVIDIA_MODEL = "meta/llama-3.1-70b-instruct"
```

Older CSV-prep variables:

```powershell
$env:LABS_SAMPLE = "data\mimic_labs_20_for_testing.csv"
$env:LABS_GENERATION = "data\mimic_labs_for_generation.csv"
$env:SILVER_RULE = "data\silver_rule_outputs.csv"
$env:LABS_TESTING = "data\mimic_labs_20_for_testing.csv"
```

## Quick Checks

Check that the Python scripts compile:

```powershell
python -m py_compile `
  scripts\finetune\prepare_sft_dataset.py `
  scripts\finetune\train_llama_lora.py `
  scripts\testing\test_llama_lora.py `
  scripts\finetune\train_medgemma_lora.py
```

Inspect the Llama dataset:

```powershell
Get-Content data\finetune_llama\train.jsonl -TotalCount 1
```

Inspect the MedGemma dataset:

```powershell
Get-Content data\finetune\train.jsonl -TotalCount 1
```

## Troubleshooting

### Hugging Face access error

Make sure:

1. You accepted the model license on Hugging Face.
2. `$env:HF_TOKEN` is set.
3. Your token has access to the model.

### CUDA out of memory

Try a shorter context:

```powershell
python scripts\finetune\train_llama_lora.py --max-seq-length 2048
```

or for MedGemma:

```powershell
python scripts\finetune\train_medgemma_lora.py --max-seq-length 768
```

Also keep:

```powershell
--batch-size 1
```

### Fine-tuned model is not better

The most likely reason is not enough training data. The current Llama set has only 100 examples. Generate more Llama outputs, review quality, export again, and rebuild `data/finetune_llama`.

### `target_text` is empty

That is expected for `data/medgemma_1000_outputs.csv`. Use `medgemma_output` as the target column unless you manually create reviewed `target_text` values.
