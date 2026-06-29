# NLP Medical Results Explanation

Utilities for creating patient-friendly explanations of MIMIC-III lab results, preparing supervised fine-tuning data, training LoRA adapters, and evaluating generated explanations.

The repo is organized by model/tool lane:

- `medgemma/`: MedGemma scripts, datasets, notebook, and adapter outputs.
- `llama/`: Llama scripts, datasets, and adapter outputs.
- `tablellm/`: TableLLM generation/evaluation scripts and results.
- `shared/`: prompt/message helpers used across lanes.
- `finetuning/`: model-agnostic SFT JSONL preparation.
- `silver_targets/`: rule/model-assisted target creation helpers.
- `data_preparation/`: raw CSV preparation and inspection helpers.
- `database/`: PostgreSQL table setup and export helpers.
- `data/`: raw/shared MIMIC lab CSV inputs.

## Setup

From the repo root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install -r requirements.txt
```

Notebook support is optional. On Windows, especially inside OneDrive paths,
Jupyter can hit long path limits while installing lab extensions. Install it
only if you need to run notebooks locally:

```powershell
pip install -r requirements-notebook.txt
```

For gated Hugging Face models:

```powershell
$env:HF_TOKEN = "hf_your_token_here"
```

GPU is strongly recommended. The training and testing scripts default to 4-bit loading when applicable.

## Core Data

```text
data/
  mimic_labs_20_for_testing.csv
  mimic_labs_for_generation.csv

medgemma/data/
  medgemma_20_outputs.csv
  medgemma_1000_outputs.csv
  finetune/train.jsonl
  finetune/validation.jsonl

llama/data/
  lab_summaries_export.csv
  lab_summaries_export_10.csv
  finetune_llama/train.jsonl
  finetune_llama/validation.jsonl
  finetune_llama_10/train.jsonl
  finetune_llama_10/validation.jsonl

llama/outputs/
  llama_tabular_outputs.csv
  llama_tabular_outputs_with_targets.csv

tablellm/outputs/trainllm_cv/
  existing TableLLM evaluation notebooks, CSVs, charts, and metadata
```

## Prepare SFT Data

Default MedGemma preparation:

```powershell
python finetuning\prepare_sft_dataset.py
```

This reads `medgemma\data\medgemma_1000_outputs.csv` and writes:

```text
medgemma/data/finetune/train.jsonl
medgemma/data/finetune/validation.jsonl
```

Prepare Llama panel-summary data:

```powershell
python finetuning\prepare_sft_dataset.py `
  --input llama\data\lab_summaries_export.csv `
  --output-dir llama\data\finetune_llama `
  --prompt-column prompt `
  --target-column generated_text
```

## Train LoRA Adapters

Train Llama:

```powershell
python llama\scripts\train_llama_lora.py
```

Defaults:

- Train file: `llama/data/finetune_llama/train.jsonl`
- Validation file: `llama/data/finetune_llama/validation.jsonl`
- Adapter output: `llama/outputs/llama-lab-lora`

Train MedGemma:

```powershell
python medgemma\scripts\train_medgemma_lora.py
```

Defaults:

- Train file: `medgemma/data/finetune/train.jsonl`
- Validation file: `medgemma/data/finetune/validation.jsonl`
- Adapter output: `medgemma/outputs/medgemma-lab-lora`

## Test And Generate

Test a Llama validation example:

```powershell
python llama\scripts\test_llama_lora.py
```

Generate row-level Llama outputs from a CSV:

```powershell
python llama\scripts\test_llama_lora.py `
  --input-csv data\mimic_labs_20_for_testing.csv `
  --output-csv llama\outputs\llama_tabular_outputs.csv `
  --max-rows 10 `
  --max-new-tokens 200
```

Generate base Llama outputs from the same rows with an explicit table-style prompt,
without loading a LoRA adapter:

```powershell
python llama\scripts\generate_llama_tabular_base.py `
  --input data\mimic_labs_20_for_testing.csv `
  --output llama\outputs\llama_tabular_base_outputs.csv `
  --max-rows 10
```

Notebook version:

```text
llama/notebooks/generate_llama_tabular_base.ipynb
```

## Silver Targets

Fill row-level target text for Llama tabular outputs:

```powershell
python silver_targets\fill_row_target_text.py
```

Defaults:

- Input: `llama/outputs/llama_tabular_outputs.csv`
- Output: `llama/outputs/llama_tabular_outputs_with_targets.csv`

## TableLLM

Generate a TableLLM `output_text` column:

```powershell
python tablellm\scripts\generate_tablellm_output_text.py `
  --load-4bit `
  --max-rows 10
```

Defaults:

- Input: `llama/outputs/llama_tabular_outputs_with_targets.csv`
- Output: `tablellm/outputs/llama_tabular_outputs_with_tablellm.csv`

Evaluate an existing prediction column:

```powershell
python tablellm\scripts\evaluate_tablellm_cv.py `
  --input llama\data\lab_summaries_export.csv `
  --prompt-column prompt `
  --target-column generated_text `
  --prediction-column generated_text `
  --folds 5
```

Default output folder:

```text
tablellm/outputs/tablellm_cv/
```

## Database Export

Create tables:

```powershell
python database\create_tables.py
```

Export the `lab_summaries` table:

```powershell
python database\export_lab_summaries_csv.py
```

Default export:

```text
llama/data/lab_summaries_export.csv
```

## Environment Variables

Database scripts and silver-standard generation use:

```powershell
$env:DB_NAME = "your_db"
$env:DB_USER = "your_user"
$env:DB_PASSWORD = "your_password"
$env:DB_HOST = "localhost"
$env:DB_PORT = "5432"
$env:DB_SCHEMA = "your_schema"
```

NVIDIA/OpenAI-compatible Llama generation uses:

```powershell
$env:NVIDIA_API_KEY = "your_key"
$env:NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
$env:NVIDIA_MODEL = "meta/llama-3.1-70b-instruct"
```

CSV preparation helpers use:

```powershell
$env:LABS_SAMPLE = "data\mimic_labs_20_for_testing.csv"
$env:LABS_GENERATION = "data\mimic_labs_for_generation.csv"
$env:SILVER_RULE = "data\silver_rule_outputs.csv"
$env:LABS_TESTING = "data\mimic_labs_20_for_testing.csv"
```

## Validation

Compile the Python files after refactors:

```powershell
python -m py_compile `
  shared\lab_prompt.py `
  database\create_tables.py `
  database\export_lab_summaries_csv.py `
  data_preparation\check_dataset.py `
  data_preparation\create_20_sample.py `
  data_preparation\generate_input.py `
  data_preparation\patient_info_from_csv.py `
  finetuning\prepare_sft_dataset.py `
  llama\scripts\generate_silver_standard_llama.py `
  llama\scripts\test_llama_lora.py `
  llama\scripts\train_llama_lora.py `
  medgemma\scripts\test_medgemma_one.py `
  medgemma\scripts\train_medgemma_lora.py `
  silver_targets\fill_row_target_text.py `
  silver_targets\generate_silver_standard.py `
  tablellm\scripts\evaluate_tablellm_cv.py `
  tablellm\scripts\generate_tablellm_output_text.py
```

## Notes

- `prepare_sft_dataset.py` is shared. Its defaults target MedGemma, and Llama uses explicit `--input`, `--output-dir`, `--prompt-column`, and `--target-column`.
- `shared/lab_prompt.py` is the central prompt builder. Model scripts import from this file.
- `tablellm/scripts/evaluate_tablellm_cv.py` can score saved outputs from any model if you pass the right `--prediction-column`.
