import argparse
import json
import os
from pathlib import Path

import torch

# This is a text-only smoke test. Avoid optional torchvision imports because a
# mismatched torch/torchvision install can crash Transformers before loading.
import transformers.utils.import_utils as transformers_import_utils
import transformers.utils as transformers_utils

transformers_import_utils.is_torchvision_available = lambda: False
transformers_utils.is_torchvision_available = lambda: False

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_jsonl(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_prompt(messages, tokenizer):
    prompt_messages = [message for message in messages if message["role"] != "assistant"]
    return tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def generate(model, tokenizer, prompt, max_new_tokens):
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


def main():
    parser = argparse.ArgumentParser(
        description="Test a fine-tuned Llama LoRA adapter on validation prompts."
    )
    parser.add_argument("--model-id", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--adapter-dir", default="outputs/llama-lab-lora")
    parser.add_argument("--validation-file", default="data/finetune_llama/validation.jsonl")
    parser.add_argument("--example-index", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=900)
    parser.add_argument("--compare-base", action="store_true")
    args = parser.parse_args()

    token = os.getenv("HF_TOKEN")
    if not token:
        raise EnvironmentError("Set HF_TOKEN before loading Llama weights.")

    records = load_jsonl(args.validation_file)
    if not records:
        raise ValueError(f"No validation records found in {args.validation_file}")
    if args.example_index >= len(records):
        raise IndexError(
            f"--example-index {args.example_index} is out of range for "
            f"{len(records)} validation records."
        )

    record = records[args.example_index]
    expected = next(
        message["content"] for message in record["messages"] if message["role"] == "assistant"
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, token=token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        token=token,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )
    base_model.eval()

    prompt = build_prompt(record["messages"], tokenizer)

    if args.compare_base:
        print("\n=== BASE MODEL OUTPUT ===\n")
        print(generate(base_model, tokenizer, prompt, args.max_new_tokens))

    tuned_model = PeftModel.from_pretrained(base_model, args.adapter_dir)
    tuned_model.eval()

    print("\n=== FINE-TUNED OUTPUT ===\n")
    print(generate(tuned_model, tokenizer, prompt, args.max_new_tokens))

    print("\n=== EXPECTED VALIDATION TARGET ===\n")
    print(expected)


if __name__ == "__main__":
    main()
