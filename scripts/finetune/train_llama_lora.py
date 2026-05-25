import argparse
import os
from pathlib import Path

import torch
from datasets import load_dataset

# This project fine-tunes text-only Llama models. Some local environments have
# incompatible torch/torchvision builds, and optional vision imports can crash
# before training starts. Disable that optional path before PEFT imports models.
import transformers.utils.import_utils as transformers_import_utils
import transformers.utils as transformers_utils

transformers_import_utils.is_torchvision_available = lambda: False
transformers_utils.is_torchvision_available = lambda: False

_read_text = Path.read_text


def _read_text_utf8(self, encoding=None, errors=None, newline=None):
    return _read_text(
        self,
        encoding=encoding or "utf-8",
        errors=errors,
        newline=newline,
    )


Path.read_text = _read_text_utf8

from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer


def format_example(example, tokenizer):
    return tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune a Llama instruct model with LoRA on lab explanations."
    )
    parser.add_argument("--model-id", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--train-file", default="data/finetune_llama/train.jsonl")
    parser.add_argument("--validation-file", default="data/finetune_llama/validation.jsonl")
    parser.add_argument("--output-dir", default="outputs/llama-lab-lora")
    parser.add_argument("--max-seq-length", type=int, default=4096)
    parser.add_argument("--epochs", type=float, default=2)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--eval-steps", type=int, default=100)
    args = parser.parse_args()

    token = os.getenv("HF_TOKEN")
    if not token:
        raise EnvironmentError(
            "Set HF_TOKEN before loading Llama weights from Hugging Face."
        )

    dataset = load_dataset(
        "json",
        data_files={
            "train": args.train_file,
            "validation": args.validation_file,
        },
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, token=token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        token=token,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )
    model.config.use_cache = False

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    training_args = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        gradient_checkpointing=True,
        max_length=args.max_seq_length,
        packing=False,
        logging_steps=10,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        eval_strategy="steps",
        save_total_limit=2,
        bf16=torch.cuda.is_available(),
        fp16=False,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        peft_config=peft_config,
        formatting_func=lambda example: format_example(example, tokenizer),
        processing_class=tokenizer,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved Llama LoRA adapter to: {args.output_dir}")


if __name__ == "__main__":
    main()
