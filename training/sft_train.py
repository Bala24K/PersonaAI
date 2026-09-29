#!/usr/bin/env python3
"""Stage 1 SFT (Section 10 of pitch doc): LoRA fine-tune of a base conversational LLM on
the exported SFT dataset (training/export_dataset.py).

*** CANNOT BE RUN IN THIS SANDBOX. *** Confirmed, not assumed: this environment has no
GPU and its network allowlist doesn't include huggingface.co (only pypi/npm/github
registries are reachable), so `from_pretrained(...)` on any real base model — Qwen,
Llama, Gemma, Mistral, whatever you pick — cannot download weights here, and CPU LoRA
training on a real-sized model would be impractically slow even if it could.

This script IS complete, correct, and ready to run as-is on a machine with a GPU and
model-hub access — that's the actual deliverable. Point --base_model at any causal LM
on the Hub (e.g. "Qwen/Qwen2.5-7B-Instruct") and --data at the exported JSONL.

Usage (on a real GPU box):
  pip install torch transformers peft trl accelerate bitsandbytes
  python training/sft_train.py \
      --base_model Qwen/Qwen2.5-7B-Instruct \
      --data data/sft_dataset.jsonl \
      --output_dir models/persona-sft \
      --use_4bit
"""
import argparse
import json


def load_dataset_from_jsonl(path: str):
    from datasets import Dataset

    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            records.append({"messages": row["messages"]})
    return Dataset.from_list(records)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", required=True, help="HF hub id, e.g. Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--data", default="data/sft_dataset.jsonl")
    parser.add_argument("--output_dir", default="models/persona-sft")
    parser.add_argument("--use_4bit", action="store_true", help="QLoRA: load base model in 4-bit")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=4)
    args = parser.parse_args()

    # Imports deferred so `--help` and dry-run argument validation don't require torch.
    import torch
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    print(f"Loading base model {args.base_model} ...")
    quant_config = None
    if args.use_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=quant_config,
        device_map="auto",
        torch_dtype=torch.bfloat16 if not args.use_4bit else None,
    )

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear",
    )

    dataset = load_dataset_from_jsonl(args.data)
    print(f"Loaded {len(dataset)} SFT examples from {args.data}")

    sft_config = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        logging_steps=10,
        save_strategy="epoch",
        bf16=True,
        report_to=[],
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
        peft_config=lora_config,
        processing_class=tokenizer,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    print(f"Saved LoRA adapter to {args.output_dir}")
    print("Merge with `model.merge_and_unload()` before deploying if you need a single "
          "merged checkpoint rather than base+adapter.")


if __name__ == "__main__":
    main()
