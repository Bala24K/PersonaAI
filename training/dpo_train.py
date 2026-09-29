#!/usr/bin/env python3
"""Stage 2 preference optimization (Section 10 of pitch doc): DPO on the exported
chosen/rejected pairs (training/export_dataset.py), starting from the SFT checkpoint.

*** CANNOT BE RUN IN THIS SANDBOX *** for the same reason as sft_train.py: no GPU, and
huggingface.co isn't in this environment's network allowlist. Complete and correct,
ready to run on a GPU box with hub access.

Usage (on a real GPU box, after sft_train.py):
  python training/dpo_train.py \
      --base_model Qwen/Qwen2.5-7B-Instruct \
      --sft_adapter models/persona-sft \
      --data data/dpo_dataset.jsonl \
      --output_dir models/persona-dpo
"""
import argparse
import json


def load_preference_dataset(path: str):
    from datasets import Dataset

    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            records.append({
                "prompt": [
                    {"role": "system", "content": row["system"]},
                    {"role": "user", "content": row["user_message"]},
                ],
                "chosen": [{"role": "assistant", "content": row["chosen"]}],
                "rejected": [{"role": "assistant", "content": row["rejected"]}],
            })
    return Dataset.from_list(records)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--sft_adapter", default=None, help="path to the LoRA adapter from sft_train.py")
    parser.add_argument("--data", default="data/dpo_dataset.jsonl")
    parser.add_argument("--output_dir", default="models/persona-dpo")
    parser.add_argument("--beta", type=float, default=0.1, help="DPO temperature")
    parser.add_argument("--epochs", type=float, default=1)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=8)
    args = parser.parse_args()

    import torch
    from peft import LoraConfig, PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig, DPOTrainer

    print(f"Loading base model {args.base_model} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=torch.bfloat16, device_map="auto")

    if args.sft_adapter:
        print(f"Loading SFT adapter from {args.sft_adapter} ...")
        model = PeftModel.from_pretrained(model, args.sft_adapter, is_trainable=True)
        lora_config = None  # already has adapters loaded and trainable
    else:
        lora_config = LoraConfig(
            r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
            task_type="CAUSAL_LM", target_modules="all-linear",
        )

    dataset = load_preference_dataset(args.data)
    print(f"Loaded {len(dataset)} preference pairs from {args.data}")

    dpo_config = DPOConfig(
        output_dir=args.output_dir,
        beta=args.beta,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        logging_steps=5,
        save_strategy="epoch",
        bf16=True,
        report_to=[],
    )

    trainer = DPOTrainer(
        model=model,
        args=dpo_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    print(f"Saved DPO-tuned adapter to {args.output_dir}")


if __name__ == "__main__":
    main()
