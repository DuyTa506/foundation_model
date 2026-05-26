#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path
from typing import Dict, List

import yaml


THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def strip_think_tags(text: str) -> str:
    return THINK_RE.sub("", text).strip()


def read_jsonl(path: Path) -> List[Dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch TRL SFT finetuning from YAML configs.")
    parser.add_argument("--training_config", required=True)
    parser.add_argument("--dataset_config", required=False)
    args = parser.parse_args()

    with open(args.training_config, "r", encoding="utf-8") as f:
        train_cfg = yaml.safe_load(f)

    model_id = train_cfg["model"]["model_id"]
    tokenizer_id = train_cfg["model"]["tokenizer_id"]
    out_dir = train_cfg["run"]["output_dir"]
    seed = int(train_cfg["run"]["seed"])
    data_manifest = Path(train_cfg["data"]["manifest_path"])
    prompt_field = train_cfg["data"]["prompt_field"]
    response_field = train_cfg["data"]["response_field"]
    strip_reasoning = bool(train_cfg["data"].get("strip_think_tags", True))

    try:
        import torch
        from datasets import Dataset
        from peft import LoraConfig
        from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
        from trl import SFTConfig, SFTTrainer
    except Exception as exc:
        raise RuntimeError(f"Missing training dependencies: {exc}") from exc

    set_seed(seed)
    rows = read_jsonl(data_manifest)
    samples = []
    for row in rows:
        p = row.get(prompt_field)
        r = row.get(response_field)
        if isinstance(p, str) and isinstance(r, str):
            if strip_reasoning:
                p = strip_think_tags(p)
                r = strip_think_tags(r)
            samples.append({"text": f"### Question\n{p}\n\n### Answer\n{r}"})
    if not samples:
        raise ValueError(f"No usable rows from {data_manifest}")

    dataset = Dataset.from_list(samples)

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16 if train_cfg["training"]["bf16"] else None,
        trust_remote_code=train_cfg["model"].get("trust_remote_code", False),
    )
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_id, trust_remote_code=train_cfg["model"].get("trust_remote_code", False)
    )

    lora_cfg = None
    if train_cfg.get("lora", {}).get("enabled", True):
        lora_cfg = LoraConfig(
            r=int(train_cfg["lora"]["r"]),
            lora_alpha=int(train_cfg["lora"]["alpha"]),
            lora_dropout=float(train_cfg["lora"]["dropout"]),
            target_modules=list(train_cfg["lora"]["target_modules"]),
            bias="none",
            task_type="CAUSAL_LM",
        )

    sft_args = SFTConfig(
        output_dir=out_dir,
        max_seq_length=int(train_cfg["training"]["max_seq_length"]),
        max_steps=int(train_cfg["training"]["max_steps"]),
        per_device_train_batch_size=int(train_cfg["training"]["per_device_train_batch_size"]),
        gradient_accumulation_steps=int(train_cfg["training"]["gradient_accumulation_steps"]),
        learning_rate=float(train_cfg["training"]["learning_rate"]),
        lr_scheduler_type=train_cfg["training"]["lr_scheduler_type"],
        warmup_steps=int(train_cfg["training"]["warmup_steps"]),
        weight_decay=float(train_cfg["training"]["weight_decay"]),
        logging_steps=int(train_cfg["training"]["logging_steps"]),
        save_steps=int(train_cfg["training"]["save_steps"]),
        bf16=bool(train_cfg["training"]["bf16"]),
        report_to=train_cfg["training"]["report_to"],
        dataset_text_field="text",
        seed=seed,
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        args=sft_args,
        peft_config=lora_cfg,
    )
    trainer.train()
    trainer.save_model(out_dir)
    print(f"[ok] finetune complete: {out_dir}")


if __name__ == "__main__":
    main()
