#!/usr/bin/env python3
"""
SFT finetuning with hybrid thinking/non-thinking mode support.

Key changes from the old version:
  - Uses the from-scratch model (outputs/pretrain or midtrain) + new tokenizer.
  - Conversational samples via tokenizer.apply_chat_template (ChatML).
  - Per-sample mode: think | no_think (controls <think> retention vs stripping).
  - Completion-only loss masking (trains on assistant tokens only).
  - Full fine-tune by default (LoRA optional for fast iteration).
  - No hardcoded ### Question/### Answer format.

Usage:
    accelerate launch scripts/launch_finetune_trl_sft.py \\
        --training_config configs/training_finetune_trl_sft.yaml \\
        --dataset_config configs/datasets_en_vi_math_finetune.yaml

Dataset format (JSONL, each row):
    {
        "prompt": "Tính giá trị của ...",
        "response": "Ta có ...\nVậy đáp số là 42.",
        "reasoning": "<optional CoT trace>",  # only used in think mode
        "mode": "think",                       # "think" | "no_think"
        "language": "vi"
    }
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import List, Dict

import yaml


THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def strip_think_tags(text: str) -> str:
    return THINK_RE.sub("", text).strip()


def build_conversation(
    row: Dict,
    mode: str,
    system_prompt: str | None = None,
) -> List[Dict[str, str]]:
    """
    Build a ChatML conversation from a dataset row.

    mode="think":    keep / inject <think>reasoning</think> in assistant turn
    mode="no_think": strip any <think> from the assistant response
    """
    messages: List[Dict[str, str]] = []

    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    messages.append({"role": "user", "content": row.get("prompt", "")})

    response: str = row.get("response", "")
    reasoning: str = row.get("reasoning", "")

    if mode == "think":
        # If response already has <think>...</think>, keep it.
        if "<think>" in response:
            assistant_content = response
        elif reasoning:
            # Inject <think> block from the dedicated reasoning field
            assistant_content = f"<think>\n{reasoning}\n</think>\n{response}"
        else:
            # No reasoning available; just keep the response without <think>
            assistant_content = response
    else:
        # no_think: strip any stray <think> tags
        assistant_content = strip_think_tags(response)

    messages.append({"role": "assistant", "content": assistant_content})
    return messages


def read_jsonl(path: Path) -> List[Dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid-thinking SFT finetuning.")
    parser.add_argument("--training_config", required=True)
    parser.add_argument("--dataset_config", required=False)
    parser.add_argument("--mode_override",
                        choices=["think", "no_think", "mixed"],
                        default=None,
                        help="Override per-sample mode. 'mixed' uses the row's mode field.")
    args = parser.parse_args()

    with open(args.training_config, "r", encoding="utf-8") as f:
        train_cfg = yaml.safe_load(f)

    model_id: str = train_cfg["model"]["model_id"]
    tokenizer_id: str = train_cfg["model"]["tokenizer_id"]
    out_dir: str = train_cfg["run"]["output_dir"]
    seed: int = int(train_cfg["run"]["seed"])
    data_manifest = Path(train_cfg["data"]["manifest_path"])

    # Default mode from config (can be overridden per sample)
    default_mode: str = train_cfg["data"].get("default_mode", "mixed")
    if args.mode_override:
        default_mode = args.mode_override

    system_prompt: str | None = train_cfg["data"].get("system_prompt")

    try:
        import torch
        from datasets import Dataset
        from peft import LoraConfig
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            set_seed,
        )
        from trl import DataCollatorForCompletionOnlyLM, SFTConfig, SFTTrainer
    except ImportError as exc:
        raise RuntimeError(f"Missing deps: {exc}") from exc

    set_seed(seed)

    # ── Load model ────────────────────────────────────────────────────────────
    print(f"[sft] loading model from: {model_id}")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16 if train_cfg["training"]["bf16"] else None,
        local_files_only=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Build conversational dataset ──────────────────────────────────────────
    rows = read_jsonl(data_manifest)
    samples: List[Dict] = []

    for row in rows:
        p = row.get("prompt") or row.get(train_cfg["data"].get("prompt_field", "prompt"))
        r = row.get("response") or row.get(train_cfg["data"].get("response_field", "response"))
        if not (isinstance(p, str) and isinstance(r, str)):
            continue

        # Per-sample mode: use row's "mode" field, or fall back to default
        mode: str = row.get("mode", default_mode)
        if mode not in ("think", "no_think"):
            mode = default_mode if default_mode in ("think", "no_think") else "no_think"

        row_for_conv = dict(row)
        row_for_conv["prompt"] = p
        row_for_conv["response"] = r
        conv = build_conversation(row_for_conv, mode, system_prompt)

        # Apply chat template to get text
        try:
            text = tokenizer.apply_chat_template(
                conv,
                tokenize=False,
                add_generation_prompt=False,
            )
        except Exception:
            # Fallback if template fails
            text = f"<|im_start|>user\n{p}<|im_end|>\n<|im_start|>assistant\n{r}<|im_end|>"

        samples.append({"text": text, "mode": mode})

    if not samples:
        raise ValueError(f"No usable rows from {data_manifest}")

    print(f"[sft] {len(samples)} samples  "
          f"think={sum(1 for s in samples if s['mode']=='think')} "
          f"no_think={sum(1 for s in samples if s['mode']=='no_think')}")

    dataset = Dataset.from_list(samples)

    # ── LoRA (optional) ───────────────────────────────────────────────────────
    lora_cfg_yaml = train_cfg.get("lora", {})
    lora_config = None
    if lora_cfg_yaml.get("enabled", False):
        lora_config = LoraConfig(
            r=int(lora_cfg_yaml["r"]),
            lora_alpha=int(lora_cfg_yaml["alpha"]),
            lora_dropout=float(lora_cfg_yaml["dropout"]),
            target_modules=list(lora_cfg_yaml["target_modules"]),
            bias="none",
            task_type="CAUSAL_LM",
        )
        print(f"[sft] LoRA enabled: r={lora_config.r} modules={lora_config.target_modules}")
    else:
        print("[sft] full fine-tune (LoRA disabled)")

    # ── Completion-only data collator ─────────────────────────────────────────
    # Response template marks the start of assistant tokens (after this, labels are unmasked).
    response_template = "<|im_start|>assistant\n"
    try:
        collator = DataCollatorForCompletionOnlyLM(
            response_template=response_template,
            tokenizer=tokenizer,
        )
    except Exception as e:
        print(f"[warn] DataCollatorForCompletionOnlyLM failed ({e}); using default collator")
        collator = None

    # ── SFT config ────────────────────────────────────────────────────────────
    t_cfg = train_cfg["training"]
    sft_args = SFTConfig(
        output_dir=out_dir,
        max_seq_length=int(t_cfg["max_seq_length"]),
        max_steps=int(t_cfg.get("max_steps", -1)),
        num_train_epochs=float(t_cfg.get("num_epochs", 2)),
        per_device_train_batch_size=int(t_cfg["per_device_train_batch_size"]),
        gradient_accumulation_steps=int(t_cfg["gradient_accumulation_steps"]),
        learning_rate=float(t_cfg["learning_rate"]),
        lr_scheduler_type=t_cfg.get("lr_scheduler_type", "cosine"),
        warmup_steps=int(t_cfg["warmup_steps"]),
        weight_decay=float(t_cfg.get("weight_decay", 0.01)),
        logging_steps=int(t_cfg.get("logging_steps", 10)),
        save_steps=int(t_cfg.get("save_steps", 500)),
        bf16=bool(t_cfg.get("bf16", True)),
        report_to=t_cfg.get("report_to", "none"),
        dataset_text_field="text",
        seed=seed,
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        args=sft_args,
        peft_config=lora_config,
        data_collator=collator,
    )
    trainer.train()
    trainer.save_model(out_dir)
    tokenizer.save_pretrained(out_dir)
    print(f"[ok] SFT complete: {out_dir}")


if __name__ == "__main__":
    main()
