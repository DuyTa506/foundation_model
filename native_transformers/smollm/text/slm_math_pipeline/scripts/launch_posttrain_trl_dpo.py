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
    parser = argparse.ArgumentParser(description="Launch DPO posttraining with automatic SFT fallback.")
    parser.add_argument("--training_config", required=True)
    parser.add_argument("--dataset_config", required=False)
    args = parser.parse_args()

    with open(args.training_config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    model_id = cfg["model"]["model_id"]
    tokenizer_id = cfg["model"]["tokenizer_id"]
    out_dir = cfg["run"]["output_dir"]
    seed = int(cfg["run"]["seed"])
    pref_manifest = Path(cfg["data"]["preference_manifest_path"])
    sft_manifest = Path(cfg["data"]["sft_manifest_path"])
    preferred_mode = cfg["mode"]["preferred"]
    strip_reasoning = bool(cfg["data"].get("strip_think_tags", True))

    try:
        import torch
        from datasets import Dataset
        from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
        from trl import DPOConfig, DPOTrainer, SFTConfig, SFTTrainer
    except Exception as exc:
        raise RuntimeError(f"Missing posttrain dependencies: {exc}") from exc

    set_seed(seed)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16 if cfg["dpo"]["bf16"] else None,
        trust_remote_code=cfg["model"].get("trust_remote_code", False),
    )
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_id, trust_remote_code=cfg["model"].get("trust_remote_code", False)
    )

    pref_rows = read_jsonl(pref_manifest) if pref_manifest.exists() else []
    use_dpo = preferred_mode == "dpo" and len(pref_rows) > 0

    if use_dpo:
        dpo_samples = []
        for row in pref_rows:
            p = row.get("prompt")
            c = row.get("chosen")
            r = row.get("rejected")
            if all(isinstance(x, str) and x for x in [p, c, r]):
                if strip_reasoning:
                    p = strip_think_tags(p)
                    c = strip_think_tags(c)
                    r = strip_think_tags(r)
                dpo_samples.append({"prompt": p, "chosen": c, "rejected": r})
        if not dpo_samples:
            use_dpo = False
        else:
            dpo_dataset = Dataset.from_list(dpo_samples)
            dpo_args = DPOConfig(
                output_dir=out_dir,
                beta=float(cfg["dpo"]["beta"]),
                max_length=int(cfg["dpo"]["max_length"]),
                max_prompt_length=int(cfg["dpo"]["max_prompt_length"]),
                learning_rate=float(cfg["dpo"]["learning_rate"]),
                per_device_train_batch_size=int(cfg["dpo"]["per_device_train_batch_size"]),
                gradient_accumulation_steps=int(cfg["dpo"]["gradient_accumulation_steps"]),
                max_steps=int(cfg["dpo"]["max_steps"]),
                warmup_steps=int(cfg["dpo"]["warmup_steps"]),
                logging_steps=int(cfg["dpo"]["logging_steps"]),
                save_steps=int(cfg["dpo"]["save_steps"]),
                bf16=bool(cfg["dpo"]["bf16"]),
                seed=seed,
            )
            trainer = DPOTrainer(
                model=model,
                processing_class=tokenizer,
                train_dataset=dpo_dataset,
                args=dpo_args,
            )
            trainer.train()
            trainer.save_model(out_dir)
            print(f"[ok] DPO posttrain complete: {out_dir}")
            return

    sft_rows = read_jsonl(sft_manifest) if sft_manifest.exists() else []
    sft_samples = []
    for row in sft_rows:
        p = row.get("prompt")
        r = row.get("response")
        if isinstance(p, str) and isinstance(r, str):
            if strip_reasoning:
                p = strip_think_tags(p)
                r = strip_think_tags(r)
            sft_samples.append({"text": f"### Prompt\n{p}\n\n### Response\n{r}"})
    if not sft_samples:
        raise ValueError("No usable DPO pairs and no SFT fallback samples were found.")

    from datasets import Dataset

    ds = Dataset.from_list(sft_samples)
    sft_args = SFTConfig(
        output_dir=out_dir,
        max_seq_length=int(cfg["sft_fallback"]["max_seq_length"]),
        learning_rate=float(cfg["sft_fallback"]["learning_rate"]),
        per_device_train_batch_size=int(cfg["sft_fallback"]["per_device_train_batch_size"]),
        gradient_accumulation_steps=int(cfg["sft_fallback"]["gradient_accumulation_steps"]),
        max_steps=int(cfg["sft_fallback"]["max_steps"]),
        warmup_steps=int(cfg["sft_fallback"]["warmup_steps"]),
        logging_steps=int(cfg["sft_fallback"]["logging_steps"]),
        save_steps=int(cfg["sft_fallback"]["save_steps"]),
        bf16=bool(cfg["sft_fallback"]["bf16"]),
        dataset_text_field="text",
        seed=seed,
    )
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=ds,
        args=sft_args,
    )
    trainer.train()
    trainer.save_model(out_dir)
    print(f"[ok] fallback SFT posttrain complete: {out_dir}")


if __name__ == "__main__":
    main()
