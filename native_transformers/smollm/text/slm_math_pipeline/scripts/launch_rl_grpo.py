#!/usr/bin/env python3
"""
GRPO / RLVR training for thinking alignment (math + science).

MiniCPM-inspired recipe:
  - TRL GRPOTrainer with vLLM rollouts
  - Three rewards: correctness (verifiable), format (<think> block), language consistency (VI)
  - Two-stage length schedule: stage 1 = standard, stage 2 (after switch_step) = penalize overlong

Usage:
    accelerate launch scripts/launch_rl_grpo.py \\
        --config configs/training_rl_grpo.yaml

Data format (prompt-only JSONL, ground-truth answers required):
    {"prompt": "Giải phương trình ...", "answer": "42", "language": "vi"}
    {"prompt": "Find the sum ...", "answer": "\\\\frac{1}{2}", "language": "en"}
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Dict

import yaml


def read_jsonl(path: Path) -> List[Dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_reward_functions(cfg: dict):
    """Build reward function callables from config."""
    from scripts.rewards.math_verify import (
        correctness_reward,
        format_reward,
        language_consistency_reward,
    )

    rewards_cfg = cfg.get("rewards", {})
    w_correct = rewards_cfg.get("correctness", {}).get("weight", 1.0)
    w_format = rewards_cfg.get("format", {}).get("weight", 0.1)
    w_lang = rewards_cfg.get("language_consistency", {}).get("weight", 0.15)
    lang_enabled = rewards_cfg.get("language_consistency", {}).get("enabled", True)

    # Two-stage length schedule
    grpo_cfg = cfg.get("grpo", {})
    length_sched = grpo_cfg.get("length_schedule", {})
    length_enabled = length_sched.get("enabled", True)
    switch_step = length_sched.get("switch_step", 2500)
    stage2_max = length_sched.get("stage2_max_tokens", 1024)
    overlong_penalty = length_sched.get("overlong_penalty_weight", 0.1)

    _current_step = [0]  # mutable reference

    def reward_fn(completions: List[str], prompts: List[str] = None,
                  ground_truths: List[str] = None, **kwargs) -> List[float]:
        # Extract ground truths from kwargs if passed as "answer"
        gts = ground_truths or kwargs.get("answer", [""] * len(completions))
        ps = prompts or kwargs.get("prompt", [""] * len(completions))

        c_r = correctness_reward(completions, gts)
        f_r = format_reward(completions)
        l_r = language_consistency_reward(completions, ps) if lang_enabled else [0.0] * len(completions)

        combined = [
            w_correct * c + w_format * f + w_lang * l
            for c, f, l in zip(c_r, f_r, l_r)
        ]

        # Two-stage length penalty
        if length_enabled and _current_step[0] >= switch_step:
            for i, comp in enumerate(completions):
                n_tokens = len(comp.split())  # rough token estimate
                if n_tokens > stage2_max:
                    combined[i] -= overlong_penalty * (n_tokens - stage2_max) / max(stage2_max, 1)

        _current_step[0] += 1
        return combined

    return reward_fn


def main() -> None:
    parser = argparse.ArgumentParser(description="GRPO/RLVR training for thinking alignment.")
    parser.add_argument("--config", default="configs/training_rl_grpo.yaml")
    parser.add_argument("--smoke_test", action="store_true", help="5 steps only.")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    try:
        import torch
        from datasets import Dataset
        from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
        from trl import GRPOConfig, GRPOTrainer
    except ImportError as exc:
        raise RuntimeError(f"Missing deps: {exc}. Install trl>=0.15") from exc

    model_id: str = cfg["model"]["model_id"]
    tokenizer_id: str = cfg["model"]["tokenizer_id"]
    out_dir: str = cfg["run"]["output_dir"]
    seed: int = cfg["run"].get("seed", 42)

    set_seed(seed)

    # ── Load model ────────────────────────────────────────────────────────────
    print(f"[grpo] loading SFT model from: {model_id}")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Load data (prompt-only) ───────────────────────────────────────────────
    data_cfg = cfg.get("data", {})
    all_rows: List[Dict] = []

    for src in data_cfg.get("en_math_prompts", []):
        src_path = Path(str(src))
        if src_path.exists():
            all_rows.extend(read_jsonl(src_path))
        else:
            # Try loading from HF
            try:
                from datasets import load_dataset
                split_name = "train"
                ds = load_dataset(src, split=split_name, trust_remote_code=True)
                for row in ds:
                    q = row.get("question") or row.get("problem") or row.get("query", "")
                    a = row.get("answer") or row.get("solution", "")
                    if q:
                        all_rows.append({"prompt": q, "answer": a, "language": "en"})
            except Exception as e:
                print(f"[warn] could not load {src}: {e}")

    for src in data_cfg.get("vi_math_prompts", []):
        src_path = Path(str(src))
        if src_path.exists():
            all_rows.extend(read_jsonl(src_path))
        else:
            print(f"[warn] VI math prompts not found: {src}. "
                  "Run scripts/data/synth_vi_reasoning.py first.")

    if not all_rows:
        raise ValueError(
            "No RLVR prompts found. Provide EN/VI math prompt files or "
            "valid HF dataset IDs in configs/training_rl_grpo.yaml"
        )

    print(f"[grpo] {len(all_rows)} prompts total  "
          f"(en={sum(1 for r in all_rows if r.get('language','en')=='en')} "
          f"vi={sum(1 for r in all_rows if r.get('language','')=='vi')})")

    # ── Apply chat template (thinking mode on) ────────────────────────────────
    prompt_format: str = data_cfg.get("prompt_format", "chatml")
    max_prompt_len: int = data_cfg.get("max_prompt_length", 1024)

    def _format_prompt(row: Dict) -> str:
        conv = [{"role": "user", "content": row["prompt"]}]
        try:
            return tokenizer.apply_chat_template(
                conv,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            return f"<|im_start|>user\n{row['prompt']}<|im_end|>\n<|im_start|>assistant\n<think>\n"

    formatted = [
        {
            "prompt": _format_prompt(r),
            "answer": r.get("answer", ""),
            "language": r.get("language", "en"),
        }
        for r in all_rows
    ]

    train_dataset = Dataset.from_list(formatted)

    # ── Reward function ───────────────────────────────────────────────────────
    reward_fn = load_reward_functions(cfg)

    # ── GRPO config ────────────────────────────────────────────────────────────
    grpo_cfg = cfg.get("grpo", {})
    total_steps = 5 if args.smoke_test else grpo_cfg.get("max_steps", 5000)

    grpo_args = GRPOConfig(
        output_dir=out_dir,
        num_generations=int(grpo_cfg.get("num_generations", 8)),
        max_new_tokens=int(grpo_cfg.get("max_new_tokens", 2048)),
        temperature=float(grpo_cfg.get("temperature", 0.9)),
        per_device_train_batch_size=int(grpo_cfg.get("per_device_train_batch_size", 1)),
        gradient_accumulation_steps=int(grpo_cfg.get("gradient_accumulation_steps", 16)),
        learning_rate=float(grpo_cfg.get("learning_rate", 1e-6)),
        warmup_steps=int(grpo_cfg.get("warmup_steps", 100)),
        max_steps=total_steps,
        logging_steps=int(grpo_cfg.get("logging_steps", 10)),
        save_steps=int(grpo_cfg.get("save_steps", 500)),
        bf16=bool(grpo_cfg.get("bf16", True)),
        beta=float(grpo_cfg.get("beta", 0.02)),   # KL penalty
        seed=seed,
    )

    # vLLM backend for generation (faster rollouts)
    vllm_cfg = cfg.get("vllm", {})
    if vllm_cfg.get("enabled", True):
        try:
            grpo_args.use_vllm = True
            grpo_args.vllm_dtype = vllm_cfg.get("dtype", "bfloat16")
            grpo_args.vllm_gpu_memory_utilization = float(
                vllm_cfg.get("gpu_memory_utilization", 0.85)
            )
        except AttributeError:
            pass  # older TRL; vLLM configured via env

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[reward_fn],
        args=grpo_args,
        train_dataset=train_dataset,
    )
    trainer.train()
    trainer.save_model(out_dir)
    tokenizer.save_pretrained(out_dir)
    print(f"[ok] GRPO/RLVR complete: {out_dir}")


if __name__ == "__main__":
    main()
