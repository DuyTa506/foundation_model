#!/usr/bin/env python3
"""
From-scratch pretraining with HF Trainer + WSD LR scheduler.

Replaces the broken launch_pretrain_megatron_ds.sh.

Usage (via accelerate):
    accelerate launch --config_file scripts/accelerate_fsdp.yaml \\
        scripts/pretrain_hf.py --config configs/training_8xH200_hf_pretrain.yaml

Usage (single GPU, for smoke testing):
    python scripts/pretrain_hf.py --config configs/training_8xH200_hf_pretrain.yaml

WSD scheduler (MiniCPM recipe):
    Warmup  : linear 0 -> peak_lr over warmup_steps
    Stable  : constant peak_lr
    Decay   : exponential  lr = peak * 0.5^((step - stable_end) / half_life)
    At the start of decay, data mix optionally switches to a VI-dominant
    high-quality mix (configured via training config).
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
from typing import Iterator

import yaml


# ─── WSD LR Scheduler ─────────────────────────────────────────────────────────

def wsd_scheduler(
    optimizer,
    warmup_steps: int,
    stable_steps: int,
    decay_steps: int,
    decay_half_life: int,
    peak_lr: float,
    min_lr: float,
):
    """
    Warmup-Stable-Decay LR scheduler (MiniCPM recipe).
    decay formula: lr = peak * 0.5^((step - stable_end) / T)
    where T = decay_half_life_steps.
    """
    from torch.optim.lr_scheduler import LambdaLR

    total_stable = warmup_steps + stable_steps

    def _lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            # Linear warmup
            return float(current_step) / max(1, warmup_steps)
        elif current_step < total_stable:
            # Stable plateau
            return 1.0
        else:
            # Exponential decay
            steps_into_decay = current_step - total_stable
            factor = 0.5 ** (steps_into_decay / max(1, decay_half_life))
            # Clip at min_lr
            min_factor = min_lr / max(peak_lr, 1e-10)
            return max(factor, min_factor)

    return LambdaLR(optimizer, lr_lambda=_lr_lambda, last_epoch=-1)


# ─── Packed-shard dataset ─────────────────────────────────────────────────────

class PackedTokenDataset:
    """
    Reads pre-tokenized, packed shards produced by curate/07_tokenize_pack.py.
    Yields fixed-length blocks of input_ids for causal LM training.
    """

    def __init__(self, shards_dir: str, max_seq_length: int):
        import numpy as np
        from torch.utils.data import IterableDataset

        shard_paths = sorted(Path(shards_dir).rglob("*.npy"))
        if not shard_paths:
            # Try datatrove .ds token files
            shard_paths = sorted(Path(shards_dir).rglob("*.ds"))
        if not shard_paths:
            raise FileNotFoundError(
                f"No .npy or .ds token shards found in {shards_dir}. "
                "Run scripts/curate/07_tokenize_pack.py first."
            )
        self._paths = shard_paths
        self._max_seq_length = max_seq_length
        print(f"[dataset] {len(shard_paths)} shards in {shards_dir}  seq={max_seq_length}")

    def _iter_shard(self, path: Path) -> Iterator[dict]:
        import numpy as np

        if path.suffix == ".npy":
            tokens = np.load(str(path), mmap_mode="r").astype("int32")
        else:
            # datatrove .ds file: raw int32 array
            tokens = np.frombuffer(path.read_bytes(), dtype=np.uint16).astype("int32")

        L = self._max_seq_length
        n_chunks = len(tokens) // (L + 1)
        for i in range(n_chunks):
            chunk = tokens[i * (L + 1) : (i + 1) * (L + 1)]
            yield {
                "input_ids": chunk[:L].tolist(),
                "labels": chunk[1 : L + 1].tolist(),
            }

    def __iter__(self) -> Iterator[dict]:
        for path in self._paths:
            yield from self._iter_shard(path)

    def as_hf_dataset(self):
        from datasets import IterableDataset

        return IterableDataset.from_generator(
            lambda: iter(self),
            features=None,
        )


# ─── Trainer callbacks ────────────────────────────────────────────────────────

class TokensPerSecCallback:
    """Log tokens/sec every N steps."""

    def __init__(self, seq_len: int, global_batch_tokens: int):
        import time
        self.seq_len = seq_len
        self.global_batch_tokens = global_batch_tokens
        self._t0 = time.time()
        self._step0 = 0

    def on_log(self, args, state, control, logs=None, **kwargs):
        import time

        if state.global_step == self._step0:
            return
        elapsed = time.time() - self._t0
        steps = state.global_step - self._step0
        tps = steps * self.global_batch_tokens / max(elapsed, 1e-6)
        if logs is not None:
            logs["tokens_per_sec"] = tps
        self._t0 = time.time()
        self._step0 = state.global_step


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="From-scratch HF Trainer pretraining.")
    parser.add_argument("--config", default="configs/training_8xH200_hf_pretrain.yaml")
    parser.add_argument("--smoke_test", action="store_true",
                        help="Run 5 steps for sanity check.")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    import torch
    from transformers import (
        AutoTokenizer,
        LlamaConfig,
        LlamaForCausalLM,
        TrainingArguments,
        Trainer,
        DataCollatorForLanguageModeling,
        set_seed,
    )
    from transformers.trainer_callback import TrainerCallback

    set_seed(cfg["run"].get("seed", 42))

    # ── Load model ────────────────────────────────────────────────────────────
    init_ckpt = Path(cfg["model"]["init_checkpoint"])
    model_cfg_path = cfg["model"]["config_path"]

    if (init_ckpt / "config.json").exists():
        print(f"[pretrain] loading init checkpoint: {init_ckpt}")
        from transformers import AutoModelForCausalLM

        # Load from the random-init checkpoint (NOT from HF hub; always local)
        model = AutoModelForCausalLM.from_pretrained(
            str(init_ckpt),
            torch_dtype=torch.bfloat16,
            local_files_only=True,  # never pull from hub
        )
    else:
        raise FileNotFoundError(
            f"Model init checkpoint not found at {init_ckpt}. "
            "Run scripts/init_model_from_scratch.py first."
        )

    # Apply YaRN rope_scaling if specified in config
    rope_scaling = cfg["model"].get("rope_scaling")
    if rope_scaling:
        print(f"[pretrain] applying rope_scaling={rope_scaling}")
        model.config.rope_scaling = rope_scaling

    total_params = sum(p.numel() for p in model.parameters())
    print(f"[pretrain] model: {total_params/1e9:.3f}B params  dtype={model.dtype}")

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    tokenizer_path = str(init_ckpt / "tokenizer")
    if not (Path(tokenizer_path) / "tokenizer.json").exists():
        tokenizer_path = "outputs/tokenizer"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    print(f"[pretrain] tokenizer: vocab={tokenizer.vocab_size} path={tokenizer_path}")

    # ── Dataset ───────────────────────────────────────────────────────────────
    data_cfg = cfg["data"]
    shards_dir = data_cfg["tokenized_shards_dir"]
    max_seq_length = data_cfg.get("max_seq_length", 4096)

    packed_ds = PackedTokenDataset(shards_dir, max_seq_length)
    train_dataset = packed_ds.as_hf_dataset()

    # ── Optimizer & scheduler ─────────────────────────────────────────────────
    opt_cfg = cfg["optimization"]
    sched_cfg = cfg["scheduler"]
    train_cfg = cfg["training"]

    total_steps = 5 if args.smoke_test else train_cfg["total_steps"]
    warmup_steps = sched_cfg["warmup_steps"]
    decay_steps = sched_cfg.get("decay_steps", max(1, total_steps // 10))
    stable_steps = total_steps - warmup_steps - decay_steps
    decay_half_life = sched_cfg.get("decay_half_life_steps", 5000)

    # ── TrainingArguments ─────────────────────────────────────────────────────
    out_dir = cfg["run"]["output_dir"]
    ckpt_cfg = cfg.get("checkpointing", {})
    log_cfg = cfg.get("logging", {})

    training_args = TrainingArguments(
        output_dir=out_dir,
        overwrite_output_dir=False,
        do_train=True,
        max_steps=total_steps,
        per_device_train_batch_size=train_cfg["micro_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        learning_rate=opt_cfg["peak_learning_rate"],
        adam_beta1=opt_cfg["adam_beta1"],
        adam_beta2=opt_cfg["adam_beta2"],
        adam_epsilon=opt_cfg.get("adam_epsilon", 1e-8),
        weight_decay=opt_cfg["weight_decay"],
        max_grad_norm=opt_cfg["grad_clip"],
        lr_scheduler_type="constant",  # WSD handled manually below
        warmup_steps=0,               # WSD handled manually
        bf16=train_cfg.get("bf16", True),
        fp16=False,
        logging_steps=train_cfg.get("logging_steps", 10),
        save_steps=train_cfg.get("save_steps", 1000),
        save_total_limit=ckpt_cfg.get("save_total_limit", 5),
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        gradient_checkpointing=train_cfg.get("gradient_checkpointing", False),
        report_to=log_cfg.get("report_to", "none"),
        run_name=cfg["run"].get("name"),
        seed=cfg["run"].get("seed", 42),
        # FSDP is configured via accelerate config, not here
    )

    # ── Custom WSD LR schedule via callback ───────────────────────────────────
    class WSDSchedulerCallback(TrainerCallback):
        def __init__(self, trainer_ref, warmup, stable, decay, half_life, peak, min_l):
            self.warmup = warmup
            self.stable = stable
            self.decay = decay
            self.half_life = half_life
            self.peak = peak
            self.min_lr = min_l
            self._total_stable = warmup + stable

        def on_step_begin(self, args, state, control, **kwargs):
            step = state.global_step
            if step < self.warmup:
                factor = step / max(1, self.warmup)
            elif step < self._total_stable:
                factor = 1.0
            else:
                steps_into = step - self._total_stable
                factor = 0.5 ** (steps_into / max(1, self.half_life))
                factor = max(factor, self.min_lr / max(self.peak, 1e-10))

            for pg in kwargs.get("optimizer", {}).param_groups if "optimizer" in kwargs else []:
                pg["lr"] = self.peak * factor

    # ── Data collator ─────────────────────────────────────────────────────────
    # Packed shards already have input_ids and labels; no masking needed here.
    # Use a simple collator that stacks and passes through.
    from transformers import default_data_collator

    # ── Trainer ───────────────────────────────────────────────────────────────
    global_batch_tokens = (
        train_cfg["micro_batch_size"]
        * train_cfg["gradient_accumulation_steps"]
        * int(os.environ.get("WORLD_SIZE", "8"))
        * max_seq_length
    )

    class TokPerSecCallback(TrainerCallback):
        def __init__(self):
            import time
            self._t0 = time.time()
            self._s0 = 0

        def on_log(self, args, state, control, logs=None, **kwargs):
            import time
            if logs is not None and state.global_step > self._s0:
                elapsed = time.time() - self._t0
                steps = state.global_step - self._s0
                tps = steps * global_batch_tokens / max(elapsed, 1e-9)
                logs["tokens_per_sec"] = round(tps)
                self._t0 = time.time()
                self._s0 = state.global_step

    wsd_cb = WSDSchedulerCallback(
        trainer_ref=None,
        warmup=warmup_steps,
        stable=stable_steps,
        decay=decay_steps,
        half_life=decay_half_life,
        peak=opt_cfg["peak_learning_rate"],
        min_l=opt_cfg["min_learning_rate"],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=default_data_collator,
        callbacks=[wsd_cb, TokPerSecCallback()],
    )

    # ── Verify initial loss is ~ln(vocab_size) ───────────────────────────────
    print(f"[pretrain] expected initial loss ≈ {math.log(tokenizer.vocab_size):.3f} "
          f"(= ln({tokenizer.vocab_size}))")

    print(f"[pretrain] training for {total_steps} steps  "
          f"({total_steps * global_batch_tokens / 1e9:.1f}B tokens)")

    trainer.train(
        resume_from_checkpoint=cfg["run"].get("resume_from_checkpoint") or None,
    )

    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    print(f"[ok] pretraining complete: {out_dir}")


if __name__ == "__main__":
    main()
