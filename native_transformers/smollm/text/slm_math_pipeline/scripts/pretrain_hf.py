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
from transformers.trainer_callback import TrainerCallback


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

class MetricsFileCallback(TrainerCallback):
    """
    Writes step metrics to {output_dir}/metrics.log (one line per logging step).
    Plain text, tail-f friendly. Not affected by tqdm carriage-return overwriting.
    Format: step=NNN  loss=X.XXXX  grad_norm=X.XXXX  lr=X.Xe-XX  tok/s=NNNN
    """

    def __init__(self, output_dir: str):
        self._path = Path(output_dir) / "metrics.log"
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "w") as f:
            f.write("# step  loss  grad_norm  lr  tokens_per_sec\n")
        print(f"[metrics] writing to {self._path}  (tail -f to follow)")

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        step = state.global_step
        loss = logs.get("loss", "")
        grad = logs.get("grad_norm", "")
        lr   = logs.get("learning_rate", "")
        tps  = logs.get("tokens_per_sec", "")

        parts = [f"step={step:6d}"]
        if loss != "":
            parts.append(f"loss={float(loss):.4f}")
        if grad != "":
            parts.append(f"grad={float(grad):.4f}")
        if lr != "":
            parts.append(f"lr={float(lr):.2e}")
        if tps != "":
            parts.append(f"tok/s={int(tps)}")

        with open(self._path, "a") as f:
            f.write("  ".join(parts) + "\n")

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


class TrainingHealthCallback(TrainerCallback):
    """
    Monitors training health every step and prints warnings for:
      - NaN / Inf loss
      - Loss spike  (loss > ema * spike_factor)
      - Loss plateau (no improvement over plateau_window steps)
      - Vanishing gradient (grad_norm < vanish_thresh)
      - Exploding gradient (grad_norm > explode_thresh, or NaN)
    Stops training on NaN loss or gradient NaN.
    """

    def __init__(
        self,
        spike_factor: float = 1.5,       # loss > ema × factor → spike warning
        ema_alpha: float = 0.1,           # smoothing for loss EMA
        plateau_window: int = 50,         # steps with < min_delta improvement
        plateau_min_delta: float = 0.01,
        vanish_thresh: float = 1e-4,      # grad_norm below this → vanishing
        explode_thresh: float = 100.0,    # grad_norm above this → exploding
    ):
        self.spike_factor = spike_factor
        self.alpha = ema_alpha
        self.plateau_window = plateau_window
        self.plateau_min_delta = plateau_min_delta
        self.vanish_thresh = vanish_thresh
        self.explode_thresh = explode_thresh

        self._ema_loss: float | None = None
        self._recent_losses: list[float] = []
        self._warned_plateau = False

    def _fmt(self, step: int, tag: str, msg: str) -> str:
        return f"\n{'!'*5} [health:{tag}] step={step}  {msg} {'!'*5}\n"

    def on_log(self, args, state, control, logs=None, **kwargs):
        import math

        if logs is None:
            return
        loss = logs.get("loss")
        grad_norm = logs.get("grad_norm")
        step = state.global_step

        # ── Loss checks ──────────────────────────────────────────────────────
        if loss is not None:
            if math.isnan(loss) or math.isinf(loss):
                print(self._fmt(step, "NaN", f"loss={loss}  STOPPING TRAINING"))
                control.should_training_stop = True
                return

            # EMA update
            if self._ema_loss is None:
                self._ema_loss = loss
            else:
                self._ema_loss = self.alpha * loss + (1 - self.alpha) * self._ema_loss

            # Spike detection (skip first few steps while EMA warms up)
            if step > 5 and loss > self._ema_loss * self.spike_factor:
                print(self._fmt(step, "SPIKE",
                    f"loss={loss:.4f}  ema={self._ema_loss:.4f}  "
                    f"ratio={loss/self._ema_loss:.2f}×"))

            # Plateau detection
            self._recent_losses.append(loss)
            if len(self._recent_losses) > self.plateau_window:
                self._recent_losses.pop(0)
            if len(self._recent_losses) == self.plateau_window:
                best_early = min(self._recent_losses[: self.plateau_window // 2])
                best_late  = min(self._recent_losses[self.plateau_window // 2 :])
                if best_early - best_late < self.plateau_min_delta and not self._warned_plateau:
                    print(self._fmt(step, "PLATEAU",
                        f"loss unchanged over last {self.plateau_window} steps  "
                        f"(best_early={best_early:.4f}  best_late={best_late:.4f})"))
                    self._warned_plateau = True
                elif best_early - best_late >= self.plateau_min_delta:
                    self._warned_plateau = False  # reset if progress resumes

        # ── Gradient checks ───────────────────────────────────────────────────
        if grad_norm is not None:
            if math.isnan(grad_norm) or math.isinf(grad_norm):
                print(self._fmt(step, "GRAD-NaN", f"grad_norm={grad_norm}  STOPPING TRAINING"))
                control.should_training_stop = True
            elif grad_norm < self.vanish_thresh:
                print(self._fmt(step, "VANISHING",
                    f"grad_norm={grad_norm:.2e}  (threshold={self.vanish_thresh:.0e})"))
            elif grad_norm > self.explode_thresh:
                print(self._fmt(step, "EXPLODING",
                    f"grad_norm={grad_norm:.1f}  (threshold={self.explode_thresh})  "
                    f"check grad_clip in config"))


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
        # dtype rules:
        #   fp16 training  → load float32; grad scaler handles AMP (loading fp16 breaks scaler)
        #   bf16 training  → load bfloat16 directly; no grad scaler, FA2 requires non-float32
        #   everything else → float32 (safe default)
        attn_impl = cfg["model"].get("attn_implementation", "eager")
        train_cfg_early = cfg.get("training", {})
        if train_cfg_early.get("bf16", False):
            load_dtype = torch.bfloat16
        elif train_cfg_early.get("fp16", False):
            load_dtype = torch.float32  # keep float32 so grad scaler works
        else:
            load_dtype = torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            str(init_ckpt),
            torch_dtype=load_dtype,
            local_files_only=True,  # never pull from hub
            attn_implementation=attn_impl,
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

    # ── Auto batch size: maintain target global_batch_tokens ─────────────────
    import torch, os as _os
    num_gpus = int(_os.environ.get("WORLD_SIZE", 1))
    micro_batch = train_cfg["micro_batch_size"]
    grad_accum = train_cfg["gradient_accumulation_steps"]
    target_global_tokens = micro_batch * grad_accum * num_gpus * max_seq_length

    if train_cfg.get("auto_batch_size", False):
        # Binary-search for the largest micro_batch that fits on this GPU,
        # then recompute grad_accum to keep global_batch_tokens constant.
        vocab_size = model.config.vocab_size
        device = next(model.parameters()).device if next(model.parameters(), None) is not None else torch.device("cuda")
        lo, hi, best = 1, micro_batch * 4, micro_batch
        while lo <= hi:
            mid = (lo + hi) // 2
            try:
                dummy = torch.randint(0, vocab_size, (mid, max_seq_length), device="cuda")
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    out = model(input_ids=dummy, labels=dummy)
                out.loss.backward()
                model.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                best = mid
                lo = mid + 1
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                hi = mid - 1
        micro_batch = best
        # Recompute grad_accum; round up to nearest int (slightly > target is fine)
        grad_accum = max(1, round(target_global_tokens / (micro_batch * num_gpus * max_seq_length)))
        actual_tokens = micro_batch * grad_accum * num_gpus * max_seq_length
        print(f"[auto_batch] micro_batch={micro_batch}  grad_accum={grad_accum}  "
              f"global_batch_tokens={actual_tokens:,}  (target={target_global_tokens:,})")

    # ── TrainingArguments ─────────────────────────────────────────────────────
    out_dir = cfg["run"]["output_dir"]
    ckpt_cfg = cfg.get("checkpointing", {})
    log_cfg = cfg.get("logging", {})

    # ── Logging backend setup ─────────────────────────────────────────────────
    report_to = log_cfg.get("report_to", "none")
    # smoke_test always disables remote logging
    if args.smoke_test:
        report_to = "none"

    if "wandb" in str(report_to):
        import os as _os
        _os.environ.setdefault("WANDB_PROJECT", log_cfg.get("wandb_project", "slm_math_vi"))
        _run_name = log_cfg.get("wandb_run_name") or cfg["run"].get("name")
        if _run_name:
            _os.environ.setdefault("WANDB_NAME", _run_name)

    if "tensorboard" in str(report_to):
        tb_dir = str(Path(out_dir) / "tensorboard")
        print(f"[pretrain] tensorboard logdir: {tb_dir}  (run: tensorboard --logdir {tb_dir})")

    training_args = TrainingArguments(
        output_dir=out_dir,
        overwrite_output_dir=False,
        do_train=True,
        max_steps=total_steps,
        per_device_train_batch_size=micro_batch,
        gradient_accumulation_steps=grad_accum,
        learning_rate=opt_cfg["peak_learning_rate"],
        adam_beta1=opt_cfg["adam_beta1"],
        adam_beta2=opt_cfg["adam_beta2"],
        adam_epsilon=opt_cfg.get("adam_epsilon", 1e-8),
        weight_decay=opt_cfg["weight_decay"],
        max_grad_norm=opt_cfg["grad_clip"],
        lr_scheduler_type="constant",  # WSD handled manually below
        warmup_steps=0,               # WSD handled manually
        bf16=train_cfg.get("bf16", False),
        fp16=train_cfg.get("fp16", False),
        logging_steps=train_cfg.get("logging_steps", 10),
        save_steps=train_cfg.get("save_steps", 1000),
        save_total_limit=ckpt_cfg.get("save_total_limit", 5),
        optim="adamw_torch_fused",
        dataloader_num_workers=data_cfg.get("dataloader_num_workers", 1),
        dataloader_pin_memory=True,
        gradient_checkpointing=train_cfg.get("gradient_checkpointing", False),
        torch_compile=train_cfg.get("torch_compile", False),
        report_to=report_to,
        run_name=log_cfg.get("wandb_run_name") or cfg["run"].get("name"),
        seed=cfg["run"].get("seed", 42),
        logging_dir=str(Path(out_dir) / "tensorboard"),  # used when report_to=tensorboard
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
        * int(os.environ.get("WORLD_SIZE", str(cfg.get("hardware", {}).get("gpus_per_node", 8))))
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

    health_cfg = cfg.get("health_monitor", {})
    health_cb = TrainingHealthCallback(
        spike_factor=health_cfg.get("spike_factor", 1.5),
        ema_alpha=health_cfg.get("ema_alpha", 0.1),
        plateau_window=health_cfg.get("plateau_window", 50),
        plateau_min_delta=health_cfg.get("plateau_min_delta", 0.01),
        vanish_thresh=health_cfg.get("vanish_thresh", 1e-4),
        explode_thresh=health_cfg.get("explode_thresh", 100.0),
    )

    metrics_cb = MetricsFileCallback(out_dir)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=default_data_collator,
        callbacks=[wsd_cb, TokPerSecCallback(), health_cb, metrics_cb],
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
