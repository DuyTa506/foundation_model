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
    Yields fixed-length blocks for causal LM training.

    Shards are written roughly one source at a time and are ~1 GB each, so naive
    sequential reading trains on one distribution for thousands of steps (a sawtooth
    / regime-block loss curve). Three ordering safeguards (all default-on) fix this:
      - shard_interleave: read ALL shards concurrently, drawing each sequence from a
        randomly chosen still-live shard. This is the important one — it dissolves
        within-shard single-source runs that shard-order shuffle alone cannot, since
        one shard dwarfs the buffer. Uniform draw over live shards also makes each
        source's share ~proportional to its token count (your corpus mix falls out).
      - a streaming shuffle buffer that locally permutes sequences (reservoir-style:
        emit a random buffer slot, refill from stream).
      - per-epoch reshuffle of the draw order (seeded by base seed + epoch).

    Labels == input_ids on purpose. LlamaForCausalLM shifts internally
    (loss compares logits[:-1] to labels[1:]); pre-shifting labels here would
    double-shift and train next-next-token prediction.
    """

    def __init__(
        self,
        shards_dir: str,
        max_seq_length: int,
        shuffle_buffer_size: int = 8192,
        seed: int = 42,
        shuffle_shards: bool = True,
        shard_interleave: bool = True,
        max_sequences: int | None = None,
    ):
        shard_paths = sorted(Path(shards_dir).rglob("*.npy"))
        if not shard_paths:
            # datatrove .ds token files (skip the _scratch working dir)
            shard_paths = sorted(
                p for p in Path(shards_dir).rglob("*.ds") if "_scratch" not in p.parts
            )
        if not shard_paths:
            raise FileNotFoundError(
                f"No .npy or .ds token shards found in {shards_dir}. "
                "Run scripts/curate/07_tokenize_pack.py first."
            )
        self._paths = shard_paths
        self._max_seq_length = max_seq_length
        self._buffer_size = max(0, int(shuffle_buffer_size))
        self._seed = int(seed)
        self._shuffle_shards = bool(shuffle_shards)
        self._shard_interleave = bool(shard_interleave)
        self._max_sequences = int(max_sequences) if max_sequences else None
        self._epoch = 0
        print(
            f"[dataset] {len(shard_paths)} shards in {shards_dir}  seq={max_seq_length}  "
            f"interleave={self._shard_interleave}  shuffle_shards={self._shuffle_shards}  "
            f"shuffle_buffer={self._buffer_size}"
        )

    def _iter_shard(self, path: Path):
        import numpy as np

        if path.suffix == ".npy":
            tokens = np.load(str(path), mmap_mode="r")
        else:
            # datatrove .ds file: raw little-endian uint16 token ids (memmap, not
            # read_bytes — a 1B-token shard is ~2 GB and read_bytes loads it all).
            tokens = np.memmap(path, dtype=np.uint16, mode="r")

        L = self._max_seq_length
        stride = L + 1
        n_chunks = len(tokens) // stride
        for i in range(n_chunks):
            start = i * stride
            # copy out of the mmap into a small int64 array (releases the page view)
            yield np.asarray(tokens[start : start + L], dtype=np.int64)

    def _raw_iter(self):
        import random

        rng = random.Random(self._seed + self._epoch)

        if not self._shard_interleave:
            # Sequential: read one shard fully before the next (shuffled order only).
            order = list(self._paths)
            if self._shuffle_shards:
                rng.shuffle(order)
            for path in order:
                yield from self._iter_shard(path)
            return

        # Interleaved multiplex: keep every shard's chunk-generator live at once and
        # draw each sequence from a random live shard. memmaps are lazy (only touched
        # pages load), so holding all shards open costs ~nothing. A shard drops out of
        # `live` when exhausted; uniform choice over live shards ⇒ each shard's share
        # is proportional to its remaining chunks ⇒ corpus mix preserved.
        gens = [self._iter_shard(p) for p in self._paths]
        live = list(range(len(gens)))
        while live:
            k = rng.randrange(len(live))
            try:
                yield next(gens[live[k]])
            except StopIteration:
                live.pop(k)

    def _emit(self) -> Iterator[dict]:
        import random

        raw = self._raw_iter()
        if self._buffer_size <= 1:
            for chunk in raw:
                ids = chunk.tolist()
                yield {"input_ids": ids, "labels": ids}
            return

        rng = random.Random(self._seed * 7919 + self._epoch)
        buffer: list = []
        for chunk in raw:
            buffer.append(chunk)
            if len(buffer) >= self._buffer_size:
                j = rng.randrange(len(buffer))
                out = buffer[j]
                buffer[j] = buffer[-1]
                buffer.pop()
                ids = out.tolist()
                yield {"input_ids": ids, "labels": ids}
        rng.shuffle(buffer)
        for out in buffer:
            ids = out.tolist()
            yield {"input_ids": ids, "labels": ids}

    def __iter__(self) -> Iterator[dict]:
        self._epoch += 1
        if self._max_sequences:  # bound eval (finite, deterministic) — never on train
            from itertools import islice
            yield from islice(self._emit(), self._max_sequences)
        else:
            yield from self._emit()

    def as_hf_dataset(self):
        from datasets import IterableDataset

        return IterableDataset.from_generator(
            lambda: iter(self),
            features=None,
        )


# ─── WSD decay-phase data anneal ──────────────────────────────────────────────

class PhaseState:
    """Mutable holder a callback writes the live global_step into, so the streaming
    dataset (which otherwise has no view of optimizer steps) can decide when to
    switch from the broad mix to the decay-phase mix."""

    def __init__(self) -> None:
        self.global_step = 0


class PhaseSwitchDataset:
    """Streams `main_ds` until the LR-decay phase starts, then switches to
    `decay_ds` (a small, high-quality VI+math mix) for the rest of training.

    The switch is keyed off the callback-updated global step, so it lands at the
    same step on every rank (a few sequences of dataloader prefetch slop near the
    boundary is irrelevant at a ~45k-step switch point). `decay_ds` repeats if
    exhausted — the decay phase intentionally over-weights HQ data, so re-reading
    it a couple of times across the decay span is fine.
    """

    def __init__(self, main_ds, decay_ds, phase_state: PhaseState, decay_start_step: int):
        self._main = main_ds
        self._decay = decay_ds
        self._phase = phase_state
        self._decay_start = int(decay_start_step)

    def _gen(self) -> Iterator[dict]:
        if self._phase.global_step < self._decay_start:
            for item in iter(self._main):
                if self._phase.global_step >= self._decay_start:
                    break
                yield item
        while True:  # decay phase (also entered directly when resuming into it)
            for item in iter(self._decay):
                yield item

    def as_hf_dataset(self):
        from datasets import IterableDataset

        return IterableDataset.from_generator(lambda: self._gen(), features=None)


# ─── Trainer callbacks ────────────────────────────────────────────────────────


class DecayPhaseCallback(TrainerCallback):
    """Publishes the live global_step into PhaseState so PhaseSwitchDataset can
    flip to the decay mix at the right step (incl. after a resume)."""

    def __init__(self, phase_state: PhaseState):
        self._phase = phase_state

    def on_train_begin(self, args, state, control, **kwargs):
        self._phase.global_step = state.global_step

    def on_step_begin(self, args, state, control, **kwargs):
        self._phase.global_step = state.global_step

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

    packed_ds = PackedTokenDataset(
        shards_dir,
        max_seq_length,
        shuffle_buffer_size=data_cfg.get("shuffle_buffer_size", 8192),
        seed=cfg["run"].get("seed", 42),
        shuffle_shards=data_cfg.get("shuffle_shards", True),
        shard_interleave=data_cfg.get("shard_interleave", True),
    )
    # train_dataset is assembled after step counts are known (decay-anneal needs
    # the decay_start step), see "WSD decay-phase data anneal" below.

    # ── Optimizer & scheduler ─────────────────────────────────────────────────
    opt_cfg = cfg["optimization"]
    sched_cfg = cfg["scheduler"]
    train_cfg = cfg["training"]

    total_steps = 5 if args.smoke_test else train_cfg["total_steps"]
    warmup_steps = sched_cfg["warmup_steps"]
    decay_steps = sched_cfg.get("decay_steps", max(1, total_steps // 10))
    stable_steps = total_steps - warmup_steps - decay_steps
    decay_half_life = sched_cfg.get("decay_half_life_steps", 5000)

    # ── Assemble train_dataset (+ optional WSD decay-phase data anneal) ────────
    # During the LR-decay phase, switch to a high-quality, VI-dominant + math mix
    # (built by scripts/data/build_decay_shards.py). LR is collapsing toward min
    # there, so whatever the model sees last is what it locks in — the MiniCPM
    # annealing trick, and the highest-value ordering lever for a single pass.
    decay_shards_dir = data_cfg.get("decay_shards_dir")
    use_decay_anneal = bool(decay_shards_dir) and sched_cfg.get("decay_phase_data_mix", False)
    phase_state = PhaseState()
    if use_decay_anneal:
        decay_start_step = warmup_steps + stable_steps
        decay_ds = PackedTokenDataset(
            decay_shards_dir,
            max_seq_length,
            shuffle_buffer_size=data_cfg.get("shuffle_buffer_size", 8192),
            seed=cfg["run"].get("seed", 42) + 1,
            shuffle_shards=data_cfg.get("shuffle_shards", True),
            shard_interleave=data_cfg.get("shard_interleave", True),
        )
        train_dataset = PhaseSwitchDataset(
            packed_ds, decay_ds, phase_state, decay_start_step
        ).as_hf_dataset()
        print(f"[pretrain] decay-phase anneal ON: switch to {decay_shards_dir} "
              f"at step {decay_start_step} (last {decay_steps} steps)")
    else:
        train_dataset = packed_ds.as_hf_dataset()
        if decay_shards_dir and not sched_cfg.get("decay_phase_data_mix", False):
            print("[pretrain] decay_shards_dir set but scheduler.decay_phase_data_mix "
                  "is false → anneal disabled")

    # ── Optional held-out eval set (val loss) ─────────────────────────────────
    # Point data.val_shards_dir at a few HELD-OUT shards (not in tokenized_shards_dir).
    # Deterministic order + a hard sequence cap so eval is fast and comparable across
    # checkpoints. Null → no eval (Trainer reports train loss only).
    val_shards_dir = data_cfg.get("val_shards_dir")
    eval_dataset = None
    if val_shards_dir:
        eval_dataset = PackedTokenDataset(
            val_shards_dir,
            max_seq_length,
            shuffle_buffer_size=0,      # deterministic: same sequences every eval
            shuffle_shards=False,
            shard_interleave=False,
            max_sequences=data_cfg.get("eval_max_sequences", 2000),
        ).as_hf_dataset()
        print(f"[pretrain] eval set: {val_shards_dir}  "
              f"(<= {data_cfg.get('eval_max_sequences', 2000)} seqs/eval)")

    # ── Auto batch size: maintain target global_batch_tokens ─────────────────
    import torch, os as _os
    num_gpus = int(_os.environ.get("WORLD_SIZE", 1))
    micro_batch = train_cfg["micro_batch_size"]
    grad_accum = train_cfg["gradient_accumulation_steps"]
    target_global_tokens = micro_batch * grad_accum * num_gpus * max_seq_length

    if train_cfg.get("auto_batch_size", False):
        # Binary-search for the largest micro_batch that fits on this GPU,
        # then recompute grad_accum to keep global_batch_tokens constant.
        # Model is still on CPU here (FSDP hasn't wrapped it yet); move it to
        # the local rank's GPU for the probe, then back to CPU afterward.
        local_rank = int(_os.environ.get("LOCAL_RANK", "0"))
        probe_device = torch.device(f"cuda:{local_rank}")
        model.to(probe_device)
        vocab_size = model.config.vocab_size
        lo, hi, best = 1, micro_batch * 4, micro_batch
        while lo <= hi:
            mid = (lo + hi) // 2
            try:
                dummy = torch.randint(0, vocab_size, (mid, max_seq_length), device=probe_device)
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
        model.to("cpu")
        torch.cuda.empty_cache()
        # Apply 50% safety margin: FSDP adds fp32 master weights + optimizer states
        # that the single-GPU probe doesn't account for.
        micro_batch = max(1, best // 2)
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
        per_device_eval_batch_size=train_cfg.get("eval_micro_batch_size", micro_batch),
        eval_strategy=("steps" if eval_dataset is not None else "no"),
        eval_steps=train_cfg.get("eval_steps", 5000),
        gradient_accumulation_steps=grad_accum,
        learning_rate=opt_cfg["peak_learning_rate"],
        adam_beta1=opt_cfg["adam_beta1"],
        adam_beta2=opt_cfg["adam_beta2"],
        adam_epsilon=opt_cfg.get("adam_epsilon", 1e-8),
        weight_decay=opt_cfg["weight_decay"],
        max_grad_norm=opt_cfg["grad_clip"],
        lr_scheduler_type="constant",  # ignored: WSDTrainer.create_scheduler overrides it
        warmup_steps=0,               # ignored: warmup is inside the WSD LambdaLR
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

    # ── Custom WSD LR schedule ────────────────────────────────────────────────
    # Implemented as a real LambdaLR owned by the Trainer (via create_scheduler),
    # NOT a callback. The previous callback set optimizer LR in on_step_begin, but
    # TrainingArguments installs a constant scheduler whose .step() runs right after
    # the optimizer and overwrote the LR back to peak every step — so the logged
    # `learning_rate` (read from lr_scheduler.get_last_lr()) was always the peak.
    # Letting HF own the WSD scheduler makes warmup/decay both effective and logged.
    class WSDTrainer(Trainer):
        def create_scheduler(self, num_training_steps: int, optimizer=None):
            if self.lr_scheduler is None:
                self.lr_scheduler = wsd_scheduler(
                    optimizer=optimizer if optimizer is not None else self.optimizer,
                    warmup_steps=warmup_steps,
                    stable_steps=stable_steps,
                    decay_steps=decay_steps,
                    decay_half_life=decay_half_life,
                    peak_lr=opt_cfg["peak_learning_rate"],
                    min_lr=opt_cfg["min_learning_rate"],
                )
            return self.lr_scheduler

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

    callbacks = [TokPerSecCallback(), health_cb, metrics_cb]
    if use_decay_anneal:
        callbacks.append(DecayPhaseCallback(phase_state))

    trainer = WSDTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=default_data_collator,
        callbacks=callbacks,
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
