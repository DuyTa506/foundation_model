#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from transformers import AutoTokenizer

from data.streaming_lm import StreamingCausalLMDataset
from models.configuration_ouro import OuroConfig
from models.modeling_ouro import OuroForCausalLM
from models.looplm_train import LoopLMLossConfig, OuroLoopLMTrain


@dataclass
class TrainConfig:
    output_dir: str
    model_name_or_path: str
    tokenizer_name_or_path: str | None
    init_from_scratch: bool
    vocab_size: int | None
    dataset_name: str
    dataset_config: str | None
    text_field: str
    split: str
    seq_len: int
    steps: int
    batch_size: int
    grad_accum: int
    lr: float
    weight_decay: float
    warmup_steps: int
    total_ut_steps: int
    kl_beta: float
    include_adaptive_exit_loss: bool
    adaptive_exit_weight: float
    save_every: int
    log_every: int
    seed: int


def _load_yaml_config(path: str) -> TrainConfig:
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return TrainConfig(**raw)


def _maybe_setup_dist() -> tuple[int, int]:
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        torch.distributed.init_process_group(backend="nccl")
        return int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    return 0, 1


def _apply_fsdp2_if_available(module: torch.nn.Module):
    # FSDP2 API mirrors HRM-Text direction when available.
    try:
        from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
    except Exception:
        return module

    policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
    return fully_shard(module, mp_policy=policy, reshard_after_forward=False)


def _build_model_from_config(cfg: TrainConfig, tokenizer_vocab_size: int) -> OuroLoopLMTrain:
    target_vocab_size = cfg.vocab_size if cfg.vocab_size is not None else tokenizer_vocab_size
    if target_vocab_size != tokenizer_vocab_size:
        raise ValueError(
            f"vocab_size mismatch: config={target_vocab_size} tokenizer={tokenizer_vocab_size}. "
            "For scratch training, keep them equal (SmolLM style: tokenizer_name_or_path + explicit vocab_size)."
        )

    model = OuroForCausalLM(
        OuroConfig(
            total_ut_steps=cfg.total_ut_steps,
            vocab_size=target_vocab_size,
        )
    )

    train_model = OuroLoopLMTrain(
        model=model,
        config=LoopLMLossConfig(
            kl_beta=cfg.kl_beta,
            include_adaptive_exit_loss=cfg.include_adaptive_exit_loss,
            adaptive_exit_weight=cfg.adaptive_exit_weight,
        ),
    )
    return train_model


def _maybe_load_non_scratch_weights(train_model: OuroLoopLMTrain, cfg: TrainConfig):
    # This project currently supports two sources:
    # 1) scratch init (default)
    # 2) local checkpoint file path in our own torch.save format
    if cfg.init_from_scratch:
        return

    path = Path(cfg.model_name_or_path)
    if not path.is_file():
        raise ValueError(
            "init_from_scratch=false expects model_name_or_path to be a local checkpoint file. "
            "HF remote loading into this custom class is not wired in this script."
        )
    ckpt = torch.load(path, map_location="cpu")
    state = ckpt["model"] if "model" in ckpt else ckpt
    train_model.load_state_dict(state, strict=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="YAML train config")
    parser.add_argument("--resume", default=None, help="Checkpoint path")
    args = parser.parse_args()

    cfg = _load_yaml_config(args.config)
    rank, world_size = _maybe_setup_dist()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.manual_seed(cfg.seed + rank)
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    out_dir = Path(cfg.output_dir)
    if rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)
        with (out_dir / "train_config.json").open("w", encoding="utf-8") as f:
            json.dump(asdict(cfg), f, indent=2)

    tokenizer_path = cfg.tokenizer_name_or_path or cfg.model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    tokenizer.model_max_length = cfg.seq_len
    tokenizer_vocab_size = len(tokenizer)
    if rank == 0:
        print(
            f"[config] tokenizer={tokenizer_path} vocab_size={tokenizer_vocab_size} "
            f"init_from_scratch={cfg.init_from_scratch}"
        )

    train_model = _build_model_from_config(cfg, tokenizer_vocab_size=tokenizer_vocab_size)
    _maybe_load_non_scratch_weights(train_model, cfg)
    train_model = train_model.to(device)
    train_model = _apply_fsdp2_if_available(train_model)

    optimizer = torch.optim.AdamW(
        train_model.parameters(),
        lr=cfg.lr,
        betas=(0.9, 0.95),
        weight_decay=cfg.weight_decay,
    )

    def lr_lambda(step: int) -> float:
        if step < cfg.warmup_steps:
            return float(step + 1) / float(max(1, cfg.warmup_steps))
        progress = (step - cfg.warmup_steps) / float(max(1, cfg.steps - cfg.warmup_steps))
        return 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.1415926535))).item()

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu")
        train_model.load_state_dict(ckpt["model"], strict=False)
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])

    dataset = StreamingCausalLMDataset(
        tokenizer=tokenizer,
        dataset_name=cfg.dataset_name,
        dataset_config=cfg.dataset_config,
        split=cfg.split,
        text_field=cfg.text_field,
        seq_len=cfg.seq_len,
        rank=rank,
        world_size=world_size,
    )
    loader = DataLoader(dataset, batch_size=cfg.batch_size, num_workers=2, pin_memory=True)
    data_iter = iter(loader)

    train_model.train()
    optimizer.zero_grad(set_to_none=True)
    global_step = 0
    for step in range(cfg.steps):
        micro_loss = 0.0
        for _ in range(cfg.grad_accum):
            batch = next(data_iter)
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            out = train_model(input_ids=input_ids, labels=labels, total_ut_steps=cfg.total_ut_steps)
            loss = out["loss"] / cfg.grad_accum
            loss.backward()
            micro_loss += loss.item()

        torch.nn.utils.clip_grad_norm_(train_model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        global_step += 1

        if rank == 0 and global_step % cfg.log_every == 0:
            print(
                f"step={global_step} loss={micro_loss:.4f} "
                f"lr={scheduler.get_last_lr()[0]:.6e} "
                f"mean_exit={out['mean_exit_step'].item():.3f}"
            )

        if rank == 0 and global_step % cfg.save_every == 0:
            torch.save(
                {
                    "model": train_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "step": global_step,
                },
                out_dir / f"checkpoint_step_{global_step}.pt",
            )

    if rank == 0:
        torch.save({"model": train_model.state_dict(), "step": global_step}, out_dir / "checkpoint_final.pt")

    if torch.distributed.is_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()

