#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

import torch

from models.configuration_ouro import OuroConfig
from models.modeling_ouro import OuroForCausalLM
from models.looplm_train import LoopLMLossConfig, OuroLoopLMTrain


def make_synth_batch(batch_size: int, seq_len: int, vocab_size: int, device: torch.device):
    x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    y = x.roll(-1, dims=1)
    y[:, -1] = -100
    return x, y


def run(total_ut_steps: int, steps: int, device: torch.device):
    cfg = OuroConfig(
        vocab_size=512,
        hidden_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        intermediate_size=256,
        max_position_embeddings=256,
        total_ut_steps=total_ut_steps,
    )
    model = OuroForCausalLM(cfg).to(device)
    train_model = OuroLoopLMTrain(
        model,
        LoopLMLossConfig(
            kl_beta=0.1,
            include_adaptive_exit_loss=True,
            adaptive_exit_weight=0.05,
        ),
    ).to(device)
    opt = torch.optim.AdamW(train_model.parameters(), lr=3e-4, betas=(0.9, 0.95), weight_decay=0.1)

    losses = []
    exits = []
    for _ in range(steps):
        x, y = make_synth_batch(batch_size=8, seq_len=128, vocab_size=cfg.vocab_size, device=device)
        out = train_model(x, y, total_ut_steps=total_ut_steps)
        loss = out["loss"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(train_model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)

        losses.append(float(loss.item()))
        exits.append(float(out["mean_exit_step"].item()))

    return {
        "total_ut_steps": total_ut_steps,
        "start_loss": losses[0],
        "end_loss": losses[-1],
        "loss_delta": losses[-1] - losses[0],
        "mean_exit_step": sum(exits) / len(exits),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)

    r1 = run(total_ut_steps=1, steps=args.steps, device=device)
    r4 = run(total_ut_steps=4, steps=args.steps, device=device)
    report = {"ut1": r1, "ut4": r4}
    print(json.dumps(report, indent=2))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()

