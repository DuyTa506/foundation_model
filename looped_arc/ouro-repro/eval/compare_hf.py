#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from models.configuration_ouro import OuroConfig
from models.modeling_ouro import OuroForCausalLM


def _load_local_checkpoint(path: str, device: torch.device):
    model = OuroForCausalLM(OuroConfig())
    ckpt = torch.load(path, map_location="cpu")
    state = ckpt["model"] if "model" in ckpt else ckpt
    model.load_state_dict(state, strict=False)
    model.to(device).eval()
    return model


@torch.no_grad()
def _logits(model, input_ids, total_ut_steps=4):
    if hasattr(model, "config") and hasattr(model.config, "total_ut_steps"):
        out = model(input_ids=input_ids)
        if isinstance(out, list):
            return out[-1]
        if hasattr(out, "logits"):
            return out.logits
        return out
    out = model(input_ids=input_ids, total_ut_steps=total_ut_steps, return_all_steps=False)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-model", default="ByteDance/Ouro-1.4B")
    parser.add_argument("--local-ckpt", required=True)
    parser.add_argument("--prompt", default="Solve 2x+3=11.")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.hf_model, trust_remote_code=True)
    hf_model = AutoModelForCausalLM.from_pretrained(
        args.hf_model,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    ).eval()
    local_model = _load_local_checkpoint(args.local_ckpt, device=device)

    batch = tokenizer(args.prompt, return_tensors="pt")
    input_ids = batch["input_ids"].to(device)

    hf_logits = _logits(hf_model, input_ids)
    local_logits = _logits(local_model, input_ids)
    min_vocab = min(hf_logits.shape[-1], local_logits.shape[-1])
    hf_logits = hf_logits[..., :min_vocab].float().cpu()
    local_logits = local_logits[..., :min_vocab].float().cpu()

    diff = (hf_logits - local_logits).abs()
    report = {
        "prompt": args.prompt,
        "max_abs_diff": diff.max().item(),
        "mean_abs_diff": diff.mean().item(),
        "hf_shape": list(hf_logits.shape),
        "local_shape": list(local_logits.shape),
    }
    print(json.dumps(report, indent=2))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()

