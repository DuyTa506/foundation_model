#!/usr/bin/env python3
"""
Initialize a Llama-like model FROM SCRATCH (random weights, NOT from_pretrained).

Usage:
    python scripts/init_model_from_scratch.py --config configs/model_llama_1b_en_vi.yaml

Reads:
    - configs/model_llama_1b_en_vi.yaml  (architecture)
    - outputs/tokenizer/                 (vocab_size auto-detected)

Writes:
    outputs/model_init/   (config.json + model.safetensors + tokenizer/)

Init method:
    - All linear/embedding weights: normal(0, initializer_range=0.02)
      (standard Llama / GPT-2 convention, matches Nanotron SmolLM2 precedent)
    - RMSNorm scale (gamma): filled to 1.0
    - Depth-scaled residual projections (o_proj, down_proj):
      additionally multiplied by 1/sqrt(2 * num_hidden_layers)
      — stabilizes residual stream variance at depth.
      This is the correct transformer analog of "Kaiming-like" init;
      literal Kaiming-He (fan-in/fan-out) is NOT used and is inappropriate
      for transformer LMs.
    - Does NOT use from_pretrained or load any checkpoint.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path

import yaml


def _round_up_vocab(size: int, multiple: int = 256) -> int:
    """Round vocab size UP to a multiple for GPU tensor-core efficiency."""
    return math.ceil(size / multiple) * multiple


def apply_depth_scaled_init(model, num_hidden_layers: int) -> None:
    """
    Scale o_proj and down_proj weights by 1/sqrt(2 * num_hidden_layers).

    This keeps the residual stream variance approximately constant as signals
    pass through stacked residual blocks (GPT-2/Llama-3 style).
    Applied AFTER the default HF _init_weights pass.
    """
    import torch

    scale = 1.0 / math.sqrt(2.0 * num_hidden_layers)
    n_scaled = 0
    for name, param in model.named_parameters():
        if name.endswith(("o_proj.weight", "down_proj.weight")):
            with torch.no_grad():
                param.mul_(scale)
            n_scaled += 1
    print(f"[init] depth-scaled {n_scaled} residual projection weights (scale={scale:.4f})")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Initialize a Llama model from scratch (random weights)."
    )
    parser.add_argument("--config", default="configs/model_llama_1b_en_vi.yaml")
    parser.add_argument("--tokenizer_path", default="outputs/tokenizer",
                        help="Path to the trained tokenizer (to read vocab_size).")
    parser.add_argument("--output_dir", default=None,
                        help="Override output_dir from config.")
    parser.add_argument("--vocab_size", type=int, default=None,
                        help="Override vocab_size (auto-detected from tokenizer if omitted).")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    model_cfg = cfg["model"]
    output_dir = Path(args.output_dir or cfg.get("output_dir", "outputs/model_init"))
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Detect vocab_size from the trained tokenizer ─────────────────────────
    tokenizer_path = Path(args.tokenizer_path)
    if args.vocab_size:
        raw_vocab = args.vocab_size
    elif (tokenizer_path / "tokenizer.json").exists():
        import json as _json
        with (tokenizer_path / "tokenizer.json").open("r") as f:
            tok_data = _json.load(f)
        raw_vocab = len(tok_data.get("model", {}).get("vocab", {}))
        if raw_vocab == 0:
            # Fallback: read tokenizer_config.json
            tc_path = tokenizer_path / "tokenizer_config.json"
            if tc_path.exists():
                with tc_path.open("r") as f:
                    tc = _json.load(f)
                raw_vocab = tc.get("vocab_size", model_cfg.get("vocab_size", 64000))
        print(f"[init] detected raw vocab_size from tokenizer: {raw_vocab}")
    else:
        raw_vocab = model_cfg.get("vocab_size", 64000)
        print(f"[init] tokenizer not found at {tokenizer_path}; using config vocab_size={raw_vocab}")

    vocab_size = _round_up_vocab(raw_vocab, 256)
    if vocab_size != raw_vocab:
        print(f"[init] rounded vocab_size {raw_vocab} -> {vocab_size} (multiple of 256)")

    # ── Build LlamaConfig ────────────────────────────────────────────────────
    try:
        from transformers import LlamaConfig, LlamaForCausalLM
    except ImportError as exc:
        raise RuntimeError("pip install transformers") from exc

    config = LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=model_cfg["hidden_size"],
        intermediate_size=model_cfg["intermediate_size"],
        num_hidden_layers=model_cfg["num_hidden_layers"],
        num_attention_heads=model_cfg["num_attention_heads"],
        num_key_value_heads=model_cfg["num_key_value_heads"],
        head_dim=model_cfg.get("head_dim", None),      # decoupled head_dim
        hidden_act=model_cfg.get("hidden_act", "silu"),
        max_position_embeddings=model_cfg["max_position_embeddings"],
        initializer_range=model_cfg.get("initializer_range", 0.02),
        rms_norm_eps=model_cfg.get("rms_norm_eps", 1e-6),
        use_cache=model_cfg.get("use_cache", True),
        tie_word_embeddings=model_cfg.get("tie_word_embeddings", False),
        rope_theta=model_cfg.get("rope_theta", 5_000_000.0),
        rope_scaling=model_cfg.get("rope_scaling", None),
        attention_bias=model_cfg.get("attention_bias", False),
        mlp_bias=model_cfg.get("mlp_bias", False),
        torch_dtype=model_cfg.get("torch_dtype", "bfloat16"),
    )
    # Some transformers versions accept unknown LlamaConfig kwargs without
    # exposing them as attributes. Keep these fields explicit so config.json and
    # init_summary.json preserve the long-context RoPE settings from YAML.
    if "rope_theta" in model_cfg:
        config.rope_theta = model_cfg["rope_theta"]
    if "rope_scaling" in model_cfg:
        config.rope_scaling = model_cfg.get("rope_scaling")

    print(f"[init] building LlamaForCausalLM from config (NOT from_pretrained)")
    print(f"[init] hidden_size={config.hidden_size} layers={config.num_hidden_layers} "
          f"heads={config.num_attention_heads} kv_heads={config.num_key_value_heads} "
          f"head_dim={getattr(config, 'head_dim', 'auto')} "
          f"vocab={config.vocab_size}")

    # ── Random init via constructor ──────────────────────────────────────────
    import torch

    with torch.device("cpu"):
        model = LlamaForCausalLM(config)

    # The HF constructor calls _init_weights: normal(0, initializer_range) for
    # linears/embeddings, 1.0 for RMSNorm. We apply depth-scaling on top.
    apply_depth_scaled_init(model, config.num_hidden_layers)

    # ── Verify it's truly random (not pretrained) ────────────────────────────
    # Check embed weight mean is close to 0 and std close to initializer_range
    embed_w = model.model.embed_tokens.weight.data
    emb_mean = embed_w.mean().item()
    emb_std = embed_w.std().item()
    print(f"[init] embed_tokens: mean={emb_mean:.4f}  std={emb_std:.4f}  "
          f"(expected ~0.0, ~{config.initializer_range:.4f})")
    assert abs(emb_mean) < 0.01, f"Embed mean too large: {emb_mean} (not random?)"

    # ── Parameter count ───────────────────────────────────────────────────────
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[init] total parameters: {total_params:,}  ({total_params/1e9:.3f}B)")
    print(f"[init] trainable:        {trainable_params:,}")

    # ── Save HF checkpoint ────────────────────────────────────────────────────
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    target_dtype = dtype_map.get(model_cfg.get("torch_dtype", "bfloat16"), torch.bfloat16)
    model = model.to(target_dtype)
    model.save_pretrained(str(output_dir), safe_serialization=True)
    print(f"[init] model saved to {output_dir}")

    # ── Copy tokenizer alongside the model ───────────────────────────────────
    if tokenizer_path.exists():
        tokenizer_out = output_dir / "tokenizer"
        if tokenizer_out.exists():
            shutil.rmtree(tokenizer_out)
        shutil.copytree(str(tokenizer_path), str(tokenizer_out))
        print(f"[init] tokenizer copied to {tokenizer_out}")
    else:
        print(f"[warn] tokenizer not found at {tokenizer_path}; copy manually before pretraining.")

    # ── Save a summary card ───────────────────────────────────────────────────
    summary = {
        "init_method": "random_normal_with_depth_scaled_residuals",
        "initializer_range": config.initializer_range,
        "depth_scale_formula": "1/sqrt(2 * num_hidden_layers)",
        "is_pretrained": False,
        "from_pretrained_source": None,
        "total_parameters": total_params,
        "total_parameters_B": round(total_params / 1e9, 3),
        "vocab_size": vocab_size,
        "hidden_size": config.hidden_size,
        "num_hidden_layers": config.num_hidden_layers,
        "num_attention_heads": config.num_attention_heads,
        "num_key_value_heads": config.num_key_value_heads,
        "head_dim": getattr(config, "head_dim", None),
        "rope_theta": getattr(config, "rope_theta", model_cfg.get("rope_theta")),
        "rope_scaling": getattr(config, "rope_scaling", model_cfg.get("rope_scaling")),
        "max_position_embeddings": config.max_position_embeddings,
    }
    with (output_dir / "init_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[ok] init summary: {output_dir / 'init_summary.json'}")
    print(f"[ok] model ready at {output_dir}  ({total_params/1e9:.3f}B params, random init)")


if __name__ == "__main__":
    main()
