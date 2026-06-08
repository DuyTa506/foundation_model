#!/usr/bin/env python3
"""
Quick inference tool — load a checkpoint and generate text.

Usage (single prompt):
    python scripts/generate.py --model outputs/smoke_train \
        --prompt "Tính 2 + 2 = ?"

Usage (interactive REPL):
    python scripts/generate.py --model outputs/pretrain

Usage (chat template mode):
    python scripts/generate.py --model outputs/sft \
        --chat --prompt "Giải phương trình x^2 - 4 = 0"

After smoke test (tiny model, expect incoherent output — only verifying pipeline):
    python scripts/generate.py --model outputs/smoke_train \
        --prompt "Hello" --max_new_tokens 50
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Generate text from a trained checkpoint.")
    p.add_argument("--model", required=True, help="Path to model checkpoint dir.")
    p.add_argument("--prompt", default=None, help="Prompt text. Omit for interactive mode.")
    p.add_argument("--chat", action="store_true",
                   help="Wrap prompt in ChatML chat template (use for SFT/RLVR checkpoints).")
    p.add_argument("--think", action="store_true",
                   help="Enable <think> reasoning block (only meaningful with --chat).")
    p.add_argument("--system", default=None,
                   help="System message (--chat only). Defaults to Vietnamese assistant prompt.")
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--repetition_penalty", type=float, default=1.1)
    p.add_argument("--greedy", action="store_true", help="Greedy decoding (temp=0).")
    p.add_argument("--device", default="auto", help="cuda | cpu | auto")
    return p


def load_model_and_tokenizer(model_path: str, device: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    path = Path(model_path)
    if not path.exists():
        print(f"[error] checkpoint not found: {model_path}", file=sys.stderr)
        sys.exit(1)

    # Tokenizer: look inside checkpoint first, then fall back to outputs/tokenizer
    tok_path = path / "tokenizer"
    if not (tok_path / "tokenizer.json").exists():
        tok_path = Path("outputs/tokenizer")
    if not (tok_path / "tokenizer.json").exists():
        tok_path = path  # some checkpoints store tokenizer at root

    print(f"[generate] loading tokenizer: {tok_path}")
    tok = AutoTokenizer.from_pretrained(str(tok_path))

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[generate] loading model: {path}  device={device}")
    model = AutoModelForCausalLM.from_pretrained(
        str(path),
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        local_files_only=True,
    ).to(device)
    model.eval()

    total = sum(p.numel() for p in model.parameters())
    print(f"[generate] {total/1e6:.1f}M params  vocab={tok.vocab_size}")
    return model, tok, device


def apply_chat(tok, prompt: str, system: str | None, think: bool) -> str:
    default_system = (
        "Bạn là một trợ lý AI thông minh, thành thạo tiếng Việt và tiếng Anh.\n"
        "Hãy trả lời bằng ngôn ngữ của người dùng.\n"
        "Với các câu hỏi toán học hoặc khoa học, hãy trình bày từng bước rõ ràng."
    )
    messages = [
        {"role": "system", "content": system or default_system},
        {"role": "user", "content": prompt},
    ]
    try:
        return tok.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=think,
        )
    except TypeError:
        # Older tokenizer without enable_thinking
        return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def generate(model, tok, text: str, args, device: str) -> str:
    import torch

    inputs = tok(text, return_tensors="pt", return_token_type_ids=False).to(device)
    input_len = inputs["input_ids"].shape[1]

    gen_kwargs = dict(
        **inputs,
        max_new_tokens=args.max_new_tokens,
        repetition_penalty=args.repetition_penalty,
        pad_token_id=tok.eos_token_id,
        eos_token_id=tok.eos_token_id,
    )
    if args.greedy:
        gen_kwargs["do_sample"] = False
    else:
        gen_kwargs["do_sample"] = True
        gen_kwargs["temperature"] = args.temperature
        gen_kwargs["top_p"] = args.top_p

    with torch.inference_mode():
        out = model.generate(**gen_kwargs)

    new_tokens = out[0][input_len:]
    return tok.decode(new_tokens, skip_special_tokens=True)


def run_once(model, tok, args, device: str, prompt: str) -> None:
    if args.chat:
        input_text = apply_chat(tok, prompt, args.system, args.think)
    else:
        input_text = prompt

    print("\n" + "─" * 60)
    print(f"[prompt]  {prompt}")
    print("─" * 60)
    response = generate(model, tok, input_text, args, device)
    print(response)
    print("─" * 60)


def run_interactive(model, tok, args, device: str) -> None:
    mode = "chat" if args.chat else "raw"
    print(f"\n[generate] interactive mode ({mode}) — Ctrl+C or 'quit' to exit\n")
    while True:
        try:
            prompt = input(">>> ").strip()
        except (KeyboardInterrupt, EOFError):
            print()
            break
        if prompt.lower() in ("quit", "exit", "q"):
            break
        if not prompt:
            continue
        run_once(model, tok, args, device, prompt)


def main() -> None:
    args = build_parser().parse_args()
    model, tok, device = load_model_and_tokenizer(args.model, args.device)

    if args.prompt:
        run_once(model, tok, args, device, args.prompt)
    else:
        run_interactive(model, tok, args, device)


if __name__ == "__main__":
    main()
