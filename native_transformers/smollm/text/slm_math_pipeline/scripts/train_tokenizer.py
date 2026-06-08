#!/usr/bin/env python3
"""
Train a from-scratch byte-level BPE tokenizer for Vietnamese + English.

Usage:
    python scripts/train_tokenizer.py --config configs/tokenizer_en_vi.yaml

Outputs (under output_dir from config):
    tokenizer.json            # HF PreTrainedTokenizerFast-compatible
    tokenizer_config.json
    special_tokens_map.json
    chat_template.jinja       # ChatML with conditional <think> rendering
    tokenizer_card.json       # fertility report per language

Design notes:
- Byte-level BPE: zero <unk> on any Unicode, including all VI diacritics.
- NFC normalization only — never NFKC (NFKC strips combining diacritics
  like tone marks that are critical for Vietnamese).
- individual_digits=True pre-tokenizer: 123 -> 1 2 3 for math.
- VI-prioritized training corpus (vi_ratio ~0.60) so VI gets the richest
  subword merges and lowest tokens/word (fertility).
- Special tokens added as atomic vocab entries (never split by BPE).
"""

from __future__ import annotations

import argparse
import json
import unicodedata
from pathlib import Path
from typing import Iterator

import yaml


# ─── Chat template (ChatML with <think> support) ─────────────────────────────

CHAT_TEMPLATE_JINJA = """\
{#-
  ChatML template — Vietnamese-first EN+VI SLM
  enable_thinking=true  -> <think>...</think> block rendered in assistant turn
  enable_thinking=false -> <think>...</think> stripped from assistant turn
  Default system: Vietnamese (model is Vietnamese-oriented)
-#}
{%- set enable_thinking = enable_thinking if enable_thinking is defined else false -%}
{%- set default_system = "Bạn là một trợ lý AI thông minh, thành thạo tiếng Việt và tiếng Anh.\\nHãy trả lời bằng ngôn ngữ của người dùng.\\nVới các câu hỏi toán học hoặc khoa học, hãy trình bày từng bước rõ ràng." -%}
{%- if messages[0]['role'] != 'system' -%}
    {{- '<|im_start|>system\\n' + default_system + '<|im_end|>\\n' -}}
{%- endif -%}
{%- for message in messages -%}
    {{- '<|im_start|>' + message['role'] + '\\n' -}}\
    {%- if message['role'] == 'assistant' and enable_thinking and\
           '<think>' not in message['content'] -%}
        {{- '<think>\\n' + message.get('reasoning', '') + '\\n</think>\\n' -}}\
        {{- message['content'] -}}
    {%- elif message['role'] == 'assistant' and not enable_thinking -%}
        {%- set content = message['content'] | regex_replace('<think>.*?</think>', '', multiline=True) | strip -%}
        {{- content -}}
    {%- else -%}
        {{- message['content'] -}}
    {%- endif -%}
    {{- '<|im_end|>\\n' -}}
{%- endfor -%}
{%- if add_generation_prompt -%}
    {{- '<|im_start|>assistant\\n' -}}
    {%- if enable_thinking -%}
        {{- '<think>\\n' -}}
    {%- endif -%}
{%- endif -%}
"""


# ─── Corpus iterator ─────────────────────────────────────────────────────────

def _iter_jsonl(path: Path) -> Iterator[str]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                text = obj.get("text") or obj.get("content") or ""
                if isinstance(text, str) and text.strip():
                    yield unicodedata.normalize("NFC", text)
            except json.JSONDecodeError:
                pass


def _iter_txt(path: Path) -> Iterator[str]:
    with path.open("r", encoding="utf-8") as f:
        buf: list[str] = []
        for line in f:
            buf.append(line)
            if len(buf) >= 1000:
                yield unicodedata.normalize("NFC", "".join(buf))
                buf = []
        if buf:
            yield unicodedata.normalize("NFC", "".join(buf))


def corpus_iterator(corpus_dirs: list[Path]) -> Iterator[str]:
    """Yield text strings from a list of directories containing .jsonl or .txt files."""
    for d in corpus_dirs:
        for p in sorted(d.rglob("*.jsonl")):
            yield from _iter_jsonl(p)
        for p in sorted(d.rglob("*.txt")):
            yield from _iter_txt(p)


# ─── Fertility measurement ────────────────────────────────────────────────────

_VI_SAMPLE = (
    "Học sinh cần hiểu bài toán trước khi giải. "
    "Việt Nam là một quốc gia có nền văn hóa phong phú và đa dạng."
)
_EN_SAMPLE = (
    "Students need to understand the problem before solving it. "
    "Mathematics is the language of the universe."
)
_LATEX_SAMPLE = (
    r"Let $x \in \mathbb{R}$. Then $\frac{d}{dx}[x^2] = 2x$ and "
    r"$\int_0^1 e^x\,dx = e - 1$."
)


def measure_fertility(tokenizer, target: dict) -> dict[str, float]:
    """tokens/word for VI, EN, and LaTeX samples."""
    results: dict[str, float] = {}
    samples = {"vi": _VI_SAMPLE, "en": _EN_SAMPLE, "latex": _LATEX_SAMPLE}
    for lang, text in samples.items():
        words = text.split()
        tokens = tokenizer.encode(text, add_special_tokens=False)
        fertility = len(tokens) / max(len(words), 1)
        results[lang] = round(fertility, 3)
        max_allowed = target.get(f"{lang}_max", 999)
        status = "OK" if fertility <= max_allowed else f"WARN (>{max_allowed})"
        print(f"  Fertility [{lang}]: {fertility:.3f} tok/word  {status}")
    return results


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a from-scratch byte-level BPE tokenizer for EN+VI."
    )
    parser.add_argument("--config", default="configs/tokenizer_en_vi.yaml")
    parser.add_argument(
        "--corpus_dirs",
        nargs="+",
        help="Directories containing .jsonl/.txt text files. "
             "Overrides config if provided.",
    )
    parser.add_argument("--output_dir", help="Override config output_dir.")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    output_dir = Path(args.output_dir or cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    vocab_size: int = cfg["vocab_size"]
    fertility_targets: dict = cfg.get("fertility_targets", {})

    # ── Build special token list ────────────────────────────────────────────
    special_tokens: list[str] = []
    st_cfg = cfg.get("special_tokens", {})
    for key in ("bos_token", "eos_token", "pad_token", "unk_token"):
        val = st_cfg.get(key)
        if val:
            special_tokens.append(val)
    special_tokens.extend(st_cfg.get("chat_tokens", []))
    special_tokens.extend(st_cfg.get("reasoning_tokens", []))
    reserved = st_cfg.get("reserved", {})
    for i in range(reserved.get("count", 64)):
        pat = reserved.get("pattern", "<|reserved_{i}|>").replace("{i}", str(i))
        special_tokens.append(pat)
    # Deduplicate preserving order
    seen: set[str] = set()
    unique_specials: list[str] = []
    for t in special_tokens:
        if t not in seen:
            seen.add(t)
            unique_specials.append(t)
    special_tokens = unique_specials

    # ── Import tokenizers (HF tokenizers library) ───────────────────────────
    try:
        from tokenizers import (
            Tokenizer,
            decoders,
            models,
            normalizers,
            pre_tokenizers,
            trainers,
        )
        from tokenizers.processors import TemplateProcessing
        from transformers import PreTrainedTokenizerFast
    except ImportError as exc:
        raise RuntimeError(
            "Missing deps: pip install tokenizers transformers"
        ) from exc

    print(f"[tokenizer] vocab_size={vocab_size}  special_tokens={len(special_tokens)}")
    print(f"[tokenizer] special tokens: {special_tokens[:6]} ... {special_tokens[-2:]}")

    # ── Build tokenizer object ───────────────────────────────────────────────
    tokenizer_obj = Tokenizer(models.BPE())

    # NFC normalization: canonical composition, never strip diacritics
    tokenizer_obj.normalizer = normalizers.NFC()

    # Byte-level pre-tokenizer with digit splitting
    tokenizer_obj.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Digits(individual_digits=True),
        pre_tokenizers.ByteLevel(add_prefix_space=True),
    ])

    # Byte-level decoder
    tokenizer_obj.decoder = decoders.ByteLevel()

    # ── Build corpus ────────────────────────────────────────────────────────
    if args.corpus_dirs:
        corpus_dirs = [Path(d) for d in args.corpus_dirs]
        print(f"[tokenizer] using corpus dirs: {corpus_dirs}")
        iterator = corpus_iterator(corpus_dirs)
    else:
        # Fallback: read from the materialized curated outputs
        default_dirs = [
            Path("outputs/curated/raw"),
            Path("outputs/curated/filtered"),
        ]
        available = [d for d in default_dirs if d.exists()]
        if not available:
            raise RuntimeError(
                "No corpus found. Pass --corpus_dirs or run the curation pipeline first "
                "(scripts/curate/00_materialize.py ... 02_language_id.py)."
            )
        print(f"[tokenizer] reading from default dirs: {available}")
        iterator = corpus_iterator(available)

    # ── Train ───────────────────────────────────────────────────────────────
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        min_frequency=2,
        show_progress=True,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )

    print(f"[tokenizer] training BPE (vocab={vocab_size}) ...")
    tokenizer_obj.train_from_iterator(iterator, trainer=trainer)
    print(f"[tokenizer] trained vocab size: {tokenizer_obj.get_vocab_size()}")

    # ── Post-process: add BOS/EOS template ─────────────────────────────────
    bos = st_cfg.get("bos_token", "<bos>")
    eos = st_cfg.get("eos_token", "<eos>")
    bos_id = tokenizer_obj.token_to_id(bos)
    eos_id = tokenizer_obj.token_to_id(eos)
    if bos_id is not None and eos_id is not None:
        tokenizer_obj.post_processor = TemplateProcessing(
            single=f"{bos} $A {eos}",
            pair=f"{bos} $A {eos} $B:1 {eos}:1",
            special_tokens=[(bos, bos_id), (eos, eos_id)],
        )

    # ── Save as HF fast tokenizer ────────────────────────────────────────────
    hf_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer_obj,
        bos_token=st_cfg.get("bos_token"),
        eos_token=st_cfg.get("eos_token"),
        pad_token=st_cfg.get("pad_token"),
        unk_token=st_cfg.get("unk_token"),
        additional_special_tokens=[
            t for t in special_tokens
            if t not in (
                st_cfg.get("bos_token"),
                st_cfg.get("eos_token"),
                st_cfg.get("pad_token"),
                st_cfg.get("unk_token"),
            )
        ],
    )
    hf_tokenizer.save_pretrained(str(output_dir))
    print(f"[tokenizer] saved to {output_dir}")

    # ── Write chat template ──────────────────────────────────────────────────
    chat_template_path = output_dir / "chat_template.jinja"
    chat_template_path.write_text(CHAT_TEMPLATE_JINJA, encoding="utf-8")
    # Bake the chat template + Vietnamese system prompt into tokenizer_config.json
    tc_path = output_dir / "tokenizer_config.json"
    if tc_path.exists():
        with tc_path.open("r", encoding="utf-8") as f:
            tc = json.load(f)
        tc["chat_template"] = CHAT_TEMPLATE_JINJA
        # Default system prompt in Vietnamese (model is Vietnamese-first)
        tc["default_system_prompt"] = cfg.get(
            "default_system_prompt",
            "Bạn là một trợ lý AI thông minh, thành thạo tiếng Việt và tiếng Anh.\n"
            "Hãy trả lời bằng ngôn ngữ của người dùng.\n"
            "Với các câu hỏi toán học hoặc khoa học, hãy trình bày từng bước rõ ràng.",
        )
        with tc_path.open("w", encoding="utf-8") as f:
            json.dump(tc, f, ensure_ascii=False, indent=2)

    # ── Fertility report ─────────────────────────────────────────────────────
    print("[tokenizer] measuring fertility ...")
    # Reload to ensure round-trip is correct
    hf_tok_loaded = PreTrainedTokenizerFast.from_pretrained(str(output_dir))
    fertility = measure_fertility(hf_tok_loaded, fertility_targets)

    # Round-trip validation
    for sample_name, sample_text in [
        ("VI", _VI_SAMPLE), ("EN", _EN_SAMPLE), ("LaTeX", _LATEX_SAMPLE)
    ]:
        ids = hf_tok_loaded.encode(sample_text, add_special_tokens=False)
        decoded = hf_tok_loaded.decode(ids)
        assert decoded == sample_text, (
            f"Round-trip FAILED for {sample_name}!\n  in:  {repr(sample_text)}\n  out: {repr(decoded)}"
        )
    print("[tokenizer] round-trip validation: OK")

    # Save tokenizer card
    card = {
        "vocab_size": hf_tok_loaded.vocab_size,
        "special_tokens_count": len(hf_tok_loaded.all_special_tokens),
        "fertility": fertility,
        "fertility_targets": fertility_targets,
        "config": str(args.config),
    }
    card_path = output_dir / "tokenizer_card.json"
    with card_path.open("w", encoding="utf-8") as f:
        json.dump(card, f, ensure_ascii=False, indent=2)
    print(f"[ok] tokenizer card: {card_path}")
    print(f"[ok] done. vocab_size={hf_tok_loaded.vocab_size}")


if __name__ == "__main__":
    main()
