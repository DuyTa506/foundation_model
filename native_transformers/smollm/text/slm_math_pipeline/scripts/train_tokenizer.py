#!/usr/bin/env python3
"""
Train a from-scratch byte-level BPE tokenizer for Vietnamese + English.

Usage:
    # Primary: read directly from HF datasets cache (no materialize step needed)
    python scripts/train_tokenizer.py \
        --config configs/tokenizer_en_vi.yaml \
        --curation_config configs/curation_pipeline.yaml \
        --cache_dir /data/hf_cache

    # Legacy: read from pre-materialized .jsonl/.txt files
    python scripts/train_tokenizer.py \
        --config configs/tokenizer_en_vi.yaml \
        --corpus_dirs outputs/curated/raw

Outputs (under output_dir from config):
    tokenizer.json            HF PreTrainedTokenizerFast-compatible
    tokenizer_config.json
    special_tokens_map.json
    chat_template.jinja
    tokenizer_card.json       fertility report per language

Design notes:
- Byte-level BPE: zero <unk> on any Unicode, including all VI diacritics.
- NFC normalization only — never NFKC (NFKC strips combining diacritics
  critical for Vietnamese tone marks).
- individual_digits=True pre-tokenizer: 123 -> 1 2 3 for math.
- VI-prioritized training corpus (vi_ratio ~0.60) so VI gets richer merges.
- Each source is budget-capped by weight so no single dataset dominates.
- Round-robin interleaving ensures VI and EN are mixed throughout training
  (matters for BPE since merge decisions reflect the distribution seen so far).
"""

from __future__ import annotations

import argparse
import json
import os
import unicodedata
from pathlib import Path
from typing import Iterator

import yaml


# ─── Chat template ────────────────────────────────────────────────────────────

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


# ─── Legacy corpus iterator (.jsonl / .txt files) ─────────────────────────────

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


def corpus_iterator_from_dirs(corpus_dirs: list[Path]) -> Iterator[str]:
    for d in corpus_dirs:
        for p in sorted(d.rglob("*.jsonl")):
            yield from _iter_jsonl(p)
        for p in sorted(d.rglob("*.txt")):
            yield from _iter_txt(p)


# ─── HF dataset iterator (primary path) ───────────────────────────────────────

# Rough chars-per-token estimate for budget calculation.
# Used to convert token budget → char budget per source.
_CHARS_PER_TOKEN = {"vi": 4.0, "en": 4.5, "default": 4.5}


def _iter_hf_source(
    src: dict,
    max_chars: int,
    cache_dir: str | None,
    hf_token: str | None,
    shuffle_seed: int = 42,
) -> Iterator[str]:
    """Stream NFC-normalized text from one HF dataset source, capped at max_chars."""
    from datasets import load_dataset

    hf_dataset = src["hf_dataset"]
    split = src.get("split", "train")
    text_field = src.get("text_field", "text")

    load_kwargs: dict = dict(
        path=hf_dataset,
        split=split,
        streaming=True,
        token=hf_token,
    )
    if src.get("subset"):
        load_kwargs["name"] = src["subset"]
    if cache_dir:
        load_kwargs["storage_options"] = {"hf_token": hf_token}

    try:
        ds = load_dataset(**load_kwargs)
        # Shuffle within streaming buffer for diversity
        ds = ds.shuffle(seed=shuffle_seed, buffer_size=10_000)
    except Exception as e:
        print(f"  [warn] could not load {hf_dataset}: {e} — skipping")
        return

    chars_yielded = 0
    for row in ds:
        text = row.get(text_field) or row.get("text") or row.get("content") or ""
        if not isinstance(text, str) or len(text) < 50:
            continue
        text = unicodedata.normalize("NFC", text)
        yield text
        chars_yielded += len(text)
        if chars_yielded >= max_chars:
            break


def build_balanced_iterator(
    tokenizer_cfg: dict,
    curation_cfg: dict,
    cache_dir: str | None,
    hf_token: str | None,
) -> Iterator[str]:
    """
    Build a VI/EN balanced corpus iterator from HF datasets.

    Respects tokenizer_cfg.training_corpus:
      vi_ratio, en_ratio, max_corpus_tokens
    Distributes budget per source proportionally by weight.
    Round-robin interleaves sources so VI and EN are mixed throughout.
    """
    train_cfg = tokenizer_cfg.get("training_corpus", {})
    max_tokens: int = train_cfg.get("max_corpus_tokens", 5_000_000_000)
    vi_ratio: float = train_cfg.get("vi_ratio", 0.60)
    en_ratio: float = train_cfg.get("en_ratio", 0.40)
    seed: int = curation_cfg.get("seed", 42)

    vi_budget_chars = int(max_tokens * vi_ratio * _CHARS_PER_TOKEN["vi"])
    en_budget_chars = int(max_tokens * en_ratio * _CHARS_PER_TOKEN["en"])

    # Only active HF sources (skip null hf_dataset and disabled)
    all_sources = [
        s for s in curation_cfg.get("sources", [])
        if s.get("hf_dataset") and s.get("enabled", True) is not False
    ]
    vi_sources = [s for s in all_sources if s.get("language") == "vi"]
    en_sources = [s for s in all_sources if s.get("language") == "en"]

    vi_w_total = sum(s.get("weight", 0) for s in vi_sources) or 1.0
    en_w_total = sum(s.get("weight", 0) for s in en_sources) or 1.0

    print(f"[tokenizer] corpus budget: {max_tokens/1e9:.1f}B tokens  "
          f"VI={vi_ratio:.0%} ({vi_budget_chars/1e9:.1f}B chars)  "
          f"EN={en_ratio:.0%} ({en_budget_chars/1e9:.1f}B chars)")
    print(f"[tokenizer] VI sources: {len(vi_sources)}  EN sources: {len(en_sources)}")

    # Build per-source iterators with individual char budgets
    source_iters: list[Iterator[str]] = []
    for src in vi_sources:
        budget = int(vi_budget_chars * src.get("weight", 0) / vi_w_total)
        if budget < 1_000_000:  # skip if <1M chars
            continue
        print(f"  vi  {src['id']:25s}  budget={budget/1e9:.2f}B chars")
        source_iters.append(_iter_hf_source(src, budget, cache_dir, hf_token, seed))

    for src in en_sources:
        budget = int(en_budget_chars * src.get("weight", 0) / en_w_total)
        if budget < 1_000_000:
            continue
        print(f"  en  {src['id']:25s}  budget={budget/1e9:.2f}B chars")
        source_iters.append(_iter_hf_source(src, budget, cache_dir, hf_token, seed))

    # Round-robin interleave: ensures VI+EN mixed throughout BPE training
    active = [iter(it) for it in source_iters]
    exhausted = [False] * len(active)
    while not all(exhausted):
        for i, it in enumerate(active):
            if exhausted[i]:
                continue
            try:
                yield next(it)
            except StopIteration:
                exhausted[i] = True


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


def measure_fertility(tokenizer, targets: dict) -> dict[str, float]:
    results: dict[str, float] = {}
    for lang, text in [("vi", _VI_SAMPLE), ("en", _EN_SAMPLE), ("latex", _LATEX_SAMPLE)]:
        words = text.split()
        tokens = tokenizer.encode(text, add_special_tokens=False)
        fertility = len(tokens) / max(len(words), 1)
        results[lang] = round(fertility, 3)
        max_allowed = targets.get(f"{lang}_max", 999)
        status = "OK" if fertility <= max_allowed else f"WARN (>{max_allowed})"
        print(f"  fertility [{lang}]: {fertility:.3f} tok/word  {status}")
    return results


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a from-scratch byte-level BPE tokenizer for EN+VI."
    )
    parser.add_argument("--config", default="configs/tokenizer_en_vi.yaml")
    parser.add_argument(
        "--curation_config", default="configs/curation_pipeline.yaml",
        help="Curation pipeline config (used to read HF dataset sources).",
    )
    parser.add_argument(
        "--corpus_dirs", nargs="+", default=None,
        help="Legacy mode: directories with .jsonl/.txt files. "
             "If set, skips HF dataset reading.",
    )
    parser.add_argument(
        "--cache_dir", default=None,
        help="HF datasets cache directory (same as used in download_datasets.py).",
    )
    parser.add_argument(
        "--hf_token", default=os.environ.get("HF_TOKEN"),
        help="HuggingFace token for gated datasets.",
    )
    parser.add_argument("--output_dir", help="Override config output_dir.")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    output_dir = Path(args.output_dir or cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    vocab_size: int = cfg["vocab_size"]
    fertility_targets: dict = cfg.get("fertility_targets", {})

    # ── Special tokens ──────────────────────────────────────────────────────
    special_tokens: list[str] = []
    st_cfg = cfg.get("special_tokens", {})
    for key in ("bos_token", "eos_token", "pad_token", "unk_token"):
        val = st_cfg.get(key)
        if val:
            special_tokens.append(val)
    special_tokens.extend(st_cfg.get("chat_tokens", []))
    special_tokens.extend(st_cfg.get("thinking_tokens", []))
    special_tokens.extend(st_cfg.get("tool_tokens", []))
    # Legacy key name kept for backward compat
    special_tokens.extend(st_cfg.get("reasoning_tokens", []))
    reserved = st_cfg.get("reserved", {})
    for i in range(reserved.get("count", 64)):
        pat = reserved.get("pattern", "<|reserved_{i}|>").replace("{i}", str(i))
        special_tokens.append(pat)
    # Deduplicate preserving order
    seen: set[str] = set()
    special_tokens = [t for t in special_tokens if not (t in seen or seen.add(t))]

    # ── Import tokenizers ───────────────────────────────────────────────────
    try:
        from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers
        from tokenizers.processors import TemplateProcessing
        from transformers import PreTrainedTokenizerFast
    except ImportError as exc:
        raise RuntimeError("pip install tokenizers transformers") from exc

    print(f"[tokenizer] vocab_size={vocab_size}  special_tokens={len(special_tokens)}")

    # ── Build tokenizer ─────────────────────────────────────────────────────
    tokenizer_obj = Tokenizer(models.BPE())
    tokenizer_obj.normalizer = normalizers.NFC()
    tokenizer_obj.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Digits(individual_digits=True),
        pre_tokenizers.ByteLevel(add_prefix_space=True),
    ])
    tokenizer_obj.decoder = decoders.ByteLevel()

    # ── Build corpus iterator ───────────────────────────────────────────────
    if args.corpus_dirs:
        # Legacy mode: read from .jsonl/.txt directories
        print(f"[tokenizer] legacy mode: reading from {args.corpus_dirs}")
        iterator = corpus_iterator_from_dirs([Path(d) for d in args.corpus_dirs])
    else:
        # Primary mode: stream directly from HF datasets
        with open(args.curation_config, "r", encoding="utf-8") as f:
            curation_cfg = yaml.safe_load(f)
        iterator = build_balanced_iterator(cfg, curation_cfg, args.cache_dir, args.hf_token)

    # ── Train ───────────────────────────────────────────────────────────────
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        min_frequency=2,
        show_progress=True,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )

    print(f"[tokenizer] training BPE vocab_size={vocab_size} ...")
    tokenizer_obj.train_from_iterator(iterator, trainer=trainer)
    print(f"[tokenizer] trained vocab size: {tokenizer_obj.get_vocab_size()}")

    # ── BOS/EOS post-processor ──────────────────────────────────────────────
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

    # ── Save as HF fast tokenizer ───────────────────────────────────────────
    hf_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer_obj,
        bos_token=st_cfg.get("bos_token"),
        eos_token=st_cfg.get("eos_token"),
        pad_token=st_cfg.get("pad_token"),
        unk_token=st_cfg.get("unk_token"),
        additional_special_tokens=[
            t for t in special_tokens
            if t not in (
                st_cfg.get("bos_token"), st_cfg.get("eos_token"),
                st_cfg.get("pad_token"), st_cfg.get("unk_token"),
            )
        ],
    )
    hf_tokenizer.save_pretrained(str(output_dir))
    print(f"[tokenizer] saved to {output_dir}")

    # ── Chat template ───────────────────────────────────────────────────────
    (output_dir / "chat_template.jinja").write_text(CHAT_TEMPLATE_JINJA, encoding="utf-8")
    tc_path = output_dir / "tokenizer_config.json"
    if tc_path.exists():
        with tc_path.open("r", encoding="utf-8") as f:
            tc = json.load(f)
        tc["chat_template"] = CHAT_TEMPLATE_JINJA
        tc["default_system_prompt"] = cfg.get(
            "default_system_prompt",
            "Bạn là một trợ lý AI thông minh, thành thạo tiếng Việt và tiếng Anh.\n"
            "Hãy trả lời bằng ngôn ngữ của người dùng.\n"
            "Với các câu hỏi toán học hoặc khoa học, hãy trình bày từng bước rõ ràng.",
        )
        with tc_path.open("w", encoding="utf-8") as f:
            json.dump(tc, f, ensure_ascii=False, indent=2)

    # ── Fertility + round-trip validation ───────────────────────────────────
    print("[tokenizer] measuring fertility ...")
    hf_tok_loaded = PreTrainedTokenizerFast.from_pretrained(str(output_dir))
    fertility = measure_fertility(hf_tok_loaded, fertility_targets)

    for name, text in [("VI", _VI_SAMPLE), ("EN", _EN_SAMPLE), ("LaTeX", _LATEX_SAMPLE)]:
        ids = hf_tok_loaded.encode(text, add_special_tokens=False)
        decoded = hf_tok_loaded.decode(ids)
        assert decoded == text, (
            f"Round-trip FAILED for {name}!\n  in:  {repr(text)}\n  out: {repr(decoded)}"
        )
    print("[tokenizer] round-trip validation: OK")

    # ── Tokenizer card ──────────────────────────────────────────────────────
    card = {
        "vocab_size": hf_tok_loaded.vocab_size,
        "special_tokens_count": len(hf_tok_loaded.all_special_tokens),
        "fertility": fertility,
        "fertility_targets": fertility_targets,
        "config": str(args.config),
    }
    with (output_dir / "tokenizer_card.json").open("w", encoding="utf-8") as f:
        json.dump(card, f, ensure_ascii=False, indent=2)

    print(f"[ok] tokenizer ready at {output_dir}  vocab={hf_tok_loaded.vocab_size}")


if __name__ == "__main__":
    main()
