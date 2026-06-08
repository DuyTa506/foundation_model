#!/usr/bin/env python3
"""
Pack long-context training shards from already-downloaded base pretrain datasets.

No new downloads needed — reuses the same HF cache from download_datasets.py.
Documents are concatenated with EOS tokens between them until a target sequence
length is reached, then saved as uint16 .npy shards.

This replaces download_longctx_datasets.py for the context-extension stages.
Run once per seq_len before each longctx training stage.

Usage:
    # Pack for 16k stage
    python scripts/pack_longctx_shards.py \
        --tokenizer_path outputs/tokenizer \
        --cache_dir /data/hf_cache \
        --seq_len 16384 \
        --output_dir outputs/curated/tokenized_16k \
        --max_tokens 8000000000   # 8B tokens

    # Pack for 32k stage
    python scripts/pack_longctx_shards.py \
        --tokenizer_path outputs/tokenizer \
        --cache_dir /data/hf_cache \
        --seq_len 32768 \
        --output_dir outputs/curated/tokenized_32k

    # Dry run — show sources and estimated token count
    python scripts/pack_longctx_shards.py --dry_run --seq_len 16384

Packed format: uint16 .npy arrays, each of length (seq_len + 1).
  input_ids = shard[:seq_len]
  labels    = shard[1:seq_len+1]
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

import numpy as np
import yaml


# ── Sources to use for long-context packing ───────────────────────────────────
# These are all already downloaded by download_datasets.py (base pretrain).
# VI-first: longer docs concatenated to fill the context window.
# Weight controls how many tokens to draw from each source.

LONGCTX_SOURCES = [
    # VI sources (target ~65% of long-ctx corpus)
    {"id": "fineweb2_hq_vi",  "weight": 0.25, "lang": "vi"},
    {"id": "c4_vi",           "weight": 0.18, "lang": "vi"},
    {"id": "culturax_vi",     "weight": 0.10, "lang": "vi"},
    {"id": "wikipedia_vi",    "weight": 0.07, "lang": "vi"},
    {"id": "vi_curated",      "weight": 0.05, "lang": "vi"},
    # EN sources — math/science heavy for long reasoning chains
    {"id": "pes2o",           "weight": 0.15, "lang": "en"},
    {"id": "finemath_4plus",  "weight": 0.10, "lang": "en"},
    {"id": "open_web_math",   "weight": 0.05, "lang": "en"},
    {"id": "fineweb_edu",     "weight": 0.05, "lang": "en"},
]


def build_source_map(curation_cfg: dict) -> dict[str, dict]:
    """Build id→source config from curation_pipeline.yaml."""
    return {s["id"]: s for s in curation_cfg.get("sources", []) if s.get("id")}


def stream_source(src_cfg: dict, cache_dir: str | None, hf_token: str | None):
    """Stream text from one HF dataset source."""
    from datasets import load_dataset

    hf_dataset = src_cfg.get("hf_dataset")
    if not hf_dataset:
        return
    subset  = src_cfg.get("subset")
    split   = re.sub(r'\[.*?\]', '', src_cfg.get("split", "train")).strip() or "train"
    field   = src_cfg.get("text_field", "text")

    load_kwargs: dict = dict(path=hf_dataset, split=split, streaming=True, token=hf_token)
    if subset:
        load_kwargs["name"] = subset
    if cache_dir:
        load_kwargs["cache_dir"] = cache_dir

    ds = load_dataset(**load_kwargs)
    for row in ds:
        text = row.get(field) or row.get("text") or row.get("content") or ""
        if isinstance(text, str) and len(text) >= 50:
            yield text


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pack long-context shards from already-downloaded base pretrain data."
    )
    parser.add_argument("--tokenizer_path", default="outputs/tokenizer")
    parser.add_argument("--curation_config", default="configs/curation_pipeline.yaml")
    parser.add_argument("--cache_dir", default=None,
                        help="HF cache dir (must match what download_datasets.py used)")
    parser.add_argument("--hf_token", default=os.environ.get("HF_TOKEN"))
    parser.add_argument("--seq_len", type=int, required=True,
                        help="Target sequence length (e.g. 16384, 32768, 65536, 131072)")
    parser.add_argument("--output_dir", default=None,
                        help="Output dir for .npy shards. Default: outputs/curated/tokenized_{seq_len//1024}k")
    parser.add_argument("--max_tokens", type=int, default=10_000_000_000,
                        help="Stop after this many tokens total (default: 10B)")
    parser.add_argument("--shard_size", type=int, default=500_000_000,
                        help="Tokens per .npy shard (default: 500M)")
    parser.add_argument("--min_doc_chars", type=int, default=500,
                        help="Skip documents shorter than this (chars). "
                             "Longer docs produce more useful long-context sequences.")
    parser.add_argument("--dry_run", action="store_true",
                        help="Print plan without writing any files")
    args = parser.parse_args()

    seq_len = args.seq_len
    output_dir = Path(args.output_dir or f"outputs/curated/tokenized_{seq_len // 1024}k")

    with open(args.curation_config) as f:
        curation_cfg = yaml.safe_load(f)
    source_map = build_source_map(curation_cfg)

    # Resolve sources + per-source token budget
    total_weight = sum(s["weight"] for s in LONGCTX_SOURCES)
    sources_with_budget = []
    for s in LONGCTX_SOURCES:
        cfg = source_map.get(s["id"])
        if not cfg or not cfg.get("hf_dataset"):
            print(f"[skip] {s['id']} — not in curation_pipeline.yaml or no hf_dataset")
            continue
        budget = int(args.max_tokens * s["weight"] / total_weight)
        sources_with_budget.append((s, cfg, budget))

    print(f"\nLong-context packing plan")
    print(f"  seq_len:    {seq_len:,} tokens  ({seq_len // 1024}k)")
    print(f"  max_tokens: {args.max_tokens / 1e9:.1f}B")
    print(f"  output_dir: {output_dir}")
    print(f"  sources ({len(sources_with_budget)}):")
    vi_budget = sum(b for s, _, b in sources_with_budget if s["lang"] == "vi")
    en_budget = sum(b for s, _, b in sources_with_budget if s["lang"] == "en")
    for s, cfg, budget in sources_with_budget:
        print(f"    {s['id']:<22} {s['lang']}  budget={budget/1e9:.2f}B tok  "
              f"weight={s['weight']:.2f}  ({cfg['hf_dataset']})")
    print(f"  VI: {vi_budget/args.max_tokens*100:.0f}%  EN: {en_budget/args.max_tokens*100:.0f}%\n")

    if args.dry_run:
        print("Dry run — no files written.")
        return

    from transformers import PreTrainedTokenizerFast

    output_dir.mkdir(parents=True, exist_ok=True)
    tok = PreTrainedTokenizerFast.from_pretrained(args.tokenizer_path)
    eos_id = tok.eos_token_id or 2

    L = seq_len
    buf: list[int] = []          # rolling token buffer
    total_tokens = 0
    shard_idx = 0
    docs_processed = 0

    def flush_shard(tokens: list[int], idx: int, count: int) -> None:
        arr = np.array(tokens, dtype=np.uint16)
        path = output_dir / f"shard_{idx:04d}.npy"
        np.save(str(path), arr)
        n_examples = len(tokens) // (L + 1)
        print(f"  [shard {idx:04d}] {path.name}  {len(tokens):,} tokens  "
              f"{n_examples:,} examples  (total docs: {count:,})")

    try:
        from tqdm import tqdm
        pbar = tqdm(total=args.max_tokens, unit="tok", unit_scale=True,
                    desc=f"packing {seq_len//1024}k")
    except ImportError:
        pbar = None

    for source_info, src_cfg, budget in sources_with_budget:
        src_tokens = 0
        print(f"\n[source] {source_info['id']}  budget={budget/1e9:.2f}B tok")

        for text in stream_source(src_cfg, args.cache_dir, args.hf_token):
            if len(text) < args.min_doc_chars:
                continue

            ids = tok.encode(text, add_special_tokens=False)
            ids.append(eos_id)   # document separator
            buf.extend(ids)
            src_tokens += len(ids)
            total_tokens += len(ids)
            docs_processed += 1

            # Flush complete shards
            while len(buf) >= args.shard_size:
                flush_shard(buf[:args.shard_size], shard_idx, docs_processed)
                buf = buf[args.shard_size:]
                shard_idx += 1

            if pbar:
                pbar.update(len(ids))

            if src_tokens >= budget:
                break
            if total_tokens >= args.max_tokens:
                break

        if total_tokens >= args.max_tokens:
            break

    if pbar:
        pbar.close()

    # Flush final partial shard (only if enough for at least 1 example)
    if len(buf) >= L + 1:
        flush_shard(buf, shard_idx, docs_processed)
        shard_idx += 1
    elif buf:
        print(f"  [warn] {len(buf):,} leftover tokens < seq_len+1={L+1}, discarding")

    shards = sorted(output_dir.glob("*.npy"))
    total_examples = sum(len(np.load(str(p), mmap_mode="r")) // (L + 1) for p in shards)

    print(f"\n[done] seq_len={seq_len}  total_tokens={total_tokens:,}  "
          f"docs={docs_processed:,}  shards={shard_idx}  examples={total_examples:,}")
    print(f"       → {output_dir}")


if __name__ == "__main__":
    main()
