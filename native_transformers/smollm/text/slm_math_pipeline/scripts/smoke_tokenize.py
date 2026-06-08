#!/usr/bin/env python3
"""
Minimal tokenize + pack script for smoke testing the pretrain pipeline.

Reads from an HF datasets cache (or streams), tokenizes with the trained
tokenizer, packs into fixed-length blocks, and saves as .npy shards.

Usage:
    python scripts/smoke_tokenize.py \
        --tokenizer_path outputs/tokenizer_test \
        --cache_dir /tmp/hf_cache_test \
        --source_id wikipedia_vi \
        --seq_len 512 \
        --max_tokens 5000000 \
        --output_dir outputs/smoke_tokenized
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer_path", default="outputs/tokenizer_test")
    parser.add_argument("--curation_config", default="configs/curation_pipeline.yaml")
    parser.add_argument("--source_id", default="wikipedia_vi",
                        help="Single source ID from curation_pipeline.yaml to tokenize.")
    parser.add_argument("--cache_dir", default=None)
    parser.add_argument("--hf_token", default=os.environ.get("HF_TOKEN"))
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--max_tokens", type=int, default=5_000_000,
                        help="Stop after this many tokens (for smoke test).")
    parser.add_argument("--output_dir", default="outputs/smoke_tokenized")
    parser.add_argument("--shard_size", type=int, default=1_000_000,
                        help="Tokens per .npy shard file.")
    args = parser.parse_args()

    from transformers import PreTrainedTokenizerFast
    from datasets import load_dataset
    import re

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load tokenizer
    print(f"[smoke_tokenize] loading tokenizer from {args.tokenizer_path}")
    tok = PreTrainedTokenizerFast.from_pretrained(args.tokenizer_path)
    eos_id = tok.eos_token_id or 2

    # Find source in curation config
    with open(args.curation_config, "r") as f:
        curation_cfg = yaml.safe_load(f)

    src = next(
        (s for s in curation_cfg["sources"] if s.get("id") == args.source_id),
        None,
    )
    if src is None:
        raise ValueError(f"source_id '{args.source_id}' not found in {args.curation_config}")

    hf_dataset = src["hf_dataset"]
    subset = src.get("subset")
    split = re.sub(r'\[.*?\]', '', src.get("split", "train")).strip() or "train"
    text_field = src.get("text_field", "text")

    print(f"[smoke_tokenize] streaming {hf_dataset} split={split}")
    load_kwargs = dict(path=hf_dataset, split=split, streaming=True, token=args.hf_token)
    if subset:
        load_kwargs["name"] = subset
    if args.cache_dir:
        load_kwargs["cache_dir"] = args.cache_dir

    ds = load_dataset(**load_kwargs)

    # Tokenize + pack
    L = args.seq_len
    buf: list[int] = []
    total_tokens = 0
    shard_idx = 0
    docs_processed = 0

    try:
        from tqdm import tqdm
        pbar = tqdm(total=args.max_tokens, unit="tok", unit_scale=True, desc="tokenizing")
    except ImportError:
        pbar = None

    def flush_shard(tokens: list[int], idx: int) -> None:
        arr = np.array(tokens, dtype=np.uint16)
        path = output_dir / f"shard_{idx:04d}.npy"
        np.save(str(path), arr)
        print(f"  saved {path}  ({len(tokens):,} tokens)")

    for row in ds:
        text = row.get(text_field) or row.get("text") or row.get("content") or ""
        if not isinstance(text, str) or len(text) < 20:
            continue

        ids = tok.encode(text, add_special_tokens=False)
        ids.append(eos_id)
        buf.extend(ids)
        docs_processed += 1

        while len(buf) >= args.shard_size:
            flush_shard(buf[:args.shard_size], shard_idx)
            buf = buf[args.shard_size:]
            shard_idx += 1

        n = len(ids)
        total_tokens += n
        if pbar:
            pbar.update(n)
        if total_tokens >= args.max_tokens:
            break

    if pbar:
        pbar.close()

    # Flush remaining tokens (must be at least seq_len+1 to form one training example)
    if len(buf) >= L + 1:
        flush_shard(buf, shard_idx)
        shard_idx += 1
    elif buf:
        print(f"  [warn] {len(buf)} leftover tokens < seq_len+1={L+1}, discarding last partial shard")

    print(f"\n[smoke_tokenize] done: {total_tokens:,} tokens, {docs_processed:,} docs, "
          f"{shard_idx} shards → {output_dir}")

    # Quick sanity: verify one shard
    shards = sorted(output_dir.glob("*.npy"))
    if shards:
        sample = np.load(str(shards[0]), mmap_mode="r")
        n_examples = len(sample) // (L + 1)
        print(f"[smoke_tokenize] shard[0]: {len(sample):,} tokens → {n_examples} examples "
              f"of seq_len={L}")


if __name__ == "__main__":
    main()
