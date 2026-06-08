#!/usr/bin/env python3
"""
Stage 7: Tokenize and pack documents into fixed-length token shards.

Replaces the Megatron preprocess_data.py path in prepare_megatron_indexed_dataset.py.
Uses datatrove DocumentTokenizer for streaming, shuffled, packed shards.

Packing: concatenate docs separated by EOS, chunk into max_seq_length blocks.
No padding — every token is real text.  The pretrainer reads these shards directly.

Usage:
    python scripts/curate/07_tokenize_pack.py \
        --config configs/curation_pipeline.yaml \
        --input_dir outputs/curated/pii_clean \
        --output_dir outputs/curated/tokenized \
        --tokenizer_path outputs/tokenizer \
        [--max_seq_length 4096]

For long-context phases, run again with --max_seq_length 32768 / 131072
pointing at the long-document input dir.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import yaml


def tokenize_and_pack(
    input_dir: str,
    output_dir: str,
    tokenizer_path: str,
    max_seq_length: int,
    max_tokens_per_file: int,
    workers: int,
    shuffle: bool,
) -> None:
    from datatrove.executor import LocalPipelineExecutor
    from datatrove.pipeline.readers import ParquetReader
    from datatrove.pipeline.tokens import DocumentTokenizer

    executor = LocalPipelineExecutor(
        pipeline=[
            ParquetReader(input_folder=input_dir, progress=True, shuffle_files=shuffle),
            DocumentTokenizer(
                output_folder=output_dir,
                tokenizer_name_or_path=tokenizer_path,
                eos_token="<eos>",        # EOS between documents for packing
                max_tokens_per_file=max_tokens_per_file,
                shuffle=shuffle,
                # DocumentTokenizer packs + chunks to max_tokens in the writer
            ),
        ],
        tasks=workers,
        workers=workers,
        logging_dir=str(Path(output_dir) / "logs"),
        skip_completed=True,
    )
    executor.run()


def main() -> None:
    parser = argparse.ArgumentParser(description="Tokenize + pack into fixed-length shards.")
    parser.add_argument("--config", default="configs/curation_pipeline.yaml")
    parser.add_argument("--input_dir", default="outputs/curated/pii_clean")
    parser.add_argument("--output_dir", default="outputs/curated/tokenized")
    parser.add_argument("--tokenizer_path", default="outputs/tokenizer")
    parser.add_argument("--max_seq_length", type=int, default=None,
                        help="Override max_seq_length from config.")
    parser.add_argument("--max_tokens_per_file", type=int, default=None,
                        help="Override max_tokens_per_file from config.")
    parser.add_argument("--workers", type=int, default=max(1, os.cpu_count() - 2))
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    tok_cfg: dict = cfg.get("tokenize", {})
    max_seq_length: int = args.max_seq_length or tok_cfg.get("max_seq_length", 4096)
    max_tokens_per_file: int = args.max_tokens_per_file or tok_cfg.get(
        "max_tokens_per_file", 1_000_000_000
    )
    shuffle: bool = tok_cfg.get("shuffle", True)
    tokenizer_path: str = args.tokenizer_path or tok_cfg.get("tokenizer_path", "outputs/tokenizer")

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    print(f"[tokenize] max_seq_length={max_seq_length}  max_tokens_per_file={max_tokens_per_file:,}")
    print(f"[tokenize] tokenizer={tokenizer_path}  shuffle={shuffle}")

    tokenize_and_pack(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        tokenizer_path=tokenizer_path,
        max_seq_length=max_seq_length,
        max_tokens_per_file=max_tokens_per_file,
        workers=args.workers,
        shuffle=shuffle,
    )
    print(f"[ok] tokenization done -> {args.output_dir}")


if __name__ == "__main__":
    main()
