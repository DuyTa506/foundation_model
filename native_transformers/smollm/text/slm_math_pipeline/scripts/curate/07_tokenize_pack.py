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
        --input_dir outputs/curated/mixed \
        --output_dir outputs/curated/tokenized \
        --tokenizer_path outputs/tokenizer \
        [--max_seq_length 4096]

For long-context phases, run again with --max_seq_length 32768 / 131072
pointing at the long-document input dir.
"""

from __future__ import annotations

import argparse
import inspect
import os
from pathlib import Path

import yaml

from _curate_utils import prune_empty_parquet


def _resolve_tokenizer_file(tokenizer_path: str) -> str:
    """datatrove's load_tokenizer() does ``Tokenizer.from_file(path)`` only when
    ``os.path.isfile(path)`` is true; otherwise it falls through to
    ``Tokenizer.from_pretrained(path)`` which treats the string as a HuggingFace
    Hub repo id and raises HFValidationError for a local dir like
    ``outputs/tokenizer``. So when given a directory, hand datatrove the concrete
    ``tokenizer.json`` file inside it."""
    p = Path(tokenizer_path)
    if p.is_dir():
        tok_json = p / "tokenizer.json"
        if not tok_json.is_file():
            raise FileNotFoundError(
                f"{tokenizer_path} is a directory but has no tokenizer.json; "
                f"datatrove's DocumentTokenizer needs a tokenizer.json file path")
        return str(tok_json)
    return tokenizer_path


def _resolve_eos_token(tokenizer_path: str) -> str:
    """The EOS string DocumentTokenizer inserts between docs must actually exist
    in the tokenizer vocab, so read it from the tokenizer rather than hardcoding."""
    from transformers import AutoTokenizer
    tk = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    eos = tk.eos_token
    if not eos:
        raise ValueError(
            f"tokenizer at {tokenizer_path} has no eos_token; set one before packing")
    return eos


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

    eos_token = _resolve_eos_token(tokenizer_path)
    tokenizer_file = _resolve_tokenizer_file(tokenizer_path)
    print(f"[tokenize] eos_token={eos_token!r} (from tokenizer)")
    print(f"[tokenize] tokenizer_file={tokenizer_file}")

    tokenizer_kwargs = {
        "output_folder": output_dir,
        "tokenizer_name_or_path": tokenizer_file,
        "local_working_dir": str(Path(output_dir) / "_scratch"),
        "eos_token": eos_token,      # EOS between documents for packing
        "max_tokens_per_file": max_tokens_per_file,
    }
    tokenizer_params = inspect.signature(DocumentTokenizer.__init__).parameters
    if "shuffle_documents" in tokenizer_params:
        tokenizer_kwargs["shuffle_documents"] = shuffle
    elif shuffle:
        print("[tokenize] installed datatrove DocumentTokenizer has no "
              "shuffle_documents kwarg; only input files will be shuffled")

    executor = LocalPipelineExecutor(
        pipeline=[
            ParquetReader(data_folder=input_dir, glob_pattern="**/*.parquet",
                          doc_progress=True, shuffle_files=shuffle),
            # DocumentTokenizer streams token .ds shards; the pretrainer's
            # PackedTokenDataset chunks them into max_seq_length blocks.
            DocumentTokenizer(**tokenizer_kwargs),
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
    parser.add_argument("--input_dir", default="outputs/curated/mixed")
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

    prune_empty_parquet(args.input_dir)
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
