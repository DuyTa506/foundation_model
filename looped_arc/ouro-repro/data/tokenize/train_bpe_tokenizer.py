#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterator

from datasets import load_dataset
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, trainers, decoders
from transformers import PreTrainedTokenizerFast


def iter_texts(
    dataset_name: str,
    dataset_config: str | None,
    split: str,
    text_field: str,
    max_rows: int,
) -> Iterator[str]:
    ds = load_dataset(dataset_name, name=dataset_config, split=split, streaming=True)
    seen = 0
    for row in ds:
        text = row.get(text_field)
        if not text:
            continue
        yield text
        seen += 1
        if seen >= max_rows:
            break


def main():
    parser = argparse.ArgumentParser(description="Train a brand-new BPE tokenizer from a pretrain corpus.")
    parser.add_argument("--dataset", required=True, help="HF dataset name, e.g. HuggingFaceFW/fineweb-edu")
    parser.add_argument("--config", default=None, help="HF dataset config/subset, e.g. sample-10BT")
    parser.add_argument("--split", default="train")
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--max-rows", type=int, default=5_000_000)
    parser.add_argument("--vocab-size", type=int, default=50_000)
    parser.add_argument("--min-frequency", type=int, default=2)
    parser.add_argument("--out-dir", required=True, help="Output tokenizer directory")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    special_tokens = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<pad>", "<unk>"]

    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.normalizer = normalizers.Sequence([normalizers.NFKC()])
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        special_tokens=special_tokens,
        show_progress=True,
    )
    tokenizer.train_from_iterator(
        iter_texts(
            dataset_name=args.dataset,
            dataset_config=args.config,
            split=args.split,
            text_field=args.text_field,
            max_rows=args.max_rows,
        ),
        trainer=trainer,
    )

    tok_fast = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token="<|im_start|>",
        eos_token="<|im_end|>",
        pad_token="<pad>",
        unk_token="<unk>",
    )
    tok_fast.model_max_length = 32768
    tok_fast.save_pretrained(out_dir.as_posix())

    meta = {
        "dataset": args.dataset,
        "config": args.config,
        "split": args.split,
        "text_field": args.text_field,
        "max_rows": args.max_rows,
        "requested_vocab_size": args.vocab_size,
        "actual_vocab_size": len(tok_fast),
        "special_tokens": special_tokens,
    }
    with (out_dir / "tokenizer_build_meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(json.dumps(meta, indent=2))
    print(f"\nTokenizer written to: {out_dir}")


if __name__ == "__main__":
    main()

