#!/usr/bin/env python3
"""
Stage 5: Decontamination — remove training docs that overlap eval benchmarks.

Implements the 13-gram overlap check that was declared in curation_pipeline.yaml
but never implemented in the old pipeline.

Strategy: build an n-gram index from all eval splits, then drop any training
document that contains any matching n-gram.

Usage:
    python scripts/curate/05_decontaminate.py \
        --config configs/curation_pipeline.yaml \
        --input_dir outputs/curated/deduped \
        --output_dir outputs/curated/decontaminated
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from typing import FrozenSet

import yaml

TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def _ngrams(text: str, n: int) -> FrozenSet[str]:
    tokens = TOKEN_RE.findall(text.lower())
    if len(tokens) < n:
        return frozenset()
    return frozenset(" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def _load_eval_ngrams(benchmark_sets: list[str], ngram_size: int) -> FrozenSet[str]:
    """Download eval splits and build their n-gram fingerprint."""
    from datasets import load_dataset

    all_ngrams: set[str] = set()
    for bench in benchmark_sets:
        if bench.startswith("harness:") or bench.startswith("ZaloAI"):
            # VI benchmarks: try loading, skip if unavailable
            hf_id = bench.split(":", 1)[-1]
        else:
            hf_id = bench

        splits_to_check = ["test", "validation", "dev"]
        field_names = ["question", "prompt", "text", "problem",
                       "query", "input", "sentence"]

        for split in splits_to_check:
            try:
                ds = load_dataset(hf_id, split=split, trust_remote_code=True)
                for field in field_names:
                    if field in ds.column_names:
                        for text in ds[field]:
                            if isinstance(text, str):
                                all_ngrams.update(_ngrams(text, ngram_size))
                print(f"[decontam] indexed {bench} {split}: +{len(all_ngrams):,} n-grams")
                break
            except Exception:
                pass

    print(f"[decontam] total eval n-grams: {len(all_ngrams):,}")
    return frozenset(all_ngrams)


def decontaminate(
    cfg: dict,
    input_dir: str,
    output_dir: str,
    workers: int,
) -> None:
    from datatrove.executor import LocalPipelineExecutor
    from datatrove.pipeline.filters import LambdaFilter
    from datatrove.pipeline.readers import ParquetReader
    from datatrove.pipeline.writers import ParquetWriter

    dc_cfg: dict = cfg.get("decontamination", {})
    ngram_size: int = dc_cfg.get("ngram_size", 13)
    benchmark_sets: list[str] = dc_cfg.get("benchmark_sets", [])

    print(f"[decontam] building eval n-gram index (ngram={ngram_size}) ...")
    eval_ngrams = _load_eval_ngrams(benchmark_sets, ngram_size)

    if not eval_ngrams:
        print("[warn] no eval n-grams loaded; skipping decontamination")
        import shutil
        shutil.copytree(input_dir, output_dir)
        return

    def _is_clean(doc) -> bool:
        text: str = doc.text or ""
        doc_ngrams = _ngrams(text, ngram_size)
        contaminated = bool(doc_ngrams & eval_ngrams)
        if contaminated:
            doc.metadata["decontam_dropped"] = True
        return not contaminated

    executor = LocalPipelineExecutor(
        pipeline=[
            ParquetReader(input_folder=input_dir, progress=True),
            LambdaFilter(filter_func=_is_clean, name="decontamination"),
            ParquetWriter(
                output_folder=output_dir,
                output_filename="${rank:04d}.parquet",
                compression="snappy",
            ),
        ],
        tasks=workers,
        workers=workers,
        logging_dir=str(Path(output_dir) / "logs"),
        skip_completed=True,
    )
    executor.run()


def main() -> None:
    parser = argparse.ArgumentParser(description="N-gram decontamination vs eval sets.")
    parser.add_argument("--config", default="configs/curation_pipeline.yaml")
    parser.add_argument("--input_dir", default="outputs/curated/deduped")
    parser.add_argument("--output_dir", default="outputs/curated/decontaminated")
    parser.add_argument("--workers", type=int, default=max(1, os.cpu_count() - 2))
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    decontaminate(cfg, args.input_dir, args.output_dir, args.workers)
    print(f"[ok] decontamination done -> {args.output_dir}")


if __name__ == "__main__":
    main()
