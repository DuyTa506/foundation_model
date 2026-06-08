#!/usr/bin/env python3
"""
Stage 4: Near-deduplication using datatrove MinHash-LSH + exact SHA-256 pass.

Replaces dedup_min_hash.py which was O(n²) and operated on manifest
descriptors (no text = no-op).

Uses datatrove's built-in MinhashDedupSignature + MinhashDedupBuckets +
MinhashDedupFilter pipeline — proper banding, scalable, memory-bounded.

Usage:
    python scripts/curate/04_dedup_minhash.py \
        --config configs/curation_pipeline.yaml \
        --input_dir outputs/curated/ultraclean \
        --output_dir outputs/curated/deduped
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import yaml


def run_dedup(cfg: dict, input_dir: str, output_dir: str, workers: int) -> None:
    from datatrove.executor import LocalPipelineExecutor
    from datatrove.pipeline.dedup import (
        MinhashDedupBuckets,
        MinhashDedupFilter,
        MinhashDedupSignature,
    )
    from datatrove.pipeline.dedup.exact_substrings import ExactSubstrFindDedups
    from datatrove.pipeline.readers import ParquetReader
    from datatrove.pipeline.writers import ParquetWriter

    dedup_cfg: dict = cfg.get("dedup", {})
    ngram_size: int = dedup_cfg.get("minhash", {}).get("ngram_size", 5)
    num_hashes: int = dedup_cfg.get("minhash", {}).get("num_hashes", 128)
    threshold: float = dedup_cfg.get("minhash", {}).get("jaccard_threshold", 0.80)
    bands: int = dedup_cfg.get("minhash", {}).get("bands", 8)
    rows_per_band: int = dedup_cfg.get("minhash", {}).get("rows_per_band", 16)

    sig_dir = str(Path(output_dir) / "_minhash_signatures")
    buckets_dir = str(Path(output_dir) / "_minhash_buckets")

    # ── Step 1: Compute MinHash signatures ──────────────────────────────────
    sig_exec = LocalPipelineExecutor(
        pipeline=[
            ParquetReader(input_folder=input_dir, progress=True),
            MinhashDedupSignature(
                output_folder=sig_dir,
                num_hashes=num_hashes,
                n_grams=ngram_size,
            ),
        ],
        tasks=workers,
        workers=workers,
        logging_dir=str(Path(output_dir) / "logs_sig"),
        skip_completed=True,
    )
    print("[dedup] computing MinHash signatures ...")
    sig_exec.run()

    # ── Step 2: Group into LSH buckets ───────────────────────────────────────
    bucket_exec = LocalPipelineExecutor(
        pipeline=[
            MinhashDedupBuckets(
                input_folder=sig_dir,
                output_folder=buckets_dir,
                num_hashes=num_hashes,
                num_buckets=bands,
            ),
        ],
        tasks=bands,
        workers=min(bands, workers),
        logging_dir=str(Path(output_dir) / "logs_buckets"),
        skip_completed=True,
    )
    print("[dedup] computing LSH buckets ...")
    bucket_exec.run()

    # ── Step 3: Filter duplicates ─────────────────────────────────────────────
    filter_exec = LocalPipelineExecutor(
        pipeline=[
            ParquetReader(input_folder=input_dir, progress=True),
            MinhashDedupFilter(
                input_folder=buckets_dir,
                jaccard_threshold=threshold,
            ),
            ParquetWriter(
                output_folder=output_dir,
                output_filename="${rank:04d}.parquet",
                compression="snappy",
            ),
        ],
        tasks=workers,
        workers=workers,
        logging_dir=str(Path(output_dir) / "logs_filter"),
        skip_completed=True,
    )
    print("[dedup] filtering near-duplicates ...")
    filter_exec.run()


def main() -> None:
    parser = argparse.ArgumentParser(description="MinHash-LSH near-deduplication.")
    parser.add_argument("--config", default="configs/curation_pipeline.yaml")
    parser.add_argument("--input_dir", default="outputs/curated/ultraclean")
    parser.add_argument("--output_dir", default="outputs/curated/deduped")
    parser.add_argument("--workers", type=int, default=max(1, os.cpu_count() - 2))
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    run_dedup(cfg, args.input_dir, args.output_dir, args.workers)
    print(f"[ok] deduplication done -> {args.output_dir}")


if __name__ == "__main__":
    main()
