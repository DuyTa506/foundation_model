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

from _curate_utils import prune_empty_parquet, stable_metadata_adapter


def run_dedup(cfg: dict, input_dir: str, output_dir: str, workers: int) -> None:
    from datatrove.executor import LocalPipelineExecutor
    from datatrove.pipeline.dedup import (
        MinhashConfig,
        MinhashDedupBuckets,
        MinhashDedupCluster,
        MinhashDedupFilter,
        MinhashDedupSignature,
    )
    from datatrove.pipeline.readers import ParquetReader
    from datatrove.pipeline.writers import ParquetWriter

    dedup_cfg: dict = cfg.get("dedup", {})
    mh_cfg: dict = dedup_cfg.get("minhash", {})
    ngram_size: int = mh_cfg.get("ngram_size", 5)
    bands: int = mh_cfg.get("bands", 8)
    rows_per_band: int = mh_cfg.get("rows_per_band", 16)

    # datatrove's banding IS the Jaccard threshold: num_buckets * hashes_per_bucket
    # total hashes; the implied similarity threshold ≈ (1/bands)^(1/rows_per_band).
    # (jaccard_threshold in the config is informational — datatrove has no such kwarg.)
    config = MinhashConfig(
        n_grams=ngram_size,
        num_buckets=bands,
        hashes_per_bucket=rows_per_band,
    )

    sig_dir = str(Path(output_dir) / "_minhash" / "signatures")
    buckets_dir = str(Path(output_dir) / "_minhash" / "buckets")
    remove_dir = str(Path(output_dir) / "_minhash" / "remove_ids")

    # ── Step 1: Compute MinHash signatures (sharded by reader tasks) ─────────
    sig_exec = LocalPipelineExecutor(
        pipeline=[
            ParquetReader(data_folder=input_dir, glob_pattern="**/*.parquet",
                          doc_progress=True),
            MinhashDedupSignature(output_folder=sig_dir, config=config),
        ],
        tasks=workers,
        workers=workers,
        logging_dir=str(Path(output_dir) / "logs_sig"),
        skip_completed=True,
    )
    print("[dedup] computing MinHash signatures ...")
    sig_exec.run()

    # ── Step 2: Group into LSH buckets (one task per bucket) ─────────────────
    bucket_exec = LocalPipelineExecutor(
        pipeline=[
            MinhashDedupBuckets(
                input_folder=sig_dir,
                output_folder=buckets_dir,
                config=config,
            ),
        ],
        tasks=bands,
        workers=min(bands, workers),
        logging_dir=str(Path(output_dir) / "logs_buckets"),
        skip_completed=True,
    )
    print("[dedup] computing LSH buckets ...")
    bucket_exec.run()

    # ── Step 3: Cluster matches → list of duplicate ids to remove ────────────
    cluster_exec = LocalPipelineExecutor(
        pipeline=[
            MinhashDedupCluster(
                input_folder=buckets_dir,
                output_folder=remove_dir,
                config=config,
            ),
        ],
        tasks=1,
        workers=1,
        logging_dir=str(Path(output_dir) / "logs_cluster"),
        skip_completed=True,
    )
    print("[dedup] clustering duplicates ...")
    cluster_exec.run()

    # ── Step 4: Drop the clustered duplicates ────────────────────────────────
    filter_exec = LocalPipelineExecutor(
        pipeline=[
            ParquetReader(data_folder=input_dir, glob_pattern="**/*.parquet",
                          doc_progress=True),
            MinhashDedupFilter(input_folder=remove_dir),
            # Keep the on-disk metadata struct uniform across all writer tasks (defensive:
            # guarantees no schema drift accumulates regardless of worker count).
            ParquetWriter(
                output_folder=output_dir,
                output_filename="${rank}.parquet",
                compression="snappy",
                adapter=stable_metadata_adapter(
                    keep_keys=("source", "dataset", "language")),
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

    prune_empty_parquet(args.input_dir)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    run_dedup(cfg, args.input_dir, args.output_dir, args.workers)
    print(f"[ok] deduplication done -> {args.output_dir}")


if __name__ == "__main__":
    main()
