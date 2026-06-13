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
        --input_dir outputs/curated/quality_filtered \
        --output_dir outputs/curated/deduped
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

import yaml

from _curate_utils import prune_empty_parquet, stable_metadata_adapter


def _exact_content(doc) -> str:
    return doc.text or ""


def run_dedup(cfg: dict, input_dir: str, output_dir: str, workers: int) -> None:
    from datatrove.executor import LocalPipelineExecutor
    from datatrove.pipeline.dedup import (
        ExactDedupConfig,
        ExactDedupFilter,
        ExactDedupSignature,
        ExactFindDedups,
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
    seed: int = int(cfg.get("seed", 1))

    exact_tmp: Path | None = None
    minhash_input_dir = input_dir
    if dedup_cfg.get("exact_first", True):
        exact_tmp = Path(output_dir).with_name(Path(output_dir).name + "_exact_tmp")
        if exact_tmp.exists():
            shutil.rmtree(exact_tmp)
        exact_sig_dir = str(exact_tmp / "signatures")
        exact_remove_dir = str(exact_tmp / "remove_ids")
        exact_filtered_dir = str(exact_tmp / "filtered")
        finder_workers = max(1, min(workers, 8))
        exact_config = ExactDedupConfig(content_getter=_exact_content)

        sig_exec = LocalPipelineExecutor(
            pipeline=[
                ParquetReader(data_folder=input_dir, glob_pattern="**/*.parquet",
                              doc_progress=True),
                ExactDedupSignature(
                    output_folder=exact_sig_dir,
                    config=exact_config,
                    finder_workers=finder_workers,
                ),
            ],
            tasks=workers,
            workers=workers,
            logging_dir=str(exact_tmp / "logs_sig"),
            skip_completed=False,
        )
        print("[dedup] computing exact-dedup signatures ...")
        sig_exec.run()

        find_exec = LocalPipelineExecutor(
            pipeline=[
                ExactFindDedups(
                    data_folder=exact_sig_dir,
                    output_folder=exact_remove_dir,
                    config=exact_config,
                ),
            ],
            tasks=finder_workers,
            workers=finder_workers,
            logging_dir=str(exact_tmp / "logs_find"),
            skip_completed=False,
        )
        print("[dedup] finding exact duplicates ...")
        find_exec.run()

        filter_exec = LocalPipelineExecutor(
            pipeline=[
                ParquetReader(data_folder=input_dir, glob_pattern="**/*.parquet",
                              doc_progress=True),
                ExactDedupFilter(data_folder=exact_remove_dir, config=exact_config),
                ParquetWriter(
                    output_folder=exact_filtered_dir,
                    output_filename="${rank}.parquet",
                    compression="snappy",
                    adapter=stable_metadata_adapter(
                        keep_keys=("source", "dataset", "language")),
                ),
            ],
            tasks=workers,
            workers=workers,
            logging_dir=str(exact_tmp / "logs_filter"),
            skip_completed=False,
        )
        print("[dedup] filtering exact duplicates ...")
        filter_exec.run()
        prune_empty_parquet(exact_filtered_dir)
        minhash_input_dir = exact_filtered_dir

    # datatrove's banding IS the Jaccard threshold: num_buckets * hashes_per_bucket
    # total hashes; the implied similarity threshold ≈ (1/bands)^(1/rows_per_band).
    # (jaccard_threshold is reported because datatrove has no direct threshold kwarg.)
    requested_threshold = mh_cfg.get("jaccard_threshold")
    implied_threshold = (1.0 / max(1, bands)) ** (1.0 / max(1, rows_per_band))
    if requested_threshold is not None and abs(float(requested_threshold) - implied_threshold) > 0.03:
        print("[dedup] warn: requested jaccard_threshold="
              f"{float(requested_threshold):.2f}, but bands={bands}, rows={rows_per_band} "
              f"imply ~{implied_threshold:.2f}")
    config = MinhashConfig(
        n_grams=ngram_size,
        num_buckets=bands,
        hashes_per_bucket=rows_per_band,
        seed=seed,
    )

    sig_dir = str(Path(output_dir) / "_minhash" / "signatures")
    buckets_dir = str(Path(output_dir) / "_minhash" / "buckets")
    remove_dir = str(Path(output_dir) / "_minhash" / "remove_ids")

    # ── Step 1: Compute MinHash signatures (sharded by reader tasks) ─────────
    sig_exec = LocalPipelineExecutor(
        pipeline=[
            ParquetReader(data_folder=minhash_input_dir, glob_pattern="**/*.parquet",
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
            ParquetReader(data_folder=minhash_input_dir, glob_pattern="**/*.parquet",
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
    if exact_tmp is not None and exact_tmp.exists():
        shutil.rmtree(exact_tmp)


def main() -> None:
    parser = argparse.ArgumentParser(description="MinHash-LSH near-deduplication.")
    parser.add_argument("--config", default="configs/curation_pipeline.yaml")
    parser.add_argument("--input_dir", default="outputs/curated/quality_filtered")
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
