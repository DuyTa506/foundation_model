#!/usr/bin/env python3
"""
Stage 1: Heuristic quality filtering.

Applies Gopher + C4 + FineWeb-style rules to remove low-quality documents.
Also applies VI-specific rules (diacritic presence, encoding check).

Replaces the absent quality-filter step in the old pipeline.

Usage:
    python scripts/curate/01_quality_filter.py \
        --config configs/curation_pipeline.yaml \
        --input_dir outputs/curated/lang_filtered \
        --output_dir outputs/curated/quality_filtered
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import yaml

from _curate_utils import build_quality_router, prune_empty_parquet, stable_metadata_adapter


def build_pipeline(cfg: dict, input_dir: str, output_dir: str, workers: int):
    from datatrove.executor import LocalPipelineExecutor
    from datatrove.pipeline.filters import LambdaFilter
    from datatrove.pipeline.readers import ParquetReader
    from datatrove.pipeline.writers import ParquetWriter

    # Language-routed: VI docs get a relaxed chain (the EN-tuned Gopher/C4/FineWeb
    # heuristics — esp. Gopher's English `min_stop_words` — reject VI en masse), EN
    # keeps the full English chain. Logic lives in _curate_utils.build_quality_router
    # so the survival-measurement script uses identical rules.
    route = build_quality_router(cfg)

    return LocalPipelineExecutor(
        pipeline=[
            # glob_pattern restricts to parquet only; without it datatrove reads
            # ALL files recursively, including the logs/ sidecar each stage writes.
            ParquetReader(data_folder=input_dir, glob_pattern="**/*.parquet",
                          doc_progress=True),
            LambdaFilter(filter_function=route),
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
        logging_dir=str(Path(output_dir) / "logs"),
        skip_completed=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Heuristic quality filter.")
    parser.add_argument("--config", default="configs/curation_pipeline.yaml")
    parser.add_argument("--input_dir", default="outputs/curated/lang_filtered")
    parser.add_argument("--output_dir", default="outputs/curated/quality_filtered")
    parser.add_argument("--workers", type=int, default=max(1, os.cpu_count() - 2))
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    prune_empty_parquet(args.input_dir)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    executor = build_pipeline(cfg, args.input_dir, args.output_dir, args.workers)
    executor.run()
    print(f"[ok] quality filtering done -> {args.output_dir}")


if __name__ == "__main__":
    main()
