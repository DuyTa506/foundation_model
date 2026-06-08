#!/usr/bin/env python3
"""
Stage 0: Materialize raw text from HuggingFace dataset sources.

Fixes the fundamental bug in build_dataset_index.py which only wrote
source descriptors (HF ids) and never fetched actual text.

Usage:
    python scripts/curate/00_materialize.py \
        --config configs/curation_pipeline.yaml \
        --output_dir outputs/curated/raw \
        [--max_rows_per_source 5000000]  # for debugging

Outputs:
    outputs/curated/raw/<source_id>/000.parquet, 001.parquet, ...

Uses datatrove streaming readers so RAM usage is bounded regardless of
dataset size.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import yaml


def _check_datatrove():
    try:
        import datatrove  # noqa: F401
    except ImportError:
        sys.exit(
            "datatrove not installed. Run: pip install datatrove[processing]"
        )


def materialize_source(
    source_cfg: dict,
    output_dir: Path,
    max_rows: int | None,
    num_workers: int,
) -> int:
    """Stream one HF source and write sharded parquet files. Returns row count."""
    from datatrove.executor import LocalPipelineExecutor
    from datatrove.pipeline.readers import HuggingFaceDatasetReader, ParquetReader
    from datatrove.pipeline.writers import ParquetWriter

    src_id: str = source_cfg["id"]
    hf_dataset: str | None = source_cfg.get("hf_dataset")
    if not hf_dataset:
        print(f"[skip] {src_id}: no hf_dataset specified")
        return 0

    src_out = output_dir / src_id
    src_out.mkdir(parents=True, exist_ok=True)

    # Skip if already done (check for sentinel file)
    sentinel = src_out / "_done"
    if sentinel.exists():
        print(f"[skip] {src_id}: already materialized ({sentinel})")
        return -1  # unknown count, was done

    text_field: str = source_cfg.get("text_field", "text")
    subset: str | None = source_cfg.get("subset")
    split: str = source_cfg.get("split", "train")

    print(f"[materialize] {src_id} <- {hf_dataset} subset={subset} split={split} "
          f"text_field={text_field}")

    reader = HuggingFaceDatasetReader(
        dataset=hf_dataset,
        dataset_options={"name": subset} if subset else {},
        split=split,
        text_key=text_field,
        progress=True,
        limit=max_rows,
    )

    writer = ParquetWriter(
        output_folder=str(src_out),
        output_filename="${rank:04d}.parquet",
        compression="snappy",
    )

    executor = LocalPipelineExecutor(
        pipeline=[reader, writer],
        tasks=num_workers,
        workers=num_workers,
        logging_dir=str(src_out / "logs"),
        skip_completed=True,
    )
    executor.run()

    # Count rows
    n = sum(
        1
        for p in src_out.rglob("*.parquet")
        for _ in __import__("pyarrow.parquet", fromlist=["read_table"])
        .read_table(str(p), columns=[text_field[:1]])
        .to_pydict()
        .values()
    )
    sentinel.write_text(json.dumps({"source": src_id, "rows": n}))
    print(f"[ok] {src_id}: {n:,} rows -> {src_out}")
    return n


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize HF dataset sources to parquet.")
    parser.add_argument("--config", default="configs/curation_pipeline.yaml")
    parser.add_argument("--output_dir", default="outputs/curated/raw")
    parser.add_argument("--max_rows_per_source", type=int, default=None,
                        help="Cap rows per source (useful for debugging).")
    parser.add_argument("--source_ids", nargs="*",
                        help="Only materialize these source IDs (default: all).")
    parser.add_argument("--workers", type=int, default=max(1, os.cpu_count() - 2))
    args = parser.parse_args()

    _check_datatrove()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sources: list[dict] = cfg.get("sources", [])
    if args.source_ids:
        sources = [s for s in sources if s["id"] in args.source_ids]

    enabled_sources = [s for s in sources if s.get("enabled", True) and s.get("hf_dataset")]
    print(f"[materialize] {len(enabled_sources)} sources to materialize")

    total = 0
    for src in enabled_sources:
        n = materialize_source(src, output_dir, args.max_rows_per_source, args.workers)
        if n > 0:
            total += n

    print(f"[ok] materialization complete. total rows ≈ {total:,}")


if __name__ == "__main__":
    main()
