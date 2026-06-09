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

from _curate_utils import run_with_hf_retry, stable_metadata_adapter


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
    streamed_root: Path | None = None,
    cache_dir: Path | None = None,
) -> int:
    """Materialize one source to sharded parquet. Returns row count.

    Source-selection priority (so an old server with data already on disk keeps
    its old behavior, and only a fresh machine pulls anything new):
      1. Pre-downloaded streamed parquet at <streamed_root>/<source_id>/ — read directly.
      2. Existing HF arrow cache under --cache_dir — reused by load_dataset (no re-download).
      3. Neither present → download from HuggingFace.
    """
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

    # ── Source selection: prefer local streamed parquet over re-downloading from HF ──
    streamed_src = (streamed_root / src_id) if streamed_root else None
    uses_hf_api = False  # True when reading directly from HF (load_dataset hits the API)
    if streamed_src and (streamed_src / ".done").exists():
        # download_datasets.py already pulled the correct fraction and normalized
        # the text column to "text". Read it directly — no network, no big arrow.
        print(f"[materialize] {src_id} <- streamed parquet {streamed_src}  "
              f"(no HF re-download)")
        reader = ParquetReader(
            data_folder=str(streamed_src),
            glob_pattern="*.parquet",
            text_key="text",
            doc_progress=True,
            limit=max_rows if max_rows else -1,
        )
    else:
        uses_hf_api = True
        if streamed_root:
            print(f"[materialize] {src_id}: no streamed cache at {streamed_src}, "
                  f"falling back to HF (reuses existing arrow cache if present)")
        print(f"[materialize] {src_id} <- {hf_dataset} subset={subset} split={split} "
              f"text_field={text_field}")
        # Pass cache_dir so load_dataset reuses an already-downloaded arrow cache
        # (the "old server" case) instead of re-downloading.
        # NOTE: datatrove's HuggingFaceDatasetReader has NO top-level `split` kwarg —
        # split/name/cache_dir all go inside dataset_options, which it forwards to
        # load_dataset(dataset, **dataset_options).
        dataset_options: dict = {"split": split}
        if subset:
            dataset_options["name"] = subset
        if cache_dir:
            dataset_options["cache_dir"] = str(cache_dir)
        reader = HuggingFaceDatasetReader(
            dataset=hf_dataset,
            dataset_options=dataset_options,
            text_key=text_field,
            doc_progress=True,
            limit=max_rows if max_rows else -1,
        )

    # Normalize EVERY source to one uniform metadata schema {source,dataset,language}.
    # Without this, heterogeneous per-source metadata (esp. numeric fields) makes a
    # later stage's writer crash when one rank batches docs from mixed sources
    # (ArrowTypeError: str cannot be converted to int). See stable_metadata_adapter.
    language: str = source_cfg.get("language", "")
    writer = ParquetWriter(
        output_folder=str(src_out),
        output_filename="${rank}.parquet",
        compression="snappy",
        adapter=stable_metadata_adapter(
            keep_keys=("source", "dataset", "language"),
            defaults={"source": src_id, "dataset": hf_dataset, "language": language},
        ),
    )

    # When reading straight from HF, EVERY task calls load_dataset and hits HF's
    # file-listing API. With tasks = cpu_count-2 (e.g. 118) that bursts past HF's
    # 1000-req/5-min cap -> 429 Too Many Requests. The download is IO-bound, not
    # CPU-bound, so cap concurrent API callers; the streamed-parquet path is local
    # and keeps full parallelism. Override the cap with MATERIALIZE_HF_TASKS.
    if uses_hf_api:
        hf_cap = int(os.environ.get("MATERIALIZE_HF_TASKS", "8"))
        n_tasks = max(1, min(num_workers, hf_cap))
        if n_tasks < num_workers:
            print(f"[materialize] {src_id}: capping HF tasks {num_workers} -> {n_tasks} "
                  f"to avoid HF API rate limits (set MATERIALIZE_HF_TASKS to change)")
    else:
        n_tasks = num_workers

    executor = LocalPipelineExecutor(
        pipeline=[reader, writer],
        tasks=n_tasks,
        workers=n_tasks,
        logging_dir=str(src_out / "logs"),
        skip_completed=True,
    )
    # Retry on transient HF 429s; skip_completed makes a retry resume, not restart.
    run_with_hf_retry(executor)

    # Count rows from parquet metadata (cheap — no full read)
    import pyarrow.parquet as pq
    n = sum(pq.read_metadata(str(p)).num_rows for p in src_out.rglob("*.parquet"))
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
    parser.add_argument("--cache_dir", default=None,
                        help="HF cache dir used by download_datasets.py. Streamed parquet "
                             "is read from <cache_dir>/streamed/ to avoid HF re-download.")
    parser.add_argument("--streamed_dir", default=None,
                        help="Explicit dir of pre-downloaded streamed parquet shards "
                             "(overrides <cache_dir>/streamed).")
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

    # Resolve where pre-downloaded streamed parquet lives (from download_datasets.py)
    streamed_root: Path | None = None
    if args.streamed_dir:
        streamed_root = Path(args.streamed_dir)
    elif args.cache_dir:
        streamed_root = Path(args.cache_dir) / "streamed"
    if streamed_root:
        status = "found" if streamed_root.exists() else "not present (will stream from HF)"
        print(f"[materialize] streamed cache: {streamed_root}  [{status}]")

    cache_dir = Path(args.cache_dir) if args.cache_dir else None

    total = 0
    for src in enabled_sources:
        n = materialize_source(src, output_dir, args.max_rows_per_source,
                               args.workers, streamed_root, cache_dir)
        if n > 0:
            total += n

    print(f"[ok] materialization complete. total rows ≈ {total:,}")


if __name__ == "__main__":
    main()
