#!/usr/bin/env python3
"""
Build the WSD decay-phase ("anneal") tokenized shards from data you've ALREADY
curated — no re-download, no re-curation.

The MiniCPM recipe spends the last ~10% of pretraining (the LR-decay phase) on a
small, high-quality, VI-dominant + math mix. Because the LR is collapsing toward
its minimum, whatever the model sees last is what it locks in, so this stage
disproportionately shapes final math/VI quality.

Every curated document carries metadata.source (set at stage 00 and preserved
through stage 06), so we just:
  1. filter outputs/curated/pii_clean to the decay sources, then
  2. run the SAME stage-07 tokenizer on that subset → a separate shard dir.

The decay sources are read from curation_pipeline.yaml:decay_phase_mix.sources.

Usage:
    python scripts/data/build_decay_shards.py \
        --curation_config configs/curation_pipeline.yaml \
        --input_dir outputs/curated/pii_clean \
        --filtered_dir outputs/curated/decay_clean \
        --output_dir outputs/curated/tokenized_decay \
        --tokenizer_path outputs/tokenizer \
        --max_seq_length 4096

Then point the trainer at it (configs/training_8xH200_hf_pretrain.yaml):
    data:
      decay_shards_dir: outputs/curated/tokenized_decay
    scheduler:
      decay_phase_data_mix: true        # already the default
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import yaml


def filter_by_source(input_dir: str, filtered_dir: str, keep_sources: set[str], workers: int) -> None:
    """Stream pii_clean parquet → keep rows whose metadata.source is a decay source."""
    from datatrove.executor import LocalPipelineExecutor
    from datatrove.pipeline.readers import ParquetReader
    from datatrove.pipeline.writers import ParquetWriter
    from datatrove.pipeline.filters import LambdaFilter

    Path(filtered_dir).mkdir(parents=True, exist_ok=True)
    print(f"[decay] filtering {input_dir} → {filtered_dir}")
    print(f"[decay] keep sources: {sorted(keep_sources)}")

    executor = LocalPipelineExecutor(
        pipeline=[
            ParquetReader(data_folder=input_dir, glob_pattern="**/*.parquet", doc_progress=True),
            LambdaFilter(
                filter_function=lambda doc: (doc.metadata or {}).get("source") in keep_sources
            ),
            ParquetWriter(output_folder=filtered_dir, output_filename="${rank}.parquet"),
        ],
        tasks=workers,
        workers=workers,
        logging_dir=str(Path(filtered_dir) / "logs"),
        skip_completed=True,
    )
    executor.run()


def main() -> None:
    ap = argparse.ArgumentParser(description="Build WSD decay-phase tokenized shards.")
    ap.add_argument("--curation_config", default="configs/curation_pipeline.yaml")
    ap.add_argument("--input_dir", default="outputs/curated/pii_clean",
                    help="Final cleaned parquet from stage 06 (has metadata.source).")
    ap.add_argument("--filtered_dir", default="outputs/curated/decay_clean")
    ap.add_argument("--output_dir", default="outputs/curated/tokenized_decay")
    ap.add_argument("--tokenizer_path", default="outputs/tokenizer")
    ap.add_argument("--max_seq_length", type=int, default=4096)
    ap.add_argument("--workers", type=int, default=max(1, os.cpu_count() - 2))
    ap.add_argument("--sources", nargs="*", default=None,
                    help="Override decay sources (default: decay_phase_mix.sources in config).")
    ap.add_argument("--skip_filter", action="store_true", help="Reuse an existing --filtered_dir.")
    args = ap.parse_args()

    with open(args.curation_config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    keep_sources = set(args.sources or cfg.get("decay_phase_mix", {}).get("sources", []))
    if not keep_sources:
        sys.exit("No decay sources: set decay_phase_mix.sources in the curation config "
                 "or pass --sources.")

    # Sanity-check the requested sources actually exist in the pipeline.
    known = {s["id"] for s in cfg.get("sources", []) if isinstance(s, dict) and "id" in s}
    unknown = keep_sources - known
    if unknown:
        print(f"[decay] WARNING: these decay sources are not in sources[]: {sorted(unknown)}")

    if not args.skip_filter:
        filter_by_source(args.input_dir, args.filtered_dir, keep_sources, args.workers)
    else:
        print(f"[decay] --skip_filter: reusing {args.filtered_dir}")

    # Reuse the exact stage-07 tokenizer/packer (numeric module name → call as CLI).
    cmd = [
        sys.executable, "scripts/curate/07_tokenize_pack.py",
        "--config", args.curation_config,
        "--input_dir", args.filtered_dir,
        "--output_dir", args.output_dir,
        "--tokenizer_path", args.tokenizer_path,
        "--max_seq_length", str(args.max_seq_length),
    ]
    print(f"[decay] tokenizing: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    print(f"[ok] decay shards → {args.output_dir}")


if __name__ == "__main__":
    main()
