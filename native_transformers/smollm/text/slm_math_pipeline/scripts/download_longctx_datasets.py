#!/usr/bin/env python3
"""
Pre-download HuggingFace datasets needed for long-context and mid-training stages.

These are NOT in download_datasets.py (which covers base pretrain only).
Run this before starting any longctx or midtrain stage.

Usage:
    HF_TOKEN=hf_xxx python scripts/download_longctx_datasets.py --cache_dir /data/hf_cache
    python scripts/download_longctx_datasets.py --dry_run
    python scripts/download_longctx_datasets.py --stages longctx --cache_dir /data/hf_cache
    python scripts/download_longctx_datasets.py --stages midtrain --cache_dir /data/hf_cache
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass

# ── Dataset registry ──────────────────────────────────────────────────────────
# hf_dataset, subset, split, ~size_gb, used_in, notes

@dataclass
class Source:
    source_id: str
    hf_dataset: str
    subset: str | None
    split: str
    approx_gb: float
    stages: list[str]        # which stages need this: longctx, midtrain
    gated: bool = False
    notes: str = ""


SOURCES: list[Source] = [
    # ── Long-context: available on HF ─────────────────────────────────────────
    Source(
        source_id="pes2o",
        hf_dataset="allenai/peS2o",
        subset=None,
        split="train",
        approx_gb=308,
        stages=["longctx", "midtrain"],
        notes="Long academic papers (EN); already in base pretrain but reused for long-ctx",
    ),
    Source(
        source_id="pg19",
        hf_dataset="deepmind/pg19",
        subset=None,
        split="train",
        approx_gb=11,
        stages=["longctx"],
        notes="Project Gutenberg books (EN); long documents up to 100k+ tokens",
    ),
    Source(
        source_id="wikipedia_vi_longctx",
        hf_dataset="wikimedia/wikipedia",
        subset="20231101.vi",
        split="train",
        approx_gb=1,
        stages=["longctx"],
        notes="Vietnamese Wikipedia — same as base but tokenized at 16k/32k seq_len",
    ),
    Source(
        source_id="wikipedia_en_longctx",
        hf_dataset="wikimedia/wikipedia",
        subset="20231101.en",
        split="train[:10%]",
        approx_gb=4,
        stages=["longctx"],
        notes="English Wikipedia subset for long-context packing",
    ),
    # ── Mid-training: math + science heavy ───────────────────────────────────
    Source(
        source_id="finemath_4plus",
        hf_dataset="HuggingFaceTB/finemath",
        subset="finemath-4plus",
        split="train",
        approx_gb=40,
        stages=["midtrain"],
        notes="Best EN math web content (quality ≥4)",
    ),
    Source(
        source_id="finemath_3plus",
        hf_dataset="HuggingFaceTB/finemath",
        subset="finemath-3plus",
        split="train[:50%]",
        approx_gb=35,
        stages=["midtrain"],
        notes="Broader EN math content (quality ≥3)",
    ),
    Source(
        source_id="open_web_math",
        hf_dataset="open-web-math/open-web-math",
        subset=None,
        split="train[:60%]",
        approx_gb=25,
        stages=["midtrain"],
        notes="EN math web corpus",
    ),
    Source(
        source_id="ultradata_math",
        hf_dataset="openbmb/UltraData-Math",
        subset=None,
        split="train",
        approx_gb=15,
        stages=["midtrain"],
        gated=True,
        notes="High-quality math SFT data (requires HF token)",
    ),
    Source(
        source_id="fineweb2_hq_vi_midtrain",
        hf_dataset="HuggingFaceFW/fineweb-2",
        subset="vie_Latn",
        split="train[:40%]",
        approx_gb=30,
        stages=["midtrain"],
        notes="Best VI web content for mid-training",
    ),
]

STAGE_LABELS = {
    "longctx": "Long-context stages (16k / 32k / 64k / 128k)",
    "midtrain": "Mid-training stage (math/science strengthening)",
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pre-download long-context and mid-training datasets."
    )
    parser.add_argument("--cache_dir", default=None,
                        help="HF cache directory (default: HF_HOME or ~/.cache/huggingface)")
    parser.add_argument("--hf_token", default=os.environ.get("HF_TOKEN"),
                        help="HuggingFace token for gated datasets (or set HF_TOKEN)")
    parser.add_argument("--stages", nargs="+", default=["longctx", "midtrain"],
                        choices=["longctx", "midtrain"],
                        help="Which stage datasets to download (default: both)")
    parser.add_argument("--source_ids", nargs="+", default=None,
                        help="Download only specific source_ids")
    parser.add_argument("--dry_run", action="store_true",
                        help="Print what would be downloaded without downloading")
    args = parser.parse_args()

    # Filter to requested stages and source_ids
    sources = [
        s for s in SOURCES
        if any(st in args.stages for st in s.stages)
        and (args.source_ids is None or s.source_id in args.source_ids)
    ]

    total_gb = sum(s.approx_gb for s in sources)
    print(f"\n{'DRY RUN — ' if args.dry_run else ''}Downloading {len(sources)} sources "
          f"(~{total_gb:.0f} GB total)\n")

    for stage in args.stages:
        stage_sources = [s for s in sources if stage in s.stages]
        if not stage_sources:
            continue
        print(f"── {STAGE_LABELS.get(stage, stage)} ──")
        for s in stage_sources:
            gated_note = "  [GATED — needs HF_TOKEN]" if s.gated else ""
            print(f"  {s.source_id:<30} ~{s.approx_gb:>5.0f} GB  {s.notes}{gated_note}")
        print()

    if args.dry_run:
        print("Dry run complete. Remove --dry_run to download.")
        return

    if not args.hf_token:
        gated = [s.source_id for s in sources if s.gated]
        if gated:
            print(f"[warn] No HF_TOKEN set — gated datasets will fail: {gated}")
            print("       Set HF_TOKEN env var or pass --hf_token hf_xxx\n")

    from datasets import load_dataset
    from pathlib import Path

    cache_dir = str(Path(args.cache_dir).expanduser()) if args.cache_dir else None

    results = []
    for src in sources:
        print(f"\n[{src.source_id}] {src.hf_dataset}"
              f"{f' ({src.subset})' if src.subset else ''} split={src.split}")
        print(f"  ~{src.approx_gb:.0f} GB  |  {src.notes}")

        load_kwargs: dict = dict(
            path=src.hf_dataset,
            split=src.split,
            streaming=True,
            token=args.hf_token,
        )
        if src.subset:
            load_kwargs["name"] = src.subset
        if cache_dir:
            load_kwargs["cache_dir"] = cache_dir

        try:
            ds = load_dataset(**load_kwargs)
            # Iterate one batch to trigger download
            batch = list(ds.take(1))
            if batch:
                print(f"  [ok] downloaded and verified (first record fields: "
                      f"{list(batch[0].keys())})")
                results.append((src.source_id, "ok"))
            else:
                print(f"  [warn] dataset appears empty")
                results.append((src.source_id, "empty"))
        except Exception as e:
            print(f"  [error] {e}")
            results.append((src.source_id, f"error: {e}"))

    # Summary
    print("\n" + "─" * 60)
    print("Download summary:")
    ok = [r for r in results if r[1] == "ok"]
    failed = [r for r in results if r[1] != "ok"]
    for sid, status in results:
        icon = "✓" if status == "ok" else "✗"
        print(f"  {icon} {sid:<30} {status}")
    print(f"\n{len(ok)}/{len(results)} succeeded")
    if failed:
        print(f"Failed: {[r[0] for r in failed]}")


if __name__ == "__main__":
    main()
