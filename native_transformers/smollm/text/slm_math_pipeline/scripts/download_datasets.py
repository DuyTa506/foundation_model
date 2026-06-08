#!/usr/bin/env python3
"""
Pre-download all HuggingFace datasets declared in curation_pipeline.yaml
to a local cache directory before running the curation pipeline.

Usage:
    python scripts/download_datasets.py
    python scripts/download_datasets.py --config configs/curation_pipeline.yaml
    python scripts/download_datasets.py --cache_dir /data/hf_cache --hf_token hf_xxx
    python scripts/download_datasets.py --dry_run        # list what would be downloaded
    python scripts/download_datasets.py --source_ids vi_hq_web vi_wikipedia  # subset only

Why run this first:
    00_materialize.py streams from HF on-the-fly, which breaks on slow/interrupted
    connections mid-curation. Pre-downloading caches every shard locally so the
    curation pipeline reads from disk at full speed with no network dependency.

Outputs:
    <cache_dir>/<dataset_id>/          HF dataset cache (arrow format)
    <cache_dir>/download_report.json   per-source status, size, row counts
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import yaml


# Sizes of the actual subset being pulled (after name/subset filtering, before split slicing).
# split slicing (e.g. train[:35%]) is applied on top in estimate_size_gb().
APPROX_SIZES_GB = {
    # Vietnamese
    "epfml/FineWeb2-HQ": 15,               # vie_Latn subset
    "uonlp/CulturaX": 8,                   # vi subset
    "VTSNLP/vietnamese_curated_dataset": 2,
    "wikimedia/wikipedia": 0.5,            # vi subset
    "Symato/c4_vi-filtered_200GB": 200,    # full dataset; split-sliced in config
    "Symato/madlad-400_vi": 8,
    "Symato/hplt-vi": 6,
    # English math+science
    "HuggingFaceTB/finemath": 25,          # finemath-4plus subset
    "open-web-math/open-web-math": 25,
    "openbmb/UltraData-Math": 5,           # L2-preview subset
    "HuggingFaceFW/fineweb-edu": 25,       # sample-10BT subset
    "allenai/peS2o": 40,
}


def parse_split_fraction(split: str) -> float:
    """Return the fraction of data selected by a split string.
    'train' -> 1.0,  'train[:35%]' -> 0.35,  'train[10%:50%]' -> 0.40
    """
    import re
    # [:N%]
    m = re.search(r'\[:(\d+(?:\.\d+)?)%\]', split)
    if m:
        return float(m.group(1)) / 100.0
    # [N%:M%]
    m = re.search(r'\[(\d+(?:\.\d+)?)%:(\d+(?:\.\d+)?)%\]', split)
    if m:
        return (float(m.group(2)) - float(m.group(1))) / 100.0
    return 1.0


def estimate_size_gb(hf_dataset: str, split: str) -> float | None:
    base = APPROX_SIZES_GB.get(hf_dataset)
    if base is None:
        return None
    return base * parse_split_fraction(split)


def parse_sources(cfg: dict) -> list[dict]:
    """Extract all non-null HF dataset sources from curation config."""
    sources = []
    for src in cfg.get("sources", []):
        hf = src.get("hf_dataset")
        if not hf:
            continue
        sources.append({
            "source_id": src.get("id", hf.replace("/", "__")),
            "hf_dataset": hf,
            "subset": src.get("subset"),
            "split": src.get("split", "train"),
            "field": src.get("text_field", "text"),
            "max_samples": src.get("max_samples"),
            "language": src.get("language", "?"),
            "weight": src.get("weight", 0),
        })
    return sources


def download_source(src: dict, cache_dir: Path, hf_token: str | None) -> dict:
    """Download one dataset source and return a status dict."""
    from datasets import load_dataset

    hf_dataset = src["hf_dataset"]
    subset = src.get("subset")
    split = src.get("split", "train")
    max_samples = src.get("max_samples")

    print(f"\n[{src['source_id']}] {hf_dataset}"
          f"{f' ({subset})' if subset else ''} split={split}")
    approx = estimate_size_gb(hf_dataset, split)
    if approx:
        print(f"  estimated size: ~{approx:.1f} GB")

    t0 = time.time()
    status = {"source_id": src["source_id"], "hf_dataset": hf_dataset,
              "subset": subset, "split": split}
    try:
        load_kwargs = dict(
            path=hf_dataset,
            split=split,
            cache_dir=str(cache_dir),
            token=hf_token,
            streaming=False,   # download to disk; streaming = no cache
        )
        if subset:
            load_kwargs["name"] = subset

        ds = load_dataset(**load_kwargs)

        # Optionally slice to max_samples (still downloads full shard set first)
        row_count = len(ds)
        if max_samples and row_count > max_samples:
            ds = ds.select(range(max_samples))
            print(f"  sliced to {max_samples:,} / {row_count:,} rows")

        elapsed = time.time() - t0
        status.update({"status": "ok", "rows": len(ds), "elapsed_s": round(elapsed, 1)})
        print(f"  done: {len(ds):,} rows in {elapsed:.0f}s")
    except Exception as e:
        elapsed = time.time() - t0
        status.update({"status": "error", "error": str(e), "elapsed_s": round(elapsed, 1)})
        print(f"  ERROR: {e}")

    return status


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pre-download HF datasets for the curation pipeline."
    )
    parser.add_argument("--config", default="configs/curation_pipeline.yaml")
    parser.add_argument(
        "--cache_dir",
        default=None,
        help="Local directory to cache datasets. Defaults to HF_HOME or ~/.cache/huggingface.",
    )
    parser.add_argument("--hf_token", default=os.environ.get("HF_TOKEN"),
                        help="HuggingFace token (or set HF_TOKEN env var).")
    parser.add_argument(
        "--source_ids", nargs="+", default=None,
        help="Download only these source_ids (from config). Omit to download all.",
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Print what would be downloaded without actually downloading.",
    )
    parser.add_argument(
        "--skip_errors", action="store_true", default=True,
        help="Continue on per-source errors (default: true).",
    )
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    sources = parse_sources(cfg)
    if not sources:
        print("No HF dataset sources found in config. Nothing to download.")
        return

    if args.source_ids:
        sources = [s for s in sources if s["source_id"] in args.source_ids]
        if not sources:
            print(f"No sources matched {args.source_ids}. Available source_ids:")
            all_sources = parse_sources(cfg)
            for s in all_sources:
                print(f"  {s['source_id']}  ({s['hf_dataset']})")
            return

    cache_dir = Path(args.cache_dir) if args.cache_dir else None
    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"Sources to download: {len(sources)}")
    total_approx = sum(
        estimate_size_gb(s["hf_dataset"], s["split"]) or 0 for s in sources
    )
    print(f"Estimated total download: ~{total_approx:.0f} GB")
    print()

    col = max(len(s["source_id"]) for s in sources)
    for s in sources:
        subset_str = f" [{s['subset']}]" if s["subset"] else ""
        split_str = s["split"] if s["split"] != "train" else ""
        size = estimate_size_gb(s["hf_dataset"], s["split"])
        size_str = f"  ~{size:.1f} GB" if size else ""
        tag = "SKIP" if args.dry_run else "    "
        print(f"  {tag}  {s['source_id']:{col}s}  {s['hf_dataset']}{subset_str}"
              f"  {split_str}{size_str}")

    if args.dry_run:
        print("\n--dry_run: no downloads performed.")
        return

    if not args.hf_token:
        print(
            "\nWARN: no HF token set. Some datasets (openbmb/*, uonlp/CulturaX) "
            "may require authentication.\n"
            "Set HF_TOKEN env var or pass --hf_token hf_xxx\n"
        )

    try:
        import datasets  # noqa: F401
    except ImportError:
        raise SystemExit("Missing: pip install datasets")

    report = []
    failed = []
    t_total = time.time()

    for src in sources:
        result = download_source(src, cache_dir, args.hf_token)
        report.append(result)
        if result["status"] == "error":
            failed.append(src["source_id"])
            if not args.skip_errors:
                break

    elapsed_total = time.time() - t_total
    print(f"\n{'='*60}")
    print(f"Done: {len(sources) - len(failed)}/{len(sources)} sources downloaded "
          f"in {elapsed_total/60:.1f} min")
    if failed:
        print(f"Failed: {failed}")

    # Save report
    report_path = (cache_dir or Path(".")) / "download_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({"sources": report, "total_elapsed_s": round(elapsed_total, 1)},
                  f, indent=2, ensure_ascii=False)
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
