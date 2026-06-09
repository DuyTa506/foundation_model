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
    python scripts/download_datasets.py --force          # re-pull even if already done

Why streaming (and why this matters for disk):
    A sliced split like `train[:5%]` with `load_dataset(streaming=False)` downloads
    the ENTIRE dataset to disk first, then slices in memory. peS2o `train[:5%]`
    still pulls all 308 GB. Summed across sources that ballooned the cache to ~1.2 TB.

    Worse: the curation/tokenizer scripts read with `streaming=True`, which uses the
    raw parquet files — NOT the arrow tables that `streaming=False` builds. So the
    full arrow download was both too big AND in the wrong format.

    This script now STREAMS each source and writes only the needed fraction as local
    parquet shards under `<cache_dir>/streamed/<source_id>/`. A `train[:5%]` source
    pulls ~5% of its bytes, not 100%. A `.done` marker makes re-runs idempotent —
    nothing is re-downloaded if it completed before (override with --force).

Downstream readers consume the shards directly:
    load_dataset("parquet", data_files="<cache_dir>/streamed/<source_id>/*.parquet",
                 streaming=True)

Outputs:
    <cache_dir>/streamed/<source_id>/*.parquet   per-source token text shards
    <cache_dir>/streamed/<source_id>/.done       completion marker (size, rows)
    <cache_dir>/download_report.json             per-source status, size, row counts
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path

import yaml


# Sizes of the actual subset being pulled (after name/subset filtering, before split slicing).
# split slicing (e.g. train[:35%]) is applied on top in estimate_size_gb().
APPROX_SIZES_GB = {
    # Vietnamese — measured from HF repo (subset size, before split slicing)
    "epfml/FineWeb2-HQ": 63,               # vie_Latn subset (measured)
    "uonlp/CulturaX": 144,                 # vi subset (measured)
    "VTSNLP/vietnamese_curated_dataset": 35,  # measured
    "wikimedia/wikipedia": 0.7,            # 20231101.vi subset (measured)
    "Symato/c4_vi-filtered_200GB": 47,     # measured (name refers to raw uncompressed)
    "Symato/madlad-400_vi": 62,            # measured
    "Symato/hplt-vi": 92,                  # measured
    # English math+science — measured from HF repo (subset size, before split slicing)
    "HuggingFaceTB/finemath": 38,          # finemath-4plus subset (measured)
    "open-web-math/open-web-math": 27,     # measured
    "openbmb/UltraData-Math": 63,          # L2-preview subset (measured)
    "HuggingFaceFW/fineweb-edu": 28,       # sample-10BT subset (measured)
    "allenai/peS2o": 308,                  # measured
}

SHARD_FLUSH_ROWS = 10_000          # rows buffered before writing a parquet shard
FULL_FRACTION = 0.999              # frac >= this → treat as "download everything"


def parse_split_fraction(split: str) -> float:
    """Return the fraction of data selected by a split string.
    'train' -> 1.0,  'train[:35%]' -> 0.35,  'train[10%:50%]' -> 0.40
    """
    # [:N%]
    m = re.search(r'\[:(\d+(?:\.\d+)?)%\]', split)
    if m:
        return float(m.group(1)) / 100.0
    # [N%:M%]
    m = re.search(r'\[(\d+(?:\.\d+)?)%:(\d+(?:\.\d+)?)%\]', split)
    if m:
        return (float(m.group(2)) - float(m.group(1))) / 100.0
    return 1.0


def base_split(split: str) -> str:
    """Strip the [...] slice — streaming does not support percentage slicing."""
    return re.sub(r'\[.*?\]', '', split).strip() or "train"


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


def _existing_bytes(out_dir: Path) -> int:
    return sum(p.stat().st_size for p in out_dir.glob("*.parquet"))


def _stream_to_parquet(src: dict, out_dir: Path, target_bytes: float,
                       cache_dir: Path | None, hf_token: str | None) -> tuple[int, int]:
    """Stream a HF source and write parquet shards until `target_bytes` are written.
    Only the consumed fraction is downloaded. Returns (rows_written, bytes_written)."""
    from datasets import load_dataset
    import pyarrow as pa
    import pyarrow.parquet as pq

    field = src.get("field", "text")
    max_samples = src.get("max_samples")

    load_kwargs: dict = dict(
        path=src["hf_dataset"],
        split=base_split(src.get("split", "train")),
        streaming=True,
        token=hf_token,
    )
    if src.get("subset"):
        load_kwargs["name"] = src["subset"]
    if cache_dir:
        load_kwargs["cache_dir"] = str(cache_dir)

    ds = load_dataset(**load_kwargs)

    out_dir.mkdir(parents=True, exist_ok=True)
    # Fresh start: clear any partial shards from an interrupted run
    for p in out_dir.glob("*.parquet"):
        p.unlink()

    rows_written = 0
    bytes_written = 0
    shard_idx = 0
    buf: list[str] = []

    def flush() -> None:
        nonlocal shard_idx, bytes_written, buf
        if not buf:
            return
        # Normalize column name to "text" so downstream ParquetReader(text_key="text")
        # works uniformly regardless of each source's original text_field.
        table = pa.table({"text": buf})
        path = out_dir / f"{shard_idx:04d}.parquet"
        pq.write_table(table, path, compression="zstd")
        bytes_written += path.stat().st_size
        shard_idx += 1
        buf = []

    for row in ds:
        text = row.get(field) or row.get("text") or row.get("content") or ""
        if not isinstance(text, str) or not text:
            continue
        buf.append(text)
        rows_written += 1

        if len(buf) >= SHARD_FLUSH_ROWS:
            flush()
            if bytes_written >= target_bytes:
                break
            if max_samples and rows_written >= max_samples:
                break

    flush()  # remaining tail
    return rows_written, bytes_written


def download_source(src: dict, out_root: Path, cache_dir: Path | None,
                    hf_token: str | None, force: bool) -> dict:
    """Stream one dataset source to local parquet shards. Returns a status dict."""
    hf_dataset = src["hf_dataset"]
    subset = src.get("subset")
    split = src.get("split", "train")
    frac = parse_split_fraction(split)
    target_gb = estimate_size_gb(hf_dataset, split)

    out_dir = out_root / src["source_id"]
    done_marker = out_dir / ".done"

    print(f"\n[{src['source_id']}] {hf_dataset}"
          f"{f' ({subset})' if subset else ''} split={split}")
    if target_gb:
        print(f"  target footprint: ~{target_gb:.1f} GB"
              f"  ({frac*100:.0f}% of {APPROX_SIZES_GB[hf_dataset]:.0f} GB full)")

    status = {"source_id": src["source_id"], "hf_dataset": hf_dataset,
              "subset": subset, "split": split}

    # Idempotent skip: already completed and not forcing
    if done_marker.exists() and not force:
        meta = json.loads(done_marker.read_text())
        print(f"  skip: already done ({meta.get('rows', '?'):,} rows, "
              f"{meta.get('gb', '?')} GB). Use --force to re-pull.")
        status.update({"status": "skipped", **meta})
        return status

    # Byte budget: full sources stream entirely; sliced sources stop at the fraction.
    if frac >= FULL_FRACTION:
        target_bytes: float = float("inf")
    elif target_gb:
        target_bytes = target_gb * 1e9
    else:
        target_bytes = float("inf")  # unknown size → cannot cap; pull all
        print("  [warn] no size estimate for this dataset — cannot cap by bytes; "
              "streaming full split")

    t0 = time.time()
    try:
        rows, wbytes = _stream_to_parquet(src, out_dir, target_bytes, cache_dir, hf_token)
        elapsed = time.time() - t0
        gb = round(wbytes / 1e9, 2)
        meta = {"rows": rows, "gb": gb, "elapsed_s": round(elapsed, 1)}
        done_marker.write_text(json.dumps(meta))
        status.update({"status": "ok", **meta})
        print(f"  done: {rows:,} rows, {gb:.2f} GB in {elapsed:.0f}s → {out_dir}")
    except Exception as e:
        elapsed = time.time() - t0
        status.update({"status": "error", "error": str(e), "elapsed_s": round(elapsed, 1)})
        print(f"  ERROR: {e}")

    return status


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pre-download HF datasets for the curation pipeline (streaming, fraction-aware)."
    )
    parser.add_argument("--config", default="configs/curation_pipeline.yaml")
    parser.add_argument(
        "--cache_dir",
        default=None,
        help="HF streaming cache dir. Parquet shards land in <cache_dir>/streamed/. "
             "Defaults to HF_HOME or ~/.cache/huggingface.",
    )
    parser.add_argument(
        "--output_dir", default=None,
        help="Where to write parquet shards (default: <cache_dir>/streamed).",
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
        "--force", action="store_true",
        help="Re-download even if a source's .done marker already exists.",
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
            for s in parse_sources(cfg):
                print(f"  {s['source_id']}  ({s['hf_dataset']})")
            return

    cache_dir = Path(args.cache_dir) if args.cache_dir else None
    if args.output_dir:
        out_root = Path(args.output_dir)
    elif cache_dir:
        out_root = cache_dir / "streamed"
    else:
        out_root = Path("outputs/curated/streamed")
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"Sources to download: {len(sources)}")
    total_approx = sum(
        estimate_size_gb(s["hf_dataset"], s["split"]) or 0 for s in sources
    )
    print(f"Estimated total footprint (streamed, fraction-aware): ~{total_approx:.0f} GB")
    print(f"Output: {out_root}")
    print()

    col = max(len(s["source_id"]) for s in sources)
    for s in sources:
        subset_str = f" [{s['subset']}]" if s["subset"] else ""
        split_str = s["split"] if s["split"] != "train" else ""
        size = estimate_size_gb(s["hf_dataset"], s["split"])
        size_str = f"  ~{size:.1f} GB" if size else "  (size unknown)"
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
        import pyarrow  # noqa: F401
    except ImportError:
        raise SystemExit("Missing deps: pip install datasets pyarrow")

    report = []
    failed = []
    t_total = time.time()

    for src in sources:
        result = download_source(src, out_root, cache_dir, args.hf_token, args.force)
        report.append(result)
        if result["status"] == "error":
            failed.append(src["source_id"])
            if not args.skip_errors:
                break

    elapsed_total = time.time() - t_total
    ok = [r for r in report if r["status"] in ("ok", "skipped")]
    total_gb = sum(r.get("gb", 0) for r in report if isinstance(r.get("gb"), (int, float)))
    print(f"\n{'='*60}")
    print(f"Done: {len(ok)}/{len(sources)} sources ready "
          f"({total_gb:.1f} GB on disk) in {elapsed_total/60:.1f} min")
    if failed:
        print(f"Failed: {failed}")

    # Save report
    report_path = out_root / "download_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({"sources": report, "total_gb": round(total_gb, 1),
                   "total_elapsed_s": round(elapsed_total, 1)},
                  f, indent=2, ensure_ascii=False)
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
