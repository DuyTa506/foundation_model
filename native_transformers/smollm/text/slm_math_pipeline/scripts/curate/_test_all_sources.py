#!/usr/bin/env python3
"""TEST-ONLY stage-0 materialize over EVERY enabled source, a tiny streamed slice each.

Purpose: flush out per-source schema surprises (e.g. a top-level string `metadata`
column, numeric extras) on real data BEFORE a full cluster run — cheaply, on a laptop.

Why not just run 00_materialize.py here: the production splits are huge
(`train[:20%]` of CulturaX etc.) and datatrove's HF reader defaults to
streaming=False, so load_dataset would DOWNLOAD the whole slice. This harness forces
streaming=True + a small `limit`, so only the first ~N rows are fetched per source.
It uses datasets.load_dataset directly because older datatrove versions can pass
world_size=0 into streaming HF sharding. It still writes through datatrove's
ParquetWriter with the SAME stable adapters 00_materialize uses, so it tests the
schema path that matters.

Usage:
    .venv/bin/python scripts/curate/_test_all_sources.py \
        --config configs/curation_pipeline.yaml \
        --output_dir outputs/_alltest/raw --limit 150
"""
from __future__ import annotations

import argparse
import os
import sys
import traceback
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _curate_utils import stable_metadata_adapter, stable_reader_adapter  # noqa: E402


def stream_one(src: dict, output_dir: Path, limit: int) -> tuple[str, bool, str]:
    from datasets import load_dataset
    from datatrove.data import Document
    from datatrove.pipeline.writers import ParquetWriter

    src_id = src["id"]
    hf_dataset = src.get("hf_dataset")
    if not hf_dataset:
        return src_id, True, "skipped (no hf_dataset)"

    text_field = src.get("text_field", "text")
    subset = src.get("subset")
    # Streaming does NOT support `train[:20%]` slice syntax — strip to the base split
    # name; `limit` bounds how many rows we actually pull.
    split = (src.get("split", "train") or "train").split("[")[0]
    language = src.get("language", "")

    dataset_options: dict = {"split": split}
    if subset:
        dataset_options["name"] = subset
    if src.get("revision"):
        dataset_options["revision"] = src["revision"]
    if os.environ.get("HF_TOKEN"):
        dataset_options["token"] = os.environ["HF_TOKEN"]

    ds = load_dataset(
        hf_dataset,
        **dataset_options,
        streaming=True,  # lazy-fetch; only `limit` rows are pulled, no full-split download
    )
    src_out = output_dir / src_id
    src_out.mkdir(parents=True, exist_ok=True)
    reader_adapter = stable_reader_adapter(
        keep_keys=("source", "dataset", "language"),
        defaults={"source": src_id, "dataset": hf_dataset, "language": language},
    )
    adapter_self = SimpleNamespace(text_key=text_field, id_key="id")

    def documents():
        for i, row in enumerate(ds):
            if i >= limit:
                break
            adapted = reader_adapter(adapter_self, dict(row), src_id, i)
            yield Document(**adapted)

    writer = ParquetWriter(
        output_folder=str(src_out),
        output_filename="${rank}.parquet",
        compression="snappy",
        adapter=stable_metadata_adapter(keep_keys=("source", "dataset", "language")),
    )
    deque(writer.run(documents(), rank=0, world_size=1), maxlen=0)

    import pyarrow.parquet as pq
    files = list(src_out.rglob("*.parquet"))
    rows = sum(pq.ParquetFile(str(p)).metadata.num_rows for p in files)
    # confirm uniform metadata schema
    schema = pq.ParquetFile(str(files[0])).schema_arrow.field("metadata").type if files else None
    return src_id, True, f"{rows} rows, metadata={schema}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/curation_pipeline.yaml")
    ap.add_argument("--output_dir", default="outputs/_alltest/raw")
    ap.add_argument("--limit", type=int, default=150)
    ap.add_argument("--source_ids", nargs="*")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    sources = [s for s in cfg.get("sources", [])
               if s.get("enabled", True) and s.get("hf_dataset")]
    if args.source_ids:
        sources = [s for s in sources if s["id"] in args.source_ids]

    print(f"[test] streaming {args.limit} rows/source over {len(sources)} sources\n")
    results = []
    for s in sources:
        sid = s["id"]
        print(f"[test] --- {sid} <- {s['hf_dataset']} ...", flush=True)
        try:
            results.append(stream_one(s, out, args.limit))
            print(f"[test]     OK  {results[-1][2]}", flush=True)
        except Exception as e:  # noqa: BLE001
            results.append((sid, False, f"{type(e).__name__}: {e}"))
            print(f"[test]     FAIL {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()

    print("\n==================== STAGE-0 SUMMARY ====================")
    ok = sum(1 for _, good, _ in results if good)
    for sid, good, msg in results:
        print(f"  {'PASS' if good else 'FAIL'}  {sid:24} {msg}")
    print(f"  {ok}/{len(results)} sources materialized")
    sys.exit(0 if ok == len(results) else 1)


if __name__ == "__main__":
    main()
