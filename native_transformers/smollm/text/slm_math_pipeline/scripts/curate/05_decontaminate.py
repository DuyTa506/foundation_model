#!/usr/bin/env python3
"""
Stage 5: Decontamination — remove training docs that overlap eval benchmarks.

Implements the 13-gram overlap check that was declared in curation_pipeline.yaml
but never implemented in the old pipeline.

Strategy: build an n-gram index from all eval splits, then drop any training
document that contains any matching n-gram.

Usage:
    python scripts/curate/05_decontaminate.py \
        --config configs/curation_pipeline.yaml \
        --input_dir outputs/curated/ultraclean \
        --output_dir outputs/curated/decontaminated
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, FrozenSet

import yaml

from _curate_utils import prune_empty_parquet, stable_metadata_adapter

TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def _ngrams(text: str, n: int) -> FrozenSet[str]:
    tokens = TOKEN_RE.findall(text.lower())
    if len(tokens) < n:
        return frozenset()
    return frozenset(" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def _iter_strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for v in value.values():
            yield from _iter_strings(v)
    elif isinstance(value, (list, tuple)):
        for v in value:
            yield from _iter_strings(v)
    elif value is not None and not isinstance(value, (int, float, bool)):
        yield json.dumps(value, ensure_ascii=False)


def _benchmark_spec(raw: str | dict[str, Any]) -> dict[str, Any]:
    if isinstance(raw, str):
        hf_id = raw.split(":", 1)[-1] if raw.startswith("harness:") else raw
        return {
            "name": raw,
            "hf_dataset": hf_id,
            "subset": None,
            "splits": ["test", "validation", "dev"],
            "optional": False,
            "streaming": False,
        }
    if isinstance(raw, dict):
        hf_id = raw.get("hf_dataset") or raw.get("dataset")
        if not hf_id:
            raise ValueError(f"benchmark entry is missing hf_dataset: {raw}")
        splits = raw.get("splits") or raw.get("split") or ["test", "validation", "dev"]
        if isinstance(splits, str):
            splits = [splits]
        return {
            "name": raw.get("name") or hf_id,
            "hf_dataset": hf_id,
            "subset": raw.get("subset") or raw.get("config"),
            "splits": list(splits),
            "optional": bool(raw.get("optional", False)),
            "streaming": bool(raw.get("streaming", False)),
        }
    raise TypeError(f"unsupported benchmark entry: {raw!r}")


def _load_eval_ngrams(
    benchmark_sets: list[str | dict[str, Any]],
    ngram_size: int,
    allow_missing: bool = False,
) -> FrozenSet[str]:
    """Download eval splits and build their n-gram fingerprint."""
    from datasets import load_dataset

    all_ngrams: set[str] = set()
    missing: list[str] = []
    optional_missing: list[str] = []
    for raw_bench in benchmark_sets:
        bench = _benchmark_spec(raw_bench)
        name = bench["name"]
        hf_id = bench["hf_dataset"]
        subset = bench["subset"]
        splits_to_check = bench["splits"]
        indexed = False
        last_error = None

        for split in splits_to_check:
            try:
                kwargs = {"split": split}
                if bench["streaming"]:
                    kwargs["streaming"] = True
                if subset:
                    ds = load_dataset(hf_id, subset, **kwargs)
                else:
                    ds = load_dataset(hf_id, **kwargs)
                before = len(all_ngrams)
                rows = 0
                for row in ds:
                    rows += 1
                    for text in _iter_strings(row):
                        all_ngrams.update(_ngrams(text, ngram_size))
                added = len(all_ngrams) - before
                print(f"[decontam] indexed {name} {split}: "
                      f"{rows:,} rows, +{added:,} n-grams")
                indexed = True
                break
            except Exception as e:  # noqa: BLE001 - aggregate all split failures
                last_error = e
        if not indexed:
            msg = f"{name} ({type(last_error).__name__}: {last_error})"
            if bench["optional"]:
                optional_missing.append(msg)
            else:
                missing.append(msg)
            print(f"[decontam] missing {msg}")

    if missing and not allow_missing:
        raise RuntimeError(
            "Could not load one or more decontamination benchmarks: "
            + "; ".join(missing)
            + ". Pass --allow_missing_benchmarks for local smoke runs only."
        )
    if optional_missing:
        print("[decontam] optional benchmarks unavailable: " + "; ".join(optional_missing))

    print(f"[decontam] total eval n-grams: {len(all_ngrams):,}")
    return frozenset(all_ngrams)


def decontaminate(
    cfg: dict,
    input_dir: str,
    output_dir: str,
    workers: int,
    allow_missing_benchmarks: bool = False,
) -> None:
    from datatrove.executor import LocalPipelineExecutor
    from datatrove.pipeline.filters import LambdaFilter
    from datatrove.pipeline.readers import ParquetReader
    from datatrove.pipeline.writers import ParquetWriter

    dc_cfg: dict = cfg.get("decontamination", {})
    ngram_size: int = dc_cfg.get("ngram_size", 13)
    benchmark_sets: list[str | dict[str, Any]] = dc_cfg.get("benchmark_sets", [])
    allow_missing = allow_missing_benchmarks or dc_cfg.get("allow_missing_benchmarks", False)

    print(f"[decontam] building eval n-gram index (ngram={ngram_size}) ...")
    eval_ngrams = _load_eval_ngrams(benchmark_sets, ngram_size, allow_missing)

    if not eval_ngrams:
        if not allow_missing:
            raise RuntimeError("No eval n-grams loaded; refusing to skip decontamination")
        print("[warn] no eval n-grams loaded; copying input for local smoke run")
        import shutil
        shutil.copytree(input_dir, output_dir, dirs_exist_ok=True)
        return

    def _is_clean(doc) -> bool:
        text: str = doc.text or ""
        doc_ngrams = _ngrams(text, ngram_size)
        contaminated = bool(doc_ngrams & eval_ngrams)
        if contaminated:
            doc.metadata["decontam_dropped"] = True
        return not contaminated

    executor = LocalPipelineExecutor(
        pipeline=[
            ParquetReader(data_folder=input_dir, glob_pattern="**/*.parquet",
                          doc_progress=True),
            LambdaFilter(filter_function=_is_clean),
            # Uniform metadata struct regardless of the (dropped-only) decontam_dropped flag.
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
    executor.run()


def main() -> None:
    parser = argparse.ArgumentParser(description="N-gram decontamination vs eval sets.")
    parser.add_argument("--config", default="configs/curation_pipeline.yaml")
    parser.add_argument("--input_dir", default="outputs/curated/ultraclean")
    parser.add_argument("--output_dir", default="outputs/curated/decontaminated")
    parser.add_argument("--workers", type=int, default=max(1, os.cpu_count() - 2))
    parser.add_argument("--allow_missing_benchmarks", action="store_true",
                        help="Smoke/local mode only: skip decontamination if eval sets "
                             "cannot be loaded.")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    prune_empty_parquet(args.input_dir)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    decontaminate(
        cfg,
        args.input_dir,
        args.output_dir,
        args.workers,
        allow_missing_benchmarks=args.allow_missing_benchmarks,
    )
    print(f"[ok] decontamination done -> {args.output_dir}")


if __name__ == "__main__":
    main()
