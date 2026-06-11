#!/usr/bin/env python3
"""
Build a small, SOURCE-STRATIFIED held-out validation shard from already-curated
data (outputs/curated/pii_clean) — for tracking `eval_loss` during pretraining.

Why stratified: a held-out val loss should mirror the TRAINING mix, so it's sampled
per source ∝ the corpus `weight`s in curation_pipeline.yaml. (Holding out one whole
.ds shard instead can be skewed toward whichever source-files happen to lead it, and
wastes ~1B train tokens to eval on ~8M — see README.)

> CAVEAT — overlap with train. These docs also live in outputs/curated/tokenized
> (train was tokenized from the same pii_clean), so this is a *monitoring* val, not a
> strict generalization val. For a single-epoch ~104B run the model sees each token
> ~once, so the optimism is mild. For a guaranteed NO-leakage val (but not source-
> stratified), carve from the .ds shards and trim them instead (README §Stage 2).

Pipeline: sample pii_clean by source ∝ weight → write one small parquet → tokenize
with the SAME stage-07 packer → outputs/curated/val/*.ds. Point the trainer at it via
`data.val_shards_dir`.

Usage:
    python scripts/data/build_val_shard.py \
      --curation_config configs/curation_pipeline.yaml \
      --input_dir outputs/curated/pii_clean \
      --val_clean_dir outputs/curated/val_clean \
      --output_dir outputs/curated/val \
      --tokenizer_path outputs/tokenizer \
      --val_tokens 16000000
"""

from __future__ import annotations

import argparse
import glob
import os
import random
import subprocess
import sys
from pathlib import Path

import yaml

# Rough EN/VI chars-per-token, used ONLY to size per-source budgets before tokenizing.
CHARS_PER_TOKEN = 4.0


def main() -> None:
    ap = argparse.ArgumentParser(description="Build a source-stratified held-out val shard.")
    ap.add_argument("--curation_config", default="configs/curation_pipeline.yaml")
    ap.add_argument("--input_dir", default="outputs/curated/pii_clean",
                    help="Final cleaned parquet (stage 06) with metadata.source.")
    ap.add_argument("--val_clean_dir", default="outputs/curated/val_clean")
    ap.add_argument("--output_dir", default="outputs/curated/val")
    ap.add_argument("--tokenizer_path", default="outputs/tokenizer")
    ap.add_argument("--max_seq_length", type=int, default=4096)
    ap.add_argument("--val_tokens", type=int, default=16_000_000,
                    help="Approx total val tokens (split across sources by weight).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip_tokenize", action="store_true",
                    help="Only sample+write val_clean parquet; skip the stage-07 tokenize.")
    args = ap.parse_args()

    import pyarrow as pa
    import pyarrow.parquet as pq

    cfg = yaml.safe_load(open(args.curation_config))
    # Budget only real, enabled, weighted sources (skip disabled/synthetic like vi_math_synth).
    weights = {
        s["id"]: float(s.get("weight", 0.0))
        for s in cfg.get("sources", [])
        if isinstance(s, dict) and s.get("id") and float(s.get("weight", 0.0)) > 0
        and s.get("enabled", True) and s.get("hf_dataset")
    }
    if not weights:
        sys.exit("No weighted sources found in curation config.")
    tot_w = sum(weights.values())
    char_budget = {sid: (w / tot_w) * args.val_tokens * CHARS_PER_TOKEN for sid, w in weights.items()}

    files = sorted(glob.glob(os.path.join(args.input_dir, "**", "*.parquet"), recursive=True))
    if not files:
        sys.exit(f"No parquet under {args.input_dir}")
    random.Random(args.seed).shuffle(files)  # spread the sample across the corpus

    kept_text: list[str] = []
    kept_meta: list[dict] = []
    chars = {sid: 0 for sid in char_budget}
    done: set[str] = set()

    for fp in files:
        if len(done) >= len(char_budget):
            break
        pf = pq.ParquetFile(fp)
        cols = [c for c in ("text", "metadata") if c in pf.schema_arrow.names]
        for batch in pf.iter_batches(columns=cols, batch_size=2048):
            d = batch.to_pydict()
            texts = d.get("text", [])
            metas = d.get("metadata", [{}] * len(texts))
            for text, meta in zip(texts, metas):
                src = (meta or {}).get("source")
                if not text or src not in char_budget or src in done:
                    continue
                kept_text.append(text)
                # write a NORMALIZED, uniform metadata struct so pyarrow infers one schema
                kept_meta.append({
                    "source": src,
                    "dataset": str((meta or {}).get("dataset", "")),
                    "language": str((meta or {}).get("language", "")),
                })
                chars[src] += len(text)
                if chars[src] >= char_budget[src]:
                    done.add(src)
            if len(done) >= len(char_budget):
                break

    if not kept_text:
        sys.exit(f"Sampled 0 docs — does {args.input_dir} have metadata.source?")

    # shuffle the kept docs so the val .ds isn't grouped by source (eval reads the head)
    order = list(range(len(kept_text)))
    random.Random(args.seed + 1).shuffle(order)
    kept_text = [kept_text[i] for i in order]
    kept_meta = [kept_meta[i] for i in order]
    kept_id = [f"val-{i:08d}" for i in range(len(kept_text))]

    Path(args.val_clean_dir).mkdir(parents=True, exist_ok=True)
    table = pa.table({"text": kept_text, "id": kept_id, "metadata": kept_meta})
    pq.write_table(table, os.path.join(args.val_clean_dir, "val.parquet"))

    est_tok = sum(chars.values()) / CHARS_PER_TOKEN
    print(f"[val] sampled {len(kept_text)} docs across {len(chars)} sources "
          f"(~{est_tok/1e6:.1f}M tok est) → {args.val_clean_dir}")
    for sid in sorted(chars, key=lambda s: -chars[s]):
        filled = "full" if sid in done else "PARTIAL (source too small)"
        print(f"    {sid:18s} ~{chars[sid]/CHARS_PER_TOKEN/1e6:5.2f}M tok  {filled}")

    if args.skip_tokenize:
        print("[val] --skip_tokenize: wrote val_clean parquet only.")
        return

    cmd = [
        sys.executable, "scripts/curate/07_tokenize_pack.py",
        "--config", args.curation_config,
        "--input_dir", args.val_clean_dir,
        "--output_dir", args.output_dir,
        "--tokenizer_path", args.tokenizer_path,
        "--max_seq_length", str(args.max_seq_length),
    ]
    print(f"[val] tokenizing: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    print(f"[ok] val shard → {args.output_dir}  (set data.val_shards_dir to this)")


if __name__ == "__main__":
    main()
