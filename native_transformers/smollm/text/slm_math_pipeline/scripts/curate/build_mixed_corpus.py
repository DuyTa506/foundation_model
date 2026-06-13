#!/usr/bin/env python3
"""
Stage 6.5: enforce the target token MIXTURE + fix source attribution.

Two problems this fixes, both on the already-cleaned `pii_clean`:
  1. The per-source `weight:` in curation_pipeline.yaml is never applied — the mix is
     "whatever survived filtering". After the (fixed) VI quality filter, the corpus is
     VI-dominant AND fineweb_edu is over-represented. This caps each source to
     `weight × target_tokens` (on deduped pii_clean = UNIQUE tokens), so the final mix
     is exactly the configured VI/EN split and no source dominates.
  2. `metadata.source` is blank on disk (only `dataset` survived). We re-stamp
     `source` from `dataset` via the config's hf_dataset→id map, so downstream
     per-source tooling (07 cap, build_val_shard, build_decay_shards) works.

Cap-at-the-END (not download): over-provision upstream, let filter/dedup cut, then trim
the survivors here to the exact budget. A source that can't fill its budget keeps all it
has (logged) — bump its download fraction or loosen its filter if that matters.

Run AFTER 06_pii_redact, BEFORE 07_tokenize_pack:
    python scripts/curate/build_mixed_corpus.py \
        --config configs/curation_pipeline.yaml \
        --input_dir outputs/curated/pii_clean \
        --output_dir outputs/curated/mixed \
        --target_tokens 50e9
    python scripts/curate/07_tokenize_pack.py --input_dir outputs/curated/mixed \
        --output_dir outputs/curated/tokenized --tokenizer_path outputs/tokenizer
"""

from __future__ import annotations

import argparse
import glob
import os
import random

import yaml


def main() -> None:
    ap = argparse.ArgumentParser(description="Enforce target token mixture + fix source attribution.")
    ap.add_argument("--config", default="configs/curation_pipeline.yaml")
    ap.add_argument("--input_dir", default="outputs/curated/pii_clean")
    ap.add_argument("--output_dir", default="outputs/curated/mixed")
    ap.add_argument("--target_tokens", type=float, default=50e9,
                    help="Total unique-token budget; per-source cap = weight×this.")
    ap.add_argument("--chars_per_token", type=float, default=3.8,
                    help="Char→token estimate for budgeting (final exact tokens come from stage 07).")
    ap.add_argument("--docs_per_shard", type=int, default=100_000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    import pyarrow as pa
    import pyarrow.parquet as pq

    cfg = yaml.safe_load(open(args.config))
    # hf_dataset → src_id, and src_id → weight (only real, weighted, enabled sources)
    ds2id, weight = {}, {}
    for s in cfg.get("sources", []):
        if not (isinstance(s, dict) and s.get("id") and s.get("hf_dataset")):
            continue
        if not s.get("enabled", True) or float(s.get("weight", 0)) <= 0:
            continue
        ds2id[s["hf_dataset"]] = s["id"]
        weight[s["id"]] = float(s["weight"])
    if not weight:
        raise SystemExit("No weighted sources in config.")
    totw = sum(weight.values())
    char_budget = {sid: (w / totw) * args.target_tokens * args.chars_per_token
                   for sid, w in weight.items()}
    print(f"[mix] target {args.target_tokens/1e9:.1f}B tok; per-source budget (tok):")
    for sid in sorted(weight, key=lambda s: -weight[s]):
        print(f"    {sid:18s} weight {weight[sid]/totw*100:5.1f}%  -> {char_budget[sid]/args.chars_per_token/1e9:5.2f}B")

    files = sorted(glob.glob(os.path.join(args.input_dir, "**", "*.parquet"), recursive=True))
    if not files:
        raise SystemExit(f"No parquet under {args.input_dir}")
    random.Random(args.seed).shuffle(files)  # spread the trim across the corpus

    os.makedirs(args.output_dir, exist_ok=True)
    chars = {sid: 0 for sid in char_budget}
    unknown = 0
    done: set[str] = set()
    buf_text, buf_id, buf_meta = [], [], []
    shard_i = 0

    def flush():
        nonlocal shard_i, buf_text, buf_id, buf_meta
        if not buf_text:
            return
        tbl = pa.table({"text": buf_text, "id": buf_id, "metadata": buf_meta})
        pq.write_table(tbl, os.path.join(args.output_dir, f"{shard_i:05d}.parquet"), compression="snappy")
        shard_i += 1
        buf_text, buf_id, buf_meta = [], [], []

    for fp in files:
        if len(done) >= len(char_budget):
            break
        pf = pq.ParquetFile(fp)
        cols = [c for c in ("text", "id", "metadata") if c in pf.schema_arrow.names]
        for batch in pf.iter_batches(columns=cols, batch_size=2048):
            d = batch.to_pydict()
            texts = d.get("text", [])
            ids = d.get("id", [None] * len(texts))
            metas = d.get("metadata", [{}] * len(texts))
            for txt, _id, m in zip(texts, ids, metas):
                if not txt:
                    continue
                m = m or {}
                sid = ds2id.get(m.get("dataset"))
                if sid is None:
                    unknown += 1
                    continue
                if sid in done or chars[sid] >= char_budget[sid]:
                    done.add(sid)
                    continue
                buf_text.append(txt)
                buf_id.append(str(_id) if _id is not None else f"{sid}-{chars[sid]}")
                buf_meta.append({"source": sid, "dataset": m.get("dataset", ""),
                                 "language": str(m.get("language", ""))})
                chars[sid] += len(txt)
                if chars[sid] >= char_budget[sid]:
                    done.add(sid)
                if len(buf_text) >= args.docs_per_shard:
                    flush()
            if len(done) >= len(char_budget):
                break
    flush()

    print(f"\n[mix] wrote {shard_i} shards → {args.output_dir}")
    got = sum(chars.values()) / args.chars_per_token
    print(f"[mix] ~{got/1e9:.1f}B tokens total (est).  unknown-dataset docs skipped: {unknown}")
    for sid in sorted(chars, key=lambda s: -chars[s]):
        est = chars[sid] / args.chars_per_token
        filled = "full" if sid in done else "SHORT (couldn't fill budget)"
        print(f"    {sid:18s} ~{est/1e9:5.2f}B  ({est/max(got,1)*100:5.1f}%)  {filled}")


if __name__ == "__main__":
    main()
