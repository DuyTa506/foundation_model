#!/usr/bin/env python3
"""
Measure how many docs survive the quality filter, per language and per dataset,
WITHOUT running the full stage. Compares the OLD uniform English chain vs the NEW
language-routed chain so you can confirm the VI recovery before committing to a full
re-run of 01→07.

Run on a sample of raw (or any stage's) parquet:
    python scripts/curate/measure_filter_survival.py \
        --config configs/curation_pipeline.yaml \
        --input_dir outputs/curated/raw \
        --max_docs 100000

Reports, per language and per dataset: total sampled, OLD survival %, NEW survival %.
"""

from __future__ import annotations

import argparse
import glob
import os
from collections import Counter

import yaml

from _curate_utils import build_quality_router, _filter_passes


def _legacy_router(cfg: dict):
    """The OLD behavior: full English Gopher+C4+FineWeb applied to EVERY doc
    (language-blind), to quantify how much VI it was wrongly rejecting."""
    from datatrove.pipeline.filters import (
        C4QualityFilter, FineWebQualityFilter, GopherQualityFilter, GopherRepetitionFilter,
    )
    qf = cfg.get("quality_filter", {})
    mwl = qf.get("mean_word_length", [3, 10])
    fs = [
        GopherQualityFilter(
            min_doc_words=qf.get("min_words", 50), min_avg_word_length=mwl[0],
            max_doc_words=None, max_avg_word_length=mwl[1],
            max_symbol_word_ratio=qf.get("symbol_word_ratio_max", 0.10),
            max_bullet_lines_ratio=qf.get("bullet_line_ratio_max", 0.90),
            max_ellipsis_lines_ratio=qf.get("ellipsis_line_ratio_max", 0.30),
            max_non_alpha_words_ratio=1.0 - qf.get("alpha_ratio_min", 0.65),
        ),
        GopherRepetitionFilter(),
        C4QualityFilter(filter_no_terminal_punct=qf.get("end_with_punctuation", True)),
        FineWebQualityFilter(),
    ]
    return lambda doc: all(_filter_passes(f, doc) for f in fs)


def main() -> None:
    ap = argparse.ArgumentParser(description="Measure quality-filter survival per language/dataset.")
    ap.add_argument("--config", default="configs/curation_pipeline.yaml")
    ap.add_argument("--input_dir", default="outputs/curated/raw")
    ap.add_argument("--max_docs", type=int, default=100_000)
    args = ap.parse_args()

    import pyarrow.parquet as pq
    from datatrove.data import Document

    cfg = yaml.safe_load(open(args.config))
    new_route = build_quality_router(cfg)
    old_route = _legacy_router(cfg)

    # tallies keyed by (level, key) → counts
    tot = Counter(); old_keep = Counter(); new_keep = Counter()

    def tally(level, key, kept_old, kept_new):
        tot[(level, key)] += 1
        old_keep[(level, key)] += int(kept_old)
        new_keep[(level, key)] += int(kept_new)

    files = sorted(glob.glob(os.path.join(args.input_dir, "**", "*.parquet"), recursive=True))
    n = 0
    for fp in files:
        if n >= args.max_docs:
            break
        t = pq.ParquetFile(fp)
        cols = [c for c in ("text", "metadata") if c in t.schema_arrow.names]
        for batch in t.iter_batches(columns=cols, batch_size=1024):
            d = batch.to_pydict()
            for txt, m in zip(d.get("text", []), d.get("metadata", [])):
                if n >= args.max_docs:
                    break
                m = m or {}
                doc = Document(text=txt or "", id=str(n), metadata=dict(m))
                ko = old_route(doc); kn = new_route(doc)
                lang = (m.get("language") or "?")
                ds = (m.get("dataset") or "?")
                tally("LANG", lang, ko, kn)
                tally("DATASET", ds, ko, kn)
                n += 1
            if n >= args.max_docs:
                break

    def show(level):
        rows = sorted([(k, tot[(l, k)]) for (l, k) in tot if l == level], key=lambda x: -x[1])
        print(f"\n== survival by {level} (sampled {n} docs) ==")
        print(f"  {'key':40s} {'total':>8s} {'OLD%':>7s} {'NEW%':>7s}")
        for k, c in rows:
            o = old_keep[(level, k)] / c * 100
            nw = new_keep[(level, k)] / c * 100
            print(f"  {k:40s} {c:>8d} {o:>6.1f}% {nw:>6.1f}%")

    show("LANG")
    show("DATASET")
    print(f"\nOVERALL: OLD {sum(old_keep[('LANG',k)] for _,k in [x for x in tot if x[0]=='LANG'])/max(n,1)*100:.1f}%"
          f"  NEW {sum(new_keep[('LANG',k)] for _,k in [x for x in tot if x[0]=='LANG'])/max(n,1)*100:.1f}%")


if __name__ == "__main__":
    main()
