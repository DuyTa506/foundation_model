#!/usr/bin/env python3
"""
Stage 1: Heuristic quality filtering.

Applies Gopher + C4 + FineWeb-style rules to remove low-quality documents.
Also applies VI-specific rules (diacritic presence, encoding check).

Replaces the absent quality-filter step in the old pipeline.

Usage:
    python scripts/curate/01_quality_filter.py \
        --config configs/curation_pipeline.yaml \
        --input_dir outputs/curated/raw \
        --output_dir outputs/curated/quality_filtered
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import yaml


def build_pipeline(cfg: dict, input_dir: str, output_dir: str, workers: int):
    from datatrove.executor import LocalPipelineExecutor
    from datatrove.pipeline.filters import (
        C4QualityFilter,
        FineWebQualityFilter,
        GopherQualityFilter,
        GopherRepetitionFilter,
        LambdaFilter,
    )
    from datatrove.pipeline.readers import ParquetReader
    from datatrove.pipeline.writers import ParquetWriter

    qf_cfg: dict = cfg.get("quality_filter", {})

    vi_diacritic_min: float = qf_cfg.get("vi_diacritic_min_ratio", 0.002)

    def _vi_diacritic_check(doc) -> bool:
        """Keep VI docs that have a meaningful fraction of diacritic characters.
        Catches garbled/mis-labeled Vietnamese that's actually Latin without marks."""
        text: str = doc.text or ""
        if not text:
            return False
        vi_chars = sum(
            1 for c in text
            if c in "ăâêôơưđĂÂÊÔƠƯĐàáâãèéêìíòóôõùúýăắặằẳẵắổỗộởờớọồốổôơ"
               "ờởớỡợặắẳẵặẻẹẽếềềểễệỉịọỏốồổỗộớờởỡợụủừứựữửúùũ"
        )
        ratio = vi_chars / max(len(text), 1)
        # Only apply the check for docs the language filter will tag as Vietnamese
        return ratio >= vi_diacritic_min

    filters = [
        GopherQualityFilter(
            min_doc_words=qf_cfg.get("min_words", 50),
            max_doc_words=None,
            min_avg_word_length=qf_cfg.get("mean_word_length", [3, 10])[0],
            max_avg_word_length=qf_cfg.get("mean_word_length", [3, 10])[1],
            max_symbol_word_ratio=qf_cfg.get("symbol_word_ratio_max", 0.10),
            max_bullet_lines_ratio=qf_cfg.get("bullet_line_ratio_max", 0.90),
            max_ellipsis_lines_ratio=qf_cfg.get("ellipsis_line_ratio_max", 0.30),
            max_non_alpha_words_ratio=1.0 - qf_cfg.get("alpha_ratio_min", 0.65),
        ),
        GopherRepetitionFilter(
            max_top_ngram_character_fraction=None,
        ),
        C4QualityFilter(
            filter_no_terminal_punct=qf_cfg.get("end_with_punctuation", True),
        ),
        FineWebQualityFilter(),
        # VI-specific: requires some diacritics (catches garbled VI docs)
        LambdaFilter(
            filter_func=lambda doc: (
                # Only apply to docs tagged vi; skip if lang unknown yet
                True if doc.metadata.get("language") not in ("vi", "vie_Latn")
                else _vi_diacritic_check(doc)
            ),
            name="vi_diacritic_check",
        ),
    ]

    return LocalPipelineExecutor(
        pipeline=[
            ParquetReader(input_folder=input_dir, progress=True),
            *filters,
            ParquetWriter(
                output_folder=output_dir,
                output_filename="${rank:04d}.parquet",
                compression="snappy",
            ),
        ],
        tasks=workers,
        workers=workers,
        logging_dir=str(Path(output_dir) / "logs"),
        skip_completed=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Heuristic quality filter.")
    parser.add_argument("--config", default="configs/curation_pipeline.yaml")
    parser.add_argument("--input_dir", default="outputs/curated/raw")
    parser.add_argument("--output_dir", default="outputs/curated/quality_filtered")
    parser.add_argument("--workers", type=int, default=max(1, os.cpu_count() - 2))
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    executor = build_pipeline(cfg, args.input_dir, args.output_dir, args.workers)
    executor.run()
    print(f"[ok] quality filtering done -> {args.output_dir}")


if __name__ == "__main__":
    main()
