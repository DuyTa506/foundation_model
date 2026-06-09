#!/usr/bin/env python3
"""
Stage 2: Per-document language identification.

Replaces filter_language_en_vi.py which operated on manifest descriptors
(no text payload → was a no-op).

Uses GlotLID-M (or fastText lid.176.bin as fallback) for per-document
language detection with confidence thresholding.  Keeps only English
(eng_Latn) and Vietnamese (vie_Latn) documents.

Usage:
    python scripts/curate/02_language_id.py \
        --config configs/curation_pipeline.yaml \
        --input_dir outputs/curated/quality_filtered \
        --output_dir outputs/curated/lang_filtered
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import yaml

from _curate_utils import prune_empty_parquet


def _load_langid_model(backend: str, model_name: str):
    """Load language identification model; falls back gracefully."""
    if backend == "glotlid":
        try:
            import fasttext
            from huggingface_hub import hf_hub_download

            model_path = hf_hub_download(
                repo_id="cis-lmu/glotlid", filename="model.bin"
            )
            model = fasttext.load_model(model_path)
            print(f"[langid] loaded GlotLID from {model_path}")
            return model, "glotlid"
        except Exception as e:
            print(f"[langid] GlotLID failed ({e}); trying fastText")

    # Fallback: fastText LID
    try:
        import fasttext
        from huggingface_hub import hf_hub_download

        model_path = hf_hub_download(
            repo_id="facebook/fasttext-language-identification",
            filename="model.bin",
        )
        model = fasttext.load_model(model_path)
        print(f"[langid] loaded fastText LID from {model_path}")
        return model, "fasttext"
    except Exception as e:
        print(f"[langid] fastText failed ({e}); falling back to heuristic")
        return None, "heuristic"


def _detect_heuristic(text: str) -> tuple[str, float]:
    """Last-resort heuristic: diacritics for VI, ASCII for EN."""
    import re

    vi_pat = re.compile(r"[ăâêôơưđĂÂÊÔƠƯĐ]")
    if vi_pat.search(text):
        return "vie_Latn", 0.70
    en_pat = re.compile(r"[a-zA-Z]")
    if en_pat.search(text):
        return "eng_Latn", 0.60
    return "other", 0.50


def build_pipeline(
    cfg: dict,
    input_dir: str,
    output_dir: str,
    workers: int,
):
    from datatrove.executor import LocalPipelineExecutor
    from datatrove.pipeline.filters import LanguageFilter, LambdaFilter
    from datatrove.pipeline.readers import ParquetReader
    from datatrove.pipeline.writers import ParquetWriter

    lid_cfg = cfg.get("language_id", {})
    backend = lid_cfg.get("backend", "glotlid")
    model_name = lid_cfg.get("model", "GlotLID-M")
    min_confidence = lid_cfg.get("min_confidence", 0.65)
    keep_languages = lid_cfg.get("keep_languages", ["eng_Latn", "vie_Latn"])

    # datatrove's built-in LanguageFilter uses fastText under the hood
    # with the same HF model; configure it directly.
    # datatrove backend choices are exactly {"ft176", "glotlid"}. glotlid emits
    # script-tagged codes (eng_Latn/vie_Latn) matching keep_languages; ft176 emits
    # bare ISO-639-1 (en/vi). Pick the one matching the configured keep_languages.
    dt_backend = "glotlid" if backend == "glotlid" else "ft176"
    try:
        lang_filter = LanguageFilter(
            languages=keep_languages,
            language_threshold=min_confidence,
            backend=dt_backend,
        )
        use_builtin = True
    except Exception:
        use_builtin = False

    if use_builtin:
        filters = [lang_filter]
    else:
        # Manual fallback. Load the fastText model lazily, once per worker process:
        # datatrove pickles this closure to its workers and a loaded
        # `fasttext_pybind.fasttext` object is not picklable. Capture only the
        # (string) backend/model_name and cache the model per process.
        _model_cache: dict = {}

        def _get_model():
            if not _model_cache:
                _model_cache["m"], _ = _load_langid_model(backend, model_name)
            return _model_cache["m"]

        def _lang_filter_func(doc) -> bool:
            text = doc.text or ""
            if not text.strip():
                return False
            model = _get_model()
            if model is not None:
                try:
                    labels, scores = model.predict(text.replace("\n", " ")[:512], k=1)
                    label = labels[0].replace("__label__", "")
                    score = float(scores[0])
                    # Map fastText codes to glotlid-style
                    label_map = {"en": "eng_Latn", "vi": "vie_Latn"}
                    label = label_map.get(label, label)
                    doc.metadata["language"] = label
                    doc.metadata["language_score"] = score
                    return label in keep_languages and score >= min_confidence
                except Exception:
                    pass
            # Pure heuristic
            lang, conf = _detect_heuristic(text)
            doc.metadata["language"] = lang
            doc.metadata["language_score"] = conf
            return lang in keep_languages and conf >= min_confidence

        filters = [LambdaFilter(filter_function=_lang_filter_func)]

    return LocalPipelineExecutor(
        pipeline=[
            ParquetReader(data_folder=input_dir, glob_pattern="**/*.parquet",
                          doc_progress=True),
            *filters,
            ParquetWriter(
                output_folder=output_dir,
                output_filename="${rank}.parquet",
                compression="snappy",
            ),
        ],
        tasks=workers,
        workers=workers,
        logging_dir=str(Path(output_dir) / "logs"),
        skip_completed=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Language ID filter (keep en/vi).")
    parser.add_argument("--config", default="configs/curation_pipeline.yaml")
    parser.add_argument("--input_dir", default="outputs/curated/quality_filtered")
    parser.add_argument("--output_dir", default="outputs/curated/lang_filtered")
    parser.add_argument("--workers", type=int, default=max(1, os.cpu_count() - 2))
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    prune_empty_parquet(args.input_dir)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    executor = build_pipeline(cfg, args.input_dir, args.output_dir, args.workers)
    executor.run()
    print(f"[ok] language filtering done -> {args.output_dir}")


if __name__ == "__main__":
    main()
