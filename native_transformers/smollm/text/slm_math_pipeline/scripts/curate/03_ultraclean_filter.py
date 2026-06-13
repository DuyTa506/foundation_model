#!/usr/bin/env python3
"""
Stage 3: UltraClean quality classifier filter (MiniCPM recipe).

For English: uses the released openbmb/Ultra-FineWeb-classifier (fastText).
For Vietnamese: trains a VI quality classifier from HQ VI seeds using the
    UltraClean efficient-verification loop, then classifies all VI docs.

The UltraClean recipe (from MiniCPM4 / Ultra-FineWeb paper, arXiv 2505.05427):
1. Select HQ seed documents from FineWeb2-HQ (already quality-filtered).
2. Train a fastText classifier on HQ seeds (+) vs low-quality docs (-).
3. Run efficient verification: fine-tune a 1B model on a small budget of
   candidate data; if eval improves, the classifier threshold is good.
4. Iterate (re-select seeds, re-train classifier) if needed.

This script handles steps 1-2 for VI, uses the released EN classifier,
and applies both to produce a classifier-filtered corpus.

Usage:
    # First run (trains VI classifier if not present):
    python scripts/curate/03_ultraclean_filter.py \
        --config configs/curation_pipeline.yaml \
        --input_dir outputs/curated/deduped \
        --output_dir outputs/curated/ultraclean

    # If VI classifier already trained:
    python scripts/curate/03_ultraclean_filter.py ... \
        --vi_classifier_path outputs/ultraclean_vi/vi_classifier.bin
"""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

import yaml

from _curate_utils import prune_empty_parquet, stable_metadata_adapter


def _train_vi_fasttext_classifier(
    input_dir: Path,
    output_path: Path,
    hq_source: str = "fineweb2_hq_vi",
    max_samples: int = 500_000,
    min_seeds: int = 200,
) -> bool:
    """Train a fastText HQ/LQ classifier for Vietnamese from the stage's OWN input.

    Returns True on success, False if it could not train (too few seeds, fastText
    error, …). Callers MUST treat False as "use heuristics" and continue — this never
    raises, so an unattended pipeline never dies here.

    Seed selection is layout-agnostic: after stage 01 the per-source subdirs are gone
    (docs are mixed into ``${rank}.parquet`` and tagged by ``metadata['source']``), so we
    read the mixed parquet in ``input_dir`` and pick positives by metadata — preferring
    the configured HQ source, then ANY Vietnamese doc. Negatives are synthetic word-shuffle
    degradations of the positives (no second hardcoded source dir needed).
    """
    try:
        import fasttext  # noqa: F401
    except ImportError:
        print("[ultraclean_vi] fasttext not installed; cannot train, will use heuristics")
        return False

    import random
    import unicodedata
    import pyarrow.parquet as pq

    def _iter_rows():
        for p in sorted(Path(input_dir).rglob("*.parquet")):
            if f"{os.sep}logs{os.sep}" in str(p):
                continue
            try:
                tbl = pq.read_table(str(p), columns=["text", "metadata"])
            except Exception:
                continue
            texts = tbl.column("text").to_pylist()
            metas = (tbl.column("metadata").to_pylist()
                     if "metadata" in tbl.column_names else [{}] * len(texts))
            for t, m in zip(texts, metas):
                yield t, (m or {})

    def _is_vi(meta: dict) -> bool:
        lang = str(meta.get("language", ""))
        return lang == "vi" or "vie" in lang

    def _clean(t: str) -> str:
        return unicodedata.normalize("NFC", t).replace("\n", " ")[:512]

    # Pass 1: HQ source only. Pass 2 (if too few): any VI doc.
    pos: list[str] = []
    for prefer_hq in (True, False):
        if len(pos) >= min_seeds and prefer_hq:
            break
        for t, m in _iter_rows():
            if not (isinstance(t, str) and t.strip()) or not _is_vi(m):
                continue
            if prefer_hq and str(m.get("source", "")) != hq_source:
                continue
            pos.append(f"__label__pos {_clean(t)}")
            if len(pos) >= max_samples // 2:
                break
        if len(pos) >= min_seeds:
            break
        if prefer_hq:
            print(f"[ultraclean_vi] only {len(pos)} HQ-source seeds; widening to any VI doc")
            pos = []  # restart, collect from all VI docs in pass 2

    print(f"[ultraclean_vi] {len(pos):,} positive (VI) samples")
    if len(pos) < min_seeds:
        print(f"[ultraclean_vi] too few VI seeds (<{min_seeds}); skipping training "
              f"(heuristic pass-through for VI)")
        return False

    def _degrade(text: str) -> str:
        words = text.split()
        random.shuffle(words)
        return " ".join(words[:50])

    neg = [f"__label__neg {_degrade(t.split(' ', 1)[-1])}" for t in pos]
    combined = pos + neg
    random.shuffle(combined)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    train_file = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt",
                                         delete=False, encoding="utf-8") as f:
            f.write("\n".join(combined) + "\n")
            train_file = f.name
        model = fasttext.train_supervised(
            input=train_file,
            epoch=10,
            lr=0.5,
            wordNgrams=2,
            dim=256,
            minCount=1,   # tiny corpora must not trip "Empty vocabulary"
            loss="softmax",
            verbose=2,
        )
        model.save_model(str(output_path))
        print(f"[ultraclean_vi] classifier saved -> {output_path}")
        return True
    except Exception as e:  # noqa: BLE001 - training must never kill the pipeline
        print(f"[ultraclean_vi] training failed ({type(e).__name__}: {e}); "
              f"using heuristic pass-through for VI")
        return False
    finally:
        if train_file and os.path.exists(train_file):
            os.unlink(train_file)


def classify_and_filter(
    input_dir: str,
    output_dir: str,
    en_classifier_path: str | None,
    vi_classifier_path: str | None,
    threshold: float,
    workers: int,
) -> None:
    from datatrove.executor import LocalPipelineExecutor
    from datatrove.pipeline.filters import LambdaFilter
    from datatrove.pipeline.readers import ParquetReader
    from datatrove.pipeline.writers import ParquetWriter

    # IMPORTANT: do NOT load the fastText models here. datatrove's
    # LocalPipelineExecutor pickles the whole pipeline (incl. this LambdaFilter's
    # closure) to dispatch it to worker processes, and a loaded
    # `fasttext_pybind.fasttext` object is NOT picklable
    # (TypeError: cannot pickle 'fasttext_pybind.fasttext' object). So capture only
    # the (string) paths and load each model lazily, once per worker process, into
    # a per-process cache. The empty dict pickles fine; the model never crosses the
    # process boundary.
    _model_cache: dict = {}

    def _get_models():
        if not _model_cache:
            try:
                import fasttext

                _model_cache["en"] = (
                    fasttext.load_model(en_classifier_path) if en_classifier_path else None
                )
                _model_cache["vi"] = (
                    fasttext.load_model(vi_classifier_path) if vi_classifier_path else None
                )
            except Exception as e:
                print(f"[warn] could not load classifier(s): {e}; falling back to heuristics")
                _model_cache["en"] = None
                _model_cache["vi"] = None
        return _model_cache["en"], _model_cache["vi"]

    def _classify(doc) -> bool:
        text: str = doc.text or ""
        if not text.strip():
            return False
        lang = doc.metadata.get("language", "")
        clean = text.replace("\n", " ")[:512]

        en_model, vi_model = _get_models()
        model = None
        if "eng" in lang or lang == "en":
            model = en_model
        elif "vie" in lang or lang == "vi":
            model = vi_model

        if model is None:
            # No classifier available for this language; pass through
            return True

        try:
            labels, scores = model.predict(clean, k=1)
            label = labels[0].replace("__label__", "")
            score = float(scores[0])
            doc.metadata["ultraclean_label"] = label
            doc.metadata["ultraclean_score"] = score
            return label == "pos" and score >= threshold
        except Exception:
            return True  # if classifier fails, pass through

    executor = LocalPipelineExecutor(
        pipeline=[
            ParquetReader(data_folder=input_dir, glob_pattern="**/*.parquet",
                          doc_progress=True),
            LambdaFilter(filter_function=_classify),
            # _classify mutates metadata (ultraclean_label/score) only for docs it
            # actually classifies — pass-through docs (no model for their language) get
            # nothing. That makes the per-doc metadata struct heterogeneous within one
            # writer batch -> pyarrow "Table schema does not match schema used to create
            # file". Re-project to the uniform {source,dataset,language} schema so every
            # row is identical regardless of which branch _classify took.
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
    parser = argparse.ArgumentParser(description="UltraClean quality classifier filter.")
    parser.add_argument("--config", default="configs/curation_pipeline.yaml")
    parser.add_argument("--input_dir", default="outputs/curated/deduped")
    parser.add_argument("--output_dir", default="outputs/curated/ultraclean")
    parser.add_argument("--vi_classifier_path", default=None,
                        help="Path to trained VI fastText classifier (.bin). "
                             "Will train if not provided.")
    parser.add_argument("--hq_vi_source", default=None,
                        help="metadata['source'] id whose VI docs seed the HQ classifier "
                             "(default: ultraclean.vi.hq_seed_source or 'fineweb2_hq_vi').")
    parser.add_argument("--skip_train_vi", action="store_true",
                        help="Skip VI classifier training (use heuristics only).")
    parser.add_argument("--workers", type=int, default=max(1, os.cpu_count() - 2))
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    uc_cfg: dict = cfg.get("ultraclean", {})
    threshold: float = uc_cfg.get("vi", {}).get("threshold", 0.5)

    # ── EN classifier ────────────────────────────────────────────────────────
    en_classifier_path: str | None = None
    en_hf_id: str = uc_cfg.get("en", {}).get("classifier_hf_id", "openbmb/Ultra-FineWeb-classifier")
    # The released repo stores the fastText classifiers under classifiers/, NOT
    # a top-level model.bin (that name 404s). EN = ultra_fineweb_en.bin.
    en_filename: str = uc_cfg.get("en", {}).get("classifier_filename",
                                                "classifiers/ultra_fineweb_en.bin")
    try:
        from huggingface_hub import hf_hub_download
        en_classifier_path = hf_hub_download(repo_id=en_hf_id, filename=en_filename)
        print(f"[ultraclean] loaded EN classifier: {en_classifier_path}")
    except Exception as e:
        print(f"[warn] could not load EN classifier ({e}); skipping EN classification")

    # ── VI classifier ────────────────────────────────────────────────────────
    vi_classifier_path: str | None = args.vi_classifier_path or uc_cfg.get(
        "vi", {}
    ).get("classifier_path")

    fallback_ok: bool = uc_cfg.get("vi", {}).get("fallback_to_heuristics", True)
    if not vi_classifier_path or not Path(vi_classifier_path).exists():
        if args.skip_train_vi:
            print("[ultraclean] skipping VI classifier training; using heuristics")
            vi_classifier_path = None
        else:
            hq_source = (args.hq_vi_source
                         or uc_cfg.get("vi", {}).get("hq_seed_source", "fineweb2_hq_vi"))
            vi_out = Path("outputs/ultraclean_vi/vi_classifier.bin")
            print(f"[ultraclean] training VI classifier -> {vi_out} "
                  f"(HQ seed source='{hq_source}')")
            # Never fatal: on any failure fall back to heuristic pass-through so an
            # unattended large-scale run keeps going instead of dying at this stage.
            try:
                trained = _train_vi_fasttext_classifier(
                    input_dir=Path(args.input_dir),
                    output_path=vi_out,
                    hq_source=hq_source,
                )
            except Exception as e:  # noqa: BLE001 - belt-and-suspenders
                print(f"[warn] VI classifier training errored ({e})")
                trained = False
            if trained:
                vi_classifier_path = str(vi_out)
            elif fallback_ok:
                print("[ultraclean] VI classifier unavailable; heuristic pass-through for VI")
                vi_classifier_path = None
            else:
                raise SystemExit("VI classifier training failed and "
                                 "fallback_to_heuristics is disabled")

    prune_empty_parquet(args.input_dir)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    classify_and_filter(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        en_classifier_path=en_classifier_path,
        vi_classifier_path=vi_classifier_path,
        threshold=threshold,
        workers=args.workers,
    )
    print(f"[ok] ultraclean filtering done -> {args.output_dir}")


if __name__ == "__main__":
    main()
