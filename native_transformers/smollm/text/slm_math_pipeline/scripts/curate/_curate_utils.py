"""Shared helpers for the curation stages (00–07).

When run as ``python scripts/curate/0X_*.py`` the script's own directory is on
``sys.path[0]``, so a plain ``from _curate_utils import prune_empty_parquet``
resolves without any package setup.
"""

from __future__ import annotations

import os
from pathlib import Path


def run_with_hf_retry(executor, max_retries: int = 8, base_delay: float = 10.0):
    """Run a datatrove executor, retrying on HuggingFace 429 rate-limit errors.

    When ``00_materialize`` reads directly from HF, every datatrove task calls
    ``load_dataset`` and hits HF's file-listing API. With many tasks this bursts past
    HF's 1000-requests / 5-min cap -> ``HfHubHTTPError: 429 Too Many Requests`` (the
    response carries a ``Retry-After`` seconds hint). The executor uses
    ``skip_completed=True``, so a retry resumes from where it stopped rather than
    redoing finished shards. We honor ``Retry-After`` when present, else exponential
    backoff. The real cure is fewer concurrent API callers (cap tasks) and/or the
    streamed-parquet path; this is the safety net.
    """
    import re
    import time

    for attempt in range(max_retries + 1):
        try:
            executor.run()
            return
        except Exception as e:  # noqa: BLE001 - inspect message for rate-limit shape
            msg = str(e)
            is_rate_limit = "429" in msg or "Too Many Requests" in msg or "rate limit" in msg.lower()
            if not is_rate_limit or attempt == max_retries:
                raise
            m = re.search(r"[Rr]etry after (\d+)", msg)
            delay = int(m.group(1)) if m else base_delay * (2 ** attempt)
            delay = min(delay, 300)  # cap a single wait at 5 min
            print(f"[hf-retry] 429 rate limit (attempt {attempt + 1}/{max_retries}); "
                  f"sleeping {delay:.0f}s then resuming (skip_completed keeps progress)")
            time.sleep(delay)


def stable_reader_adapter(keep_keys=("source", "dataset", "language"), defaults=None):
    """Return a datatrove **Reader** ``adapter`` (signature ``(self, data, path, id_in_file)``)
    that builds a Document with a FIXED, type-stable metadata schema directly at READ time.

    Why a *reader* adapter (not just the writer one): datatrove's default reader adapter
    does ``data.pop("metadata", {}) | data`` to fold leftover columns into metadata. Some
    HF sources (e.g. ``open-web-math``) ship a top-level column literally named
    ``metadata`` whose value is a JSON **string**, so on older datatrove versions that do
    NOT guard the type, ``str | dict`` raises::

        TypeError: unsupported operand type(s) for |: 'str' and 'dict'

    This blows up inside the reader, BEFORE our writer ``stable_metadata_adapter`` ever
    runs, so the writer fix alone can't catch it. Supplying our own reader adapter means
    we never touch ``_default_adapter`` at all — version-proof — and we emit the same
    uniform ``{source,dataset,language}`` string schema the rest of the chain expects.

    Precedence per key: ``defaults`` WINS when the key is provided there, else fall back
    to the raw row (top-level, then a nested ``metadata`` dict). This is deliberate: at
    materialize we know the authoritative source/dataset/language from the pipeline config,
    and several HF sources ship their OWN ``source``/``language`` columns (e.g. CulturaX's
    ``source`` is "mC4"/"OSCAR", FineWeb's ``language`` is "en") — letting those override
    would corrupt source attribution and break stage-03 language routing. Values are
    coerced to ``str``; missing keys become ``""`` so every document of every source is
    identical.
    """
    defaults = defaults or {}

    def _adapter(self, data, path, id_in_file):  # bound via MethodType -> self is the reader
        # A source may carry a 'metadata' column that is a str/JSON/dict — never assume dict.
        nested = data.get("metadata")
        if isinstance(nested, str):
            import json
            try:
                nested = json.loads(nested)
            except (json.JSONDecodeError, ValueError):
                nested = {}
        if not isinstance(nested, dict):
            nested = {}
        out = {}
        for k in keep_keys:
            # config-provided defaults are authoritative; only consult the row when the
            # pipeline config is silent about this key.
            if k in defaults and defaults[k] not in (None, ""):
                v = defaults[k]
            else:
                v = data.get(k, nested.get(k, defaults.get(k, "")))
            out[k] = "" if v is None else str(v)
        text = data.get(self.text_key, "")
        doc_id = data.get(self.id_key)
        return {
            "text": text or "",
            "id": str(doc_id) if doc_id is not None else f"{path}/{id_in_file}",
            "media": [],
            "metadata": out,
        }

    return _adapter


def stable_metadata_adapter(keep_keys=("source", "dataset", "language"), defaults=None):
    """Return a datatrove ParquetWriter ``adapter`` that projects every document to a
    FIXED, type-stable schema: ``{text:str, id:str, metadata:{<keep_keys>:str}}``.

    Why this is needed: different HuggingFace sources ship wildly different metadata
    (url/title/timestamp, and some carry NUMERIC fields). datatrove shards by file and
    round-robins files across writer tasks, so a single writer rank routinely batches
    documents from MULTIPLE sources. pyarrow then infers the ``metadata`` struct type
    from that mixed batch and collides, e.g.:
        ArrowTypeError: object of type <class 'str'> cannot be converted to int
    (a field that was int in one source's docs and str in another's), or a schema
    mismatch when later batches introduce new keys. Forcing one uniform set of string
    keys removes both failure modes. Values are coerced to str and missing keys default
    to "" so the struct is identical for every document of every source.

    The default keeps {source, dataset, language} — the only metadata the curation
    chain actually reads downstream (stage 03 routes on ``metadata['language']``); the
    noisy per-source fields (url, title, …) are dropped. Pass ``defaults`` to seed keys
    the raw source lacks (e.g. ``{"source": src_id, "language": "vi"}`` at materialize).
    """
    defaults = defaults or {}

    def _adapter(self, document):  # datatrove binds this as a method -> (self, document)
        meta = document.metadata or {}
        out = {}
        for k in keep_keys:
            v = meta.get(k, defaults.get(k, ""))
            out[k] = "" if v is None else str(v)
        return {
            "text": document.text or "",
            "id": str(document.id) if document.id is not None else "",
            "metadata": out,
        }

    return _adapter


def _is_readable_parquet(path: Path) -> bool:
    """True iff ``path`` is a structurally valid parquet file (footer present)."""
    try:
        import pyarrow.parquet as pq

        # Opening reads + validates the footer/metadata without loading row data.
        with pq.ParquetFile(str(path)):
            return True
    except Exception:
        return False


def prune_empty_parquet(folder: str | Path) -> int:
    """Delete empty / structurally-invalid ``*.parquet`` files under ``folder``.

    datatrove's ParquetWriter leaves a broken file behind for any shard/rank
    whose documents were all filtered out, or whose worker was interrupted mid
    write: the handle is opened (and a Parquet header may be written) but the
    footer with the magic bytes is never flushed. datatrove's ParquetReader then
    raises either ``ArrowInvalid: Parquet file size is 0 bytes`` (truly empty) or
    ``ArrowInvalid: Parquet magic bytes not found in footer`` (header-only). Both
    cases are normal in a curation pipeline, so prune them before the next stage
    reads. We validate the footer rather than only checking ``size == 0`` so the
    header-only case is caught too.

    Returns the number of files removed.
    """
    root = Path(folder)
    if not root.exists():
        return 0
    removed = 0
    for p in root.rglob("*.parquet"):
        try:
            if not p.is_file():
                continue
            if p.stat().st_size == 0 or not _is_readable_parquet(p):
                p.unlink()
                removed += 1
        except OSError:
            pass
    if removed:
        print(f"[prune] removed {removed} empty/corrupt parquet file(s) under {root}")
    return removed


# ─── Language-routed quality filter (shared by stage 01 + survival measurement) ──

# Vietnamese diacritic chars (precomposed + base modified letters): catches garbled
# / mis-labeled VI that's actually Latin without marks.
_VI_DIACRITICS = (
    "ăâêôơưđĂÂÊÔƠƯĐàáảãạằắẳẵặầấẩẫậèéẻẽẹềếểễệìíỉĩị"
    "òóỏõọồốổỗộờớởỡợùúủũụừứửữựỳýỷỹỵĐ"
)


def _filter_passes(f, doc) -> bool:
    """A datatrove filter's .filter() returns a bool or (keep, reason) tuple — normalize."""
    r = f.filter(doc)
    return bool(r[0]) if isinstance(r, tuple) else bool(r)


def build_quality_router(cfg: dict):
    """Return ``route(doc) -> bool`` applying language-appropriate quality filters.

    The EN-tuned Gopher/C4/FineWeb heuristics reject Vietnamese en masse — most
    fatally Gopher's ``min_stop_words=2`` against ENGLISH stop words (a pure-VI doc
    has zero). So VI gets a relaxed chain; EN keeps the full English chain. Used by
    both stage 01 and scripts/curate/measure_filter_survival.py so they stay in sync.
    """
    from datatrove.pipeline.filters import (
        C4QualityFilter, FineWebQualityFilter, GopherQualityFilter, GopherRepetitionFilter,
    )

    qf = cfg.get("quality_filter", {})
    vi_diacritic_min = qf.get("vi_diacritic_min_ratio", 0.002)
    mwl = qf.get("mean_word_length", [3, 10])
    common = dict(
        max_doc_words=None,
        max_avg_word_length=mwl[1],
        max_symbol_word_ratio=qf.get("symbol_word_ratio_max", 0.10),
        max_bullet_lines_ratio=qf.get("bullet_line_ratio_max", 0.90),
        max_ellipsis_lines_ratio=qf.get("ellipsis_line_ratio_max", 0.30),
        max_non_alpha_words_ratio=1.0 - qf.get("alpha_ratio_min", 0.65),
    )

    def _vi_diacritic_ok(doc) -> bool:
        text = doc.text or ""
        if not text:
            return False
        return sum(1 for c in text if c in _VI_DIACRITICS) / len(text) >= vi_diacritic_min

    en_filters = [
        GopherQualityFilter(min_doc_words=qf.get("min_words", 50),
                            min_avg_word_length=mwl[0], **common),
        GopherRepetitionFilter(),
        C4QualityFilter(filter_no_terminal_punct=qf.get("end_with_punctuation", True)),
        FineWebQualityFilter(),
    ]
    vi_filters = [
        GopherQualityFilter(min_doc_words=qf.get("vi_min_words", qf.get("min_words", 50)),
                            min_avg_word_length=qf.get("vi_min_avg_word_length", 2.0),
                            min_stop_words=0, **common),
        GopherRepetitionFilter(),
        C4QualityFilter(filter_no_terminal_punct=False, min_num_sentences=1,
                        min_words_per_line=1, remove_citations=False),
    ]

    def route(doc) -> bool:
        lang = (doc.metadata.get("language") or "").lower()
        if lang.startswith("vi"):
            return all(_filter_passes(f, doc) for f in vi_filters) and _vi_diacritic_ok(doc)
        return all(_filter_passes(f, doc) for f in en_filters)

    return route
