"""Shared helpers for the curation stages (00–07).

When run as ``python scripts/curate/0X_*.py`` the script's own directory is on
``sys.path[0]``, so a plain ``from _curate_utils import prune_empty_parquet``
resolves without any package setup.
"""

from __future__ import annotations

import os
from pathlib import Path


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
