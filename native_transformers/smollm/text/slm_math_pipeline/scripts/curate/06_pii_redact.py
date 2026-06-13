#!/usr/bin/env python3
"""
Stage 6: PII redaction (emails, IPs, phone numbers).

Usage:
    python scripts/curate/06_pii_redact.py \
        --config configs/curation_pipeline.yaml \
        --input_dir outputs/curated/decontaminated \
        --output_dir outputs/curated/pii_clean
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

import yaml

from _curate_utils import prune_empty_parquet, stable_metadata_adapter

# PII patterns
_EMAIL_RE = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")
_IP_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
_PHONE_RE = re.compile(
    # Vietnamese mobile/landline forms plus compact international forms. This avoids
    # matching arbitrary math spans like "1 2 3 4 5 6 7" by requiring a phone prefix.
    r"(?<!\w)(?:\+?84|0)(?:[\s.\-]?\d){8,10}(?!\w)",
    re.IGNORECASE,
)


def redact(
    text: str,
    replacement: str = "<|pii|>",
    redact_email: bool = True,
    redact_ip: bool = True,
    redact_phone: bool = True,
) -> tuple[str, int]:
    """Redact PII and return (cleaned_text, count_replaced)."""
    count = 0
    patterns = []
    if redact_email:
        patterns.append(_EMAIL_RE)
    if redact_ip:
        patterns.append(_IP_RE)
    if redact_phone:
        patterns.append(_PHONE_RE)
    for pat in patterns:
        replaced = pat.sub(replacement, text)
        count += len(pat.findall(text))
        text = replaced
    return text, count


def run_pii(cfg: dict, input_dir: str, output_dir: str, workers: int) -> None:
    from datatrove.executor import LocalPipelineExecutor
    from datatrove.pipeline.filters import LambdaFilter
    from datatrove.pipeline.readers import ParquetReader
    from datatrove.pipeline.writers import ParquetWriter

    pii_cfg: dict = cfg.get("pii", {})
    replacement: str = pii_cfg.get("replacement", "<|pii|>")
    redact_email = pii_cfg.get("redact_email", True)
    redact_ip = pii_cfg.get("redact_ip", True)
    redact_phone = pii_cfg.get("redact_phone", True)

    def _redact(doc) -> bool:
        text, _ = redact(
            doc.text or "",
            replacement,
            redact_email=redact_email,
            redact_ip=redact_ip,
            redact_phone=redact_phone,
        )
        doc.text = text
        return True  # never drop; only redact

    executor = LocalPipelineExecutor(
        pipeline=[
            ParquetReader(data_folder=input_dir, glob_pattern="**/*.parquet",
                          doc_progress=True),
            LambdaFilter(filter_function=_redact),
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
    parser = argparse.ArgumentParser(description="PII redaction.")
    parser.add_argument("--config", default="configs/curation_pipeline.yaml")
    parser.add_argument("--input_dir", default="outputs/curated/decontaminated")
    parser.add_argument("--output_dir", default="outputs/curated/pii_clean")
    parser.add_argument("--workers", type=int, default=max(1, os.cpu_count() - 2))
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    prune_empty_parquet(args.input_dir)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    run_pii(cfg, args.input_dir, args.output_dir, args.workers)
    print(f"[ok] PII redaction done -> {args.output_dir}")


if __name__ == "__main__":
    main()
