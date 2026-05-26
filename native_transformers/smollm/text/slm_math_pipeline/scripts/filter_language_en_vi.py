#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path
from typing import Dict, Iterable, Optional


VI_CHAR_HINT = re.compile(r"[ăâêôơưđĂÂÊÔƠƯĐ]")
EN_CHAR_HINT = re.compile(r"[a-zA-Z]")


def detect_lang_heuristic(text: str) -> Optional[str]:
    if not text or not text.strip():
        return None
    if VI_CHAR_HINT.search(text):
        return "vi"
    if EN_CHAR_HINT.search(text):
        return "en"
    return None


def detect_lang(text: str) -> Optional[str]:
    try:
        from langdetect import detect  # type: ignore

        return detect(text)
    except Exception:
        return detect_lang_heuristic(text)


def read_jsonl(path: Path) -> Iterable[Dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: Path, rows: Iterable[Dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter manifest rows to EN/VI by detected language.")
    parser.add_argument("--input_manifest", required=True)
    parser.add_argument("--output_manifest", required=True)
    parser.add_argument("--text_field", default="text")
    args = parser.parse_args()

    input_path = Path(args.input_manifest)
    output_path = Path(args.output_manifest)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    kept = []
    dropped = 0

    for row in read_jsonl(input_path):
        text = row.get(args.text_field)
        if not isinstance(text, str):
            # Keep source rows that do not contain text payload yet.
            row["lang_detected"] = row.get("language")
            row["lang_filter_status"] = "skipped_no_text"
            kept.append(row)
            continue

        lang = detect_lang(text)
        row["lang_detected"] = lang
        if lang in {"en", "vi"}:
            row["lang_filter_status"] = "kept"
            kept.append(row)
        else:
            dropped += 1

    write_jsonl(output_path, kept)
    print(f"[ok] kept={len(kept)} dropped={dropped} output={output_path}")


if __name__ == "__main__":
    main()
