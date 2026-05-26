#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from datasets import load_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="HF dataset name")
    parser.add_argument("--config", default=None, help="HF dataset config")
    parser.add_argument("--split", default="train")
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--max-rows", type=int, default=1000000)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(args.dataset, name=args.config, split=args.split, streaming=True)
    with out_path.open("w", encoding="utf-8") as f:
        for idx, row in enumerate(ds):
            if idx >= args.max_rows:
                break
            text = row.get(args.text_field)
            if not text:
                continue
            f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()

