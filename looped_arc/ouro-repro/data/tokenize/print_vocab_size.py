#!/usr/bin/env python3
from __future__ import annotations

import argparse
from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser(description="Print tokenizer vocab size for train config.")
    parser.add_argument("--tokenizer", required=True, help="Path or HF tokenizer id")
    args = parser.parse_args()

    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    print(len(tok))


if __name__ == "__main__":
    main()

