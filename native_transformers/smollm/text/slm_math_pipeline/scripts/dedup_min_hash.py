#!/usr/bin/env python3
import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple


TOKEN_RE = re.compile(r"\w+", re.UNICODE)


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


def shingles(text: str, ngram: int) -> Set[str]:
    tokens = TOKEN_RE.findall(text.lower())
    if len(tokens) < ngram:
        return {" ".join(tokens)} if tokens else set()
    return {" ".join(tokens[i : i + ngram]) for i in range(0, len(tokens) - ngram + 1)}


def minhash_signature(shingle_set: Set[str], num_hashes: int) -> List[int]:
    if not shingle_set:
        return [0] * num_hashes
    sig = []
    vals = [int(hashlib.md5(s.encode("utf-8")).hexdigest(), 16) for s in shingle_set]
    mod = (1 << 61) - 1
    for i in range(num_hashes):
        a = 2 * i + 1
        b = 3 * i + 7
        sig.append(min((a * v + b) % mod for v in vals))
    return sig


def jaccard_from_minhash(sig_a: List[int], sig_b: List[int]) -> float:
    if not sig_a or not sig_b or len(sig_a) != len(sig_b):
        return 0.0
    eq = sum(1 for a, b in zip(sig_a, sig_b) if a == b)
    return eq / float(len(sig_a))


def dedup_rows(
    rows: Iterable[Dict],
    text_field: str,
    ngram_size: int,
    num_hashes: int,
    jaccard_threshold: float,
) -> Tuple[List[Dict], int]:
    kept: List[Dict] = []
    seen: List[Tuple[List[int], str]] = []
    dropped = 0

    for row in rows:
        text = row.get(text_field)
        if not isinstance(text, str):
            row["dedup_status"] = "skipped_no_text"
            kept.append(row)
            continue

        sh = shingles(text, ngram_size)
        sig = minhash_signature(sh, num_hashes)
        row_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()

        is_dup = False
        for old_sig, _ in seen:
            if jaccard_from_minhash(sig, old_sig) >= jaccard_threshold:
                is_dup = True
                break

        if is_dup:
            dropped += 1
            continue

        row["content_sha256"] = row_hash
        row["dedup_status"] = "kept"
        kept.append(row)
        seen.append((sig, row_hash))

    return kept, dropped


def main() -> None:
    parser = argparse.ArgumentParser(description="Simple MinHash-LSH style near-dedup for JSONL manifests.")
    parser.add_argument("--input_manifest", required=True)
    parser.add_argument("--output_manifest", required=True)
    parser.add_argument("--text_field", default="text")
    parser.add_argument("--ngram_size", type=int, default=13)
    parser.add_argument("--num_hashes", type=int, default=128)
    parser.add_argument("--jaccard_threshold", type=float, default=0.85)
    args = parser.parse_args()

    inp = Path(args.input_manifest)
    out = Path(args.output_manifest)
    out.parent.mkdir(parents=True, exist_ok=True)

    kept, dropped = dedup_rows(
        rows=read_jsonl(inp),
        text_field=args.text_field,
        ngram_size=args.ngram_size,
        num_hashes=args.num_hashes,
        jaccard_threshold=args.jaccard_threshold,
    )
    write_jsonl(out, kept)
    print(f"[ok] kept={len(kept)} dropped={dropped} output={out}")


if __name__ == "__main__":
    main()
