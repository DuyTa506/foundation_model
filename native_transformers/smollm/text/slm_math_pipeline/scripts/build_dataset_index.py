#!/usr/bin/env python3
import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List

import yaml


def stable_hash(obj: Any) -> str:
    payload = json.dumps(obj, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def normalize_source(src: Dict[str, Any], seed: int) -> Dict[str, Any]:
    item = {
        "id": src["id"],
        "hf_dataset": src["hf_dataset"],
        "subset": src.get("subset"),
        "split": src.get("split", "train"),
        "text_field": src.get("text_field"),
        "prompt_field": src.get("prompt_field"),
        "response_field": src.get("response_field"),
        "chosen_field": src.get("chosen_field"),
        "rejected_field": src.get("rejected_field"),
        "language": src.get("language"),
        "weight": float(src.get("weight", 1.0)),
        "role": src.get("role", "unspecified"),
        "seed": seed,
    }
    item["source_hash"] = stable_hash(item)
    return item


def write_jsonl(path: Path, records: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in records:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build deterministic dataset manifest from YAML config.")
    parser.add_argument("--config", required=True, help="Path to dataset config YAML.")
    parser.add_argument("--output_dir", required=True, help="Output directory for manifest and data card.")
    args = parser.parse_args()

    config_path = Path(args.config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    seed = int(cfg.get("seed", 42))
    output_cfg = cfg.get("output", {})
    manifest_name = output_cfg.get("manifest_name", "dataset_manifest.jsonl")
    data_card_name = output_cfg.get("data_card_name", "data_card.json")

    manifest_records: List[Dict[str, Any]] = []

    if "sources" in cfg:
        for src in cfg["sources"]:
            manifest_records.append(normalize_source(src, seed))
    else:
        post_sources = cfg.get("sources", {})
        pref_sources = post_sources.get("preference_pairs", [])
        sft_sources = post_sources.get("sft_fallback", [])
        for src in pref_sources + sft_sources:
            if src.get("enabled", True):
                manifest_records.append(normalize_source(src, seed))

    write_jsonl(output_dir / manifest_name, manifest_records)

    data_card = {
        "created_at_utc": dt.datetime.utcnow().isoformat() + "Z",
        "config_path": str(config_path),
        "seed": seed,
        "num_sources": len(manifest_records),
        "languages_allow": cfg.get("curation", {}).get("languages", {}).get("allow", []),
        "dedup": cfg.get("curation", {}).get("dedup", {}),
        "decontamination": cfg.get("curation", {}).get("decontamination", {}),
        "manifest_file": manifest_name,
        "manifest_sha256": stable_hash(manifest_records),
        "sources": manifest_records,
    }
    with (output_dir / data_card_name).open("w", encoding="utf-8") as f:
        json.dump(data_card, f, ensure_ascii=False, indent=2)

    print(f"[ok] wrote manifest: {output_dir / manifest_name}")
    print(f"[ok] wrote data card: {output_dir / data_card_name}")


if __name__ == "__main__":
    main()
