#!/usr/bin/env python3
import argparse
import json
import subprocess
from pathlib import Path
from typing import Dict, Iterable, List, Optional


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


def check_model_compatibility(base_model: str, trust_remote_code: bool) -> Dict:
    result = {"base_model": base_model, "compatible": False, "issues": []}
    try:
        from transformers import AutoConfig, AutoTokenizer
    except Exception as exc:
        result["issues"].append(f"transformers import failed: {exc}")
        return result

    try:
        cfg = AutoConfig.from_pretrained(base_model, trust_remote_code=trust_remote_code)
        tok = AutoTokenizer.from_pretrained(base_model, trust_remote_code=trust_remote_code)
    except Exception as exc:
        result["issues"].append(f"cannot load model/tokenizer: {exc}")
        return result

    model_type = getattr(cfg, "model_type", "unknown")
    vocab_size = getattr(cfg, "vocab_size", None)
    max_pos = getattr(cfg, "max_position_embeddings", None)

    result.update(
        {
            "model_type": model_type,
            "vocab_size": vocab_size,
            "max_position_embeddings": max_pos,
            "tokenizer_vocab_size": len(tok),
        }
    )

    allowed_types = {"llama", "minicpm"}
    if model_type not in allowed_types:
        result["issues"].append(f"unsupported model_type={model_type} for current Megatron baseline")
    if not vocab_size or vocab_size < 32000:
        result["issues"].append(f"unexpected vocab_size={vocab_size}")

    result["compatible"] = len(result["issues"]) == 0
    return result


def build_text_corpus(manifest: Path, output_jsonl: Path, text_field: str) -> int:
    rows: List[Dict] = []
    for row in read_jsonl(manifest):
        text = row.get(text_field)
        if isinstance(text, str) and text.strip():
            rows.append({"text": text})

    write_jsonl(output_jsonl, rows)
    return len(rows)


def build_preprocess_command(
    megatron_tools_dir: str,
    input_jsonl: Path,
    output_prefix: Path,
    tokenizer: str,
) -> List[str]:
    script = Path(megatron_tools_dir) / "tools" / "preprocess_data.py"
    return [
        "python",
        str(script),
        "--input",
        str(input_jsonl),
        "--output-prefix",
        str(output_prefix),
        "--tokenizer-type",
        "HFTokenizer",
        "--tokenizer-name-or-path",
        tokenizer,
        "--dataset-impl",
        "mmap",
        "--workers",
        "8",
        "--append-eod",
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare JSONL corpus + Megatron indexed dataset command.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--output_prefix", required=True)
    parser.add_argument("--text_field", default="text")
    parser.add_argument("--megatron_tools_dir", default="Megatron-LM")
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--run_preprocess", action="store_true")
    args = parser.parse_args()

    manifest = Path(args.manifest)
    output_prefix = Path(args.output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    compatibility = check_model_compatibility(args.base_model, args.trust_remote_code)
    compatibility_file = output_prefix.parent / "model_compatibility.json"
    with compatibility_file.open("w", encoding="utf-8") as f:
        json.dump(compatibility, f, ensure_ascii=False, indent=2)
    print(f"[ok] compatibility report: {compatibility_file}")

    if not compatibility.get("compatible", False):
        print("[warn] compatibility checks reported issues; verify before launching pretraining.")

    corpus_jsonl = output_prefix.parent / f"{output_prefix.name}.jsonl"
    n_rows = build_text_corpus(manifest, corpus_jsonl, args.text_field)
    print(f"[ok] prepared corpus rows={n_rows} at {corpus_jsonl}")

    cmd = build_preprocess_command(
        megatron_tools_dir=args.megatron_tools_dir,
        input_jsonl=corpus_jsonl,
        output_prefix=output_prefix,
        tokenizer=args.base_model,
    )

    cmd_file = output_prefix.parent / "run_megatron_preprocess.sh"
    with cmd_file.open("w", encoding="utf-8") as f:
        f.write("#!/usr/bin/env bash\nset -euo pipefail\n")
        f.write(" ".join(cmd) + "\n")
    print(f"[ok] preprocess command written to {cmd_file}")

    if args.run_preprocess:
        subprocess.run(cmd, check=True)
        print("[ok] megatron indexed dataset created")
    else:
        print("[info] dry-run mode: pass --run_preprocess to execute command")


if __name__ == "__main__":
    main()
