#!/usr/bin/env python3
"""
Synthesize Vietnamese reasoning data for thinking-mode SFT and RLVR.

Strategy:
    Tier 2: translate EN CoT datasets to Vietnamese, verify answer integrity.
    Tier 3: generate VI CoT traces from a teacher model (DeepSeek-R1 / Qwen3),
            reject-sample on the verifiable answer, apply language-ID filter
            to drop traces that drift to English/Chinese.

Outputs:
    - outputs/vi_reasoning/thinking_sft.jsonl  (VI CoT traces for SFT, mode=think)
    - outputs/rl_data/vi_math_prompts.jsonl    (VI prompt + ground-truth answer for RLVR)

Usage:
    # Tier 2: translate + verify
    python scripts/data/synth_vi_reasoning.py --mode translate \
        --source_dataset AI-MO/NuminaMath-CoT \
        --output_dir outputs/vi_reasoning \
        --max_samples 50000

    # Tier 3: teacher distillation
    python scripts/data/synth_vi_reasoning.py --mode distill \
        --teacher deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
        --vi_prompts_jsonl outputs/rl_data/vi_math_prompts.jsonl \
        --output_dir outputs/vi_reasoning \
        --max_samples 100000
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Iterator

import unicodedata


# ─── Answer extraction (reuse from rewards) ───────────────────────────────────

_BOXED_RE = re.compile(r"\\boxed\{([^}]*)\}", re.DOTALL)
_LAST_NUM_RE = re.compile(r"([+-]?\d[\d\s,./]*\d|\d)")
_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)


def extract_answer(text: str) -> str | None:
    m = _BOXED_RE.findall(text)
    if m:
        return m[-1].strip()
    nums = _LAST_NUM_RE.findall(text)
    return nums[-1].strip().replace(",", "") if nums else None


def _answers_close(a: str | None, b: str | None) -> bool:
    if a is None or b is None:
        return False
    a = a.strip().replace(",", "")
    b = b.strip().replace(",", "")
    if a == b:
        return True
    try:
        return abs(float(a) - float(b)) < 1e-5
    except (ValueError, TypeError):
        pass
    try:
        from sympy import simplify, sympify
        return simplify(sympify(a) - sympify(b)) == 0
    except Exception:
        return False


# ─── Language ID ──────────────────────────────────────────────────────────────

def _is_mostly_vietnamese(text: str, min_conf: float = 0.50) -> bool:
    """Return True if the majority of the text is Vietnamese."""
    try:
        import fasttext
        from huggingface_hub import hf_hub_download
        model_path = hf_hub_download("cis-lmu/glotlid", "model.bin")
        _mdl = fasttext.load_model(model_path)
        labels, scores = _mdl.predict(text.replace("\n", " ")[:512], k=1)
        label = labels[0].replace("__label__", "")
        return ("vie" in label or "vi" == label) and float(scores[0]) >= min_conf
    except Exception:
        vi_chars = set("ăâêôơưđĂÂÊÔƠƯĐ")
        vi_frac = sum(1 for c in text if c in vi_chars) / max(len(text), 1)
        return vi_frac > 0.005


# ─── Tier 2: Translate EN CoT -> VI with answer verification ─────────────────

def translate_and_verify(
    source_dataset: str,
    output_dir: Path,
    max_samples: int,
    translator_model: str,
) -> None:
    """
    Load EN CoT dataset, translate question+solution to Vietnamese using an LLM,
    keep only samples where the translated solution still yields the correct answer.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    sft_out = output_dir / "thinking_sft_translated.jsonl"
    prompt_out = output_dir.parent / "rl_data" / "vi_math_prompts_translated.jsonl"
    prompt_out.parent.mkdir(parents=True, exist_ok=True)

    print(f"[synth] loading {source_dataset} ...")
    try:
        from datasets import load_dataset
        ds = load_dataset(source_dataset, split="train", trust_remote_code=True)
    except Exception as e:
        print(f"[error] could not load {source_dataset}: {e}")
        sys.exit(1)

    print(f"[synth] {len(ds)} EN samples to translate")

    try:
        from vllm import LLM, SamplingParams
        llm = LLM(
            model=translator_model,
            dtype="bfloat16",
            gpu_memory_utilization=0.85,
            max_model_len=4096,
        )
        params = SamplingParams(temperature=0.3, max_tokens=2048)
    except ImportError:
        print("[warn] vLLM not available; translation requires vLLM")
        sys.exit(1)

    sft_f = sft_out.open("w", encoding="utf-8")
    prompt_f = prompt_out.open("w", encoding="utf-8")
    kept = 0
    dropped = 0

    for i, row in enumerate(ds):
        if i >= max_samples:
            break

        question = row.get("problem") or row.get("question") or row.get("query", "")
        solution = row.get("solution") or row.get("answer", "")
        gold = extract_answer(solution) or solution

        if not question or not gold:
            continue

        # Translate question + solution to Vietnamese
        translate_prompt = (
            f"Dịch câu hỏi và lời giải toán học sau sang tiếng Việt. "
            f"Giữ nguyên các ký hiệu toán học LaTeX, số, và đáp án cuối.\n\n"
            f"Câu hỏi: {question}\n\nLời giải: {solution}"
        )
        outputs = llm.generate([translate_prompt], params)
        translated = outputs[0].outputs[0].text.strip()

        # Split back into question/solution (heuristic: split at "Lời giải:")
        parts = re.split(r"Lời giải[:：]", translated, maxsplit=1)
        if len(parts) == 2:
            vi_q, vi_sol = parts[0].strip(), parts[1].strip()
        else:
            vi_q = translated[:len(translated)//2]
            vi_sol = translated[len(translated)//2:]

        # Remove "Câu hỏi:" prefix if present
        vi_q = re.sub(r"^(?:Câu hỏi|Question)[:：]\s*", "", vi_q, flags=re.IGNORECASE)

        # Verify: extract answer from translated solution and check vs gold
        pred = extract_answer(vi_sol)
        if not _answers_close(pred, gold):
            dropped += 1
            continue

        # Write SFT sample
        sft_row = {
            "prompt": vi_q,
            "response": vi_sol,
            "mode": "think",
            "language": "vi",
            "source": source_dataset,
            "source_type": "translated_verified",
        }
        sft_f.write(json.dumps(sft_row, ensure_ascii=False) + "\n")

        # Write RLVR prompt
        prompt_row = {"prompt": vi_q, "answer": gold, "language": "vi"}
        prompt_f.write(json.dumps(prompt_row, ensure_ascii=False) + "\n")

        kept += 1
        if kept % 100 == 0:
            print(f"[synth] translated+verified: kept={kept} dropped={dropped}")

    sft_f.close()
    prompt_f.close()
    print(f"[ok] translated: kept={kept} dropped={dropped}  -> {sft_out}")


# ─── Tier 3: Teacher distillation with reject sampling ───────────────────────

def distill_from_teacher(
    teacher_model: str,
    vi_prompts_jsonl: Path,
    output_dir: Path,
    max_samples: int,
    samples_per_prompt: int = 4,
) -> None:
    """
    Generate Vietnamese CoT from a teacher model for VI prompts.
    Reject-sample on answer correctness and Vietnamese language consistency.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "thinking_sft_distilled.jsonl"

    if not vi_prompts_jsonl.exists():
        print(f"[error] VI prompts not found: {vi_prompts_jsonl}")
        print("  Run with --mode translate first, or provide the prompts manually.")
        sys.exit(1)

    prompts = [r for r in (json.loads(l) for l in vi_prompts_jsonl.read_text().splitlines() if l.strip())]
    print(f"[synth] {len(prompts)} VI prompts for distillation from {teacher_model}")

    try:
        from vllm import LLM, SamplingParams
        llm = LLM(
            model=teacher_model,
            dtype="bfloat16",
            gpu_memory_utilization=0.85,
            max_model_len=8192,
        )
        params = SamplingParams(
            temperature=0.8,
            top_p=0.95,
            max_tokens=2048,
        )
    except ImportError:
        print("[error] vLLM required for teacher distillation")
        sys.exit(1)

    out_f = out_path.open("w", encoding="utf-8")
    kept = 0
    dropped_wrong = 0
    dropped_lang = 0

    for i, row in enumerate(prompts):
        if kept >= max_samples:
            break

        prompt = row["prompt"]
        gold = row.get("answer", "")

        # Format prompt for teacher (thinking-mode)
        teacher_input = (
            f"<|im_start|>user\n{prompt}<|im_end|>\n"
            f"<|im_start|>assistant\n<think>\n"
        )

        completions = llm.generate([teacher_input] * samples_per_prompt, params)

        for out in completions:
            if kept >= max_samples:
                break
            text = out.outputs[0].text.strip()
            full_response = "<think>\n" + text  # re-add think prefix

            # 1. Answer correctness check
            pred = extract_answer(full_response)
            if gold and not _answers_close(pred, gold):
                dropped_wrong += 1
                continue

            # 2. Language consistency: the <think> trace should be Vietnamese
            think_m = _THINK_RE.search(full_response)
            trace = think_m.group(1) if think_m else full_response
            if not _is_mostly_vietnamese(trace[:500]):
                dropped_lang += 1
                continue

            # Clean up
            final_answer = pred or ""
            sft_row = {
                "prompt": prompt,
                "response": full_response,
                "mode": "think",
                "language": "vi",
                "source": teacher_model,
                "source_type": "teacher_distilled",
                "verified_answer": final_answer,
            }
            out_f.write(json.dumps(sft_row, ensure_ascii=False) + "\n")
            kept += 1

        if (i + 1) % 50 == 0:
            print(f"[synth] prompt {i+1}/{len(prompts)}  kept={kept} "
                  f"dropped_wrong={dropped_wrong} dropped_lang={dropped_lang}")

    out_f.close()
    print(f"[ok] distilled: kept={kept}  -> {out_path}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Synthesize Vietnamese reasoning data for thinking-mode SFT + RLVR."
    )
    parser.add_argument(
        "--mode", choices=["translate", "distill"], required=True,
        help="translate: translate EN CoT to VI + verify. "
             "distill: generate VI CoT from teacher model.",
    )
    parser.add_argument("--source_dataset", default="AI-MO/NuminaMath-CoT",
                        help="(translate mode) HF dataset to translate.")
    parser.add_argument("--translator_model",
                        default="Qwen/Qwen2.5-7B-Instruct",
                        help="(translate mode) LLM to use for translation.")
    parser.add_argument("--teacher",
                        default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
                        help="(distill mode) Teacher model for CoT generation.")
    parser.add_argument("--vi_prompts_jsonl",
                        default="outputs/rl_data/vi_math_prompts_translated.jsonl",
                        help="(distill mode) VI prompts with ground-truth answers.")
    parser.add_argument("--output_dir", default="outputs/vi_reasoning")
    parser.add_argument("--max_samples", type=int, default=50_000)
    parser.add_argument("--samples_per_prompt", type=int, default=4,
                        help="(distill mode) number of rollouts per prompt for reject sampling.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    if args.mode == "translate":
        translate_and_verify(
            source_dataset=args.source_dataset,
            output_dir=output_dir,
            max_samples=args.max_samples,
            translator_model=args.translator_model,
        )
    else:
        distill_from_teacher(
            teacher_model=args.teacher,
            vi_prompts_jsonl=Path(args.vi_prompts_jsonl),
            output_dir=output_dir,
            max_samples=args.max_samples,
            samples_per_prompt=args.samples_per_prompt,
        )


if __name__ == "__main__":
    main()
