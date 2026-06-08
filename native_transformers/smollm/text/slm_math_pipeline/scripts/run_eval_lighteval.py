#!/usr/bin/env python3
"""
Evaluation runner for EN+VI math/science + long-context benchmarks.

Fixes the broken lighteval invocation in the original script:
  - Correct task spec format: "suite|task|num_fewshot|truncate"
  - --custom-tasks only when a custom task file is provided
  - Adds Vietnamese benchmarks (VMLU, ViMMLU, VI-translated GSM8K)
  - Adds long-context eval (RULER, needle-in-a-haystack) at 32k/128k

Usage:
    python scripts/run_eval_lighteval.py \\
        --model_path outputs/rl \\
        --stage final

    python scripts/run_eval_lighteval.py \\
        --model_path outputs/pretrain_32k \\
        --stage longctx_32k

    python scripts/run_eval_lighteval.py \\
        --model_path outputs/pretrain \\
        --tasks math,vi_math
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


# ─── Task registries ──────────────────────────────────────────────────────────

# Standard lighteval task format: "suite|task_name|num_fewshot|truncate"
TASK_SETS = {
    "math": [
        "lighteval|gsm8k|5|0",
        "lighteval|math|4|0",
    ],
    "science": [
        "lighteval|arc:challenge|25|0",
        "lighteval|arc:easy|25|0",
        "lighteval|sciq|0|0",
        "lighteval|mmlu:STEM|5|0",
        "lighteval|openbookqa|0|0",
    ],
    "vi_math": [
        # VI-translated/native math; requires lighteval[multilingual]
        "community|vi_gsm8k|5|0",           # VI-translated GSM8K
        "community|vi_math_word|3|0",        # VI math word problems
    ],
    "vi_general": [
        "community|vmlu|5|0",               # VMLU / ViMMLU
        "community|vi_mmlu|5|0",
    ],
    "longctx_32k": [
        "custom|ruler_32k|0|0",
        "custom|needle_32k|0|0",
    ],
    "longctx_128k": [
        "custom|ruler_128k|0|0",
        "custom|needle_128k|0|0",
        "custom|longbench|0|0",
    ],
}

# Stage -> which task sets to run
STAGE_TASKS = {
    "pretrain": ["math", "science"],
    "midtrain": ["math", "science", "vi_general"],
    "sft": ["math", "science", "vi_math", "vi_general"],
    "rl": ["math", "science", "vi_math", "vi_general"],
    "final": ["math", "science", "vi_math", "vi_general"],
    "longctx_32k": ["math", "longctx_32k"],
    "longctx_128k": ["math", "longctx_32k", "longctx_128k"],
}


def build_task_list(task_keys: list[str], custom_task_file: str | None) -> tuple[list[str], bool]:
    """Return (task_list, uses_custom_tasks)."""
    tasks: list[str] = []
    has_custom = False
    for key in task_keys:
        task_set = TASK_SETS.get(key, [f"lighteval|{key}|0|0"])
        for t in task_set:
            if t.startswith("custom|") or t.startswith("community|"):
                has_custom = True
            tasks.append(t)
    return tasks, has_custom


def run_eval(
    model_path: str,
    task_keys: list[str],
    output_dir: str,
    backend: str,
    max_context: int,
    custom_task_file: str | None,
    dry_run: bool,
) -> None:
    tasks, needs_custom = build_task_list(task_keys, custom_task_file)

    # Model args string for lighteval
    model_args = (
        f"model_name={model_path},"
        f"dtype=bfloat16,"
        f"max_model_length={max_context},"
        "gpu_memory_utilization=0.85,"
        "generation_parameters={max_new_tokens:2048,temperature:0.0}"
    )

    # Task string: comma-separated
    task_str = ",".join(tasks)

    cmd = [
        "lighteval",
        backend,
        model_args,
    ]

    if needs_custom and custom_task_file and Path(custom_task_file).exists():
        cmd.extend(["--custom-tasks", custom_task_file])
    elif needs_custom:
        print(f"[eval] WARNING: custom/community tasks requested but "
              f"--custom_task_file not provided. "
              f"Skipping community tasks.")
        # Remove custom tasks from the list
        tasks = [t for t in tasks if not t.startswith(("custom|", "community|"))]
        task_str = ",".join(tasks)
        if not tasks:
            print("[eval] no standard tasks left; nothing to run.")
            return

    cmd.extend([task_str, "--output-dir", output_dir])

    print(f"[eval] command: {' '.join(cmd)}")
    print(f"[eval] tasks ({len(tasks)}): {task_str}")

    if dry_run:
        print("[eval] dry-run; not executing.")
        return

    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f"[eval] lighteval returned non-zero exit code {result.returncode}")
    else:
        print(f"[ok] eval artifacts at {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run lighteval tasks for EN+VI math/science SLM evaluation."
    )
    parser.add_argument("--model_path", required=True, help="Model dir or HF repo id.")
    parser.add_argument(
        "--stage",
        default="final",
        choices=list(STAGE_TASKS.keys()),
        help="Eval stage (determines task suite).",
    )
    parser.add_argument(
        "--tasks",
        help="Comma-separated task set keys (overrides --stage). "
             "E.g. math,vi_math,science",
    )
    parser.add_argument("--backend", default="vllm", choices=["vllm", "accelerate"])
    parser.add_argument("--output_dir", default="outputs/eval")
    parser.add_argument("--max_context", type=int, default=4096,
                        help="Max model context length for eval. "
                             "Set 32768 or 131072 for long-context stages.")
    parser.add_argument("--custom_task_file",
                        help="Path to custom lighteval task definitions Python file.")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    if args.tasks:
        task_keys = [t.strip() for t in args.tasks.split(",")]
    else:
        task_keys = STAGE_TASKS.get(args.stage, ["math", "science"])

    print(f"[eval] model={args.model_path}  stage={args.stage}  "
          f"backend={args.backend}  max_ctx={args.max_context}")

    run_eval(
        model_path=args.model_path,
        task_keys=task_keys,
        output_dir=args.output_dir,
        backend=args.backend,
        max_context=args.max_context,
        custom_task_file=args.custom_task_file,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
