#!/usr/bin/env python3
import argparse
import subprocess


def main() -> None:
    parser = argparse.ArgumentParser(description="Run lighteval tasks for math/science checkpoints.")
    parser.add_argument("--model_path", required=True, help="Model path or HF repo id.")
    parser.add_argument("--tasks", default="gsm8k,math,arc_challenge,sciq")
    parser.add_argument("--backend", default="vllm", choices=["vllm", "accelerate"])
    parser.add_argument("--output_dir", default="outputs/eval")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    model_args = (
        f"model_name={args.model_path},dtype=bfloat16,max_model_length=4096,"
        "gpu_memory_utilization=0.85,generation_parameters={max_new_tokens:2048,temperature:0.0}"
    )

    cmd = [
        "lighteval",
        args.backend,
        model_args,
        "custom",
        args.tasks,
        "--output-dir",
        args.output_dir,
    ]

    print("[eval] " + " ".join(cmd))
    if not args.dry_run:
        subprocess.run(cmd, check=True)
        print(f"[ok] evaluation artifacts at {args.output_dir}")
    else:
        print("[info] dry-run only")


if __name__ == "__main__":
    main()
