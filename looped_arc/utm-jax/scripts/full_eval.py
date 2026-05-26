"""Evaluate checkpoints on the FULL 423K test set.

Resolves the ±10pp training-curve oscillation by evaluating once on the
complete test set. Pulls checkpoints from GCS if not available locally.
Uploads results to GCS.

Usage:
    # Evaluate all known checkpoints (pulls from GCS as needed)
    python scripts/full_eval.py --all

    # Evaluate a single checkpoint
    python scripts/full_eval.py --checkpoint_dir checkpoints/phase-2-bias-T16-S0

    # Custom GCS bucket
    python scripts/full_eval.py --all --gcs_bucket gs://utm-checkpoints
"""
import os
import sys
import subprocess
import argparse
import json
import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx
import orbax.checkpoint as ocp
import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.ut import UniversalTransformer


# ─── Checkpoint configs ─────────────────────────────────────────────
# Maps run name → model config overrides (defaults: T=16, ACT, DerfNorm)
KNOWN_CONFIGS = {
    # Phase 1b: bias=-3, S=123
    "phase-1b-bias-T0-S123":   {"T": 0},
    "phase-1b-bias-T8-S123":   {"T": 8},
    "phase-1b-bias-T16-S123":  {"T": 16},
    "phase-1b-bias-T32-S123":  {"T": 32},
    "phase-1b-bias-T64-S123":  {"T": 64},
    # Phase 1d
    "phase-1d-T4-S123":        {"T": 4},
    "phase-1d-T8-S42":         {"T": 8},
    "phase-1d-warmup-T16-S123":{"T": 16},
    # Phase 1c
    "phase-1c-lambda001-T16-S123": {"T": 16},
    # Phase 2: bias=-3, S=0
    "phase-2-bias-T0-S0":      {"T": 0},
    "phase-2-bias-T8-S0":      {"T": 8},
    "phase-2-bias-T16-S0":     {"T": 16},
    "phase-2-bias-T16-S42":    {"T": 16},
    "phase-2-bias-T32-S0":     {"T": 32},
    "phase-2-bias-T32-S42":    {"T": 32},
    "phase-2-bias-T64-S0":     {"T": 64},
    # Long runs
    "long-run-v2-T16-S123":    {"T": 16},
    "long-run-T16-S123-warmup":{"T": 16},
    # Ablations
    "ablation-fixed18-T16-S123":      {"T": 16, "disable_act": True},
    "ablation-rmsnorm-bias0-T16-S42": {"T": 16, "use_rmsnorm": True},
    # Submission fixes (when available)
    "sub-fixed18-T16-S0":      {"T": 16, "disable_act": True},
    "sub-fixed18-T16-S42":     {"T": 16, "disable_act": True},
    "sub-warmup-T16-S0":       {"T": 16},
    "sub-warmup-T16-S42":      {"T": 16},
    "sub-T0-S42":              {"T": 0},
    "sub-T4-S0":               {"T": 4},
    "sub-T4-S42":              {"T": 4},
    "sub-T64-S42":             {"T": 64},
}

# GCS subdirectories to search for checkpoints
GCS_SUBDIRS = ["phase2", "submission", "final", "long-run-v2", "small-data-reg",
               "long-run", "trm-matched"]


def gsutil_available():
    try:
        subprocess.run(["gsutil", "version"], capture_output=True, timeout=5)
        return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def pull_checkpoint_from_gcs(run_name, local_dir, gcs_bucket):
    """Try to pull a checkpoint from GCS if not available locally."""
    local_path = os.path.join(local_dir, run_name)
    # Check if we already have a checkpoint locally
    if os.path.exists(local_path):
        steps = [d for d in os.listdir(local_path) if d.isdigit()]
        if steps:
            return True  # already have it

    if not gsutil_available():
        return False

    # Search GCS subdirectories for this run
    for subdir in GCS_SUBDIRS:
        gcs_path = f"{gcs_bucket}/{subdir}/{run_name}"
        try:
            result = subprocess.run(
                ["gsutil", "-q", "stat", f"{gcs_path}/.completed"],
                capture_output=True, timeout=10)
            if result.returncode == 0:
                print(f"  Pulling {run_name} from {gcs_path}...")
                os.makedirs(local_path, exist_ok=True)
                subprocess.run(
                    ["gsutil", "-m", "rsync", "-r", f"{gcs_path}/", f"{local_path}/"],
                    capture_output=True, timeout=600)
                return True
        except subprocess.TimeoutExpired:
            continue

    # Also try flat bucket path
    gcs_path = f"{gcs_bucket}/{run_name}"
    try:
        result = subprocess.run(
            ["gsutil", "ls", f"{gcs_path}/"],
            capture_output=True, timeout=10)
        if result.returncode == 0 and result.stdout:
            print(f"  Pulling {run_name} from {gcs_path}...")
            os.makedirs(local_path, exist_ok=True)
            subprocess.run(
                ["gsutil", "-m", "rsync", "-r", f"{gcs_path}/", f"{local_path}/"],
                capture_output=True, timeout=600)
            return True
    except subprocess.TimeoutExpired:
        pass

    return False


def load_model(checkpoint_dir, num_memory_tokens=16, max_ponder_steps=18,
               disable_act=False, use_rmsnorm=False):
    """Load model from checkpoint using raw restore."""
    rngs = nnx.Rngs(0)
    model = UniversalTransformer(
        vocab_size=11, hidden_size=512, num_heads=8, max_len=81,
        num_memory_tokens=num_memory_tokens, max_ponder_steps=max_ponder_steps,
        epsilon=0.05, rngs=rngs, disable_act=disable_act,
        router_init_bias=-3.0, use_rmsnorm=use_rmsnorm,
    )
    ckpt_base = os.path.abspath(checkpoint_dir)
    manager = ocp.CheckpointManager(ckpt_base, ocp.StandardCheckpointer())
    step = manager.latest_step()
    if step is None:
        return None, None

    ckpt_path = os.path.join(ckpt_base, str(step), "default")
    if not os.path.exists(ckpt_path):
        return None, None

    raw = ocp.StandardCheckpointer().restore(ckpt_path)
    raw_state = raw.get("ema_model", raw.get("model"))
    _load_raw_into_module(model, raw_state)
    return model, step


def _load_raw_into_module(module, raw_dict):
    """Recursively load raw checkpoint dict into an NNX module."""
    for key, value in raw_dict.items():
        attr = getattr(module, key, None)
        if attr is None:
            continue
        if isinstance(value, dict) and 'value' in value:
            if isinstance(attr, nnx.Variable):
                attr.value = jnp.array(value['value'])
        elif isinstance(value, dict):
            _load_raw_into_module(attr, value)


@jax.jit
def predict_batch(model, batch_inp, pad_mask):
    logits, _, halt_steps, _ = model(batch_inp, pad_mask)
    return logits, halt_steps


def evaluate_full(model, data_dir, num_memory_tokens, batch_size=256):
    """Evaluate on the full test set."""
    inputs = np.load(os.path.join(data_dir, "test", "all__inputs.npy"), mmap_mode="r")
    labels = np.load(os.path.join(data_dir, "test", "all__labels.npy"), mmap_mode="r")
    total = len(inputs)

    total_correct_cells = 0
    total_exact = 0
    total_puzzles = 0
    halt_sum = 0.0

    for start in tqdm.tqdm(range(0, total, batch_size), desc="  Eval", ncols=80):
        end = min(start + batch_size, total)
        batch_inp = jnp.array(inputs[start:end], dtype=jnp.int32)
        B = batch_inp.shape[0]
        pad_mask = jnp.ones((B, 81), dtype=jnp.bool_)

        logits, halt_steps = predict_batch(model, batch_inp, pad_mask)
        preds = np.array(jnp.argmax(logits[:, num_memory_tokens:, :], axis=-1))

        correct = (preds == labels[start:end]).sum(axis=1)
        total_correct_cells += int(correct.sum())
        total_exact += int((correct == 81).sum())
        total_puzzles += B
        halt_sum += float(jnp.mean(halt_steps)) * B

    return {
        "total_puzzles": total_puzzles,
        "exact_match": total_exact / total_puzzles,
        "cell_accuracy": total_correct_cells / (total_puzzles * 81),
        "mean_halt": halt_sum / total_puzzles,
        "puzzles_solved": total_exact,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", type=str, default=None)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--data_dir", type=str, default="data/sudoku-extreme-full")
    parser.add_argument("--checkpoint_root", type=str, default="checkpoints")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--output", type=str, default="full_eval_results.json")
    parser.add_argument("--gcs_bucket", type=str, default="gs://utm-checkpoints")
    args = parser.parse_args()

    results = {}

    if args.all:
        checkpoints_to_eval = []
        for name, cfg in sorted(KNOWN_CONFIGS.items()):
            local_path = os.path.join(args.checkpoint_root, name)
            # Try local first, then GCS
            if not os.path.exists(local_path) or not any(
                d.isdigit() for d in os.listdir(local_path) if os.path.isdir(os.path.join(local_path, d))
            ):
                if not pull_checkpoint_from_gcs(name, args.checkpoint_root, args.gcs_bucket):
                    continue
            # Verify we have a step directory
            steps = [d for d in os.listdir(local_path) if d.isdigit()]
            if steps:
                checkpoints_to_eval.append((name, local_path, cfg))

        print(f"Found {len(checkpoints_to_eval)} checkpoints to evaluate")
    elif args.checkpoint_dir:
        name = os.path.basename(args.checkpoint_dir)
        cfg = KNOWN_CONFIGS.get(name, {"T": 16})
        checkpoints_to_eval = [(name, args.checkpoint_dir, cfg)]
    else:
        print("Specify --checkpoint_dir or --all")
        return

    for name, path, cfg in checkpoints_to_eval:
        print(f"\n{'='*60}")
        print(f"Evaluating: {name} (T={cfg.get('T', 16)})")
        T = cfg.get("T", 16)
        model, step = load_model(
            path, num_memory_tokens=T,
            disable_act=cfg.get("disable_act", False),
            use_rmsnorm=cfg.get("use_rmsnorm", False),
        )
        if model is None:
            print(f"  No valid checkpoint, skipping.")
            continue

        print(f"  Loaded step {step}")
        metrics = evaluate_full(model, args.data_dir, T, batch_size=args.batch_size)
        results[name] = {**metrics, "step": step, "T": T, **{k: v for k, v in cfg.items()}}
        print(f"  RESULT: EM={metrics['exact_match']*100:.2f}% "
              f"({metrics['puzzles_solved']}/{metrics['total_puzzles']}) "
              f"cell_acc={metrics['cell_accuracy']*100:.2f}% "
              f"halt={metrics['mean_halt']:.1f}")

    # Save results locally
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")

    # Upload to GCS
    if gsutil_available():
        gcs_output = f"{args.gcs_bucket}/eval_results/{args.output}"
        subprocess.run(["gsutil", "cp", args.output, gcs_output], capture_output=True, timeout=30)
        print(f"Results uploaded to {gcs_output}")

    # Summary table
    print(f"\n{'='*70}")
    print(f"FULL EVALUATION RESULTS (423K test set)")
    print(f"{'='*70}")
    print(f"{'Run':42s} {'T':>3} {'EM%':>7} {'Solved':>8} {'Halt':>6}")
    print(f"{'-'*70}")
    for name, m in sorted(results.items(), key=lambda x: (x[1].get('T', 0), x[0])):
        print(f"{name:42s} {m.get('T','?'):>3} {m['exact_match']*100:>6.2f}% "
              f"{m['puzzles_solved']:>6}/{m['total_puzzles']} {m['mean_halt']:>6.1f}")


if __name__ == "__main__":
    main()
