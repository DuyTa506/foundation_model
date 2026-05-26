"""Analyze a trained UTM model: puzzle solving visualization, extended inference, attention patterns.

Usage:
    # Basic puzzle visualization (show input → output, per-step predictions)
    python scripts/analyze_model.py --checkpoint_dir checkpoints/phase-1b-bias-T16-S123 --mode solve

    # Extended inference (run for more steps than trained)
    python scripts/analyze_model.py --checkpoint_dir checkpoints/phase-1b-bias-T16-S123 --mode extended --max_steps 32

    # Attention analysis (per-step, per-head attention patterns)
    python scripts/analyze_model.py --checkpoint_dir checkpoints/phase-1b-bias-T16-S123 --mode attention

    # All analyses
    python scripts/analyze_model.py --checkpoint_dir checkpoints/phase-1b-bias-T16-S123 --mode all
"""
import os
import sys
import json
import argparse
import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx
import orbax.checkpoint as ocp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.ut import UniversalTransformer
from models.layers import RoPEMultiHeadAttention


def load_model_from_checkpoint(checkpoint_dir, config_override=None):
    """Load model from orbax checkpoint."""
    # Load config from wandb or infer from checkpoint
    # Default config (can be overridden)
    config = {
        "vocab_size": 11, "hidden_size": 512, "num_heads": 8,
        "max_len": 81, "num_memory_tokens": 16, "max_ponder_steps": 18,
        "epsilon": 0.05, "router_init_bias": -3.0, "disable_act": False,
        "use_rmsnorm": False,
    }
    if config_override:
        config.update(config_override)

    rngs = nnx.Rngs(0)
    model = UniversalTransformer(
        vocab_size=config["vocab_size"],
        hidden_size=config["hidden_size"],
        num_heads=config["num_heads"],
        max_len=config["max_len"],
        num_memory_tokens=config["num_memory_tokens"],
        max_ponder_steps=config["max_ponder_steps"],
        epsilon=config["epsilon"],
        rngs=rngs,
        disable_act=config["disable_act"],
        router_init_bias=config["router_init_bias"],
        use_rmsnorm=config["use_rmsnorm"],
    )

    # Find latest checkpoint step
    checkpointer = ocp.StandardCheckpointer()
    manager = ocp.CheckpointManager(os.path.abspath(checkpoint_dir), checkpointer)
    latest = manager.latest_step()
    if latest is None:
        raise ValueError(f"No checkpoint found in {checkpoint_dir}")

    print(f"Loading checkpoint from step {latest}...")
    target = {"model": nnx.state(model)}
    restored = manager.restore(latest, args=ocp.args.StandardRestore(target))
    nnx.update(model, restored["model"])
    print(f"Model loaded ({config['num_memory_tokens']} mem tokens, {config['max_ponder_steps']} max steps)")
    return model, config


def load_test_puzzles(data_dir="data/sudoku-extreme-full", n=10, seed=42):
    """Load n test puzzles."""
    rng = np.random.default_rng(seed)
    inputs = np.load(os.path.join(data_dir, "test", "all__inputs.npy"), mmap_mode="r")
    labels = np.load(os.path.join(data_dir, "test", "all__labels.npy"), mmap_mode="r")
    indices = rng.choice(len(inputs), size=n, replace=False)
    return inputs[indices].copy(), labels[indices].copy()


def format_sudoku(cells, blank_token=1):
    """Format 81-cell array as 9x9 grid. Tokens: 1=blank, 2-10=digits 1-9."""
    grid = ""
    for i in range(9):
        if i > 0 and i % 3 == 0:
            grid += "------+-------+------\n"
        row = ""
        for j in range(9):
            if j > 0 and j % 3 == 0:
                row += "| "
            val = int(cells[i * 9 + j])
            if val == blank_token:
                row += ". "
            else:
                row += f"{val - 1} "  # token 2-10 → digit 1-9
        grid += row.strip() + "\n"
    return grid


def run_per_step_inference(model, inputs, num_memory_tokens, max_steps=None):
    """Run model and capture per-step predictions and halting probabilities."""
    B, L = inputs.shape
    pad_mask = jnp.ones((B, L), dtype=jnp.bool_)

    # Manually unroll to capture per-step outputs
    h = model.embed(inputs)
    h = model.mem_prepender(h)
    B, total_len, H = h.shape

    type_indices = jnp.concatenate([
        jnp.zeros((num_memory_tokens,), dtype=jnp.int32),
        jnp.ones((L,), dtype=jnp.int32)
    ])
    h = h + model.type_embed(type_indices)[None, :, :]

    rotary_indices = jnp.concatenate([
        jnp.arange(num_memory_tokens, dtype=jnp.int32),
        jnp.arange(L, dtype=jnp.int32)
    ])

    mem_mask = jnp.ones((B, num_memory_tokens), dtype=jnp.bool_)
    full_mask = jnp.concatenate([mem_mask, pad_mask], axis=1)
    q_mask = full_mask[:, None, :, None]
    k_mask = full_mask[:, None, None, :]
    attn_mask = q_mask & k_mask

    if max_steps is None:
        max_steps = model.max_ponder_steps

    step_predictions = []  # per-step argmax predictions
    step_probs = []  # per-step halting probability
    step_em = []  # per-step exact match (if we decoded at this step)

    halting_probabilities = jnp.zeros((B, total_len, 1), dtype=h.dtype)
    halted = jnp.zeros((B, total_len, 1), dtype=jnp.bool_)
    accumulated_states = jnp.zeros_like(h)

    for step in range(max_steps):
        # Use modular step embedding (wraps around if exceeding trained steps)
        step_idx = step % model.max_ponder_steps
        h_step = h + model.step_embed(jnp.array(step_idx))[None, None, :]

        h_next, _ = model.block(h_step, mask=attn_mask, rotary_indices=rotary_indices,
                                num_memory_tokens=num_memory_tokens)

        p = model.router(h_next)
        if model.disable_act:
            p = jnp.zeros_like(p)

        # Get per-step prediction (what the model would output at this step)
        step_logits = model.out_proj(h_next)
        step_pred = jnp.argmax(step_logits[:, num_memory_tokens:, :], axis=-1)
        step_predictions.append(step_pred)
        step_probs.append(float(jnp.mean(p)))

        # ACT accumulation
        p_masked = jnp.where(halted, 0.0, p)
        new_hp = halting_probabilities + p_masked
        natural_halt = (new_hp >= (1.0 - model.epsilon)) & (~halted)
        if step == max_steps - 1:
            just_halted = ~halted
        else:
            just_halted = natural_halt
        step_weight = jnp.where(just_halted, 1.0 - halting_probabilities, p_masked)
        accumulated_states = accumulated_states + step_weight * h_next
        halted = halted | just_halted
        halting_probabilities = new_hp
        h = jnp.where(halted, h, h_next)

    # Final ACT output
    final_logits = model.out_proj(accumulated_states)
    final_pred = jnp.argmax(final_logits[:, num_memory_tokens:, :], axis=-1)

    return step_predictions, step_probs, final_pred


def analyze_solve(model, config, n_puzzles=5):
    """Show how the model solves specific puzzles, step by step."""
    inputs, labels = load_test_puzzles(n=n_puzzles)
    N = config["num_memory_tokens"]

    step_preds, step_probs, final_pred = run_per_step_inference(
        model, jnp.array(inputs, dtype=jnp.int32), N)

    for i in range(n_puzzles):
        print(f"\n{'='*50}")
        print(f"PUZZLE {i+1}")
        print(f"{'='*50}")

        print("\nInput (. = blank):")
        print(format_sudoku(inputs[i]))

        print("Solution:")
        print(format_sudoku(labels[i]))

        # Show predictions at key steps
        for step_idx in [0, 4, 8, 12, 17]:
            if step_idx >= len(step_preds):
                break
            pred = np.array(step_preds[step_idx][i])
            correct = (pred == labels[i]).sum()
            print(f"Step {step_idx:>2} prediction (p={step_probs[step_idx]:.3f}, {correct}/81 correct):")
            print(format_sudoku(pred))

        # Final ACT output
        fp = np.array(final_pred[i])
        correct = (fp == labels[i]).sum()
        exact = "SOLVED" if correct == 81 else f"WRONG ({81-correct} errors)"
        print(f"Final ACT output ({correct}/81 correct — {exact}):")
        print(format_sudoku(fp))


def analyze_extended(model, config, max_steps=32, n_puzzles=20):
    """Run model for more steps than trained and measure quality at each step."""
    inputs, labels = load_test_puzzles(n=n_puzzles)
    N = config["num_memory_tokens"]

    print(f"\nExtended inference: trained for {model.max_ponder_steps} steps, running for {max_steps}")
    print(f"{'step':>4} {'p_mean':>7} {'correct':>8} {'exact_match':>12}")

    step_preds, step_probs, _ = run_per_step_inference(
        model, jnp.array(inputs, dtype=jnp.int32), N, max_steps=max_steps)

    for step_idx in range(max_steps):
        pred = np.array(step_preds[step_idx])
        per_puzzle_correct = (pred == labels).sum(axis=1)
        mean_correct = per_puzzle_correct.mean()
        exact_match = (per_puzzle_correct == 81).mean()
        trained = "  (trained)" if step_idx < model.max_ponder_steps else "  (EXTENDED)"
        print(f"{step_idx:>4} {step_probs[step_idx]:>7.4f} {mean_correct:>8.1f}/81 {exact_match:>11.1%}{trained}")


def analyze_attention(model, config, n_puzzles=20):
    """Analyze per-step, per-head attention patterns."""
    inputs, labels = load_test_puzzles(n=n_puzzles)
    N = config["num_memory_tokens"]
    B = len(inputs)
    L = 81

    if N == 0:
        print("No memory tokens — skipping attention analysis.")
        return

    # We need to capture attention weights at each step.
    # Monkey-patch the MHA to return weights.
    original_call = RoPEMultiHeadAttention.__call__

    captured_weights = []

    def capturing_call(self, q_inputs, mask=None, rotary_indices=None, num_memory_tokens=0):
        B_local, L_local, _ = q_inputs.shape
        q = self.q_proj(q_inputs).reshape((B_local, L_local, self.num_heads, self.head_dim))
        k = self.k_proj(q_inputs).reshape((B_local, L_local, self.num_heads, self.head_dim))
        v = self.v_proj(q_inputs).reshape((B_local, L_local, self.num_heads, self.head_dim))
        q = self.q_norm(q)
        k = self.k_norm(k)
        if rotary_indices is not None:
            from models.layers import apply_rope
            q = apply_rope(q, rotary_indices)
            k = apply_rope(k, rotary_indices)
        logits = jnp.einsum('bqhd,bkhd->bhqk', q, k) / jnp.sqrt(self.head_dim)
        if mask is not None:
            logits = jnp.where(mask, logits, -1e9)
        weights = jax.nn.softmax(logits, axis=-1)

        # Capture per-head attention quadrants
        quadrant = {
            "seq_to_mem": weights[:, :, N:, :N].sum(axis=-1).mean(axis=(0, 2)),  # (H,)
            "seq_to_seq": weights[:, :, N:, N:].sum(axis=-1).mean(axis=(0, 2)),
            "mem_to_mem": weights[:, :, :N, :N].sum(axis=-1).mean(axis=(0, 2)),
            "mem_to_seq": weights[:, :, :N, N:].sum(axis=-1).mean(axis=(0, 2)),
        }
        captured_weights.append(quadrant)

        output = jnp.einsum('bhqk,bkhd->bqhd', weights, v)
        output = output.reshape((B_local, L_local, self.in_features))

        attn_diag = {}
        if num_memory_tokens > 0:
            attn_diag["attn_seq_to_mem"] = weights[:, :, N:, :N].sum(axis=-1).mean()
            attn_diag["attn_seq_to_seq"] = weights[:, :, N:, N:].sum(axis=-1).mean()
            attn_diag["attn_mem_to_mem"] = weights[:, :, :N, :N].sum(axis=-1).mean()
            attn_diag["attn_mem_to_seq"] = weights[:, :, :N, N:].sum(axis=-1).mean()

        return self.out_proj(output), attn_diag

    # Patch and run
    RoPEMultiHeadAttention.__call__ = capturing_call
    pad_mask = jnp.ones((B, L), dtype=jnp.bool_)
    _ = model(jnp.array(inputs, dtype=jnp.int32), pad_mask)
    RoPEMultiHeadAttention.__call__ = original_call

    num_heads = config["num_heads"]
    print(f"\nAttention analysis ({len(captured_weights)} steps, {num_heads} heads)")
    print(f"\nPer-step attention quadrants (averaged across heads):")
    print(f"{'step':>4} {'s→m':>6} {'s→s':>6} {'m→m':>6} {'m→s':>6}")
    for step, q in enumerate(captured_weights):
        print(f"{step:>4} {float(q['seq_to_mem'].mean()):>6.3f} {float(q['seq_to_seq'].mean()):>6.3f} "
              f"{float(q['mem_to_mem'].mean()):>6.3f} {float(q['mem_to_seq'].mean()):>6.3f}")

    print(f"\nPer-head specialization (averaged across steps):")
    print(f"{'head':>4} {'s→m':>6} {'s→s':>6} {'m→m':>6} {'m→s':>6} {'role':>15}")
    for h in range(num_heads):
        s2m = np.mean([float(q["seq_to_mem"][h]) for q in captured_weights])
        s2s = np.mean([float(q["seq_to_seq"][h]) for q in captured_weights])
        m2m = np.mean([float(q["mem_to_mem"][h]) for q in captured_weights])
        m2s = np.mean([float(q["mem_to_seq"][h]) for q in captured_weights])
        # Classify head role
        if s2m > 0.4:
            role = "mem-reader"
        elif m2s > 0.7:
            role = "mem-writer"
        elif m2m > 0.6:
            role = "mem-internal"
        else:
            role = "puzzle-focused"
        print(f"{h:>4} {s2m:>6.3f} {s2s:>6.3f} {m2m:>6.3f} {m2s:>6.3f} {role:>15}")


def main():
    parser = argparse.ArgumentParser(description="Analyze trained UTM model")
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--mode", type=str, default="all", choices=["solve", "extended", "attention", "all"])
    parser.add_argument("--max_steps", type=int, default=32, help="Max steps for extended inference")
    parser.add_argument("--n_puzzles", type=int, default=5, help="Number of puzzles to analyze")
    parser.add_argument("--num_memory_tokens", type=int, default=16)
    parser.add_argument("--data_dir", type=str, default="data/sudoku-extreme-full")

    # Model config overrides
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--max_ponder_steps", type=int, default=18)
    parser.add_argument("--disable_act", action="store_true")
    parser.add_argument("--use_rmsnorm", action="store_true")
    args = parser.parse_args()

    config_override = {
        "num_memory_tokens": args.num_memory_tokens,
        "hidden_size": args.hidden_size,
        "num_heads": args.num_heads,
        "max_ponder_steps": args.max_ponder_steps,
        "disable_act": args.disable_act,
        "use_rmsnorm": args.use_rmsnorm,
    }

    model, config = load_model_from_checkpoint(args.checkpoint_dir, config_override)

    if args.mode in ("solve", "all"):
        print("\n" + "=" * 60)
        print("PUZZLE SOLVING VISUALIZATION")
        print("=" * 60)
        analyze_solve(model, config, n_puzzles=args.n_puzzles)

    if args.mode in ("extended", "all"):
        print("\n" + "=" * 60)
        print("EXTENDED INFERENCE")
        print("=" * 60)
        analyze_extended(model, config, max_steps=args.max_steps, n_puzzles=20)

    if args.mode in ("attention", "all"):
        print("\n" + "=" * 60)
        print("ATTENTION ANALYSIS")
        print("=" * 60)
        analyze_attention(model, config, n_puzzles=20)


if __name__ == "__main__":
    main()
