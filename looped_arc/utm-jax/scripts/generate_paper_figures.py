"""Generate paper-quality figures and analysis from trained UTM checkpoints.

Produces:
  1. Puzzle solving visualization (input → per-step → output grids)
  2. Extended inference analysis (what happens beyond trained steps)
  3. Per-step, per-head attention heatmaps
  4. Step-weight distribution plots
  5. Comparison across models (T=8, T=16, T=0, trapped)

Usage:
    python scripts/generate_paper_figures.py --output_dir analysis_output
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

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import ListedColormap

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.ut import UniversalTransformer
from models.layers import RoPEMultiHeadAttention, apply_rope


# ─── Styling ───────────────────────────────────────────────────────
plt.rcParams.update({
    'font.size': 11, 'axes.titlesize': 13, 'axes.labelsize': 11,
    'xtick.labelsize': 9, 'ytick.labelsize': 9, 'legend.fontsize': 9,
    'figure.dpi': 150, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
    'font.family': 'sans-serif',
})

COLORS = {
    'success': '#2ecc71', 'fail': '#e74c3c', 'neutral': '#3498db',
    'mem': '#9b59b6', 'seq': '#2980b9', 'act': '#e67e22',
}


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


def load_model(checkpoint_dir, num_memory_tokens=16, max_ponder_steps=18,
               disable_act=False, use_rmsnorm=False, hidden_size=512, num_heads=8):
    """Load a model from checkpoint."""
    rngs = nnx.Rngs(0)
    model = UniversalTransformer(
        vocab_size=11, hidden_size=hidden_size, num_heads=num_heads,
        max_len=81, num_memory_tokens=num_memory_tokens,
        max_ponder_steps=max_ponder_steps, epsilon=0.05, rngs=rngs,
        disable_act=disable_act, router_init_bias=-3.0, use_rmsnorm=use_rmsnorm,
    )
    ckpt_base = os.path.abspath(checkpoint_dir)
    manager = ocp.CheckpointManager(ckpt_base, ocp.StandardCheckpointer())
    step = manager.latest_step()
    if step is None:
        raise ValueError(f"No checkpoint in {checkpoint_dir}")

    # Raw restore bypasses strict tree matching (avoids optimizer version mismatch)
    ckpt_path = os.path.join(ckpt_base, str(step), "default")
    raw = ocp.StandardCheckpointer().restore(ckpt_path)
    raw_state = raw.get("ema_model", raw.get("model"))
    # Direct recursive attribute assignment — JAX arrays are immutable so
    # flatten-and-zip doesn't work (assignments are no-ops on array leaves)
    _load_raw_into_module(model, raw_state)
    print(f"  Loaded {checkpoint_dir} (step {step}, T={num_memory_tokens})")
    return model, step


def load_puzzles(data_dir="data/sudoku-extreme-full", n=50, seed=42):
    """Load n test puzzles, stratified by difficulty (number of blanks).

    Returns puzzles from three difficulty buckets (easy/medium/hard)
    so the visualization shows a representative spread.
    """
    rng = np.random.default_rng(seed)
    inputs = np.load(os.path.join(data_dir, "test", "all__inputs.npy"), mmap_mode="r")
    labels = np.load(os.path.join(data_dir, "test", "all__labels.npy"), mmap_mode="r")

    # Count blanks (token=1) per puzzle as difficulty proxy
    blanks = (inputs == 1).sum(axis=1)
    terciles = np.percentile(blanks, [33, 67])

    easy = np.where(blanks <= terciles[0])[0]
    medium = np.where((blanks > terciles[0]) & (blanks <= terciles[1]))[0]
    hard = np.where(blanks > terciles[1])[0]

    per_bucket = max(1, n // 3)
    selected = np.concatenate([
        rng.choice(easy, size=min(per_bucket, len(easy)), replace=False),
        rng.choice(medium, size=min(per_bucket, len(medium)), replace=False),
        rng.choice(hard, size=min(n - 2 * per_bucket, len(hard)), replace=False),
    ])
    rng.shuffle(selected)

    print(f"  Loaded {len(selected)} puzzles: blanks range [{int(blanks[selected].min())}-{int(blanks[selected].max())}], "
          f"terciles at {int(terciles[0])}/{int(terciles[1])} blanks")

    return inputs[selected].copy(), labels[selected].copy()


def run_inference_detailed(model, inputs, num_mem, max_steps=None):
    """Run model capturing per-step predictions, p values, and attention weights."""
    B, L = inputs.shape
    if max_steps is None:
        max_steps = model.max_ponder_steps

    h = model.embed(jnp.array(inputs, dtype=jnp.int32))
    h = model.mem_prepender(h)
    B, total_len, H = h.shape

    type_idx = jnp.concatenate([jnp.zeros(num_mem, dtype=jnp.int32), jnp.ones(L, dtype=jnp.int32)])
    h = h + model.type_embed(type_idx)[None, :, :]
    rot_idx = jnp.concatenate([jnp.arange(num_mem, dtype=jnp.int32), jnp.arange(L, dtype=jnp.int32)])

    mask = jnp.ones((B, total_len), dtype=jnp.bool_)
    attn_mask = mask[:, None, :, None] & mask[:, None, None, :]

    preds_per_step = []
    p_per_step = []
    weights_per_step = []  # attention weight quadrants per step

    halting_prob = jnp.zeros((B, total_len, 1))
    halted = jnp.zeros((B, total_len, 1), dtype=jnp.bool_)
    accumulated = jnp.zeros_like(h)

    for step in range(max_steps):
        step_idx = step % model.max_ponder_steps
        h_step = h + model.step_embed(jnp.array(step_idx))[None, None, :]
        h_next, attn_diag = model.block(h_step, mask=attn_mask, rotary_indices=rot_idx,
                                         num_memory_tokens=num_mem)

        p = model.router(h_next)
        if model.disable_act:
            p = jnp.zeros_like(p)

        # Per-step prediction
        logits = model.out_proj(h_next)
        pred = jnp.argmax(logits[:, num_mem:, :], axis=-1)
        preds_per_step.append(np.array(pred))
        p_per_step.append(float(jnp.mean(p)))

        # Attention quadrants
        if num_mem > 0:
            weights_per_step.append({k: float(v) for k, v in attn_diag.items()})

        # ACT
        p_masked = jnp.where(halted, 0.0, p)
        new_hp = halting_prob + p_masked
        natural = (new_hp >= 0.95) & (~halted)
        just_halted = (~halted) if step == max_steps - 1 else natural
        sw = jnp.where(just_halted, 1.0 - halting_prob, p_masked)
        accumulated = accumulated + sw * h_next
        halted = halted | just_halted
        halting_prob = new_hp
        h = jnp.where(halted, h, h_next)

    final_logits = model.out_proj(accumulated)
    final_pred = np.array(jnp.argmax(final_logits[:, num_mem:, :], axis=-1))

    return {
        'preds_per_step': preds_per_step,
        'p_per_step': p_per_step,
        'weights_per_step': weights_per_step,
        'final_pred': final_pred,
    }


def token_to_digit(t):
    """Convert token (1=blank, 2-10=digits 1-9) to display string."""
    if t == 1: return '.'
    return str(t - 1)


def plot_sudoku_grid(ax, cells, labels=None, title="", highlight_errors=True):
    """Draw a 9x9 Sudoku grid on a matplotlib axes."""
    ax.set_xlim(0, 9)
    ax.set_ylim(0, 9)
    ax.set_aspect('equal')
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=10, pad=4)

    # Draw grid lines
    for i in range(10):
        lw = 2 if i % 3 == 0 else 0.5
        ax.axhline(i, color='black', linewidth=lw)
        ax.axvline(i, color='black', linewidth=lw)

    for i in range(9):
        for j in range(9):
            val = int(cells[i * 9 + j])
            txt = token_to_digit(val)
            color = 'black'
            if highlight_errors and labels is not None:
                if val != int(labels[i * 9 + j]) and val != 1:
                    color = COLORS['fail']
                elif val == int(labels[i * 9 + j]) and val != 1:
                    color = COLORS['success']
            ax.text(j + 0.5, i + 0.5, txt, ha='center', va='center',
                    fontsize=8, fontweight='bold', color=color)


def figure_puzzle_solving(model, inputs, labels, num_mem, output_dir, name):
    """Figure 1: Step-by-step puzzle solving visualization.

    Selects puzzles by outcome: 1 solved, 1 partially correct, 1 failed
    (or best available if the model doesn't solve any).
    """
    # Run on all available puzzles to classify outcomes
    result = run_inference_detailed(model, inputs, num_mem)
    final_correct = np.array([(result['final_pred'][i] == labels[i]).sum() for i in range(len(inputs))])

    # Pick representative puzzles: solved (81/81), partial (60-80), failed (<60)
    solved_idx = np.where(final_correct == 81)[0]
    partial_idx = np.where((final_correct >= 60) & (final_correct < 81))[0]
    failed_idx = np.where(final_correct < 60)[0]

    selected = []
    blanks_per = (inputs == 1).sum(axis=1)
    for pool, label in [(solved_idx, "solved"), (partial_idx, "partial"), (failed_idx, "failed")]:
        if len(pool) > 0:
            # Pick the one with most blanks (hardest) from this pool
            hardest = pool[blanks_per[pool].argmax()]
            selected.append((hardest, label))
    # Fall back if not enough categories
    if len(selected) < 3:
        remaining = [i for i in range(len(inputs)) if i not in [s[0] for s in selected]]
        for i in remaining[:3 - len(selected)]:
            selected.append((i, "other"))

    print(f"  Selected puzzles: {[(l, int(final_correct[i]), int(blanks_per[i])) for i, l in selected]}")

    for sel_idx, (puzzle_idx, category) in enumerate(selected):
        n_blanks = int(blanks_per[puzzle_idx])
        # 2×3 grid layout for readability
        fig, axes = plt.subplots(2, 3, figsize=(10, 8.5))
        fig.subplots_adjust(hspace=0.45, wspace=0.3)
        fig.suptitle(f"{name} — {category} puzzle ({n_blanks} blanks)", fontsize=13, y=0.98)

        # Input
        plot_sudoku_grid(axes[0, 0], inputs[puzzle_idx], title=f"Input\n({81-n_blanks} givens)")

        # Steps at ~0%, 25%, 50%, 75% of max depth
        max_s = len(result['preds_per_step']) - 1
        show_steps = [0, max(1, max_s // 4), max(1, max_s // 2), max(1, 3 * max_s // 4)]
        grid_positions = [(0, 1), (0, 2), (1, 0), (1, 1)]
        for (row, col), step in zip(grid_positions, show_steps):
            pred = result['preds_per_step'][step][puzzle_idx]
            correct = int((pred == labels[puzzle_idx]).sum())
            plot_sudoku_grid(axes[row, col], pred, labels[puzzle_idx],
                           title=f"Step {step}\n({correct}/81)")

        # Final ACT output
        fp = result['final_pred'][puzzle_idx]
        correct = int((fp == labels[puzzle_idx]).sum())
        status = "SOLVED" if correct == 81 else f"{81-correct} errors"
        plot_sudoku_grid(axes[1, 2], fp, labels[puzzle_idx], title=f"ACT Output\n({status})")

        path = os.path.join(output_dir, f"{name}_{category}_puzzle.png")
        fig.savefig(path)
        plt.close(fig)
        print(f"  Saved {path}")


def figure_extended_inference(model, inputs, labels, num_mem, output_dir, name, max_steps=32):
    """Figure 2: What happens when running beyond trained steps."""
    result = run_inference_detailed(model, inputs[:50], num_mem, max_steps=max_steps)
    trained_steps = model.max_ponder_steps

    steps = list(range(max_steps))
    correct_per_step = []
    em_per_step = []
    for s in steps:
        pred = result['preds_per_step'][s]
        per_puzzle = (pred == labels[:50]).sum(axis=1)
        correct_per_step.append(per_puzzle.mean())
        em_per_step.append((per_puzzle == 81).mean() * 100)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # Cells correct
    ax1.plot(steps, correct_per_step, 'o-', markersize=3, color=COLORS['neutral'])
    ax1.axvline(trained_steps - 0.5, color='red', linestyle='--', alpha=0.7, label=f'Trained limit ({trained_steps})')
    ax1.set_xlabel('Ponder Step')
    ax1.set_ylabel('Mean Cells Correct (of 81)')
    ax1.set_title('Per-Step Prediction Quality')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Exact match
    ax2.plot(steps, em_per_step, 's-', markersize=3, color=COLORS['success'])
    ax2.axvline(trained_steps - 0.5, color='red', linestyle='--', alpha=0.7, label=f'Trained limit ({trained_steps})')
    ax2.set_xlabel('Ponder Step')
    ax2.set_ylabel('Exact Match (%)')
    ax2.set_title('Per-Step Exact Match Rate')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # p values on twin axis
    ax3 = ax1.twinx()
    ax3.plot(steps, result['p_per_step'], '--', color=COLORS['act'], alpha=0.5, label='mean p')
    ax3.set_ylabel('Mean Halting p', color=COLORS['act'])
    ax3.tick_params(axis='y', labelcolor=COLORS['act'])

    path = os.path.join(output_dir, f"{name}_extended_inference.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")

    # Also save text summary
    with open(os.path.join(output_dir, f"{name}_extended_inference.txt"), 'w') as f:
        f.write(f"Extended inference: trained={trained_steps}, running={max_steps}\n\n")
        f.write(f"{'Step':>4} {'p':>7} {'Correct':>8} {'EM%':>6} {'Zone':>10}\n")
        for s in steps:
            zone = "trained" if s < trained_steps else "EXTENDED"
            f.write(f"{s:>4} {result['p_per_step'][s]:>7.4f} {correct_per_step[s]:>8.1f} {em_per_step[s]:>6.1f} {zone:>10}\n")


def figure_attention_heatmap(model, inputs, labels, num_mem, output_dir, name, num_heads=8):
    """Figure 3: Per-step attention quadrant heatmap + per-head specialization."""
    result = run_inference_detailed(model, inputs[:20], num_mem)

    if not result['weights_per_step']:
        print(f"  No attention data for {name} (T=0?), skipping.")
        return

    n_steps = len(result['weights_per_step'])
    quadrants = ['attn_seq_to_mem', 'attn_seq_to_seq', 'attn_mem_to_mem', 'attn_mem_to_seq']
    labels_q = ['Seq→Mem', 'Seq→Seq', 'Mem→Mem', 'Mem→Seq']

    # Per-step quadrant evolution
    fig, ax = plt.subplots(figsize=(10, 4))
    for q, label in zip(quadrants, labels_q):
        vals = [result['weights_per_step'][s].get(q, 0) for s in range(n_steps)]
        ax.plot(range(n_steps), vals, 'o-', markersize=3, label=label)
    ax.set_xlabel('Ponder Step')
    ax.set_ylabel('Attention Fraction')
    ax.set_title(f'Attention Patterns Across Depth ({name})')
    ax.legend(loc='center right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.5, n_steps - 0.5)

    path = os.path.join(output_dir, f"{name}_attention_quadrants.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")


def figure_step_weights(model, inputs, labels, num_mem, output_dir, name):
    """Figure 4: Step-weight distribution — where does the output come from?"""
    result = run_inference_detailed(model, inputs[:20], num_mem)

    # Compute step weights from p values (approximate — using mean p)
    # The actual weights depend on per-token halting, but mean p gives the distribution shape
    n_steps = len(result['p_per_step'])

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(range(n_steps), result['p_per_step'], color=COLORS['neutral'], alpha=0.7, label='Mean p (halting prob)')
    ax.set_xlabel('Ponder Step')
    ax.set_ylabel('Mean Halting Probability')
    ax.set_title(f'Router Output Distribution ({name})')
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend()

    path = os.path.join(output_dir, f"{name}_step_weights.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")


def figure_attention_maps(model, inputs, labels, num_mem, output_dir, name, num_heads=8):
    """Full attention heatmaps at key ponder steps.

    Shows the (L+N) × (L+N) attention matrix with mem/seq quadrants
    clearly delineated, for 3 ponder steps and head-averaged.
    """
    if num_mem == 0:
        print(f"  No memory tokens for {name}, skipping attention maps.")
        return

    N = num_mem
    L = 81

    # Monkey-patch to capture full attention weights
    original_call = RoPEMultiHeadAttention.__call__
    all_weights = []

    def capturing_call(self, q_inputs, mask=None, rotary_indices=None, num_memory_tokens=0):
        B_l, L_l, _ = q_inputs.shape
        q = self.q_proj(q_inputs).reshape((B_l, L_l, self.num_heads, self.head_dim))
        k = self.k_proj(q_inputs).reshape((B_l, L_l, self.num_heads, self.head_dim))
        v = self.v_proj(q_inputs).reshape((B_l, L_l, self.num_heads, self.head_dim))
        q = self.q_norm(q); k = self.k_norm(k)
        if rotary_indices is not None:
            q = apply_rope(q, rotary_indices)
            k = apply_rope(k, rotary_indices)
        logits = jnp.einsum('bqhd,bkhd->bhqk', q, k) / jnp.sqrt(self.head_dim)
        if mask is not None:
            logits = jnp.where(mask, logits, -1e9)
        weights = jax.nn.softmax(logits, axis=-1)
        all_weights.append(np.array(weights))  # (B, H, Q, K)
        output = jnp.einsum('bhqk,bkhd->bqhd', weights, v)
        output = output.reshape((B_l, L_l, self.in_features))
        attn_diag = {}
        if num_memory_tokens > 0:
            attn_diag["attn_seq_to_mem"] = weights[:, :, N:, :N].sum(axis=-1).mean()
            attn_diag["attn_seq_to_seq"] = weights[:, :, N:, N:].sum(axis=-1).mean()
            attn_diag["attn_mem_to_mem"] = weights[:, :, :N, :N].sum(axis=-1).mean()
            attn_diag["attn_mem_to_seq"] = weights[:, :, :N, N:].sum(axis=-1).mean()
        return self.out_proj(output), attn_diag

    RoPEMultiHeadAttention.__call__ = capturing_call
    # Run on a single puzzle for clarity
    pad_mask = jnp.ones((1, L), dtype=jnp.bool_)
    _ = model(jnp.array(inputs[:1], dtype=jnp.int32), pad_mask)
    RoPEMultiHeadAttention.__call__ = original_call

    n_steps = len(all_weights)
    # Show steps at early, mid, late
    show_steps = [0, n_steps // 2, n_steps - 1]

    # Head-averaged heatmaps at 3 steps — vertical layout for readability
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    for ax_idx, step in enumerate(show_steps):
        w = all_weights[step][0].mean(axis=0)  # (Q, K) averaged over heads
        ax = axes[ax_idx]
        im = ax.imshow(w, cmap='viridis', aspect='auto', vmin=0, vmax=w.max())
        ax.axhline(N - 0.5, color='red', linewidth=1, linestyle='--')
        ax.axvline(N - 0.5, color='red', linewidth=1, linestyle='--')
        ax.set_title(f'Step {step} (head-avg)', fontsize=11)
        ax.set_xlabel('Key position')
        ax.set_ylabel('Query position')
        # Label quadrants
        ax.text(N/2, N/2, 'M→M', ha='center', va='center', color='white', fontsize=8, fontweight='bold')
        ax.text(N + L/2, N/2, 'M→S', ha='center', va='center', color='white', fontsize=8, fontweight='bold')
        ax.text(N/2, N + L/2, 'S→M', ha='center', va='center', color='white', fontsize=8, fontweight='bold')
        ax.text(N + L/2, N + L/2, 'S→S', ha='center', va='center', color='white', fontsize=8, fontweight='bold')
    fig.colorbar(im, ax=axes, shrink=0.6, label='Attention weight')
    fig.suptitle(f'Attention Maps — {name} (N={N} mem tokens)', fontsize=13)
    path = os.path.join(output_dir, f"{name}_attention_maps.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")

    # Per-head heatmaps at the last step — 2×4 grid (wide, fits near discussion)
    w_last = all_weights[-1][0]  # (H, Q, K)
    n_cols = 4
    n_rows = (num_heads + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    axes_flat = axes.flatten() if num_heads > 1 else [axes]
    for h in range(num_heads):
        ax = axes_flat[h]
        im = ax.imshow(w_last[h], cmap='viridis', aspect='auto', vmin=0, vmax=w_last[h].max())
        ax.axhline(N - 0.5, color='red', linewidth=0.8, linestyle='--')
        ax.axvline(N - 0.5, color='red', linewidth=0.8, linestyle='--')
        s2m = float(w_last[h, N:, :N].sum(axis=-1).mean())
        ax.set_title(f'Head {h} (s→m={s2m:.2f})', fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    for h in range(num_heads, len(axes_flat)):
        axes_flat[h].set_visible(False)
    fig.suptitle(f'Per-Head Attention — {name}, Step {n_steps-1}', fontsize=13)
    fig.tight_layout()
    path = os.path.join(output_dir, f"{name}_attention_per_head.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")


def figure_comparison(all_results, output_dir):
    """Figure 5: Compare extended inference across models."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for name, data in all_results.items():
        steps = list(range(len(data['em_per_step'])))
        ax1.plot(steps, data['correct_per_step'], '-', label=name, linewidth=1.5)
        ax2.plot(steps, data['em_per_step'], '-', label=name, linewidth=1.5)

    ax1.axvline(17.5, color='red', linestyle='--', alpha=0.5, label='Train limit')
    ax2.axvline(17.5, color='red', linestyle='--', alpha=0.5, label='Train limit')
    ax1.set_xlabel('Ponder Step')
    ax1.set_ylabel('Mean Cells Correct')
    ax1.set_title('Prediction Quality vs Depth')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax2.set_xlabel('Ponder Step')
    ax2.set_ylabel('Exact Match (%)')
    ax2.set_title('Exact Match vs Depth')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    path = os.path.join(output_dir, "comparison_extended_inference.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="analysis_output")
    parser.add_argument("--data_dir", default="data/sudoku-extreme-full")
    parser.add_argument("--max_extended_steps", type=int, default=32)
    parser.add_argument("--n_puzzles", type=int, default=50)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Define models to analyze
    models_config = {
        "T16_ACT_warmup": {
            "checkpoint": "checkpoints/long-run-v2-T16-S123",
            "T": 16, "label": "T=16, ACT+warmup (best)", "disable_act": False,
        },
        "T8_ACT": {
            "checkpoint": "checkpoints/phase-1d-T8-S42",
            "T": 8, "label": "T=8, ACT (halt=18)", "disable_act": False,
        },
        "T16_ACT_plain": {
            "checkpoint": "checkpoints/phase-2-bias-T16-S0",
            "T": 16, "label": "T=16, ACT, no lambda", "disable_act": False,
        },
        "T16_trapped": {
            "checkpoint": "checkpoints/ablation-rmsnorm-bias0-T16-S42",
            "T": 16, "label": "T=16, RMSNorm+bias=0 (trapped)", "disable_act": False,
            "use_rmsnorm": True,
        },
    }

    # Load puzzles
    print("Loading test puzzles...")
    inputs, labels = load_puzzles(args.data_dir, n=args.n_puzzles)

    all_extended = {}

    for name, cfg in models_config.items():
        ckpt = cfg["checkpoint"]
        if not os.path.exists(ckpt):
            print(f"\n[SKIP] {name}: {ckpt} not found")
            continue

        print(f"\n{'='*60}")
        print(f"Analyzing: {cfg['label']}")
        print(f"{'='*60}")

        try:
            model, ckpt_step = load_model(
                ckpt, num_memory_tokens=cfg["T"],
                disable_act=cfg.get("disable_act", False),
                use_rmsnorm=cfg.get("use_rmsnorm", False),
            )
        except Exception as e:
            print(f"  [ERROR] Failed to load: {e}")
            continue

        model_dir = os.path.join(args.output_dir, f"{name}_step{ckpt_step}")
        os.makedirs(model_dir, exist_ok=True)

        # 1. Puzzle solving visualization
        print("  Generating puzzle solving figures...")
        figure_puzzle_solving(model, inputs, labels, cfg["T"], model_dir, name)

        # 2. Extended inference
        print("  Running extended inference...")
        result = run_inference_detailed(model, inputs, cfg["T"], max_steps=args.max_extended_steps)
        correct_list = []
        em_list = []
        for s in range(args.max_extended_steps):
            pred = result['preds_per_step'][s]
            per_puzzle = (pred == labels).sum(axis=1)
            correct_list.append(float(per_puzzle.mean()))
            em_list.append(float((per_puzzle == 81).mean() * 100))
        all_extended[cfg['label']] = {
            'correct_per_step': correct_list,
            'em_per_step': em_list,
        }
        figure_extended_inference(model, inputs, labels, cfg["T"], model_dir, name,
                                  max_steps=args.max_extended_steps)

        # 3. Attention analysis
        if cfg["T"] > 0:
            print("  Analyzing attention patterns...")
            figure_attention_heatmap(model, inputs, labels, cfg["T"], model_dir, name)
            print("  Generating attention heatmaps...")
            figure_attention_maps(model, inputs, labels, cfg["T"], model_dir, name)

        # 4. Step weights
        print("  Plotting step-weight distribution...")
        figure_step_weights(model, inputs, labels, cfg["T"], model_dir, name)

    # 5. Cross-model comparison
    if len(all_extended) > 1:
        print(f"\n{'='*60}")
        print("Generating cross-model comparison...")
        figure_comparison(all_extended, args.output_dir)

    print(f"\nAll figures saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
