"""Build a small (1000 puzzle) dataset by subsampling from sudoku-extreme-full.

Creates augmented copies using Sudoku symmetry transforms (digit permutation,
row/col band shuffles, transpose) to match the TRM/HRM training protocol.

Usage:
    python scripts/build_small_dataset.py \
        --source_dir data/sudoku-extreme-full \
        --output_dir data/sudoku-extreme-1k-aug100 \
        --num_puzzles 1000 \
        --num_aug 100
"""
import os
import json
import argparse
import numpy as np


def shuffle_sudoku(board: np.ndarray, solution: np.ndarray):
    """Apply a random Sudoku-preserving symmetry transform."""
    # Random digit remapping (1-9, keeping 0/pad unchanged if any)
    digit_map = np.zeros(11, dtype=np.uint8)  # our vocab is 1-10, 0=pad
    perm = np.random.permutation(9) + 1  # permute digits 1-9
    digit_map[1:10] = perm
    digit_map[10] = 10  # keep token 10 unchanged if present

    # Random row permutation (preserve band structure)
    bands = np.random.permutation(3)
    row_perm = np.concatenate([b * 3 + np.random.permutation(3) for b in bands])

    # Random column permutation (preserve stack structure)
    stacks = np.random.permutation(3)
    col_perm = np.concatenate([s * 3 + np.random.permutation(3) for s in stacks])

    # Whether to transpose
    do_transpose = np.random.rand() < 0.5

    def apply(seq):
        grid = seq.reshape(9, 9)
        if do_transpose:
            grid = grid.T
        grid = grid[row_perm][:, col_perm]
        return digit_map[grid.flatten()]

    return apply(board), apply(solution)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_dir", type=str, default="data/sudoku-extreme-full")
    parser.add_argument("--output_dir", type=str, default="data/sudoku-extreme-1k-aug100")
    parser.add_argument("--num_puzzles", type=int, default=1000)
    parser.add_argument("--num_aug", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    # Load source training data
    src_train = os.path.join(args.source_dir, "train")
    inputs = np.load(os.path.join(src_train, "all__inputs.npy"), mmap_mode="r")
    labels = np.load(os.path.join(src_train, "all__labels.npy"), mmap_mode="r")
    print(f"Source: {len(inputs)} puzzles")

    # Subsample base puzzles
    indices = rng.choice(len(inputs), size=args.num_puzzles, replace=False)
    base_inputs = inputs[indices].copy()
    base_labels = labels[indices].copy()
    print(f"Subsampled {args.num_puzzles} base puzzles")

    # Generate augmented training set
    all_inputs = []
    all_labels = []
    for i in range(args.num_puzzles):
        # Original
        all_inputs.append(base_inputs[i])
        all_labels.append(base_labels[i])
        # Augments
        for _ in range(args.num_aug):
            aug_inp, aug_lab = shuffle_sudoku(base_inputs[i], base_labels[i])
            all_inputs.append(aug_inp)
            all_labels.append(aug_lab)

    all_inputs = np.array(all_inputs, dtype=np.uint8)
    all_labels = np.array(all_labels, dtype=np.uint8)
    total = len(all_inputs)
    print(f"Generated {total} training examples ({args.num_puzzles} × {1 + args.num_aug})")

    # Build index arrays (one puzzle per group for simplicity)
    puzzle_indices = np.arange(total + 1, dtype=np.int32)
    group_indices = np.arange(total + 1, dtype=np.int32)
    puzzle_identifiers = np.zeros(total, dtype=np.int32)

    # Save training split
    train_dir = os.path.join(args.output_dir, "train")
    os.makedirs(train_dir, exist_ok=True)
    np.save(os.path.join(train_dir, "all__inputs.npy"), all_inputs)
    np.save(os.path.join(train_dir, "all__labels.npy"), all_labels)
    np.save(os.path.join(train_dir, "all__puzzle_indices.npy"), puzzle_indices)
    np.save(os.path.join(train_dir, "all__group_indices.npy"), group_indices)
    np.save(os.path.join(train_dir, "all__puzzle_identifiers.npy"), puzzle_identifiers)

    metadata = {
        "seq_len": 81,
        "vocab_size": 11,
        "pad_id": 0,
        "ignore_label_id": 0,
        "blank_identifier_id": 0,
        "num_puzzle_identifiers": 1,
        "total_groups": total,
        "mean_puzzle_examples": 1.0,
        "total_puzzles": total,
        "sets": ["all"]
    }
    with open(os.path.join(train_dir, "dataset.json"), "w") as f:
        json.dump(metadata, f)

    # Copy test split directly (full 423k test set for generalization eval)
    src_test = os.path.join(args.source_dir, "test")
    test_dir = os.path.join(args.output_dir, "test")
    if not os.path.exists(test_dir):
        os.symlink(os.path.abspath(src_test), test_dir)
        print(f"Symlinked test set from {src_test}")
    else:
        print(f"Test dir already exists: {test_dir}")

    print(f"Done. Dataset saved to {args.output_dir}")
    print(f"  Train: {total} examples ({args.num_puzzles} base × {1+args.num_aug} augments)")
    print(f"  Test: full sudoku-extreme test set (symlinked)")


if __name__ == "__main__":
    main()
