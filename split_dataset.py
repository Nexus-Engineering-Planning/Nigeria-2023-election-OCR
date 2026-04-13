"""
split_dataset.py

Splits the annotations.csv produced by build_training_data.py into
train / validation / test sets using a stratified split on state_name.

Output files (written to the same directory as annotations.csv):
    annotations_train.csv
    annotations_val.csv
    annotations_test.csv
    annotations_holdout.csv   ← fixed benchmark, never used in training

Split ratios (configurable via flags):
    --train  0.70   (default)
    --val    0.15   (default)
    --test   0.10   (default)
    --holdout 0.05  (default)
    (must sum to 1.0)

Stratification ensures every state is proportionally represented in all
four splits, which matters because polling unit density varies enormously
by state (Lagos >> Kogi, for example).

Usage:
    py split_dataset.py [--input DIR] [--seed N]
"""

import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path


# ---------------------------------------------------------------------------
# Stratified split
# ---------------------------------------------------------------------------

def stratified_split(rows, group_col, fracs, seed):
    """
    Split rows into len(fracs) parts, each proportional to fracs[i],
    stratified by group_col.

    fracs: list of floats summing to 1.0, e.g. [0.70, 0.15, 0.10, 0.05]

    Returns a list of lists (one per fraction).
    """
    rng = random.Random(seed)

    groups = defaultdict(list)
    for r in rows:
        groups[r[group_col]].append(r)

    splits = [[] for _ in fracs]

    for group_rows in groups.values():
        shuffled = group_rows[:]
        rng.shuffle(shuffled)
        n = len(shuffled)

        boundaries = []
        cumulative = 0.0
        for f in fracs[:-1]:
            cumulative += f
            boundaries.append(round(cumulative * n))

        prev = 0
        for i, b in enumerate(boundaries):
            splits[i].extend(shuffled[prev:b])
            prev = b
        splits[-1].extend(shuffled[prev:])

    return splits


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input",    type=str,   default="training_data",
                        help="Directory containing annotations.csv (default: training_data)")
    parser.add_argument("--train",    type=float, default=0.70,
                        help="Fraction for training set   (default: 0.70)")
    parser.add_argument("--val",      type=float, default=0.15,
                        help="Fraction for validation set (default: 0.15)")
    parser.add_argument("--test",     type=float, default=0.10,
                        help="Fraction for test set       (default: 0.10)")
    parser.add_argument("--holdout",  type=float, default=0.05,
                        help="Fraction for fixed hold-out (default: 0.05)")
    parser.add_argument("--seed",     type=int,   default=42,
                        help="Random seed (default: 42)")
    args = parser.parse_args()

    fracs = [args.train, args.val, args.test, args.holdout]
    total = sum(fracs)
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Fractions must sum to 1.0 — got {total:.4f}")

    root             = Path(args.input)
    annotations_path = root / "annotations.csv"

    if not annotations_path.exists():
        raise FileNotFoundError(f"annotations.csv not found at {annotations_path}")

    # Load
    with open(annotations_path, newline="", encoding="utf-8") as f:
        reader    = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows      = list(reader)

    print(f"Loaded {len(rows):,} rows from {annotations_path}")
    print(f"Split ratios — train:{args.train}  val:{args.val}  "
          f"test:{args.test}  holdout:{args.holdout}  (seed={args.seed})")

    split_names = ["train", "val", "test", "holdout"]
    splits      = stratified_split(rows, "state_name", fracs, args.seed)

    for name, split in zip(split_names, splits):
        out_path = root / f"annotations_{name}.csv"
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(split)
        print(f"  {name:8s}: {len(split):>4} rows → {out_path.name}")

    # Sanity check: every row appears in exactly one split
    all_pucs = [r["polling_unit_code"] for split in splits for r in split]
    if len(all_pucs) == len(rows) and len(set(all_pucs)) == len(set(r["polling_unit_code"] for r in rows)):
        print("\nSanity check passed — all rows accounted for, no duplicates across splits.")
    else:
        print(f"\nWARNING: row count mismatch. Original={len(rows)}, "
              f"Split total={len(all_pucs)}")

    print("\nDone.")


if __name__ == "__main__":
    main()
