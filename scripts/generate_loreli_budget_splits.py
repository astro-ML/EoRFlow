#!/usr/bin/env python3
"""
Generate per-budget Loreli train/val/test split files for the data-scaling
experiment.

Protocol
--------
- The held-out test set is fixed across all budgets: the same 1,150 indices
  in EoRFlow-dev/data_splits/loreli_split.npz['test'] (which was derived from
  the supervised ViT's seed=1729 split).

- The "pool" of indices available for training/validation is the union of the
  canonical split's train+val: 5,751 + 766 = 6,517 indices, kept in the order
  produced by `torch.utils.data.random_split` with `manual_seed(1729)`.
  Nesting is therefore deterministic: every smaller-budget train/val set is a
  strict prefix of every larger-budget train/val set.

- For each budget N_sims, the split file contains:
      train : pool[:n_train]
      val   : pool[n_train : n_train + n_val]
      test  : same 1,150 indices as the canonical split
  where n_val = max(min_val, int(val_frac * N_sims)) and n_train = N_sims - n_val.

Output
------
For each requested budget N, writes:
  EoRFlow-dev/data_splits/loreli_split_N{N}.npz

with fields {train, val, test, total, N_sims, n_train, n_val, val_frac, seed,
pool_size}.
"""

import argparse
import os
from pathlib import Path

import numpy as np


DEFAULT_BUDGETS = [100, 500, 1000, 2500, 5000, 6517]
DEFAULT_VAL_FRAC = 0.10
DEFAULT_MIN_VAL = 20
DEFAULT_CANONICAL = "/pfs/10/work/hd_pt254-skatr/EoRFlow-dev/data_splits/loreli_split.npz"
DEFAULT_OUT_DIR = "/pfs/10/work/hd_pt254-skatr/EoRFlow-dev/data_splits"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--canonical", type=str, default=DEFAULT_CANONICAL,
                   help="Path to the canonical loreli_split.npz (provides test set + pool ordering).")
    p.add_argument("--out_dir", type=str, default=DEFAULT_OUT_DIR,
                   help="Directory to write per-budget split files.")
    p.add_argument("--budgets", type=int, nargs="+", default=DEFAULT_BUDGETS,
                   help="List of N_sims values (train+val budgets).")
    p.add_argument("--val_frac", type=float, default=DEFAULT_VAL_FRAC,
                   help="Fraction of N_sims reserved for validation.")
    p.add_argument("--min_val", type=int, default=DEFAULT_MIN_VAL,
                   help="Minimum number of validation samples regardless of val_frac.")
    return p.parse_args()


def main():
    args = parse_args()

    canonical = np.load(args.canonical)
    test_idx = np.asarray(canonical["test"], dtype=np.int64)
    pool_train = np.asarray(canonical["train"], dtype=np.int64)
    pool_val   = np.asarray(canonical["val"],   dtype=np.int64)
    seed = int(canonical["seed"].item()) if "seed" in canonical.files else 1729
    total = int(canonical["total"].item()) if "total" in canonical.files else (pool_train.size + pool_val.size + test_idx.size)

    # Pool = train + val from canonical, in the deterministic seed=1729 order
    # (concatenate train then val so that smaller-budget train sets are prefixes
    # of larger-budget train sets).
    pool = np.concatenate([pool_train, pool_val])
    pool_size = pool.size

    # Sanity: every test index must be disjoint from the pool.
    overlap = set(pool.tolist()) & set(test_idx.tolist())
    if overlap:
        raise RuntimeError(f"Test set overlaps the pool — corrupt canonical split file. Overlap size: {len(overlap)}.")
    if pool.size + test_idx.size != total:
        raise RuntimeError(f"pool ({pool.size}) + test ({test_idx.size}) != total ({total}).")

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Canonical: pool_size={pool_size}, test_size={test_idx.size}, total={total}, seed={seed}")
    print(f"Generating budget files in {args.out_dir}")
    print()
    print(f"{'budget':>8} {'n_train':>8} {'n_val':>6} {'n_test':>6} {'path'}")

    for N in args.budgets:
        if N > pool_size:
            print(f"  skipping N={N}: exceeds pool_size={pool_size}")
            continue
        if N < args.min_val + 1:
            print(f"  skipping N={N}: smaller than min_val+1={args.min_val + 1}")
            continue

        n_val   = max(args.min_val, int(args.val_frac * N))
        n_train = N - n_val
        train_subset = pool[:n_train]
        val_subset   = pool[n_train : n_train + n_val]

        out_path = Path(args.out_dir) / f"loreli_split_N{N}.npz"
        np.savez(
            out_path,
            train=train_subset.astype(np.int64),
            val=val_subset.astype(np.int64),
            test=test_idx.astype(np.int64),
            total=np.int64(total),
            N_sims=np.int64(N),
            n_train=np.int64(n_train),
            n_val=np.int64(n_val),
            val_frac=np.float32(args.val_frac),
            seed=np.int64(seed),
            pool_size=np.int64(pool_size),
        )
        print(f"  N={N:>5}  n_train={n_train:>5}  n_val={n_val:>3}  n_test={test_idx.size:>4}  -> {out_path.name}")

    print()
    print("Done. Use these files via `split_indices_path` in EoRFlow's CFM config")
    print("and via the new `data.split_indices_path` in skatr's training config")
    print("(see the skatr/src/experiments/training.py patch from this commit).")


if __name__ == "__main__":
    main()
