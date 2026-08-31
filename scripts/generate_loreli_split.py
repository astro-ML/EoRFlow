#!/usr/bin/env python3
"""
Generate a canonical Loreli train/val/test split that aligns with the supervised
ViT's split (seed=1729, fractions train=0.75 / val=0.10 / test=0.15 of 7667).

Writes a single .npz with three integer arrays: train, val, test. All CFM
training runs that reference this file via `split_indices_path` will share the
same test set, eliminating the leakage between the supervised ViT's training
set and the CFM's evaluation set.

Why these exact numbers:
- The ViT used `random_split(dset, [1 - val_frac - test_frac, val_frac, test_frac], seed=1729)`
  with val_frac=0.10 and test_frac=0.15.
- File ordering is `sorted(glob("lightcone_*.npz"))` and is identical between
  /pfs/10/work/hd_pt254-skatr/21cmSims/loreli_downsampled/ (ViT input) and
  /pfs/10/work/hd_pt254-skatr/21cmSims/loreli_summaries_*/ (CFM input), so
  index N refers to the same physical lightcone in both pipelines.

Run once; commit the resulting .npz to the repo so all collaborators / runs
reproduce the same split.
"""

import argparse
import os
import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--total", type=int, default=7667,
                        help="Total number of Loreli lightcones")
    parser.add_argument("--val_frac", type=float, default=0.10)
    parser.add_argument("--test_frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=1729,
                        help="Must match the supervised ViT split seed in "
                             "skatr/src/experiments/training.py (currently 1729)")
    parser.add_argument("--output", type=str, required=True,
                        help="Output .npz path")
    args = parser.parse_args()

    train_frac = 1.0 - args.val_frac - args.test_frac
    gen = torch.Generator().manual_seed(args.seed)
    splits = torch.utils.data.random_split(
        range(args.total),
        [train_frac, args.val_frac, args.test_frac],
        generator=gen,
    )
    train_idx = np.asarray(splits[0].indices, dtype=np.int64)
    val_idx   = np.asarray(splits[1].indices, dtype=np.int64)
    test_idx  = np.asarray(splits[2].indices, dtype=np.int64)

    assert len(set(train_idx) & set(val_idx))  == 0
    assert len(set(train_idx) & set(test_idx)) == 0
    assert len(set(val_idx)   & set(test_idx)) == 0
    assert len(train_idx) + len(val_idx) + len(test_idx) == args.total

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    np.savez(
        args.output,
        train=train_idx, val=val_idx, test=test_idx,
        total=args.total, val_frac=args.val_frac, test_frac=args.test_frac, seed=args.seed,
    )
    print(f"Saved split to {args.output}")
    print(f"  train: {len(train_idx)}")
    print(f"  val:   {len(val_idx)}")
    print(f"  test:  {len(test_idx)}")
    print(f"  first 5 test indices: {test_idx[:5].tolist()}")


if __name__ == "__main__":
    main()
