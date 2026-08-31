#!/usr/bin/env python3
"""
Compute and cache all the statistics we want for the data-scaling × ensemble
study.

For each (budget N, pipeline P) we load the M=5 ensemble member samples.npz
files and compute:

  Per-member (M values, one per ensemble member):
    - r2_per_param          (5,)   per-parameter R^2
    - r2_mean               scalar mean across params
    - rel_unc_per_param     (5,)   mean(posterior_std)/std(labels) per param
    - tarp_joint_mae        scalar mean |ECP - alpha| of 5-D Euclidean TARP
    - tarp_joint_bias       scalar mean (ECP - alpha)
    - tarp_per_param_mae    (5,)   marginal 1-D TARP MAE per param
    - tarp_per_param_bias   (5,)   marginal 1-D TARP bias per param

  Combined-ensemble (concatenate all M members' samples -> M*N_samples per obs):
    - same metric set, all computed once on the combined 2,500-sample posterior

Output: a single .npz at the location given by --output (default
EoRFlow-dev/output/skatr/loreli/ensembles/ensemble_stats.npz).

Structure of the saved file:
    budgets       (B,)             N values, in order
    pipelines     list of B strings ["skatr", "vit_21cmfast", "vit_loreli"]
    param_names   list of 5 strings
    For each (pipe, key) we save TWO arrays:
        per_member.{pipe}.{key}   shape (B, M, ...)
        combined.{pipe}.{key}     shape (B, ...)
    where ... matches the per-member shape minus the M axis.

Missing per-member files are filled with NaN and tracked in a boolean mask
array `present.{pipe}` of shape (B, M).

Wall time: this iterates over 90 sample files and runs ~108 TARP coverage
calls (90 joint + 90 per-param × 5 + 18 combined-joint + 18 combined-per-param
× 5 = 108 + 558 = ~666 TARP calls). Each TARP is a few seconds, so total
~30-60 minutes. Submit as a SLURM job.
"""

import argparse
import os
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import r2_score
from tarp import get_tarp_coverage


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
ENS_BASE_DEFAULT = "/pfs/10/work/hd_pt254-skatr/EoRFlow-dev/output/skatr/loreli/ensembles"
OUTPUT_DEFAULT   = "/pfs/10/work/hd_pt254-skatr/EoRFlow-dev/output/skatr/loreli/ensembles/ensemble_stats.npz"

BUDGETS_DEFAULT  = [100, 500, 1000, 2500, 5000, 6517]
PIPES_DEFAULT    = ["skatr", "vit_21cmfast", "vit_loreli"]
N_MEMBERS_DEFAULT = 5
PARAM_NAMES = ["f_X", "tau", "r_H", "logMmin", "f_esc_post"]
N_PARAMS = len(PARAM_NAMES)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ens_base", type=str, default=ENS_BASE_DEFAULT)
    p.add_argument("--output", type=str, default=OUTPUT_DEFAULT)
    p.add_argument("--budgets", type=int, nargs="+", default=BUDGETS_DEFAULT)
    p.add_argument("--pipelines", type=str, nargs="+", default=PIPES_DEFAULT)
    p.add_argument("--members", type=int, default=N_MEMBERS_DEFAULT)
    p.add_argument("--tarp_seed", type=int, default=42,
                   help="Numpy global seed set before each TARP call for reproducibility.")
    return p.parse_args()


def tarp_metrics(preds, labels, seed=42):
    """Return (mae, bias) of the joint TARP coverage."""
    np.random.seed(seed)
    ecp, alpha = get_tarp_coverage(
        preds.copy(), labels.copy(),
        references="random", norm=True, bootstrap=True, metric="euclidean",
    )
    mean_curve = ecp.mean(0)
    mae  = float(np.abs(mean_curve - alpha).mean())
    bias = float((mean_curve - alpha).mean())
    return mae, bias


def per_param_tarp(preds, labels, seed=42):
    """Run a 1-D TARP on each parameter dimension separately. Returns
    (5,) arrays of mae and bias."""
    mae_arr  = np.empty(N_PARAMS, dtype=np.float32)
    bias_arr = np.empty(N_PARAMS, dtype=np.float32)
    for i in range(N_PARAMS):
        np.random.seed(seed)
        ecp, alpha = get_tarp_coverage(
            preds[:, :, i:i+1].copy(),
            labels[:, i:i+1].copy(),
            references="random", norm=True, bootstrap=True, metric="euclidean",
        )
        mean_curve = ecp.mean(0)
        mae_arr[i]  = float(np.abs(mean_curve - alpha).mean())
        bias_arr[i] = float((mean_curve - alpha).mean())
    return mae_arr, bias_arr


def metrics_of(preds, labels, seed):
    """Compute the full metric set on a single (preds, labels) pair.

    preds: (N_samples, n_test, n_params) -- can be a single CFM run or the
           combined ensemble.
    labels: (n_test, n_params)
    """
    pmean = preds.mean(axis=0)
    pstd  = preds.std(axis=0)
    r2_pp = np.array([r2_score(labels[:, i], pmean[:, i]) for i in range(N_PARAMS)], dtype=np.float32)
    ru_pp = (pstd.mean(axis=0) / labels.std(axis=0)).astype(np.float32)
    mae_j, bias_j = tarp_metrics(preds, labels, seed=seed)
    mae_pp, bias_pp = per_param_tarp(preds, labels, seed=seed)
    return dict(
        r2_per_param=r2_pp,
        r2_mean=float(np.mean(r2_pp)),
        rel_unc_per_param=ru_pp,
        tarp_joint_mae=float(mae_j),
        tarp_joint_bias=float(bias_j),
        tarp_per_param_mae=mae_pp.astype(np.float32),
        tarp_per_param_bias=bias_pp.astype(np.float32),
    )


def main():
    args = parse_args()
    ens_base = Path(args.ens_base)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    B = len(args.budgets)
    M = args.members
    pipes = args.pipelines

    # Allocate per-member and combined containers; per-member is shape (B, M, ...)
    nans = lambda *shape: np.full(shape, np.nan, dtype=np.float32)
    per_member = {pipe: dict(
        r2_per_param        = nans(B, M, N_PARAMS),
        r2_mean             = nans(B, M),
        rel_unc_per_param   = nans(B, M, N_PARAMS),
        tarp_joint_mae      = nans(B, M),
        tarp_joint_bias     = nans(B, M),
        tarp_per_param_mae  = nans(B, M, N_PARAMS),
        tarp_per_param_bias = nans(B, M, N_PARAMS),
    ) for pipe in pipes}
    combined = {pipe: dict(
        r2_per_param        = nans(B, N_PARAMS),
        r2_mean             = nans(B,),
        rel_unc_per_param   = nans(B, N_PARAMS),
        tarp_joint_mae      = nans(B,),
        tarp_joint_bias     = nans(B,),
        tarp_per_param_mae  = nans(B, N_PARAMS),
        tarp_per_param_bias = nans(B, N_PARAMS),
    ) for pipe in pipes}
    present = {pipe: np.zeros((B, M), dtype=bool) for pipe in pipes}

    t0 = time.time()
    total_cells = B * len(pipes)
    cells_done = 0
    for bi, N in enumerate(args.budgets):
        for pipe in pipes:
            cells_done += 1
            member_preds = []
            member_labels = None
            for mi in range(M):
                d = ens_base / f"N{N}" / pipe / f"ens{mi+1}"
                sp = d / "samples.npz"
                if not sp.exists():
                    print(f"  [skip] {sp} missing")
                    continue
                s = np.load(sp, allow_pickle=True)
                preds = s["preds"]
                labels_loc = s["labels"]
                # Per-member metrics
                stats = metrics_of(preds, labels_loc, seed=args.tarp_seed)
                for k, v in stats.items():
                    per_member[pipe][k][bi, mi] = v
                present[pipe][bi, mi] = True
                # Stash for combined ensemble
                member_preds.append(preds)
                if member_labels is None:
                    member_labels = labels_loc
            if member_preds:
                # Combined posterior across available members
                combined_preds = np.concatenate(member_preds, axis=0)
                stats_c = metrics_of(combined_preds, member_labels, seed=args.tarp_seed)
                for k, v in stats_c.items():
                    combined[pipe][k][bi] = v

            elapsed = time.time() - t0
            print(f"  [{cells_done}/{total_cells}] N={N} {pipe} done in cumulative {elapsed/60:.1f} min")

    # -- save --
    save_kwargs = dict(
        budgets=np.asarray(args.budgets, dtype=np.int64),
        pipelines=np.asarray(pipes),
        param_names=np.asarray(PARAM_NAMES),
        n_members=np.int64(M),
    )
    for pipe in pipes:
        for k, v in per_member[pipe].items():
            save_kwargs[f"per_member.{pipe}.{k}"] = v
        for k, v in combined[pipe].items():
            save_kwargs[f"combined.{pipe}.{k}"] = v
        save_kwargs[f"present.{pipe}"] = present[pipe]

    np.savez(out_path, **save_kwargs)
    print(f"\nSaved ensemble stats to {out_path}")
    print(f"Total wall-clock: {(time.time() - t0)/60:.1f} min")


if __name__ == "__main__":
    main()
