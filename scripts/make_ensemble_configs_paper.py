#!/usr/bin/env python3
"""
Generate CFM ensemble configs for the paper-final scaling study on full-z
Loreli (z=5..15) with the data-correct ShiftMedian per dataset (-25 mK clean,
0 mK noisy) and no augmentations anywhere.

6 budgets x 3 pipelines x M=5 = 90 configs per condition x 2 conditions
(clean + noisy) = 180 configs total.

Output:
    config/scaling/ensembles_paper/{N{N}_{cond}_{pipe}_ens{i}.yaml}
    output_tag stem: results_paper/{cond}/N{N}/{pipe}/ens{i}

Conditions and their pipelines:
    cond=clean: pipelines = skatr, vit_21cmfast, vit_loreli
    cond=noisy: pipelines = skatr, vit_21cmfast, vit_loreli_noiseTrained
"""

import os
from pathlib import Path

import yaml

TEMPLATE = "/pfs/10/work/hd_pt254-skatr/EoRFlow-dev/config/train_skatr_loreli_trial.yaml"
OUT_DIR  = "/pfs/10/work/hd_pt254-skatr/EoRFlow-dev/config/scaling/ensembles_paper"
DATA_SPLITS = "/pfs/10/work/hd_pt254-skatr/EoRFlow-dev/data_splits"
SUM_BASE = "/pfs/10/work/hd_pt254-skatr/21cmSims"

BUDGETS    = [100, 500, 1000, 2500, 5000, 6517]
N_MEMBERS  = 5
CONDITIONS = ("clean", "noisy")


def summary_dir(cond: str, pipe: str, N: int) -> str:
    """Map (cond, pipe, N) to the matching summary directory on disk."""
    if cond == "clean":
        if pipe == "skatr":
            return f"{SUM_BASE}/loreli_summaries_skatr_zfull"
        if pipe == "vit_21cmfast":
            return f"{SUM_BASE}/loreli_summaries_vit_21cmfast_zfull"
        if pipe == "vit_loreli":
            return f"{SUM_BASE}/loreli_summaries_supervised_vit_loreli_N{N}_zfull"
    elif cond == "noisy":
        if pipe == "skatr":
            return f"{SUM_BASE}/loreli_summaries_skatr_zfull_aastar_mod"
        if pipe == "vit_21cmfast":
            return f"{SUM_BASE}/loreli_summaries_vit_21cmfast_zfull_aastar_mod"
        if pipe == "vit_loreli_noiseTrained":
            return f"{SUM_BASE}/loreli_summaries_aastar_mod_vit_loreli_N{N}_zfull_noiseTrained"
    raise ValueError(f"Unknown (cond, pipe) = ({cond}, {pipe})")


def pipelines_for(cond: str):
    if cond == "clean":
        return ("skatr", "vit_21cmfast", "vit_loreli")
    if cond == "noisy":
        return ("skatr", "vit_21cmfast", "vit_loreli_noiseTrained")
    raise ValueError(cond)


def main():
    with open(TEMPLATE) as f:
        base = yaml.safe_load(f)
    os.makedirs(OUT_DIR, exist_ok=True)

    n_written = 0
    for cond in CONDITIONS:
        for N in BUDGETS:
            split = f"{DATA_SPLITS}/loreli_split_N{N}.npz"
            if not os.path.exists(split):
                print(f"  warn: missing split {split}; skipping N={N}")
                continue
            for pipe in pipelines_for(cond):
                sdir = summary_dir(cond, pipe, N)
                if not os.path.isdir(sdir):
                    raise SystemExit(f"ERROR: summary dir missing: {sdir}")
                for i in range(1, N_MEMBERS + 1):
                    cfg = yaml.safe_load(yaml.safe_dump(base))
                    cfg.setdefault("skatr", {}).setdefault("source_dirs", {})["loreli"] = sdir
                    cfg["output_tag"] = f"results_paper/{cond}/N{N}/{pipe}/ens{i}"
                    cfg["split_indices_path"] = split
                    cfg["num_files"] = None

                    out_path = Path(OUT_DIR) / f"N{N}_{cond}_{pipe}_ens{i}.yaml"
                    with open(out_path, "w") as f:
                        yaml.safe_dump(cfg, f, sort_keys=False)
                    n_written += 1

    print(f"Wrote {n_written} configs to {OUT_DIR}")


if __name__ == "__main__":
    main()
