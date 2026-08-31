# EoRFlow

Simulation-based inference for 21cm reionization signals: a set of 21cm summary
statistics paired with neural density estimators that infer astrophysical
parameters or the neutral-fraction history from a lightcone.

## Summaries

- spherically (1D) and cylindrically (2D) averaged power spectra
- CNN-learned summary from 21cm lightcones
- transformer-learned self-supervised summary, precomputed with
  [SKATR](https://arxiv.org/pdf/2410.18899)

## Inference models

- conditional normalizing flow (affine coupling, FrEIA)
- conditional flow matching (CFM)

## Layout

    src/models/        density estimators (flow, CFM) and the CNN summary
    src/training/      training entry point
    src/evaluation/    posterior sampling, metrics, plotting
    src/data_tools/    dataset loaders, power spectra, noise injection
    config/            configuration templates
    scripts/           split generation, config generation, cluster submission

## Getting started

Training and evaluation are two separate passes. Training writes a checkpoint;
it does **not** write posterior samples.

    # 1. edit the templates -- every path in them is a <placeholder>
    cp config/train.yaml config/my_train.yaml

    # 2. train
    python src/training/train.py --config config/my_train.yaml

    # 3. sample posteriors from the trained model
    python src/evaluation/sample.py --config config/my_eval.yaml

Outputs are written to
`{project_root}/{output_subpath}/{mode}/{data}/{output_tag}/`.

## Splits

When several models must be compared, they have to share one held-out test set,
otherwise the comparison leaks. `scripts/generate_loreli_split.py` writes an
explicit `split.npz` of train/val/test indices, and
`scripts/generate_loreli_budget_splits.py` derives nested per-budget subsets
from it for data-scaling studies. Point `split_indices_path` at the result
rather than relying on `test_fraction`.

Note that `sample.py` writes `samples.npz` whose row order does **not** follow
`test_indices.npy`. Rows are aligned with the `labels` stored in the same file,
so per-run metrics are correct, but joining two runs by row index compares
different simulations. Match on the label vector instead.

## Note on paths

This code was developed on a specific HPC system and the scripts under
`scripts/` contain absolute paths and SLURM directives from that environment.
They are included as working examples of the full pipeline rather than as
portable tools; adapt the paths before use.
