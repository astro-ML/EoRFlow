# EoRFlow — SKA Science Data Challenge 3b

The version of EoRFlow used for the
[SKA Science Data Challenge 3b](https://sdc3.skao.int/challenges/inference/results)
inference task.

This branch is a **code-only** snapshot. The challenge run also produced ~780 MB
of intermediate power spectra, posterior FITS files, and figures; those are kept
in the private development repository rather than here.

## Layout

    SDC_inference/       challenge-specific inference, result and contour scripts
    src/models/          normalizing flow used for the challenge
    src/training/        flow training and Optuna hyperparameter search
    src/evaluation/      posterior sampling
    src/data_tools/      dataset loading and consistency checks
    scripts/             SLURM submission wrappers

## Notes

The pipeline is: prepare the power spectra (`SDC_inference/prepare_ps.ipynb`),
train the flow (`src/training/train_flow_only.py`, tuned with
`src/training/optuna_flow.py`), sample posteriors
(`src/evaluation/inference_flow_only.py`), then produce results and contours
(`SDC_inference/create_results.py`, `SDC_inference/create_contours.py`).

The SLURM scripts under `scripts/` and `SDC_inference/ska_inference.sh` carry
absolute paths, module loads, and account settings from the Jean Zay cluster
where the challenge entry was run. They are included as a record of the exact
setup rather than as portable tools. Note that `ska_inference.sh` invokes
`create_results.py`, not a script of its own name.

For the general-purpose version of EoRFlow, see the `main` branch.
