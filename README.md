# EoRFlow — SKA Science Data Challenge 3b

The version of EoRFlow used for the
[SKA Science Data Challenge 3b](https://sdc3.skao.int/challenges/inference/results)
inference task.

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

For the general-purpose version of EoRFlow, see the `main` branch.
