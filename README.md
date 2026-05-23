# SL-IBPM

**Title:** A Mechanism-Guided Source Localization Framework Based on Information Back-Propagation

SL-IBPM is a deep learning framework for locating information sources in graph
propagation processes. The core idea is an Information Back Propagation Mechanism
that reconstructs likely source nodes from observed propagation states through two
components: self-information damping and neighbor-information aggregation. The
BackProp module can be embedded into unidirectional propagation models and is
evaluated against classical and neural baselines on synthetic and real cascade
datasets.

## Abstract

The rapid expansion of social networks has amplified the spread of rumors and misinformation, making accurate and efficient information source localization a critical research problem. Traditional source localization methods are highly sensitive to propagation randomness and incomplete observations. Existing deep learning-based approaches mainly rely on empirical message-passing architectures to fit diffusion snapshots, which leads to three major limitations: strong dependence on large-scale observational data, difficulty in balancing localization accuracy and computational complexity, and insufficient theoretical alignment with the intrinsic dynamics of graph diffusion. To address these limitations, we propose a novel mechanism-guided deep learning framework termed SL-IBPM (Source Localization via Information back-propagation Mechanism). Unlike empirical fitting approaches, SL-IBPM is developed through a mechanism-guided reverse modeling strategy motivated by the forward dynamic message-passing algorithm. It explicitly formulates a reverse state transition process for graph diffusion via two interpretable components. The proposed Information back-propagation Mechanism (IBPM) decomposes the reverse diffusion process into two fundamental components: self-information Damping and neighbor-information Aggregation. These components are integrated into a lightweight BackProp module that can be universally embedded into various unidirectional propagation models, including SI, SIR, LT, and IC. Extensive experiments on six social networks and four real propagation cascade datasets demonstrate that SL-IBPM consistently outperforms state-of-the-art methods in source localization accuracy. The framework maintains low model complexity, requires shorter training time, and exhibits strong robustness under limited observational data, validating its practical effectiveness in real-world applications.

## Repository Structure

- `ibpm.py`: Main SL-IBPM model, trainer, synthetic experiments, cascade experiments,
  and layer-sensitivity analysis.
- `util.py`: Shared graph loading, propagation simulation, dataset generation, data
  loading, and CSV-saving utilities.
- `main.py`: Convenience entry point for running selected experiment groups.
- `gcnsi.py`, `hfsd.py`, `ivgd.py`, `lpsi.py`, `mpnn.py`, `rdgin.py`, `slvae.py`,
  `ajc.py`: Baseline implementations.
- `wo_aggr.py`, `wo_damp.py`, `wo_damp_aggr.py`, `wo_epsilon.py`, `wo_lambda.py`,
  `plain_gnn.py`: Ablation variants.
- `mask.py`: Baseline mask-robustness experiments.
- `others.py`: SL-IBPM mask-robustness and lambda-sensitivity experiments.
- `figure/`: Scripts for plotting runtime, robustness, lambda sensitivity, and
  layer/epoch/loss figures.

## Dataset

Download the dataset from Zenodo:

<https://zenodo.org/records/20222437>

After downloading and extracting the archive, place the `data/` directory in the
project root:

```text
SL-IBPM/
  data/
    SIR/
    SI/
    LT/
    IC/
    android/
    christianity/
    douban/
    twitter/
  ibpm.py
  util.py
  ...
```

Synthetic propagation data are expected under `data/{MODEL}/{DATASET}/` with files
such as `edge_index.npy`, `state.npy`, and `seed.npy`. Real cascade datasets are
expected to contain preprocessed `train_state.npy`, `train_seed.npy`,
`valid_state.npy`, `valid_seed.npy`, `test_state.npy`, `test_seed.npy`, and
`edge_index.npy`.

## Requirements

The project uses Python with PyTorch and PyTorch Geometric. Install versions that
match your CUDA/PyTorch environment.

Core Python packages:

- `torch`
- `torch-geometric`
- `torch-scatter`
- `numpy`
- `networkx`
- `scikit-learn`
- `pandas`
- `matplotlib`
- `seaborn`

## Running Experiments

Run the main SL-IBPM experiments:

```bash
python ibpm.py
```

Run the convenience experiment entry point:

```bash
python main.py
```

Run baseline mask-robustness experiments:

```bash
python mask.py
```

Run SL-IBPM mask-robustness and lambda-sensitivity experiments:

```bash
python others.py
```

Generate datasets from graph edge files:

```bash
python util.py
```

Each script exposes callable functions with configurable arguments such as
`epochs`, `device`, `lr`, `alpha`, `lambda_`, `mask_ratios`, and `save_dir`.

## Outputs

Experiment results are written under `result/`, including:

- `result/SL-IBPM.csv`: Aggregated SL-IBPM metrics.
- `result/history/` or `result/history_aggregated/`: Per-epoch aggregated training
  history.
- `result/layer_sensitivity/`: Layer-sensitivity summaries.
- `result/mask_robustness/`: SL-IBPM mask-robustness results.
- `result/baseline_mask/`: Baseline mask-robustness results.
- `result/lambda_sensitivity/`: Lambda-sensitivity results.

Figure scripts in `figure/` read these result files and save PDF outputs.

## Notes

- Default experiments use five seeds: `0, 1, 2, 3, 4`.
- Synthetic data use an 8:1:1 train/validation/test split.
- Source nodes are sampled as 10% of graph nodes by default.
- The default propagation target ratio is 30%.
