"""
Experiment utilities for lambda-weighted loss, mask robustness, and lambda sensitivity.
"""

import csv
import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from ibpm import SL_IBPM, IBPMTrainer
from utils import SEEDS, compute_stats, load_data

# ===================== Lambda-weighted loss (paper Eq. 23) =====================


class LambdaWeightedLoss(nn.Module):
    """
    Implement the weighted loss from paper Eq. (23):
        L = sum_{i in S} L_i + lambda * sum_{j not in S} L_j

    S is the source-node set, and lambda controls the loss weight of non-source nodes.
    When lambda < 1.0, non-source penalties are reduced so the model focuses more on sources.
    """

    def __init__(self, lambda_: float = 0.1, reduction: str = "mean") -> None:
        super().__init__()
        self.lambda_ = lambda_
        self.reduction = reduction
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, pred: Tensor, true: Tensor) -> Tensor:
        """
        pred: [batch_size, num_nodes] or [batch_size, *] - model logits.
        true: [batch_size, num_nodes] or [batch_size, *] - labels (0 or 1).
        """
        # Compute unreduced BCE loss for each element.
        loss_per_element = self.bce(
            pred, true.float()
        )  # shape: [batch_size, num_nodes]

        # Build a weight matrix: source nodes use 1, non-source nodes use lambda.
        weights = torch.where(
            true > 0.5, torch.ones_like(true), torch.full_like(true, self.lambda_)
        )

        # Weighted loss.
        weighted_loss = loss_per_element * weights

        if self.reduction == "mean":
            return weighted_loss.mean()
        elif self.reduction == "sum":
            return weighted_loss.sum()
        else:
            return weighted_loss


# ===================== Lambda-aware trainer variant =====================


class IBPMTrainerWithLambda(IBPMTrainer):
    """
    Extend IBPMTrainer by replacing BiasedEstimator with LambdaWeightedLoss.
    """

    def __init__(
        self,
        model: SL_IBPM,
        lr: float,
        weight_decay: float,
        lambda_: float,
        reduction: str,
        device: str,
    ) -> None:
        # Override initialization directly instead of calling the parent __init__.
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.estimator = LambdaWeightedLoss(lambda_=lambda_, reduction=reduction)


# ===================== Saving utilities =====================


def save_mask_robustness_csv(
    mask_ratios: list,
    results_by_method: dict,
    save_path: str,
):
    """
    Save mask robustness experiment results.

    results_by_method: {
        method_name: {
            dataset_name: {
                'auc': [auc_at_mask_0, auc_at_mask_0.1, ...],
                'f1': [f1_at_mask_0, f1_at_mask_0.1, ...]
            }
        }
    }
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fieldnames = ["mask_ratio"] + list(results_by_method.keys())

    with open(save_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["dataset", "metric", "mask_ratio"] + list(results_by_method.keys())
        )

        for dataset_name in next(iter(results_by_method.values())).keys():
            for metric in ["auc", "f1"]:
                for i, mr in enumerate(mask_ratios):
                    row = [dataset_name, metric, f"{mr:.1f}"]
                    for method_name in results_by_method:
                        row.append(
                            f"{results_by_method[method_name][dataset_name][metric][i]:.4f}"
                        )
                    writer.writerow(row)


def save_lambda_sensitivity_csv(
    lambda_values: list,
    results: dict,
    save_path: str,
):
    """
    Save lambda sensitivity experiment results.

    results: {
        (dataset_name, model_type): {
            'auc': [auc_at_lambda_0.1, auc_at_lambda_0.2, ...],
            'f1': [f1_at_lambda_0.1, f1_at_lambda_0.2, ...]
        }
    }
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset", "type", "lambda", "auc", "f1"])
        for (name, type_), metrics in results.items():
            for i, lam in enumerate(lambda_values):
                writer.writerow(
                    [
                        name,
                        type_,
                        f"{lam:.2f}",
                        f"{metrics['auc'][i]:.4f}",
                        f"{metrics['f1'][i]:.4f}",
                    ]
                )


# ===================== Mask robustness analysis =====================


def mask_robustness_analysis(
    name: str,
    type_: str,
    lr: float = 0.001,
    lambda_: float = 0.1,
    weight_decay: float = 0.0,
    device: str = "cuda:0",
    mask_ratios: list | None = None,
    epochs: int = 100,
    seeds: list | None = None,
    n_mask_trials: int = 5,  # Number of independent mask trials per seed and ratio.
    save_dir: str = "result/mask_robustness",
):
    """
    Mask robustness analysis with repeated random masks.

    Gradually mask different proportions of observed nodes to test SL-IBPM under
    limited observations. For each (seed, mask_ratio), run n_mask_trials independent
    random masks, average trials first, then average across seeds.
    """
    if mask_ratios is None:
        mask_ratios = [i / 10 for i in range(10)]  # [0.0, 0.1, ..., 0.9]
    if seeds is None:
        seeds = SEEDS  # [0, 1, 2, 3, 4]

    # Load data only to get graph metadata; each seed reloads its own split.
    (
        train_loader,
        valid_loader,
        test_loader,
        edge_index,
        num_nodes,
        num_edges,
        num_states,
    ) = load_data(name, type_, seed=seeds[0])

    auc_results_mean = []  # Cross-seed mean AUC.
    auc_results_std = []  # Cross-seed AUC std.
    f1_results_mean = []  # Cross-seed mean F1.
    f1_results_std = []  # Cross-seed F1 std.

    for mask_ratio in mask_ratios:
        num_masked = int(num_nodes * mask_ratio)

        # Collect results across seeds.
        seed_aucs = []
        seed_f1s = []

        for seed in seeds:
            # Load the data split for the current seed.
            (
                train_loader,
                valid_loader,
                test_loader,
                edge_index,
                num_nodes,
                num_edges,
                num_states,
            ) = load_data(name, type_, seed=seed)

            # Run n_mask_trials independent random masks for the current seed.
            trial_aucs = []
            trial_f1s = []

            for trial in range(n_mask_trials):
                # Bind the random seed to both seed and trial for reproducible independent masks.
                mask_rng_seed = seed * 10000 + trial
                rng = np.random.RandomState(mask_rng_seed)

                if num_masked > 0:
                    masked_nodes = sorted(
                        rng.choice(num_nodes, size=num_masked, replace=False).tolist()
                    )
                else:
                    masked_nodes = None

                # Reset torch/NumPy seeds so model initialization and training randomness match per trial.
                # This isolates the variance introduced by mask sampling.
                torch.manual_seed(seed)
                np.random.seed(seed)
                model = SL_IBPM(num_nodes, num_edges, num_states)
                trainer = IBPMTrainerWithLambda(
                    model,
                    lr=lr,
                    weight_decay=weight_decay,
                    lambda_=lambda_,
                    reduction="mean",
                    device=device,
                )

                # Train.
                trainer.fit(
                    train_loader,
                    valid_loader,
                    test_loader,
                    edge_index,
                    epochs=epochs,
                    mask=masked_nodes,
                )

                # Test.
                roc, precision, recall, f1 = trainer.evaluate(
                    test_loader, edge_index, mask=masked_nodes
                )
                trial_aucs.append(roc)
                trial_f1s.append(f1)

            # Average mask trials under the current seed for a stable estimate.
            seed_auc_mean = float(np.mean(trial_aucs))
            seed_f1_mean = float(np.mean(trial_f1s))
            seed_aucs.append(seed_auc_mean)
            seed_f1s.append(seed_f1_mean)

        # Compute cross-seed statistics.
        auc_mean, auc_std = compute_stats(seed_aucs)
        f1_mean, f1_std = compute_stats(seed_f1s)
        auc_results_mean.append(auc_mean)
        auc_results_std.append(auc_std)
        f1_results_mean.append(f1_mean)
        f1_results_std.append(f1_std)

        print(
            f"  [{name}-{type_}] Mask ratio {mask_ratio:.1f} "
            f"(trials={n_mask_trials}x{len(seeds)}): "
            f"AUC={auc_mean:.4f}+/-{auc_std:.4f}, F1={f1_mean:.4f}+/-{f1_std:.4f}"
        )

    # Save results with std columns to inspect variance suppression.
    save_path = f"{save_dir}/{name}_{type_}_mask_robustness.csv"
    os.makedirs(save_dir, exist_ok=True)
    with open(save_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["mask_ratio", "auc_mean", "auc_std", "f1_mean", "f1_std"])
        for mr, auc_m, auc_s, f1_m, f1_s in zip(
            mask_ratios,
            auc_results_mean,
            auc_results_std,
            f1_results_mean,
            f1_results_std,
        ):
            writer.writerow(
                [
                    f"{mr:.1f}",
                    f"{auc_m:.4f}",
                    f"{auc_s:.4f}",
                    f"{f1_m:.4f}",
                    f"{f1_s:.4f}",
                ]
            )

    return mask_ratios, auc_results_mean, f1_results_mean


def run_mask_robustness_for_paper(
    lr: float = 0.001,
    lambda_: float = 0.1,
    weight_decay: float = 0.0,
    device: str = "cuda:0",
    mask_ratios: list | None = None,
    epochs: int = 100,
    n_mask_trials: int = 5,  # Number of trials for mask robustness experiments.
    save_dir: str = "result/mask_robustness",
):
    """
    Run the mask robustness experiments corresponding to Fig. 2.
    """
    if mask_ratios is None:
        mask_ratios = [i / 10 for i in range(10)]

    datasets = ["karate", "jazz", "net_science", "cora_ml"]
    model_type = "SIR"

    print("=" * 70)
    print("Mask robustness analysis (corresponding to Fig. 2, with repeated random masks)")
    print("=" * 70)

    for name in datasets:
        print(f"\n--- Dataset: {name} ---")
        mask_robustness_analysis(
            name=name,
            type_=model_type,
            lr=lr,
            lambda_=lambda_,
            weight_decay=weight_decay,
            device=device,
            mask_ratios=mask_ratios,
            epochs=epochs,
            n_mask_trials=n_mask_trials,  # Pass through.
            save_dir=save_dir,
        )

    print("\nMask robustness experiments completed.")


# ===================== Lambda sensitivity analysis =====================


def lambda_sensitivity_analysis(
    name: str,
    type_: str,
    lr: float = 0.001,
    weight_decay: float = 0.0,
    device: str = "cuda:0",
    lambda_values: list | None = None,
    epochs: int = 100,
    seeds: list | None = None,
    save_dir: str = "result/lambda_sensitivity",
):
    """
    Lambda sensitivity analysis.

    Test how different lambda values affect model performance.

    Args:
        name: Dataset name.
        type_: Propagation model type.
        lambda_values: Lambda values. Defaults to [0.01, 0.05, 0.1, 0.2, 0.5, 1.0].
        epochs: Number of training epochs.
        seeds: Random seed list.

    Returns:
        lambda_values: Lambda value list.
        auc_results: AUC list for each lambda value (cross-seed mean).
        f1_results: F1 list for each lambda value (cross-seed mean).
    """
    if lambda_values is None:
        lambda_values = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
    if seeds is None:
        seeds = SEEDS

    auc_results = []
    f1_results = []

    for lam in lambda_values:
        # Collect results across seeds.
        seed_aucs = []
        seed_f1s = []

        for seed in seeds:
            # Load data.
            (
                train_loader,
                valid_loader,
                test_loader,
                edge_index,
                num_nodes,
                num_edges,
                num_states,
            ) = load_data(name, type_, seed=seed)

            # Set up model and trainer.
            torch.manual_seed(seed)
            np.random.seed(seed)
            model = SL_IBPM(num_nodes, num_edges, num_states)
            trainer = IBPMTrainerWithLambda(
                model,
                lr=lr,
                weight_decay=weight_decay,
                lambda_=lam,
                reduction="mean",
                device=device,
            )

            # Train.
            trainer.fit(
                train_loader, valid_loader, test_loader, edge_index, epochs=epochs
            )

            # Test.
            roc, precision, recall, f1 = trainer.evaluate(test_loader, edge_index)
            seed_aucs.append(roc)
            seed_f1s.append(f1)

        # Compute cross-seed mean.
        auc_mean, auc_std = compute_stats(seed_aucs)
        f1_mean, f1_std = compute_stats(seed_f1s)
        auc_results.append(auc_mean)
        f1_results.append(f1_mean)

        print(
            f"  [{name}-{type_}] lambda={lam:.2f}: "
            f"AUC={auc_mean:.4f}+/-{auc_std:.4f}, F1={f1_mean:.4f}+/-{f1_std:.4f}"
        )

    return lambda_values, auc_results, f1_results


def run_lambda_sensitivity_for_paper(
    lr: float = 0.001,
    weight_decay: float = 0.0,
    device: str = "cuda:0",
    lambda_values: list | None = None,
    epochs: int = 100,
    save_dir: str = "result/lambda_sensitivity",
):
    """
    Run the lambda sensitivity experiments corresponding to Fig. 4.

    Test how different lambda values affect F1 on six datasets.

    The recommended lambda search range is {0.05, 0.1, 0.2, 0.5}.
    """
    if lambda_values is None:
        # Fig. 4 uses 0.1, 0.2, ..., 1.0.
        lambda_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    file_list = ["karate", "jazz", "net_science", "cora_ml", "power_grid", "lastFM"]
    type_list = ["SIR"]

    print("=" * 70)
    print("Lambda sensitivity analysis (corresponding to Fig. 4)")
    print("=" * 70)

    all_results = {}

    for name in file_list:
        for type_ in type_list:
            print(f"\n--- Dataset: {name}, model: {type_} ---")
            _, auc_results, f1_results = lambda_sensitivity_analysis(
                name=name,
                type_=type_,
                lr=lr,
                weight_decay=weight_decay,
                device=device,
                lambda_values=lambda_values,
                epochs=epochs,
                save_dir=save_dir,
            )
            all_results[(name, type_)] = {
                "auc": auc_results,
                "f1": f1_results,
            }

    # Save summary results.
    save_lambda_sensitivity_csv(
        lambda_values,
        all_results,
        f"{save_dir}/lambda_sensitivity.csv",
    )

    print("\nLambda sensitivity experiments completed.")
    return all_results


# ===================== Fast lambda search for new datasets =====================


def lambda_grid_search(
    name: str,
    type_: str,
    lr: float = 0.001,
    weight_decay: float = 0.0,
    device: str = "cuda:0",
    lambda_candidates: list | None = None,
    epochs: int = 100,
    seed: int = 0,
):
    """
    Run a lambda grid search for a new dataset and choose the best lambda.

    Search over {0.05, 0.1, 0.2, 0.5}. Since the model usually converges in fewer
    than 20 epochs, this search is inexpensive.

    Args:
        name: Dataset name.
        type_: Propagation model type.
        lambda_candidates: Candidate lambda values. Defaults to [0.05, 0.1, 0.2, 0.5].
        epochs: Number of training epochs; fewer epochs can speed up the search.
        seed: Random seed.

    Returns:
        best_lambda: Best lambda value.
        best_f1: Best F1 score.
        results: Results for all candidate lambda values.
    """
    if lambda_candidates is None:
        lambda_candidates = [0.05, 0.1, 0.2, 0.5]

    print(f"Lambda grid search: {name}-{type_}")
    print(f"Candidates: {lambda_candidates}")

    # Load data.
    (
        train_loader,
        valid_loader,
        test_loader,
        edge_index,
        num_nodes,
        num_edges,
        num_states,
    ) = load_data(name, type_, seed=seed)

    results = {}
    best_lambda = None
    best_f1 = -1.0

    for lam in lambda_candidates:
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = SL_IBPM(num_nodes, num_edges, num_states)
        trainer = IBPMTrainerWithLambda(
            model,
            lr=lr,
            weight_decay=weight_decay,
            lambda_=lam,
            reduction="mean",
            device=device,
        )

        # Train.
        trainer.fit(train_loader, valid_loader, test_loader, edge_index, epochs=epochs)

        # Evaluate on the validation set with F1 as the primary metric.
        roc, precision, recall, f1 = trainer.evaluate(valid_loader, edge_index)
        results[lam] = {"auc": roc, "f1": f1}

        print(f"  lambda={lam:.2f}: Valid AUC={roc:.4f}, Valid F1={f1:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            best_lambda = lam

    print(f"\nBest lambda: {best_lambda:.2f}, corresponding F1: {best_f1:.4f}")
    return best_lambda, best_f1, results


# ===================== Main entry point =====================

if __name__ == "__main__":
    run_mask_robustness_for_paper(
        lr=0.001,
        lambda_=0.1,
        device="cuda:0",
        epochs=100,
    )

    run_lambda_sensitivity_for_paper(
        lr=0.001,
        device="cuda:0",
        epochs=100,
    )

    # best_lambda, best_f1, results = lambda_grid_search(
    #     name="karate",
    #     type_="SIR",
    #     epochs=20,
    # )
