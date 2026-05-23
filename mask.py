"""
Multi-algorithm mask robustness experiments (GCNSI / HFSD / IVGD / MPNN / RSDGIN)

Runs the robustness experiments corresponding to Fig. 2 and evaluates several
baseline methods under different observation ratios.
Experiments repeat masking across multiple seeds and average the results.

Usage:
    from baseline_mask_experiments import run_all_baselines_mask_robustness

    # Run mask robustness experiments for all baselines.
    run_all_baselines_mask_robustness(
        algorithms=["gcnsi", "hfsd", "ivgd", "mpnn", "rdgin"],
        datasets=["karate", "jazz", "net_science", "cora_ml"],
        model_type="SIR",
        mask_ratios=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        seeds=[0, 1, 2, 3, 4],
        epochs=100,
        device="cuda:0",
    )
"""

import csv
import os
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from utils import load_data, SEEDS, compute_stats

# ---- Import baseline algorithms ----
from gcnsi import (
    GCNSI,
    GCNSITrainer,
    InputGenerator,
    preprocess_dataloader as gcnsi_preprocess,
)
from hfsd import (
    HFSDModel,
    HFSDTrainer,
    HFSDFeatureGenerator,
    preprocess_dataloader as hfsd_preprocess,
)
from ivgd import IVGD, IVGDTrainer
from mpnn import ResGCN, ResGCNTrainer
from rdgin import RSDGIN, RSDGINTrainer

# ===================== Unified training/evaluation interfaces =====================


def _run_gcnsi_mask_experiment(
    file_name: str,
    type_: str,
    seed: int,
    mask: list | None,
    epochs: int,
    device: str,
):
    """Run one masked GCNSI experiment."""
    (
        train_loader,
        valid_loader,
        test_loader,
        edge_index,
        num_nodes,
        _,
        _,
    ) = load_data(file_name, type_, seed=seed)

    # GCNSI needs preprocessing; InputGenerator depends on num_nodes.
    generator = InputGenerator(edge_index=edge_index.to(device), num_nodes=num_nodes)
    train_loader = gcnsi_preprocess(train_loader, generator, device)
    valid_loader = gcnsi_preprocess(valid_loader, generator, device)
    test_loader = gcnsi_preprocess(test_loader, generator, device)

    torch.manual_seed(seed)
    np.random.seed(seed)
    model = GCNSI()
    trainer = GCNSITrainer(model=model, lr=0.001, reduction="mean", device=device)
    trainer.fit(train_loader, valid_loader, edge_index, epochs=epochs, mask=mask)
    auc, pre, rec, f1 = trainer.evaluate(test_loader, edge_index, mask=mask)
    return auc, pre, rec, f1


def _run_hfsd_mask_experiment(
    file_name: str,
    type_: str,
    seed: int,
    mask: list | None,
    epochs: int,
    device: str,
):
    """Run one masked HFSD experiment."""
    (
        train_loader,
        valid_loader,
        test_loader,
        edge_index,
        num_nodes,
        _,
        _,
    ) = load_data(file_name, type_, seed=seed)

    # HFSD needs preprocessing; HFSDFeatureGenerator depends on edge_index and num_nodes.
    generator = HFSDFeatureGenerator(
        edge_index=edge_index.to(device), num_nodes=num_nodes
    )
    train_loader = hfsd_preprocess(train_loader, generator, device)
    valid_loader = hfsd_preprocess(valid_loader, generator, device)
    test_loader = hfsd_preprocess(test_loader, generator, device)

    torch.manual_seed(seed)
    np.random.seed(seed)
    model = HFSDModel()
    trainer = HFSDTrainer(model, num_nodes, lr=1e-3, lambda_=0.05, device=device)
    trainer.fit(train_loader, valid_loader, edge_index, epochs=epochs, mask=mask)
    auc, pre, rec, f1 = trainer.evaluate(test_loader, edge_index, mask=mask)
    return auc, pre, rec, f1


def _run_ivgd_mask_experiment(
    file_name: str,
    type_: str,
    seed: int,
    mask: list | None,
    epochs: int,
    device: str,
):
    """Run one masked IVGD experiment."""
    (
        train_loader,
        valid_loader,
        test_loader,
        edge_index,
        _,
        _,
        num_states,
    ) = load_data(file_name, type_, seed=seed)

    torch.manual_seed(seed)
    np.random.seed(seed)
    model = IVGD(num_states)
    trainer = IVGDTrainer(model=model, lr=0.001, reduction="mean", device=device)
    trainer.fit(train_loader, valid_loader, edge_index, epochs=epochs, mask=mask)
    auc, pre, rec, f1 = trainer.evaluate(test_loader, edge_index, mask=mask)
    return auc, pre, rec, f1


def _run_mpnn_mask_experiment(
    file_name: str,
    type_: str,
    seed: int,
    mask: list | None,
    epochs: int,
    device: str,
):
    """Run one masked MPNN (ResGCN) experiment."""
    (
        train_loader,
        valid_loader,
        test_loader,
        edge_index,
        _,
        _,
        num_states,
    ) = load_data(file_name, type_, seed=seed)

    torch.manual_seed(seed)
    np.random.seed(seed)
    model = ResGCN(in_dim=num_states)
    trainer = ResGCNTrainer(model=model, lr=0.001, device=device)
    trainer.fit(train_loader, valid_loader, edge_index, epochs=epochs, mask=mask)
    auc, pre, rec, f1 = trainer.evaluate(test_loader, edge_index, mask=mask)
    return auc, pre, rec, f1


def _run_rdgin_mask_experiment(
    file_name: str,
    type_: str,
    seed: int,
    mask: list | None,
    epochs: int,
    device: str,
):
    """Run one masked RSDGIN experiment."""
    (
        train_loader,
        valid_loader,
        test_loader,
        edge_index,
        _,
        _,
        num_states,
    ) = load_data(file_name, type_, seed=seed)

    torch.manual_seed(seed)
    np.random.seed(seed)
    model = RSDGIN(num_states, 256)
    trainer = RSDGINTrainer(model, lr=0.001, reduction="mean", device=device)
    trainer.fit(train_loader, valid_loader, edge_index, epochs=epochs, mask=mask)
    auc, pre, rec, f1 = trainer.evaluate(test_loader, edge_index, mask=mask)
    return auc, pre, rec, f1


# Map algorithm names to experiment runners.
ALGORITHM_RUNNERS = {
    "gcnsi": _run_gcnsi_mask_experiment,
    "hfsd": _run_hfsd_mask_experiment,
    "ivgd": _run_ivgd_mask_experiment,
    "mpnn": _run_mpnn_mask_experiment,
    "rdgin": _run_rdgin_mask_experiment,
}


# ===================== Core: mask robustness for one algorithm =====================


def baseline_mask_robustness(
    algorithm: str,
    name: str,
    type_: str,
    mask_ratios: list = None,
    seeds: list = None,
    epochs: int = 100,
    device: str = "cuda:0",
    save_dir: str = "result/baseline_mask",
):
    """
    Run mask robustness analysis for one baseline algorithm.

    Args:
        algorithm: Algorithm name; one of "gcnsi"/"hfsd"/"ivgd"/"mpnn"/"rdgin".
        name: Dataset name, such as "karate".
        type_: Propagation model type, such as "SIR".
        mask_ratios: Mask ratios. Defaults to [0.0, 0.1, ..., 0.9].
        seeds: Random seeds. Defaults to [0, 1, 2, 3, 4].
        epochs: Number of training epochs.
        device: Compute device.

    Returns:
        result_dict: {
            "mask_ratios": [...],
            "auc_mean": [...],   # cross-seed mean
            "auc_std": [...],    # cross-seed std
            "f1_mean": [...],
            "f1_std": [...],
        }
    """
    if mask_ratios is None:
        mask_ratios = [i / 10 for i in range(10)]  # [0.0, 0.1, ..., 0.9]
    if seeds is None:
        seeds = SEEDS

    if algorithm not in ALGORITHM_RUNNERS:
        raise ValueError(
            f"Unknown algorithm: {algorithm}. "
            f"Choose from {list(ALGORITHM_RUNNERS.keys())}"
        )

    runner = ALGORITHM_RUNNERS[algorithm]

    # First get num_nodes to compute the number of masked nodes.
    _, _, _, edge_index, num_nodes, _, _ = load_data(name, type_, seed=seeds[0])

    auc_means, auc_stds = [], []
    f1_means, f1_stds = [], []

    for mask_ratio in mask_ratios:
        num_masked = int(num_nodes * mask_ratio)

        seed_aucs, seed_f1s = [], []

        for seed in seeds:
            # Generate the masked nodes for this seed, ensuring reproducibility.
            rng = np.random.RandomState(seed)
            if num_masked > 0:
                masked_nodes = sorted(
                    rng.choice(num_nodes, size=num_masked, replace=False).tolist()
                )
            else:
                masked_nodes = None

            # Run one experiment for this algorithm.
            try:
                auc, pre, rec, f1 = runner(
                    file_name=name,
                    type_=type_,
                    seed=seed,
                    mask=masked_nodes,
                    epochs=epochs,
                    device=device,
                )
                seed_aucs.append(auc)
                seed_f1s.append(f1)
            except Exception as e:
                print(
                    f"    [Warning] {algorithm.upper()} {name}-{type_} "
                    f"mask={mask_ratio:.1f} seed={seed} failed: {e}"
                )
                continue

        # Cross-seed statistics.
        auc_mean, auc_std = compute_stats(seed_aucs)
        f1_mean, f1_std = compute_stats(seed_f1s)
        auc_means.append(auc_mean)
        auc_stds.append(auc_std)
        f1_means.append(f1_mean)
        f1_stds.append(f1_std)

        print(
            f"  [{algorithm.upper()}] {name}-{type_} mask={mask_ratio:.1f}: "
            f"AUC={auc_mean:.4f}+/-{auc_std:.4f}, F1={f1_mean:.4f}+/-{f1_std:.4f}"
        )

    result = {
        "mask_ratios": mask_ratios,
        "auc_mean": auc_means,
        "auc_std": auc_stds,
        "f1_mean": f1_means,
        "f1_std": f1_stds,
    }

    # Save results to CSV.
    os.makedirs(save_dir, exist_ok=True)
    csv_path = f"{save_dir}/{algorithm}_{name}_{type_}_mask.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["mask_ratio", "auc_mean", "auc_std", "f1_mean", "f1_std"])
        for i in range(len(mask_ratios)):
            writer.writerow(
                [
                    f"{mask_ratios[i]:.1f}",
                    f"{auc_means[i]:.4f}",
                    f"{auc_stds[i]:.4f}",
                    f"{f1_means[i]:.4f}",
                    f"{f1_stds[i]:.4f}",
                ]
            )
    print(f"  Results saved: {csv_path}")

    return result


# ===================== Batch runner =====================


def run_all_baselines_mask_robustness(
    algorithms: list = None,
    datasets: list = None,
    model_type: str = "SIR",
    mask_ratios: list = None,
    seeds: list = None,
    epochs: int = 100,
    device: str = "cuda:0",
    save_dir: str = "result/baseline_mask",
):
    """
    Run mask robustness experiments for multiple baselines and datasets.

    Args:
        algorithms: Algorithm list. Defaults to all five baselines.
        datasets: Dataset list. Defaults to ["karate", "jazz", "net_science", "cora_ml"].
        model_type: Propagation model type. Defaults to "SIR".
        mask_ratios: Mask ratio list.
        seeds: Random seed list.
        epochs: Number of training epochs.
        device: Compute device.
        save_dir: Directory for saved results.

    Returns:
        all_results: {
            algorithm_name: {
                dataset_name: {
                    "mask_ratios": [...],
                    "auc_mean": [...],
                    "f1_mean": [...],
                }
            }
        }
    """
    if algorithms is None:
        algorithms = ["gcnsi", "hfsd", "ivgd", "mpnn", "rdgin"]
    if datasets is None:
        datasets = ["karate", "jazz", "net_science", "cora_ml"]
    if mask_ratios is None:
        mask_ratios = [i / 10 for i in range(10)]

    print("=" * 70)
    print("Multi-algorithm mask robustness experiments (corresponding to Fig. 2)")
    print("=" * 70)
    print(f"Algorithms: {algorithms}")
    print(f"Datasets: {datasets}")
    print(f"Propagation model: {model_type}")
    print(f"Mask ratios: {mask_ratios}")
    print(f"Seeds: {seeds if seeds else SEEDS}")
    print(f"Epochs: {epochs}")
    print("=" * 70)

    all_results = {}

    for alg in algorithms:
        print(f"\n{'='*70}")
        print(f"Algorithm: {alg.upper()}")
        print(f"{'='*70}")
        all_results[alg] = {}

        for dataset in datasets:
            print(f"\n--- Dataset: {dataset} ---")
            result = baseline_mask_robustness(
                algorithm=alg,
                name=dataset,
                type_=model_type,
                mask_ratios=mask_ratios,
                seeds=seeds,
                epochs=epochs,
                device=device,
                save_dir=save_dir,
            )
            all_results[alg][dataset] = result

    # Save combined results for all algorithms to one CSV.
    _save_combined_results(
        all_results, algorithms, datasets, model_type, mask_ratios, save_dir
    )

    print("\n" + "=" * 70)
    print("All mask robustness experiments completed.")
    print("=" * 70)
    return all_results


def _save_combined_results(
    all_results: dict,
    algorithms: list,
    datasets: list,
    model_type: str,
    mask_ratios: list,
    save_dir: str,
):
    """Save all algorithm mask robustness results to one summary CSV."""
    os.makedirs(save_dir, exist_ok=True)
    csv_path = f"{save_dir}/all_baselines_{model_type}_mask_summary.csv"

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        # Write header.
        header = ["dataset", "mask_ratio"]
        for alg in algorithms:
            header += [f"{alg}_auc", f"{alg}_f1"]
        writer.writerow(header)

        # Write rows by dataset and mask ratio.
        for dataset in datasets:
            for i, mr in enumerate(mask_ratios):
                row = [dataset, f"{mr:.1f}"]
                for alg in algorithms:
                    result = all_results[alg][dataset]
                    row += [
                        f"{result['auc_mean'][i]:.4f}",
                        f"{result['f1_mean'][i]:.4f}",
                    ]
                writer.writerow(row)

    print(f"\nSummary results saved: {csv_path}")


# ===================== Convenience entry points for individual algorithms =====================


def run_gcnsi_mask(
    datasets: list = None,
    model_type: str = "SIR",
    mask_ratios: list = None,
    seeds: list = None,
    epochs: int = 100,
    device: str = "cuda:0",
    save_dir: str = "result/baseline_mask",
):
    """Convenience function for GCNSI mask robustness experiments."""
    if datasets is None:
        datasets = ["karate", "jazz", "net_science", "cora_ml"]
    results = {}
    for name in datasets:
        results[name] = baseline_mask_robustness(
            algorithm="gcnsi",
            name=name,
            type_=model_type,
            mask_ratios=mask_ratios,
            seeds=seeds,
            epochs=epochs,
            device=device,
            save_dir=save_dir,
        )
    return results


def run_hfsd_mask(
    datasets: list = None,
    model_type: str = "SIR",
    mask_ratios: list = None,
    seeds: list = None,
    epochs: int = 100,
    device: str = "cuda:0",
    save_dir: str = "result/baseline_mask",
):
    """Convenience function for HFSD mask robustness experiments."""
    if datasets is None:
        datasets = ["karate", "jazz", "net_science", "cora_ml"]
    results = {}
    for name in datasets:
        results[name] = baseline_mask_robustness(
            algorithm="hfsd",
            name=name,
            type_=model_type,
            mask_ratios=mask_ratios,
            seeds=seeds,
            epochs=epochs,
            device=device,
            save_dir=save_dir,
        )
    return results


def run_ivgd_mask(
    datasets: list = None,
    model_type: str = "SIR",
    mask_ratios: list = None,
    seeds: list = None,
    epochs: int = 100,
    device: str = "cuda:0",
    save_dir: str = "result/baseline_mask",
):
    """Convenience function for IVGD mask robustness experiments."""
    if datasets is None:
        datasets = ["karate", "jazz", "net_science", "cora_ml"]
    results = {}
    for name in datasets:
        results[name] = baseline_mask_robustness(
            algorithm="ivgd",
            name=name,
            type_=model_type,
            mask_ratios=mask_ratios,
            seeds=seeds,
            epochs=epochs,
            device=device,
            save_dir=save_dir,
        )
    return results


def run_mpnn_mask(
    datasets: list = None,
    model_type: str = "SIR",
    mask_ratios: list = None,
    seeds: list = None,
    epochs: int = 100,
    device: str = "cuda:0",
    save_dir: str = "result/baseline_mask",
):
    """Convenience function for MPNN mask robustness experiments."""
    if datasets is None:
        datasets = ["karate", "jazz", "net_science", "cora_ml"]
    results = {}
    for name in datasets:
        results[name] = baseline_mask_robustness(
            algorithm="mpnn",
            name=name,
            type_=model_type,
            mask_ratios=mask_ratios,
            seeds=seeds,
            epochs=epochs,
            device=device,
            save_dir=save_dir,
        )
    return results


def run_rdgin_mask(
    datasets: list = None,
    model_type: str = "SIR",
    mask_ratios: list = None,
    seeds: list = None,
    epochs: int = 100,
    device: str = "cuda:0",
    save_dir: str = "result/baseline_mask",
):
    """Convenience function for RSDGIN mask robustness experiments."""
    if datasets is None:
        datasets = ["karate", "jazz", "net_science", "cora_ml"]
    results = {}
    for name in datasets:
        results[name] = baseline_mask_robustness(
            algorithm="rdgin",
            name=name,
            type_=model_type,
            mask_ratios=mask_ratios,
            seeds=seeds,
            epochs=epochs,
            device=device,
            save_dir=save_dir,
        )
    return results


# ===================== Main entry point =====================

if __name__ == "__main__":
    # Example: run mask robustness experiments for all baselines.
    run_all_baselines_mask_robustness(
        algorithms=["gcnsi", "hfsd", "ivgd", "mpnn", "rdgin"],
        datasets=["karate", "jazz", "net_science", "cora_ml"],
        model_type="SIR",
        mask_ratios=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        seeds=[0, 1, 2, 3, 4],
        epochs=100,
        device="cuda:3",
    )
