"""
多算法 Mask 鲁棒性实验 (GCNSI / HFSD / IVGD / MPNN / RSDGIN)

对应论文 Fig. 2 的鲁棒性实验，对多个基线方法进行不同观测比例下的性能测试。
实验在多个 seed 下重复 mask 并取平均，确保结果稳定。

用法:
    from baseline_mask_experiments import run_all_baselines_mask_robustness

    # 运行所有基线的 mask 鲁棒性实验
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

# ---- 导入各基线算法 ----
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

# ===================== 统一接口：各算法的训练与评估 =====================


def _run_gcnsi_mask_experiment(
    file_name: str,
    type_: str,
    seed: int,
    mask: list | None,
    epochs: int,
    device: str,
):
    """运行 GCNSI 的单个 mask 实验。"""
    (
        train_loader,
        valid_loader,
        test_loader,
        edge_index,
        num_nodes,
        _,
        _,
    ) = load_data(file_name, type_, seed=seed)

    # GCNSI 需要预处理（InputGenerator 依赖 num_nodes）
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
    """运行 HFSD 的单个 mask 实验。"""
    (
        train_loader,
        valid_loader,
        test_loader,
        edge_index,
        num_nodes,
        _,
        _,
    ) = load_data(file_name, type_, seed=seed)

    # HFSD 需要预处理（HFSDFeatureGenerator 依赖 edge_index 和 num_nodes）
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
    """运行 IVGD 的单个 mask 实验。"""
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
    """运行 MPNN (ResGCN) 的单个 mask 实验。"""
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
    """运行 RSDGIN 的单个 mask 实验。"""
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


# 算法名称到实验函数的映射
ALGORITHM_RUNNERS = {
    "gcnsi": _run_gcnsi_mask_experiment,
    "hfsd": _run_hfsd_mask_experiment,
    "ivgd": _run_ivgd_mask_experiment,
    "mpnn": _run_mpnn_mask_experiment,
    "rdgin": _run_rdgin_mask_experiment,
}


# ===================== 核心：单个算法的 mask 鲁棒性分析 =====================


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
    对指定基线算法进行 Mask 鲁棒性分析。

    参数:
        algorithm: 算法名称，可选 "gcnsi"/"hfsd"/"ivgd"/"mpnn"/"rdgin"
        name: 数据集名称，如 "karate"
        type_: 传播模型类型，如 "SIR"
        mask_ratios: mask 比例列表，默认 [0.0, 0.1, ..., 0.9]
        seeds: 随机种子列表，默认 [0,1,2,3,4]
        epochs: 训练轮数
        device: 计算设备

    返回:
        result_dict: {
            "mask_ratios": [...],
            "auc_mean": [...],   # 跨 seed 均值
            "auc_std": [...],    # 跨 seed 标准差
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

    # 先获取 num_nodes（用于计算 mask 节点数）
    _, _, _, edge_index, num_nodes, _, _ = load_data(name, type_, seed=seeds[0])

    auc_means, auc_stds = [], []
    f1_means, f1_stds = [], []

    for mask_ratio in mask_ratios:
        num_masked = int(num_nodes * mask_ratio)

        seed_aucs, seed_f1s = [], []

        for seed in seeds:
            # 生成该 seed 对应的 mask 节点（确保可复现）
            rng = np.random.RandomState(seed)
            if num_masked > 0:
                masked_nodes = sorted(
                    rng.choice(num_nodes, size=num_masked, replace=False).tolist()
                )
            else:
                masked_nodes = None

            # 运行该算法的单个实验
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

        # 跨 seed 统计
        auc_mean, auc_std = compute_stats(seed_aucs)
        f1_mean, f1_std = compute_stats(seed_f1s)
        auc_means.append(auc_mean)
        auc_stds.append(auc_std)
        f1_means.append(f1_mean)
        f1_stds.append(f1_std)

        print(
            f"  [{algorithm.upper()}] {name}-{type_} mask={mask_ratio:.1f}: "
            f"AUC={auc_mean:.4f}±{auc_std:.4f}, F1={f1_mean:.4f}±{f1_std:.4f}"
        )

    result = {
        "mask_ratios": mask_ratios,
        "auc_mean": auc_means,
        "auc_std": auc_stds,
        "f1_mean": f1_means,
        "f1_std": f1_stds,
    }

    # 保存结果到 CSV
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
    print(f"  结果已保存: {csv_path}")

    return result


# ===================== 批量运行 =====================


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
    批量运行多个基线算法在多个数据集上的 Mask 鲁棒性实验。

    参数:
        algorithms: 算法列表，默认全部5个 ["gcnsi", "hfsd", "ivgd", "mpnn", "rdgin"]
        datasets: 数据集列表，默认 ["karate", "jazz", "net_science", "cora_ml"]
        model_type: 传播模型类型，默认 "SIR"
        mask_ratios: mask 比例列表
        seeds: 随机种子列表
        epochs: 训练轮数
        device: 计算设备
        save_dir: 结果保存目录

    返回:
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
    print("多算法 Mask 鲁棒性分析实验 (对应论文 Fig. 2)")
    print("=" * 70)
    print(f"算法: {algorithms}")
    print(f"数据集: {datasets}")
    print(f"传播模型: {model_type}")
    print(f"Mask 比例: {mask_ratios}")
    print(f"Seeds: {seeds if seeds else SEEDS}")
    print(f"Epochs: {epochs}")
    print("=" * 70)

    all_results = {}

    for alg in algorithms:
        print(f"\n{'='*70}")
        print(f"算法: {alg.upper()}")
        print(f"{'='*70}")
        all_results[alg] = {}

        for dataset in datasets:
            print(f"\n--- 数据集: {dataset} ---")
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

    # 保存汇总结果（所有算法合并到一个 CSV）
    _save_combined_results(
        all_results, algorithms, datasets, model_type, mask_ratios, save_dir
    )

    print("\n" + "=" * 70)
    print("所有 Mask 鲁棒性实验完成！")
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
    """将所有算法的 mask 鲁棒性结果保存为一个汇总 CSV。"""
    os.makedirs(save_dir, exist_ok=True)
    csv_path = f"{save_dir}/all_baselines_{model_type}_mask_summary.csv"

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        # 写入表头
        header = ["dataset", "mask_ratio"]
        for alg in algorithms:
            header += [f"{alg}_auc", f"{alg}_f1"]
        writer.writerow(header)

        # 按数据集和 mask 比例写入数据
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

    print(f"\n汇总结果已保存: {csv_path}")


# ===================== 单独运行某个算法的便捷入口 =====================


def run_gcnsi_mask(
    datasets: list = None,
    model_type: str = "SIR",
    mask_ratios: list = None,
    seeds: list = None,
    epochs: int = 100,
    device: str = "cuda:0",
    save_dir: str = "result/baseline_mask",
):
    """便捷函数：运行 GCNSI 的 mask 鲁棒性实验。"""
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
    """便捷函数：运行 HFSD 的 mask 鲁棒性实验。"""
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
    """便捷函数：运行 IVGD 的 mask 鲁棒性实验。"""
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
    """便捷函数：运行 MPNN 的 mask 鲁棒性实验。"""
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
    """便捷函数：运行 RSDGIN 的 mask 鲁棒性实验。"""
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


# ===================== 主入口 =====================

if __name__ == "__main__":
    # 示例：运行所有基线的 mask 鲁棒性实验
    run_all_baselines_mask_robustness(
        algorithms=["gcnsi", "hfsd", "ivgd", "mpnn", "rdgin"],
        datasets=["karate", "jazz", "net_science", "cora_ml"],
        model_type="SIR",
        mask_ratios=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        seeds=[0, 1, 2, 3, 4],
        epochs=100,
        device="cuda:3",
    )
