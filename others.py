"""
实验代码：包含 lambda 加权损失函数、Mask 鲁棒性实验和 Lambda 敏感性实验
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

# ===================== Lambda 加权损失函数 (论文公式 23) =====================


class LambdaWeightedLoss(nn.Module):
    """
    实现论文公式 (23) 的加权损失函数:
        L = sum_{i in S} L_i + lambda * sum_{j not in S} L_j

    其中 S 是源节点集合，lambda 是控制非源节点损失权重的超参数。
    当 lambda < 1.0 时，模型对非源节点的惩罚降低，从而更关注源节点的识别。
    """

    def __init__(self, lambda_: float = 0.1, reduction: str = "mean") -> None:
        super().__init__()
        self.lambda_ = lambda_
        self.reduction = reduction
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, pred: Tensor, true: Tensor) -> Tensor:
        """
        pred: [batch_size, num_nodes] 或 [batch_size, *] — 模型预测的对数几率
        true: [batch_size, num_nodes] 或 [batch_size, *] — 真实标签 (0 或 1)
        """
        # 计算每个元素的 BCE 损失（不减约）
        loss_per_element = self.bce(
            pred, true.float()
        )  # shape: [batch_size, num_nodes]

        # 创建权重矩阵：源节点权重为 1，非源节点权重为 lambda
        weights = torch.where(
            true > 0.5, torch.ones_like(true), torch.full_like(true, self.lambda_)
        )

        # 加权损失
        weighted_loss = loss_per_element * weights

        if self.reduction == "mean":
            return weighted_loss.mean()
        elif self.reduction == "sum":
            return weighted_loss.sum()
        else:
            return weighted_loss


# ===================== 支持 Lambda 的 Trainer 变体 =====================


class IBPMTrainerWithLambda(IBPMTrainer):
    """
    扩展 IBPMTrainer，使用 LambdaWeightedLoss 替代 BiasedEstimator
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
        # 不再调用父类的 __init__，而是直接复写
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.estimator = LambdaWeightedLoss(lambda_=lambda_, reduction=reduction)


# ===================== 保存工具函数 =====================


def save_mask_robustness_csv(
    mask_ratios: list,
    results_by_method: dict,
    save_path: str,
):
    """
    保存 Mask 鲁棒性实验结果。

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
    保存 Lambda 敏感性实验结果。

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


# ===================== Mask 鲁棒性分析实验 =====================


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
    n_mask_trials: int = 5,  # 新增：每个 seed 下对同一个 mask_ratio 做多少次独立 mask 实验
    save_dir: str = "result/mask_robustness",
):
    """
    Mask 鲁棒性分析实验（带多次随机 mask 平均）。

    逐渐 mask 不同比例的观测节点，测试 SL-IBPM 在有限观测数据下的性能。
    为消除“某次 mask 恰好保留关键节点”带来的偶然波动，对每个 (seed, mask_ratio)
    执行 n_mask_trials 次独立随机 mask 采样，先对 trial 平均，再跨 seed 平均。
    """
    if mask_ratios is None:
        mask_ratios = [i / 10 for i in range(10)]  # [0.0, 0.1, ..., 0.9]
    if seeds is None:
        seeds = SEEDS  # [0, 1, 2, 3, 4]

    # 加载数据（仅用于获取 num_nodes 等图结构信息，实际训练在每个 seed 内重新加载）
    (
        train_loader,
        valid_loader,
        test_loader,
        edge_index,
        num_nodes,
        num_edges,
        num_states,
    ) = load_data(name, type_, seed=seeds[0])

    auc_results_mean = []  # 跨 seed 平均后的 AUC
    auc_results_std = []  # 跨 seed 的 AUC 标准差
    f1_results_mean = []  # 跨 seed 平均后的 F1
    f1_results_std = []  # 跨 seed 的 F1 标准差

    for mask_ratio in mask_ratios:
        num_masked = int(num_nodes * mask_ratio)

        # 跨 seed 收集结果
        seed_aucs = []
        seed_f1s = []

        for seed in seeds:
            # 加载当前 seed 的数据划分
            (
                train_loader,
                valid_loader,
                test_loader,
                edge_index,
                num_nodes,
                num_edges,
                num_states,
            ) = load_data(name, type_, seed=seed)

            # 在当前 seed 下进行 n_mask_trials 次独立随机 mask
            trial_aucs = []
            trial_f1s = []

            for trial in range(n_mask_trials):
                # 使用与 seed 和 trial 都绑定的独立随机种子，确保可复现且不同 trial 的 mask 独立
                mask_rng_seed = seed * 10000 + trial
                rng = np.random.RandomState(mask_rng_seed)

                if num_masked > 0:
                    masked_nodes = sorted(
                        rng.choice(num_nodes, size=num_masked, replace=False).tolist()
                    )
                else:
                    masked_nodes = None

                # 重新固定 torch/np 全局种子，确保每个 trial 的模型初始化、训练随机性完全一致
                # 从而纯粹地分离“mask 采样”带来的方差
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

                # 训练
                trainer.fit(
                    train_loader,
                    valid_loader,
                    test_loader,
                    edge_index,
                    epochs=epochs,
                    mask=masked_nodes,
                )

                # 测试
                roc, precision, recall, f1 = trainer.evaluate(
                    test_loader, edge_index, mask=masked_nodes
                )
                trial_aucs.append(roc)
                trial_f1s.append(f1)

            # 对当前 seed 下的多次 mask trial 取平均，得到该 seed 在该 mask_ratio 下的稳定估计
            seed_auc_mean = float(np.mean(trial_aucs))
            seed_f1_mean = float(np.mean(trial_f1s))
            seed_aucs.append(seed_auc_mean)
            seed_f1s.append(seed_f1_mean)

        # 计算跨 seed 的统计量
        auc_mean, auc_std = compute_stats(seed_aucs)
        f1_mean, f1_std = compute_stats(seed_f1s)
        auc_results_mean.append(auc_mean)
        auc_results_std.append(auc_std)
        f1_results_mean.append(f1_mean)
        f1_results_std.append(f1_std)

        print(
            f"  [{name}-{type_}] Mask ratio {mask_ratio:.1f} "
            f"(trials={n_mask_trials}×{len(seeds)}): "
            f"AUC={auc_mean:.4f}±{auc_std:.4f}, F1={f1_mean:.4f}±{f1_std:.4f}"
        )

    # 保存结果（增加 std 列，便于观察波动是否被抑制）
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
    n_mask_trials: int = 5,  # 新增：传入 mask 鲁棒性实验的 trial 次数
    save_dir: str = "result/mask_robustness",
):
    """
    运行论文中 Fig. 2 对应的 Mask 鲁棒性实验。
    """
    if mask_ratios is None:
        mask_ratios = [i / 10 for i in range(10)]

    datasets = ["karate", "jazz", "net_science", "cora_ml"]
    model_type = "SIR"

    print("=" * 70)
    print("Mask 鲁棒性分析实验 (对应论文 Fig. 2，带多次随机 mask 平均)")
    print("=" * 70)

    for name in datasets:
        print(f"\n--- 数据集: {name} ---")
        mask_robustness_analysis(
            name=name,
            type_=model_type,
            lr=lr,
            lambda_=lambda_,
            weight_decay=weight_decay,
            device=device,
            mask_ratios=mask_ratios,
            epochs=epochs,
            n_mask_trials=n_mask_trials,  # 传入
            save_dir=save_dir,
        )

    print("\nMask 鲁棒性实验完成！")


# ===================== Lambda 参数敏感性分析实验 =====================


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
    Lambda 参数敏感性分析实验。

    测试不同 lambda 值对模型性能的影响，帮助选择最优的 lambda。

    参数:
        name: 数据集名称
        type_: 传播模型类型
        lambda_values: lambda 值列表，默认 [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
        epochs: 训练轮数
        seeds: 随机种子列表

    返回:
        lambda_values: lambda 值列表
        auc_results: 每个 lambda 对应的 AUC 列表 (跨 seed 均值)
        f1_results: 每个 lambda 对应的 F1 列表 (跨 seed 均值)
    """
    if lambda_values is None:
        lambda_values = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
    if seeds is None:
        seeds = SEEDS

    auc_results = []
    f1_results = []

    for lam in lambda_values:
        # 跨 seed 收集结果
        seed_aucs = []
        seed_f1s = []

        for seed in seeds:
            # 加载数据
            (
                train_loader,
                valid_loader,
                test_loader,
                edge_index,
                num_nodes,
                num_edges,
                num_states,
            ) = load_data(name, type_, seed=seed)

            # 设置模型和训练器
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

            # 训练
            trainer.fit(
                train_loader, valid_loader, test_loader, edge_index, epochs=epochs
            )

            # 测试
            roc, precision, recall, f1 = trainer.evaluate(test_loader, edge_index)
            seed_aucs.append(roc)
            seed_f1s.append(f1)

        # 计算跨 seed 均值
        auc_mean, auc_std = compute_stats(seed_aucs)
        f1_mean, f1_std = compute_stats(seed_f1s)
        auc_results.append(auc_mean)
        f1_results.append(f1_mean)

        print(
            f"  [{name}-{type_}] lambda={lam:.2f}: "
            f"AUC={auc_mean:.4f}±{auc_std:.4f}, F1={f1_mean:.4f}±{f1_std:.4f}"
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
    运行论文中 Fig. 4 对应的 Lambda 参数敏感性实验。

    在 6 个数据集上，测试不同 lambda 值对 F1 分数的影响。

    根据论文回复意见，建议的 lambda 搜索范围是 {0.05, 0.1, 0.2, 0.5}。
    """
    if lambda_values is None:
        # 论文 Fig. 4 使用 0.1, 0.2, ..., 1.0
        lambda_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    file_list = ["karate", "jazz", "net_science", "cora_ml", "power_grid", "lastFM"]
    type_list = ["SIR"]

    print("=" * 70)
    print("Lambda 参数敏感性分析实验 (对应论文 Fig. 4)")
    print("=" * 70)

    all_results = {}

    for name in file_list:
        for type_ in type_list:
            print(f"\n--- 数据集: {name}, 模型: {type_} ---")
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

    # 保存汇总结果
    save_lambda_sensitivity_csv(
        lambda_values,
        all_results,
        f"{save_dir}/lambda_sensitivity.csv",
    )

    print("\nLambda 敏感性实验完成！")
    return all_results


# ===================== 快速lambda搜索（用于新数据集）=====================


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
    对新的数据集进行 lambda 网格搜索，选择最优 lambda。

    根据论文回复意见的建议，在 {0.05, 0.1, 0.2, 0.5} 范围内搜索。
    由于模型收敛极快（通常 < 20 epochs），此搜索计算开销很低。

    参数:
        name: 数据集名称
        type_: 传播模型类型
        lambda_candidates: lambda 候选值列表，默认 [0.05, 0.1, 0.2, 0.5]
        epochs: 训练轮数（可以使用较少的 epochs 加速搜索）
        seed: 随机种子

    返回:
        best_lambda: 最优 lambda 值
        best_f1: 最优 F1 分数
        results: 所有候选 lambda 的结果
    """
    if lambda_candidates is None:
        lambda_candidates = [0.05, 0.1, 0.2, 0.5]

    print(f"Lambda 网格搜索: {name}-{type_}")
    print(f"候选值: {lambda_candidates}")

    # 加载数据
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

        # 训练
        trainer.fit(train_loader, valid_loader, test_loader, edge_index, epochs=epochs)

        # 在验证集上评估（使用 F1 作为主要指标）
        roc, precision, recall, f1 = trainer.evaluate(valid_loader, edge_index)
        results[lam] = {"auc": roc, "f1": f1}

        print(f"  lambda={lam:.2f}: Valid AUC={roc:.4f}, Valid F1={f1:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            best_lambda = lam

    print(f"\n最优 lambda: {best_lambda:.2f}, 对应 F1: {best_f1:.4f}")
    return best_lambda, best_f1, results


# ===================== 主入口 =====================

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
