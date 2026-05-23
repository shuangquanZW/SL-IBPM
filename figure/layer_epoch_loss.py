import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from util_figure import set_global_style, get_style_scheme, save_figure


def plot_layer_epoch_auroc():
    set_global_style()

    # ==================== 数据读取配置 ====================
    datasets = ["Karate", "Jazz", "NetScience", "CoraML", "PowerGrid", "LastFM"]
    file_prefixes = ["karate", "jazz", "net_science", "cora_ml", "power_grid", "lastFM"]
    data_dir = "../result/layer_sensitivity"  # <-- 根据你实际存放 CSV 的路径修改

    # -------------------- Layer 实验数据 --------------------
    auroc_layer = []
    for prefix in file_prefixes:
        df = pd.read_csv(f"{data_dir}/{prefix}_SIR_layer_sensitivity.csv")
        auroc_layer.append(df["test_auc"].values)

    # -------------------- Epoch 实验 + Loss 数据 --------------------
    auroc_epoch = []
    loss_data = {}
    epoch_points = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

    for prefix, name in zip(file_prefixes, datasets):
        df = pd.read_csv(f"../result/history_aggregated/{prefix}_SIR.csv")

        # Epoch 实验：采样指定 epoch 的 test_auc_mean
        sampled = df[df["epoch"].isin(epoch_points)]["test_auc_mean"].values
        auroc_epoch.append(sampled)

        # Loss：保留全部 100 个 epoch 的均值（如需和旧版完全一致只画 10 个点，
        # 可改为 df[df["epoch"].isin(range(10,101,10))]...）
        loss_data[name] = (
            df["train_loss_mean"].values,
            df["valid_loss_mean"].values,
        )

    # ==================== 坐标轴数据 ====================
    layers = [1, 2, 3, 4, 5, 6, 7, 8]  # 新版只有 8 层
    epochs = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    all_epochs = np.arange(1, 101)  # Loss 图使用全部 epoch

    # ==================== 绘图 ====================
    _, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(24, 4))
    palette, markers, _ = get_style_scheme(6, "line")

    # ---------- (a) Layer Experiment ----------
    for i in range(6):
        ax1.plot(
            layers,
            auroc_layer[i],
            color=palette[i],
            marker=markers[i],
            linewidth=2.5,
            markersize=8,
            markeredgewidth=1.5,
            markeredgecolor="white",
            label=datasets[i],
        )
    ax1.set_xlabel("Layer", fontweight="bold")
    ax1.set_ylabel("AUROC", fontweight="bold")
    ax1.set_title("(a) Layer Experiment", fontweight="bold")
    ax1.set_ylim(0.8, 1.0)
    ax1.grid(alpha=0.3, linestyle="--")

    # ---------- (b) Epoch Experiment ----------
    for i in range(6):
        ax2.plot(
            epochs,
            auroc_epoch[i],
            color=palette[i],
            marker=markers[i],
            linewidth=2.5,
            markersize=8,
            markeredgewidth=1.5,
            markeredgecolor="white",
            label=datasets[i],
        )
    ax2.set_xlabel("Epoch", fontweight="bold")
    ax2.set_ylabel("AUROC", fontweight="bold")
    ax2.set_title("(b) Epoch Experiment", fontweight="bold")
    ax2.set_ylim(0.8, 1.0)
    ax2.grid(alpha=0.3, linestyle="--")

    # ---------- (c)(d) Loss Curves ----------
    loss_palette, loss_markers, _ = get_style_scheme(len(datasets), "line")
    titles = ["(c) Training loss", "(d) Validation loss"]

    for j, dataset in enumerate(datasets):
        # Training loss
        ax3.plot(
            all_epochs,
            loss_data[dataset][0],
            label=dataset,
            color=loss_palette[j],
            marker=loss_markers[j],
            linewidth=2.5,
            markersize=8,  # 100 个点较密，适当缩小
            markeredgewidth=1.5,
            markeredgecolor="white",
            markevery=10,  # 每 10 个 epoch 显示一个 marker
        )
        ax3.set_title(f"{titles[0]}", fontweight="bold")
        ax3.grid(linestyle="--", alpha=0.3)
        ax3.set_xlabel("Epoch", fontweight="bold")
        ax3.set_ylabel("Loss", fontweight="bold")
        ax3.set_ylim(0.0, 2.0)

        # Validation loss
        ax4.plot(
            all_epochs,
            loss_data[dataset][1],
            label=dataset,
            color=loss_palette[j],
            marker=loss_markers[j],
            linewidth=2.5,
            markersize=5,
            markeredgewidth=1.5,
            markeredgecolor="white",
            markevery=10,
        )
        ax4.set_title(f"{titles[1]}", fontweight="bold")
        ax4.grid(linestyle="--", alpha=0.3)
        ax4.set_xlabel("Epoch", fontweight="bold")
        ax4.set_ylabel("Loss", fontweight="bold")
        ax4.set_ylim(0.0, 2.0)
        ax4.legend(loc="upper right", fontsize=14)

    save_figure("layer_epoch_loss.pdf")


if __name__ == "__main__":
    plot_layer_epoch_auroc()
