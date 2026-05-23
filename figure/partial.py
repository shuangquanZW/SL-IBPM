import pandas as pd
import matplotlib.pyplot as plt
from util_figure import set_global_style, get_style_scheme, save_figure


def plot_partial():
    set_global_style()

    # === 读取 baselines 数据 ===
    baselines = pd.read_csv(
        "../result/baseline_mask/all_baselines_SIR_mask_summary.csv"
    )

    # === 读取 SL-IBPM 各数据集 robustness 数据 ===
    slibpm_files = {
        "karate": "../result/mask_robustness/karate_SIR_mask_robustness.csv",
        "jazz": "../result/mask_robustness/jazz_SIR_mask_robustness.csv",
        "net_science": "../result/mask_robustness/net_science_SIR_mask_robustness.csv",
        "cora_ml": "../result/mask_robustness/cora_ml_SIR_mask_robustness.csv",
    }
    slibpm_data = {k: pd.read_csv(v) for k, v in slibpm_files.items()}

    mask = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    methods = ["SL-IBPM", "GCNSI", "MPNN", "IVGD", "HFSD", "RSDGIN"]
    datasets_key = ["karate", "jazz", "net_science", "cora_ml"]
    datasets = ["Karate", "Jazz", "NetScience", "CoraML"]

    # 方法 -> baselines csv 中的列前缀
    baseline_prefix = {
        "GCNSI": "gcnsi",
        "MPNN": "mpnn",
        "IVGD": "ivgd",
        "HFSD": "hfsd",
        "RSDGIN": "rdgin",
    }

    def get_method_f1(method, dkey):
        """返回 f1_array 长度=len(mask)"""
        if method == "SL-IBPM":
            df = slibpm_data[dkey].sort_values("mask_ratio")
            return df["f1_mean"].to_numpy()
        else:
            sub = baselines[baselines["dataset"] == dkey].sort_values("mask_ratio")
            p = baseline_prefix[method]
            return sub[f"{p}_f1"].to_numpy()

    # === 样式 ===
    palette, markers, _ = get_style_scheme(len(methods), "line")
    styles = {
        method: {
            "color": palette[i],
            "marker": markers[i],
            "linewidth": 2.5,
            "markersize": 8,
            "markeredgewidth": 1.5,
            "markeredgecolor": "white",
        }
        for i, method in enumerate(methods)
    }

    fig, axes = plt.subplots(1, 4, figsize=(24, 4))

    handles_collected, labels_collected = [], []

    for idx, ax in enumerate(axes):
        dkey = datasets_key[idx]

        for method in methods:
            f1_vals = get_method_f1(method, dkey)

            (l1,) = ax.plot(
                mask,
                f1_vals,
                linestyle="-",
                **styles[method],
            )

            if idx == 0:
                handles_collected.append(l1)
                labels_collected.append(method)

        ax.set_title(f"{datasets[idx]}", fontweight="bold")
        ax.set_xlabel("Mask Ratio", fontweight="bold")
        ax.set_ylabel("F1", fontweight="bold")
        ax.set_ylim(0.0, 0.5)
        ax.grid(alpha=0.3, linestyle="--")
        ax.minorticks_on()

    # 统一图例（仅方法）
    fig.legend(
        handles_collected,
        labels_collected,
        loc="lower center",
        ncol=6,
        frameon=True,
        framealpha=0.9,
        edgecolor="lightgray",
        bbox_to_anchor=(0.5, -0.2),
    )

    save_figure("partial.pdf")


if __name__ == "__main__":
    plot_partial()
