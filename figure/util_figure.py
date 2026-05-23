import matplotlib.pyplot as plt
import seaborn as sns


# 全局样式设置
def set_global_style():
    """设置全局绘图样式"""
    sns.set_style("whitegrid")
    sns.set_context("talk", font_scale=1)
    plt.rcParams["font.family"] = ["Times New Roman"]
    plt.rcParams["axes.titlesize"] = 24
    plt.rcParams["axes.labelsize"] = 20
    plt.rcParams["legend.fontsize"] = 20
    plt.rcParams["xtick.labelsize"] = 16
    plt.rcParams["ytick.labelsize"] = 16
    plt.rcParams["xtick.direction"] = "in"
    plt.rcParams["ytick.direction"] = "in"
    plt.rcParams["xtick.major.size"] = 6
    plt.rcParams["ytick.major.size"] = 6
    plt.rcParams["xtick.minor.size"] = 3
    plt.rcParams["ytick.minor.size"] = 3
    plt.rcParams["figure.dpi"] = 600


# 颜色和标记方案
def get_style_scheme(n_items, plot_type):
    """获取颜色和标记方案"""
    schemes = {
        "bar": {
            "palette": sns.color_palette("gray", n_items),
            "hatches": ["", "//", "xx", "..", "++", "**"],
        },
        "line": {
            "palette": sns.color_palette("tab10", n_items),
            "markers": ["o", "s", "^", "D", "v", "p", "<", ">", "X", "*"],
        },
        "multi": {
            "palette": sns.color_palette("flare", n_items),
            "markers": ["P", "1", "2", "3", "4", "d", "h", "|"],
        },
    }

    scheme = schemes.get(plot_type, schemes["line"])
    return scheme["palette"], scheme.get("markers", []), scheme.get("hatches", [])


# 保存图像
def save_figure(filename, bbox_inches="tight", format="pdf"):
    plt.savefig(
        filename,
        bbox_inches=bbox_inches,
        format=format,
    )
    plt.close()


# 创建图例
def create_legend(ax, ncol=1, loc="best", framealpha=0.9):
    """创建图例"""
    handles, labels = ax.get_legend_handles_labels()
    return plt.legend(
        handles,
        labels,
        loc=loc,
        ncol=ncol,
        frameon=True,
        framealpha=framealpha,
        edgecolor="lightgray",
    )


# 双Y轴折线图
def plot_twin_axis(
    data_left,
    data_right,
    x_values,
    labels,
    title,
    left_label,
    right_label,
    figsize=(8, 5),
):
    """绘制双Y轴折线图"""
    palette_left, markers_left, _ = get_style_scheme(1, "line")
    palette_right, markers_right, _ = get_style_scheme(1, "multi")

    fig, ax_left = plt.subplots(figsize=figsize)
    ax_right = ax_left.twinx()

    # 左侧轴
    ax_left.plot(
        x_values,
        data_left,
        color=palette_left[0],
        marker=markers_left[0],
        linewidth=2.5,
        markersize=8,
        markeredgewidth=1.5,
        markeredgecolor="white",
        label=left_label,
    )

    # 右侧轴
    ax_right.plot(
        x_values,
        data_right,
        color=palette_right[0],
        marker=markers_right[0],
        linewidth=2.5,
        markersize=8,
        markeredgewidth=1.5,
        markeredgecolor="white",
        label=right_label,
    )

    # 设置标签
    ax_left.set_xlabel(labels["x"], fontweight="bold")
    ax_left.set_ylabel(labels["left"], fontweight="bold", color=palette_left[0])
    ax_right.set_ylabel(labels["right"], fontweight="bold", color=palette_right[0])
    ax_left.set_title(title, fontweight="bold")

    # 设置刻度颜色
    ax_left.tick_params(axis="y", colors=palette_left[0])
    ax_right.tick_params(axis="y", colors=palette_right[0])

    # 添加图例
    lines = [ax_left.get_lines()[0], ax_right.get_lines()[0]]
    ax_left.legend(lines, [left_label, right_label])

    return fig, (ax_left, ax_right)
