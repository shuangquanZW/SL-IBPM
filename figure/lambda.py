import pandas as pd
import matplotlib.pyplot as plt
from util_figure import set_global_style, get_style_scheme, save_figure


def plot_lambda():
    set_global_style()
    plt.figure(figsize=(8, 5))

    # === Load data ===
    df = pd.read_csv("../result/lambda_sensitivity/lambda_sensitivity.csv")

    datasets_key = ["karate", "jazz", "net_science", "cora_ml", "power_grid", "lastFM"]
    datasets = ["Karate", "Jazz", "NetScience", "CoraML", "PowerGrid", "LastFM"]
    key2name = dict(zip(datasets_key, datasets))

    data_dict = {}
    ratio = None
    for dkey in datasets_key:
        sub = df[df["dataset"] == dkey].sort_values("lambda")
        if ratio is None:
            ratio = sub["lambda"].to_numpy()
        data_dict[key2name[dkey]] = sub["f1"].to_numpy()

    # === Style ===
    palette, markers, _ = get_style_scheme(len(datasets), "line")

    for j, dataset in enumerate(datasets):
        plt.plot(
            ratio,  # type: ignore
            data_dict[dataset],
            label=dataset,
            color=palette[j],
            marker=markers[j],
            linewidth=2.5,
            markersize=8,
            markeredgewidth=1.5,
            markeredgecolor="white",
        )

    plt.ylabel("F1", fontweight="bold")
    plt.xlabel("Lambda", fontweight="bold")
    plt.ylim(-0.05, 1.05)
    plt.grid(alpha=0.3, linestyle="--")
    plt.legend(fontsize=12, loc="upper center", ncol=3)

    save_figure("lambda.pdf")


if __name__ == "__main__":
    plot_lambda()
