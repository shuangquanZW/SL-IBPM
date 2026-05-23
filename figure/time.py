import pandas as pd
import matplotlib.pyplot as plt
from util_figure import set_global_style, get_style_scheme, save_figure


def plot_time():
    set_global_style()
    plt.figure(figsize=(8, 5))

    # Runtime data.
    df = pd.read_excel("./result/study_result.xlsx", sheet_name="Time")
    gcnsi = df.iloc[0].to_numpy()[1:]
    mpnn = df.iloc[1].to_numpy()[1:]
    slvae = df.iloc[2].to_numpy()[1:]
    ivgd = df.iloc[3].to_numpy()[1:]
    hfsd = df.iloc[4].to_numpy()[1:]
    rsdgin = df.iloc[5].to_numpy()[1:]
    slibpm = df.iloc[6].to_numpy()[1:]

    dataset_names = ["Karate", "Jazz", "NetScience", "CoraML", "PowerGrid", "LastFM"]
    methods = ["GCNSI", "MPNN", "SLVAE", "IVGD", "HFSD", "RSDGIN", "SL-IBPM"]

    runtime_palette, runtime_markers, _ = get_style_scheme(len(methods), "line")
    runtime_styles = {
        method: {
            "color": runtime_palette[i],
            "marker": runtime_markers[i],
            "linewidth": 2.5,
            "markersize": 8,
            "markeredgewidth": 1.5,
            "markeredgecolor": "white",
        }
        for i, method in enumerate(methods)
    }

    for method, data in zip(methods, [gcnsi, mpnn, slvae, ivgd, hfsd, rsdgin, slibpm]):
        plt.plot(dataset_names, data, label=method, **runtime_styles[method])

    # plt.title("Runime analysis", fontweight="bold")
    plt.ylabel("Runtime (s)", fontweight="bold")
    plt.grid(linestyle="--", alpha=0.3)
    plt.legend(fontsize=12)

    save_figure("figure/time.pdf")


if __name__ == "__main__":
    plot_time()
