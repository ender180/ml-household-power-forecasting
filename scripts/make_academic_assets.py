from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


RESULTS_DIR = Path("results_weather")
ASSET_DIR = Path("reports/academic_assets")
MODEL_ORDER = ["lstm", "transformer", "conv_transformer"]
MODEL_LABELS = {
    "lstm": "LSTM",
    "transformer": "Transformer",
    "conv_transformer": "MSConv-Transformer",
}


def setup_font() -> None:
    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def box(ax, xy, text, w=1.55, h=0.58, fontsize=10, fc="#F5F5F5") -> None:
    x, y = xy
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.03,rounding_size=0.04",
        linewidth=1.1,
        edgecolor="#222222",
        facecolor=fc,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize, color="#111111")


def arrow(ax, start, end) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=13,
            linewidth=1.1,
            color="#222222",
            shrinkA=4,
            shrinkB=4,
        )
    )


def save_diagram(fig, name: str) -> None:
    fig.tight_layout(pad=0.4)
    fig.savefig(ASSET_DIR / name, dpi=220, bbox_inches="tight")
    plt.close(fig)


def make_lstm_diagram() -> None:
    fig, ax = plt.subplots(figsize=(8, 1.85))
    ax.set_axis_off()
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 1.85)
    labels = ["90天多变量输入", "标准化\n特征序列", "两层 LSTM\n时序编码", "最后隐藏状态", "全连接预测头", "未来H天\n功率序列"]
    xs = [0.2, 1.55, 2.85, 4.1, 5.35, 6.65]
    for x, label in zip(xs, labels):
        box(ax, (x, 0.72), label, w=1.08, h=0.58, fontsize=9.5, fc="#E8F2FF")
    for x in xs[:-1]:
        arrow(ax, (x + 1.08, 1.01), (x + 1.34, 1.01))
    save_diagram(fig, "model_lstm.png")


def make_transformer_diagram() -> None:
    fig, ax = plt.subplots(figsize=(8, 1.95))
    ax.set_axis_off()
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 1.95)
    labels = ["90天多变量输入", "线性嵌入", "位置编码", "Transformer\nEncoder", "末时刻表示", "预测头", "未来H天输出"]
    xs = [0.15, 1.25, 2.25, 3.25, 4.55, 5.65, 6.65]
    widths = [0.95, 0.82, 0.82, 1.1, 0.9, 0.75, 0.95]
    for x, label, w in zip(xs, labels, widths):
        box(ax, (x, 0.78), label, w=w, h=0.58, fontsize=9.5, fc="#EAF7EA")
    for x, w, nx in zip(xs[:-1], widths[:-1], xs[1:]):
        arrow(ax, (x + w, 1.07), (nx, 1.07))
    save_diagram(fig, "model_transformer.png")


def make_msconv_diagram() -> None:
    fig, ax = plt.subplots(figsize=(8, 2.75))
    ax.set_axis_off()
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 2.75)
    box(ax, (0.25, 1.15), "90天多变量输入", w=1.1, h=0.58, fc="#FFF3D7")
    for y, label in [(2.05, "Conv1D\nk=3"), (1.15, "Conv1D\nk=7"), (0.25, "Conv1D\nk=15")]:
        box(ax, (1.85, y), label, w=0.95, h=0.54, fontsize=9.2, fc="#FCE4D6")
        arrow(ax, (1.35, 1.44), (1.85, y + 0.27))
    box(ax, (1.85, 1.70), "线性投影", w=0.95, h=0.34, fontsize=8.9, fc="#E8F2FF")
    arrow(ax, (1.35, 1.44), (1.85, 1.87))
    box(ax, (3.35, 1.15), "特征拼接\n与融合", w=1.02, h=0.58, fc="#EADCF8")
    for y in [2.32, 1.42, 0.52, 1.87]:
        arrow(ax, (2.8, y), (3.35, 1.44))
    box(ax, (4.75, 1.15), "Transformer\nEncoder", w=1.15, h=0.58, fc="#EAF7EA")
    arrow(ax, (4.37, 1.44), (4.75, 1.44))
    box(ax, (6.25, 1.15), "注意力池化\n预测头", w=1.05, h=0.58, fc="#E8F2FF")
    arrow(ax, (5.9, 1.44), (6.25, 1.44))
    box(ax, (6.35, 0.25), "未来H天输出", w=1.0, h=0.48, fontsize=9.2, fc="#FFF3D7")
    arrow(ax, (6.78, 1.15), (6.85, 0.73))
    save_diagram(fig, "model_msconv_transformer.png")


def make_metric_charts() -> None:
    summary = pd.read_csv(RESULTS_DIR / "summary.csv")
    summary["model_label"] = summary["model"].map(MODEL_LABELS)
    colors = ["#4C78A8", "#59A14F", "#F28E2B"]
    for metric, ylabel, filename in [
        ("mse", "MSE", "metric_mse.png"),
        ("mae", "MAE", "metric_mae.png"),
    ]:
        fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
        for ax, horizon in zip(axes, [90, 365]):
            part = summary[summary["horizon"] == horizon].set_index("model").loc[MODEL_ORDER].reset_index()
            ax.bar(
                part["model_label"],
                part[f"{metric}_mean"],
                yerr=part[f"{metric}_std"],
                capsize=4,
                color=colors,
                edgecolor="#222222",
                linewidth=0.8,
            )
            ax.set_title(f"预测长度 = {horizon} 天", color="#111111")
            ax.set_ylabel(ylabel)
            ax.grid(axis="y", linestyle="--", color="#DDDDDD", linewidth=0.8)
            ax.tick_params(axis="x", rotation=15)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
        fig.tight_layout()
        fig.savefig(ASSET_DIR / filename, dpi=220, bbox_inches="tight")
        plt.close(fig)


def make_prediction_plots() -> None:
    best = pd.read_csv(ASSET_DIR / "best_plot_seeds.csv")
    for _, row in best.iterrows():
        model = row["model"]
        horizon = int(row["horizon"])
        seed = int(row["seed"])
        source = (
            RESULTS_DIR
            / model
            / f"horizon_{horizon}"
            / f"seed_{seed}"
            / "first_window_prediction.csv"
        )
        df = pd.read_csv(source)
        fig, ax = plt.subplots(figsize=(8.2, 3.1))
        ax.plot(df["day"], df["ground_truth"], color="#1F77B4", linewidth=1.8, label="真实值")
        ax.plot(df["day"], df["prediction"], color="#FF7F0E", linewidth=1.8, linestyle="--", label="预测值")
        ax.set_xlabel("预测天数")
        ax.set_ylabel("每日总有功功率")
        ax.set_title(f"{MODEL_LABELS[model]}，预测长度 = {horizon} 天，seed = {seed}", color="#111111")
        ax.grid(axis="y", linestyle="--", color="#DDDDDD", linewidth=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(frameon=False, loc="upper left")
        fig.tight_layout()
        fig.savefig(ASSET_DIR / f"prediction_{model}_{horizon}.png", dpi=220, bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    setup_font()
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    make_lstm_diagram()
    make_transformer_diagram()
    make_msconv_diagram()
    make_metric_charts()

    all_runs = pd.read_csv(RESULTS_DIR / "all_runs.csv")
    best = (
        all_runs.sort_values("test_mse")
        .groupby(["model", "horizon"], as_index=False)
        .first()[["model", "horizon", "seed", "test_mse", "test_mae"]]
    )
    best.to_csv(ASSET_DIR / "best_plot_seeds.csv", index=False)
    make_prediction_plots()
    print(f"Wrote academic assets to {ASSET_DIR}")


if __name__ == "__main__":
    main()
