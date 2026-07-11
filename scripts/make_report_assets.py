from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


MODEL_LABELS = {
    "lstm": "LSTM",
    "transformer": "Transformer",
    "conv_transformer": "MSConv-Transformer",
}


def main() -> None:
    results_dir = Path("results_weather")
    out_dir = Path("reports/assets")
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = pd.read_csv(results_dir / "summary.csv")
    summary["model_label"] = summary["model"].map(MODEL_LABELS)

    for metric, ylabel, filename in [
        ("mse", "MSE", "mse_comparison.png"),
        ("mae", "MAE", "mae_comparison.png"),
    ]:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=False)
        for ax, horizon in zip(axes, [90, 365]):
            part = summary[summary["horizon"] == horizon].copy()
            part = part.set_index("model").loc[["lstm", "transformer", "conv_transformer"]].reset_index()
            ax.bar(
                part["model_label"],
                part[f"{metric}_mean"],
                yerr=part[f"{metric}_std"],
                capsize=5,
                color=["#4C78A8", "#59A14F", "#F28E2B"],
                alpha=0.88,
            )
            ax.set_title(f"Horizon = {horizon} days")
            ax.set_ylabel(ylabel)
            ax.grid(axis="y", linestyle="--", alpha=0.35)
            ax.tick_params(axis="x", rotation=18)
        fig.tight_layout()
        fig.savefig(out_dir / filename, dpi=220)
        plt.close(fig)

    all_runs = pd.read_csv(results_dir / "all_runs.csv")
    best = (
        all_runs.sort_values("test_mse")
        .groupby(["model", "horizon"], as_index=False)
        .first()[["model", "horizon", "seed", "test_mse", "test_mae"]]
    )
    best.to_csv(out_dir / "best_plot_seeds.csv", index=False)

    print("Wrote report assets to", out_dir)
    print(best.to_string(index=False))


if __name__ == "__main__":
    main()
