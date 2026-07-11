from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import TARGET_COL, prepare_datasets
from .models import build_model


@dataclass
class RunResult:
    model: str
    horizon: int
    seed: int
    test_mse: float
    test_mae: float
    best_val_loss: float
    epochs_ran: int


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, device: torch.device) -> float:
    model.train()
    criterion = nn.MSELoss()
    total = 0.0
    count = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad(set_to_none=True)
        pred = model(x)
        loss = criterion(pred, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total += loss.item() * len(x)
        count += len(x)
    return total / max(count, 1)


@torch.no_grad()
def evaluate_scaled(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    criterion = nn.MSELoss(reduction="sum")
    total = 0.0
    count = 0
    preds = []
    trues = []
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        pred = model(x)
        total += criterion(pred, y).item()
        count += y.numel()
        preds.append(pred.cpu().numpy())
        trues.append(y.cpu().numpy())
    return total / max(count, 1), np.concatenate(preds, axis=0), np.concatenate(trues, axis=0)


def plot_prediction(pred: np.ndarray, true: np.ndarray, out_path: Path, title: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(12, 4.5))
    plt.plot(true, label="Ground Truth", linewidth=2)
    plt.plot(pred, label="Prediction", linewidth=2)
    plt.xlabel("Forecast day")
    plt.ylabel("Daily global active power")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def run_single(args: argparse.Namespace, model_name: str, horizon: int, seed: int) -> RunResult:
    set_seed(seed)
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    (
        train_ds,
        val_ds,
        test_ds,
        scaler,
        target_index,
        feature_cols,
        train_daily,
        test_daily,
    ) = prepare_datasets(args.train_csv, args.test_csv, args.input_len, horizon, args.val_ratio)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model_kwargs = {
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
    }
    if model_name != "lstm":
        model_kwargs = {
            "d_model": args.hidden_dim,
            "nhead": args.nhead,
            "num_layers": args.num_layers,
            "dim_feedforward": args.ffn_dim,
            "dropout": args.dropout,
        }
    model = build_model(model_name, input_dim=len(feature_cols), horizon=horizon, **model_kwargs).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=5, factor=0.5)

    out_dir = Path(args.out_dir) / model_name / f"horizon_{horizon}" / f"seed_{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "feature_columns.json").open("w", encoding="utf-8") as f:
        json.dump(feature_cols, f, ensure_ascii=False, indent=2)
    train_daily.to_csv(out_dir / "train_daily.csv", index=False)
    test_daily.to_csv(out_dir / "test_daily.csv", index=False)

    best_val = float("inf")
    best_state = None
    bad_epochs = 0
    history = []
    epochs_iter = range(1, args.epochs + 1)
    if not args.quiet:
        epochs_iter = tqdm(epochs_iter, desc=f"{model_name}-h{horizon}-s{seed}")

    epochs_ran = 0
    for epoch in epochs_iter:
        epochs_ran = epoch
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss, _, _ = evaluate_scaled(model, val_loader, device)
        scheduler.step(val_loss)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
        if bad_epochs >= args.patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save(model.state_dict(), out_dir / "best_model.pth")
    pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)

    _, pred_scaled, true_scaled = evaluate_scaled(model, test_loader, device)
    pred = scaler.inverse_transform_target(pred_scaled, target_index)
    true = scaler.inverse_transform_target(true_scaled, target_index)
    mse = float(np.mean((pred - true) ** 2))
    mae = float(np.mean(np.abs(pred - true)))

    first_pred = pred[0]
    first_true = true[0]
    pd.DataFrame({"day": np.arange(1, horizon + 1), "prediction": first_pred, "ground_truth": first_true}).to_csv(
        out_dir / "first_window_prediction.csv", index=False
    )
    plot_prediction(
        first_pred,
        first_true,
        out_dir / "prediction_vs_ground_truth.png",
        f"{model_name} horizon={horizon} seed={seed}",
    )

    result = RunResult(model_name, horizon, seed, mse, mae, best_val, epochs_ran)
    with (out_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(asdict(result), f, ensure_ascii=False, indent=2)
    return result


def summarize(results: list[RunResult], out_dir: Path) -> None:
    df = pd.DataFrame([asdict(r) for r in results])
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "all_runs.csv", index=False)
    summary = (
        df.groupby(["model", "horizon"])
        .agg(
            mse_mean=("test_mse", "mean"),
            mse_std=("test_mse", "std"),
            mae_mean=("test_mae", "mean"),
            mae_std=("test_mae", "std"),
            val_loss_mean=("best_val_loss", "mean"),
            runs=("seed", "count"),
        )
        .reset_index()
    )
    summary.to_csv(out_dir / "summary.csv", index=False)
    with (out_dir / "summary.md").open("w", encoding="utf-8") as f:
        f.write(summary.to_markdown(index=False))
        f.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Household power forecasting experiments.")
    parser.add_argument("--train-csv", default="data/raw/train.csv")
    parser.add_argument("--test-csv", default="data/raw/test.csv")
    parser.add_argument("--out-dir", default="results")
    parser.add_argument("--models", nargs="+", default=["lstm", "transformer", "conv_transformer"])
    parser.add_argument("--horizons", nargs="+", type=int, default=[90, 365])
    parser.add_argument("--seeds", nargs="+", type=int, default=[2026, 2027, 2028, 2029, 2030])
    parser.add_argument("--input-len", type=int, default=90)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--ffn-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = []
    for model_name in args.models:
        for horizon in args.horizons:
            for seed in args.seeds:
                results.append(run_single(args, model_name, horizon, seed))
                summarize(results, Path(args.out_dir))
    summarize(results, Path(args.out_dir))


if __name__ == "__main__":
    main()
