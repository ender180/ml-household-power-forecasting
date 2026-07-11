from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


TARGET_COL = "global_active_power"


@dataclass
class Standardizer:
    mean_: np.ndarray | None = None
    std_: np.ndarray | None = None

    def fit(self, values: np.ndarray) -> "Standardizer":
        self.mean_ = np.nanmean(values, axis=0)
        self.std_ = np.nanstd(values, axis=0)
        self.std_ = np.where(self.std_ < 1e-8, 1.0, self.std_)
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("Standardizer must be fitted before transform.")
        return (values - self.mean_) / self.std_

    def inverse_transform_target(self, values: np.ndarray, target_index: int) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("Standardizer must be fitted before inverse transform.")
        return values * self.std_[target_index] + self.mean_[target_index]


class WindowDataset(Dataset):
    def __init__(
        self,
        features: np.ndarray,
        target: np.ndarray,
        input_len: int,
        horizon: int,
        start_indices: Iterable[int] | None = None,
    ) -> None:
        self.features = features.astype(np.float32)
        self.target = target.astype(np.float32)
        self.input_len = input_len
        self.horizon = horizon
        max_start = len(self.features) - input_len - horizon + 1
        if max_start <= 0:
            raise ValueError(
                f"Not enough daily rows ({len(self.features)}) for input_len={input_len}, "
                f"horizon={horizon}."
            )
        self.indices = list(range(max_start)) if start_indices is None else list(start_indices)
        if not self.indices:
            raise ValueError("No window samples were created.")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = self.indices[item]
        x_end = start + self.input_len
        y_end = x_end + self.horizon
        x = self.features[start:x_end]
        y = self.target[x_end:y_end]
        return torch.from_numpy(x), torch.from_numpy(y)


def read_raw_csv(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    df = pd.read_csv(
        path,
        sep=None,
        engine="python",
        na_values=["?", "NA", "N/A", "nan", "NaN", ""],
    )
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df


def _build_datetime(df: pd.DataFrame) -> pd.Series:
    if "datetime" in df.columns:
        return pd.to_datetime(df["datetime"], errors="coerce", dayfirst=True)
    if "date" in df.columns and "time" in df.columns:
        return pd.to_datetime(
            df["date"].astype(str) + " " + df["time"].astype(str),
            errors="coerce",
            dayfirst=True,
        )
    if "date" in df.columns:
        return pd.to_datetime(df["date"], errors="coerce", dayfirst=True)
    raise ValueError("CSV must contain Date/Time, date/time, or datetime columns.")


def aggregate_daily(path: str | Path) -> pd.DataFrame:
    df = read_raw_csv(path)
    df["datetime"] = _build_datetime(df)
    df = df.dropna(subset=["datetime"]).copy()

    for col in df.columns:
        if col not in {"date", "time", "datetime"}:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    meter_cols = ["sub_metering_1", "sub_metering_2", "sub_metering_3"]
    if TARGET_COL in df.columns and all(col in df.columns for col in meter_cols):
        df["sub_metering_remainder"] = (
            df[TARGET_COL] * 1000.0 / 60.0 - df[meter_cols].sum(axis=1)
        )

    df = df.set_index("datetime").sort_index()
    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns]

    sum_cols = [
        c
        for c in [
            TARGET_COL,
            "global_reactive_power",
            "sub_metering_1",
            "sub_metering_2",
            "sub_metering_3",
            "sub_metering_remainder",
        ]
        if c in numeric_cols
    ]
    mean_cols = [c for c in ["voltage", "global_intensity"] if c in numeric_cols]
    weather_cols = [
        c
        for c in ["rr", "nbjrr1", "nbjrr5", "nbjrr10", "nbjbrou"]
        if c in numeric_cols
    ]
    other_cols = [c for c in numeric_cols if c not in set(sum_cols + mean_cols + weather_cols)]

    daily_parts: list[pd.DataFrame] = []
    if sum_cols:
        daily_parts.append(df[sum_cols].resample("D").sum(min_count=1))
    if mean_cols:
        daily_parts.append(df[mean_cols].resample("D").mean())
    if weather_cols:
        daily_parts.append(df[weather_cols].resample("D").first())
    if other_cols:
        daily_parts.append(df[other_cols].resample("D").mean())

    if not daily_parts:
        raise ValueError("No numeric columns found after reading the CSV.")

    daily = pd.concat(daily_parts, axis=1)
    daily.index.name = "date"
    daily = daily.sort_index()
    daily = daily.interpolate(method="time", limit_direction="both").ffill().bfill()
    daily = daily.dropna(axis=1, how="all")
    daily = daily.fillna(daily.median(numeric_only=True))

    if TARGET_COL not in daily.columns:
        raise ValueError(f"Required target column {TARGET_COL!r} was not found.")
    return daily.reset_index()


def chronological_split(
    daily: pd.DataFrame,
    val_ratio: float = 0.2,
    min_val_len: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    val_len = max(int(len(daily) * val_ratio), min_val_len)
    val_len = min(val_len, len(daily) - 1)
    split = len(daily) - val_len
    split = max(1, min(split, len(daily) - 1))
    return daily.iloc[:split].copy(), daily.iloc[split:].copy()


def make_feature_frame(train_daily: pd.DataFrame, other_daily: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    feature_cols = [c for c in train_daily.columns if c != "date"]
    common_cols = [c for c in feature_cols if c in other_daily.columns]
    if TARGET_COL not in common_cols:
        raise ValueError("Target column must exist in both train and test data.")
    return train_daily[["date", *common_cols]].copy(), other_daily[["date", *common_cols]].copy(), common_cols


def build_scaled_arrays(
    fit_daily: pd.DataFrame,
    transform_daily: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[np.ndarray, np.ndarray, Standardizer, int]:
    scaler = Standardizer().fit(fit_daily[feature_cols].to_numpy(dtype=np.float32))
    values = scaler.transform(transform_daily[feature_cols].to_numpy(dtype=np.float32))
    target_index = feature_cols.index(TARGET_COL)
    target = values[:, target_index]
    return values, target, scaler, target_index


def prepare_datasets(
    train_csv: str | Path,
    test_csv: str | Path,
    input_len: int,
    horizon: int,
    val_ratio: float = 0.2,
) -> tuple[WindowDataset, WindowDataset, WindowDataset, Standardizer, int, list[str], pd.DataFrame, pd.DataFrame]:
    train_daily_raw = aggregate_daily(train_csv)
    test_daily_raw = aggregate_daily(test_csv)
    train_daily_raw, test_daily_raw, feature_cols = make_feature_frame(train_daily_raw, test_daily_raw)
    train_part, val_part = chronological_split(train_daily_raw, val_ratio, min_val_len=horizon)

    train_values, train_target, scaler, target_index = build_scaled_arrays(train_part, train_part, feature_cols)
    train_ds = WindowDataset(train_values, train_target, input_len, horizon)

    val_context = pd.concat([train_part.tail(input_len), val_part], ignore_index=True)
    val_values = scaler.transform(val_context[feature_cols].to_numpy(dtype=np.float32))
    val_target = val_values[:, target_index]
    val_ds = WindowDataset(val_values, val_target, input_len, horizon, start_indices=range(max(1, len(val_context) - input_len - horizon + 1)))

    test_context = pd.concat([train_daily_raw.tail(input_len), test_daily_raw], ignore_index=True)
    test_values = scaler.transform(test_context[feature_cols].to_numpy(dtype=np.float32))
    test_target = test_values[:, target_index]
    test_start_count = len(test_context) - input_len - horizon + 1
    test_ds = WindowDataset(test_values, test_target, input_len, horizon, start_indices=range(test_start_count))
    return train_ds, val_ds, test_ds, scaler, target_index, feature_cols, train_daily_raw, test_daily_raw
