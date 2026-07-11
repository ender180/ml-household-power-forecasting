from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split the UCI household power txt file into train/test CSV files.")
    parser.add_argument("input_txt", help="Path to household_power_consumption.txt")
    parser.add_argument("--out-dir", default="data/raw", help="Output directory.")
    parser.add_argument("--test-days", type=int, default=455, help="Days reserved for test. 455 = 90 context + 365 target.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input_txt, sep=";", na_values=["?"], low_memory=False)
    dt = pd.to_datetime(df["Date"].astype(str) + " " + df["Time"].astype(str), dayfirst=True, errors="coerce")
    df = df.loc[dt.notna()].copy()
    df["_datetime"] = dt[dt.notna()].values
    cutoff = df["_datetime"].max().normalize() - pd.Timedelta(days=args.test_days - 1)

    train = df.loc[df["_datetime"] < cutoff].drop(columns=["_datetime"])
    test = df.loc[df["_datetime"] >= cutoff].drop(columns=["_datetime"])

    train.to_csv(out_dir / "train.csv", index=False)
    test.to_csv(out_dir / "test.csv", index=False)
    print(f"train rows: {len(train):,} -> {out_dir / 'train.csv'}")
    print(f"test rows:  {len(test):,} -> {out_dir / 'test.csv'}")
    print(f"cutoff date: {cutoff.date()}")


if __name__ == "__main__":
    main()
