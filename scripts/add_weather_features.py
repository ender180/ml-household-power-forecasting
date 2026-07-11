from __future__ import annotations

import argparse
import gzip
import math
import shutil
import urllib.request
from pathlib import Path

import pandas as pd


DEFAULT_WEATHER_URL = (
    "https://object.files.data.gouv.fr/meteofrance/data/synchro_ftp/BASE/MENS/"
    "MENSQ_92_previous-1950-2024.csv.gz"
)
DEFAULT_WEATHER_FILE = "data/raw/MENSQ_92_previous-1950-2024.csv.gz"
DEFAULT_DEPARTMENTS = ["75", "78", "91", "92", "94"]
UCI_SCEAUX_LAT = 48.778
UCI_SCEAUX_LON = 2.290
WEATHER_COLUMNS = ["RR", "NBJRR1", "NBJRR5", "NBJRR10", "NBJBROU"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge Météo-France monthly weather features into train/test CSV files.")
    parser.add_argument("--train-csv", default="data/raw/train.csv")
    parser.add_argument("--test-csv", default="data/raw/test.csv")
    parser.add_argument("--weather-file", default=DEFAULT_WEATHER_FILE)
    parser.add_argument("--weather-url", default=DEFAULT_WEATHER_URL)
    parser.add_argument(
        "--departments",
        nargs="+",
        default=DEFAULT_DEPARTMENTS,
        help="French department weather files to try when --weather-file is not explicitly used.",
    )
    parser.add_argument("--out-train", default="data/raw/train_weather.csv")
    parser.add_argument("--out-test", default="data/raw/test_weather.csv")
    parser.add_argument("--lat", type=float, default=UCI_SCEAUX_LAT)
    parser.add_argument("--lon", type=float, default=UCI_SCEAUX_LON)
    parser.add_argument("--force-download", action="store_true")
    return parser.parse_args()


def ensure_weather_file(path: Path, url: str, force: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not force:
        return
    tmp = path.with_suffix(path.suffix + ".download")
    print(f"Downloading weather data from {url}")
    with urllib.request.urlopen(url, timeout=120) as response, tmp.open("wb") as f:
        shutil.copyfileobj(response, f)
    tmp.replace(path)


def read_weather(path: Path) -> pd.DataFrame:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
            return pd.read_csv(f, sep=";", low_memory=False)
    return pd.read_csv(path, sep=None, engine="python")


def default_department_url(department: str) -> str:
    return (
        "https://object.files.data.gouv.fr/meteofrance/data/synchro_ftp/BASE/MENS/"
        f"MENSQ_{department}_previous-1950-2024.csv.gz"
    )


def load_department_weather(departments: list[str], base_dir: Path, force: bool) -> pd.DataFrame:
    frames = []
    for department in departments:
        path = base_dir / f"MENSQ_{department}_previous-1950-2024.csv.gz"
        ensure_weather_file(path, default_department_url(department), force)
        frame = read_weather(path)
        frame["_department"] = department
        frames.append(frame)
    return pd.concat(frames, ignore_index=True, sort=False)


def month_range_from_power(train_csv: Path, test_csv: Path) -> tuple[int, int]:
    months = []
    for csv_path in [train_csv, test_csv]:
        dates = pd.read_csv(csv_path, usecols=["Date"], dtype=str)
        dt = pd.to_datetime(dates["Date"], dayfirst=True, errors="coerce")
        months.append(dt.dt.strftime("%Y%m").astype(float))
    all_months = pd.concat(months).dropna().astype(int)
    return int(all_months.min()), int(all_months.max())


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lam = math.radians(lon2 - lon1)
    a = math.sin(d_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lam / 2) ** 2
    return 2 * radius * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def select_station(weather: pd.DataFrame, lat: float, lon: float, start_yyyymm: int, end_yyyymm: int) -> tuple[str, str]:
    needed = {"NUM_POSTE", "NOM_USUEL", "LAT", "LON", "AAAAMM", *WEATHER_COLUMNS}
    missing = needed.difference(weather.columns)
    if missing:
        raise ValueError(f"Weather file is missing required columns: {sorted(missing)}")

    weather = weather.copy()
    weather["AAAAMM"] = pd.to_numeric(weather["AAAAMM"], errors="coerce")
    span = weather[(weather["AAAAMM"] >= start_yyyymm) & (weather["AAAAMM"] <= end_yyyymm)]
    candidates = []
    for station, grp in span.groupby("NUM_POSTE"):
        available_months = grp["AAAAMM"].nunique()
        weather_non_null = grp[WEATHER_COLUMNS].notna().sum().sum()
        first = grp.iloc[0]
        distance = haversine_km(lat, lon, float(first["LAT"]), float(first["LON"]))
        fog_non_null = grp["NBJBROU"].notna().sum()
        if available_months > 0 and weather_non_null > 0:
            candidates.append(
                (
                    -available_months,
                    -weather_non_null,
                    -fog_non_null,
                    distance,
                    str(station),
                    str(first["NOM_USUEL"]),
                )
            )
    if not candidates:
        raise ValueError("No weather station has usable weather values for the requested date range.")
    candidates.sort()
    _, _, _, _, station, name = candidates[0]
    return station, name


def build_monthly_weather(weather: pd.DataFrame, station: str, start_yyyymm: int, end_yyyymm: int) -> pd.DataFrame:
    cols = ["NUM_POSTE", "NOM_USUEL", "LAT", "LON", "ALTI", "AAAAMM", *WEATHER_COLUMNS]
    selected = weather.loc[weather["NUM_POSTE"].astype(str) == station, [c for c in cols if c in weather.columns]].copy()
    selected["AAAAMM"] = pd.to_numeric(selected["AAAAMM"], errors="coerce").astype("Int64")
    selected = selected[(selected["AAAAMM"] >= start_yyyymm) & (selected["AAAAMM"] <= end_yyyymm)]
    selected = selected.sort_values("AAAAMM").drop_duplicates("AAAAMM", keep="first")
    for col in WEATHER_COLUMNS:
        selected[col] = pd.to_numeric(selected[col], errors="coerce")
    selected["RR"] = selected["RR"] / 10.0
    selected = selected.rename(columns={col: col.upper() for col in WEATHER_COLUMNS})
    return selected


def merge_weather(power_csv: Path, monthly_weather: pd.DataFrame, out_csv: Path) -> None:
    power = pd.read_csv(power_csv, low_memory=False)
    dt = pd.to_datetime(power["Date"].astype(str) + " " + power["Time"].astype(str), dayfirst=True, errors="coerce")
    power["AAAAMM"] = dt.dt.strftime("%Y%m").astype("Int64")
    merged = power.merge(monthly_weather[["AAAAMM", *WEATHER_COLUMNS]], on="AAAAMM", how="left")
    merged = merged.drop(columns=["AAAAMM"])
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_csv, index=False)
    missing_rate = merged[WEATHER_COLUMNS].isna().mean().mean()
    print(f"{power_csv} -> {out_csv} (weather missing rate: {missing_rate:.2%})")


def main() -> None:
    args = parse_args()
    train_csv = Path(args.train_csv)
    test_csv = Path(args.test_csv)
    weather_file = Path(args.weather_file)
    if args.weather_file != DEFAULT_WEATHER_FILE or args.weather_url != DEFAULT_WEATHER_URL:
        ensure_weather_file(weather_file, args.weather_url, args.force_download)
        weather = read_weather(weather_file)
    else:
        weather = load_department_weather(args.departments, weather_file.parent, args.force_download)
    start_yyyymm, end_yyyymm = month_range_from_power(train_csv, test_csv)
    station, name = select_station(weather, args.lat, args.lon, start_yyyymm, end_yyyymm)
    monthly_weather = build_monthly_weather(weather, station, start_yyyymm, end_yyyymm)
    print(f"Selected weather station: {station} {name}")
    print(f"Weather months: {monthly_weather['AAAAMM'].min()} - {monthly_weather['AAAAMM'].max()}")
    merge_weather(train_csv, monthly_weather, Path(args.out_train))
    merge_weather(test_csv, monthly_weather, Path(args.out_test))


if __name__ == "__main__":
    main()
