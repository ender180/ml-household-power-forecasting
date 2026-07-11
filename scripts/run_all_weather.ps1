param(
    [string]$TrainCsv = "data/raw/train_weather.csv",
    [string]$TestCsv = "data/raw/test_weather.csv",
    [string]$OutDir = "results_weather"
)

python -m ml_power_forecast.train `
    --train-csv $TrainCsv `
    --test-csv $TestCsv `
    --out-dir $OutDir `
    --models lstm transformer conv_transformer `
    --horizons 90 365 `
    --seeds 2026 2027 2028 2029 2030 `
    --epochs 80 `
    --batch-size 32 `
    --device auto
