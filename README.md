# 家庭电力消耗多变量时间序列预测

本项目实现了一个面向家庭电力消耗数据的多变量时间序列预测实验流程。任务目标是利用过去 90 天的多变量用电与天气特征，预测未来每日 `global_active_power` 序列，并比较 LSTM、Transformer 与改进的多尺度卷积 Transformer 在不同预测长度下的表现。

## 项目内容

主要实验设置如下：

- 输入窗口：过去 90 天
- 预测目标：未来每日 `global_active_power`
- 预测长度：90 天与 365 天
- 对比模型：LSTM、Transformer、多尺度卷积 Transformer
- 重复实验：每个模型与预测长度组合运行 5 个随机种子
- 评价指标：MSE 与 MAE，报告均值和标准差
- 外部特征：可融合 Météo-France 月度天气特征

## 目录结构

```text
ml_power_forecast/
  data.py                  # 数据读取、日级聚合、窗口构造与标准化
  models.py                # LSTM、Transformer、多尺度卷积 Transformer
  train.py                 # 训练、评估与结果保存入口

scripts/
  prepare_from_uci_txt.py  # 将 UCI 原始 txt 切分为 train/test
  add_weather_features.py  # 下载并合并天气特征
  run_all.ps1              # 运行基础实验
  run_all_weather.ps1      # 运行天气增强实验

results_weather/
  summary.csv              # 汇总结果
  summary.md               # Markdown 汇总表
  all_runs.csv             # 所有随机种子的指标
  log.txt                  # 训练日志
```

## 环境配置

建议使用 Python 3.10 或以上版本。

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

如果需要使用 GPU，请根据本机 CUDA 版本安装对应的 PyTorch。安装 PyTorch 后，也可以单独安装其余依赖：

```powershell
pip install numpy pandas scikit-learn matplotlib tqdm
```

## 数据准备

如果已经有切分好的文件，将其放入：

```text
data/raw/train.csv
data/raw/test.csv
```

如果只有 UCI 原始文件 `household_power_consumption.txt`，可以先执行：

```powershell
python scripts\prepare_from_uci_txt.py data\raw\household_power_consumption.txt
```

如需加入天气特征，执行：

```powershell
python scripts\add_weather_features.py
```

该脚本会下载巴黎及周边省份的 Météo-France 月度气候数据，选择覆盖时间完整且字段可用的站点，并生成：

```text
data/raw/train_weather.csv
data/raw/test_weather.csv
```

## 运行实验

运行天气增强版本的完整实验：

```powershell
.\scripts\run_all_weather.ps1
```

也可以直接使用 Python 命令：

```powershell
python -m ml_power_forecast.train `
  --train-csv data/raw/train_weather.csv `
  --test-csv data/raw/test_weather.csv `
  --out-dir results_weather `
  --models lstm transformer conv_transformer `
  --horizons 90 365 `
  --seeds 2026 2027 2028 2029 2030 `
  --epochs 80 `
  --batch-size 32 `
  --device auto
```

快速调试时可以只运行一个模型和一个随机种子：

```powershell
python -m ml_power_forecast.train --models lstm --horizons 90 --seeds 1 --epochs 2 --batch-size 16 --device cpu
```

## 输出结果

训练完成后，主要结果保存在 `results_weather/`：

- `summary.csv`：不同模型和预测长度的均值/标准差汇总
- `summary.md`：Markdown 版本汇总表
- `all_runs.csv`：每个模型、预测长度和随机种子的完整指标
- `log.txt`：命令行训练日志

## 模型说明

本项目包含三类模型：

1. LSTM：使用循环结构编码 90 天历史窗口，并通过全连接预测头直接输出未来序列。
2. Transformer：使用位置编码和 Transformer Encoder 建模输入窗口内的全局时间依赖。
3. 多尺度卷积 Transformer：在 Transformer 前加入卷积核大小为 3、7、15 的一维卷积分支，用于提取不同时间尺度的局部模式，再通过 Transformer 建模全局依赖。

## 说明

本仓库主要用于保存实验代码、运行脚本和核心汇总结果。原始数据、报告文件和大规模训练中间结果不包含在仓库中，需要在本地按上述步骤准备或生成。
