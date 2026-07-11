# 家庭电力消耗多变量时间序列预测实验报告

## 1. 问题介绍

本项目面向家庭电力消耗预测任务，目标是利用过去 90 天的多变量用电与气象特征，预测未来 90 天和 365 天的每日总有功功率 `global_active_power`。该任务可辅助家庭用电规划、异常用电检测和智能电网负荷调度。

原始数据以分钟为粒度记录家庭用电情况。根据课程要求，实验前将数据聚合到天级别：`global_active_power`、`global_reactive_power` 与分表能耗取日总和，`voltage` 和 `global_intensity` 取日均值，气象变量取当天任意有效值。外部天气数据来自 Météo-France 在 data.gouv.fr 发布的月度基础气候数据，在巴黎及周边省份气象站中选择覆盖 2006-2010 年且课程指定字段完整的站点。本实验预处理脚本选取 `PARIS-MONTSOURIS` 站，将 `RR`、`NBJRR1`、`NBJRR5`、`NBJRR10`、`NBJBROU` 按月份合并到每一天。缺失值使用时间插值、前后向填充和中位数填充处理。

## 2. 模型

### 2.1 LSTM

LSTM 使用过去 90 天的多变量序列作为输入，通过循环门控结构建模时间依赖关系。模型取最后一层隐藏状态，经全连接预测头一次性输出未来 `H` 天曲线，其中 `H` 分别为 90 和 365。短期和长期预测分别训练独立模型。

### 2.2 Transformer

Transformer 将每日特征投影到隐空间，加入位置编码后送入多层 Transformer Encoder。模型通过自注意力机制捕获序列中不同日期之间的全局依赖，并使用最后一个时间步的表示预测未来曲线。

### 2.3 改进模型：多尺度卷积 Transformer

改进模型在 Transformer Encoder 前加入多尺度一维卷积分支，卷积核大小分别为 3、7、15，用于提取不同时间尺度上的局部波动、周周期趋势和更长跨度的平滑模式。卷积特征与线性投影特征拼接后送入 Transformer Encoder，再通过注意力池化得到序列表示并输出预测曲线。该结构的动机是先增强局部模式抽取能力，再利用自注意力建模长距离依赖。

## 3. 结果与分析

运行 `python -m ml_power_forecast.train ...` 后，将 `results/summary.md` 中的表格粘贴到这里。

| model | horizon | mse_mean | mse_std | mae_mean | mae_std |
|---|---:|---:|---:|---:|---:|
| 待填 | 90 | 待填 | 待填 | 待填 | 待填 |
| 待填 | 365 | 待填 | 待填 | 待填 | 待填 |

将以下图片插入报告：

- LSTM, horizon=90 的 `prediction_vs_ground_truth.png`
- LSTM, horizon=365 的 `prediction_vs_ground_truth.png`
- Transformer, horizon=90 的 `prediction_vs_ground_truth.png`
- Transformer, horizon=365 的 `prediction_vs_ground_truth.png`
- 改进模型, horizon=90 的 `prediction_vs_ground_truth.png`
- 改进模型, horizon=365 的 `prediction_vs_ground_truth.png`

从结果上比较三类模型的 MSE 与 MAE。一般来说，365 天预测比 90 天预测更难，因为误差会随预测步长增加而累积，长期季节性和生活行为变化也更难由 90 天上下文完全解释。若改进模型优于基础 Transformer，可说明多尺度卷积对局部用电模式有帮助；若性能不佳，可从模型复杂度、训练数据规模、长期预测不确定性和超参数敏感性方面分析原因。

## 4. 讨论

本实验的主要挑战包括分钟级数据缺失、天级聚合后的样本数量有限、长期预测对季节变化敏感，以及天气特征与用电行为之间可能存在滞后关系。后续可尝试加入月份、星期、节假日等日历特征，或使用分解式模型分别学习趋势项、周期项和残差项。

本报告文字整理过程中使用了 ChatGPT/Codex 辅助，但实验代码、结果和分析均需以实际运行结果为准。

## 参考文献

[1] UCI Machine Learning Repository. Individual household electric power consumption dataset.

[2] Météo-France. Données climatologiques de base - mensuelles. data.gouv.fr.

[3] Vaswani, A. et al. Attention Is All You Need. NeurIPS, 2017.

[4] Hochreiter, S., and Schmidhuber, J. Long Short-Term Memory. Neural Computation, 1997.
