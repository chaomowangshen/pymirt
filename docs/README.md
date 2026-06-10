# PyMIRT 文档

欢迎来到 PyMIRT 文档！

## 目录

- [快速开始](quickstart.md)
- [API 参考](api.md)
- [使用示例](examples.md)
- [理论背景](theory.md)

## 简介

PyMIRT 是一个用于项目反应理论（Item Response Theory, IRT）参数估计的 Python 包。

## 主要特性

- 支持单维和多维 IRT 模型，包括 Rasch/1PL、2PL、3PL、GRM、M2PL 和 MGRM
- 多种估计方法（EM 算法、MCMC）
- 可选 CEN-QB 神经网络估计（`method='nn'`，用于单维 1PL/Rasch 和 2PL）
- 可选稀疏计算后端（`use_sparse=True`）
- 对象式 API（`IRT` / `MIRT`）
- 灵活的配置选项
- 完整的文档和示例

## 安装

```bash
pip install pymirt
```

## 基本用法

```python
import pandas as pd
from pymirt import irt

# 加载数据
response_df = pd.read_csv('your_data.csv')

# 估计参数
a_est, b_est, theta_est = irt(response_df, model='2PL')

# Rasch/1PL 固定区分度为 1，支持 EM、MCMC、MCEM 和 SAEM
a_est, b_est, theta_est = irt(response_df, model='rasch', method='mcmc')

# 3PL 当前支持 EM/EAP，返回 (a, b, c, theta)
a_est, b_est, c_est, theta_est = irt(response_df, model='3pl')

# CEN-QB 神经网络后端需要安装 pymirt[nn]
a_est, b_est, theta_est = irt(
    response_df,
    model='2pl',
    method='nn',
    nn_config={'epochs': 200, 'random_state': 123}
)

# 单维多级计分神经估计：step 和 stand 均可用
n_categories = [4] * response_df.shape[1]
a_est, b_est, theta_est = irt(
    response_df,
    model='grm',
    grm_type='stand',
    method='nn',
    n_categories=n_categories,
)
```

也可以使用对象式 API 获取摘要和参数表：

```python
from pymirt import IRT

result = IRT(model='2pl', use_sparse=True).fit(response_df)

print(result.summary())
print(result.item_params().head())
print(result.person_params().head())
```

更多详细信息请参考具体的文档页面。
