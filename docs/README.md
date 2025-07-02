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

- 支持单维和多维 IRT 模型
- 多种估计方法（EM 算法、MCMC）
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
```

更多详细信息请参考具体的文档页面。
