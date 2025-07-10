# PyMIRT

PyMIRT 是一个用于项目反应理论（Item Response Theory, IRT）参数估计的 Python 包。该包支持单维和多维 IRT 模型的参数估计。

[English](README.md) | 中文

## 特性

- **单维 IRT 模型**：支持 2PL 模型和等级反应模型（GRM）
- **多维 IRT 模型**：支持多维 2PL 模型（M2PL）和多维等级反应模型（MGRM）
- **多种估计方法**：支持 EM 算法、蒙特卡洛 EM（MCEM）方法、随机游走EM(SAEM)算法和马尔可夫链蒙特卡洛(MCMC)方法
- **能力估计**：支持期望后验估计（EAP）和马尔可夫链蒙特卡洛估计
- **缺失数据处理**：支持含有缺失数据的作答矩阵进行参数估计
- **灵活配置**：支持自定义求积点数、迭代次数、收敛容忍度等参数

## 安装

目前需要从源码安装：

```bash
git clone https://github.com/chaomowangshen/pymirt.git
cd pymirt
pip install -e .
```

或者直接从 GitHub 安装：

```bash
pip install git+https://github.com/chaomowangshen/pymirt.git
```

## 快速开始

### 单维 IRT 模型

```python
import pandas as pd
from pymirt import irt

# 加载数据（假设是二分数据）
response_df = pd.read_csv('your_response_data.csv')

# 使用 2PL 模型估计参数
a_est, b_est, theta_est = irt(
    response_df=response_df,
    model='2PL',
    n_quadrature=27,
    max_iter=100,
    tol=1e-4,
    verbose=True
)

print(f"区分度参数: {a_est}")
print(f"难度参数: {b_est}")
print(f"能力估计: {theta_est}")
```

### 多维 IRT 模型

```python
import numpy as np
import pandas as pd
from pymirt import mirt

# 加载数据和 Q 矩阵
response_df = pd.read_csv('your_response_data.csv')
Q = np.array([[1, 0], [1, 0], [0, 1], [0, 1]])  # 项目特征矩阵

# 使用多维 2PL 模型估计参数
a_est, b_est, theta_est = mirt(
    response_df=response_df,
    Q=Q,
    method='em',
    model='m2pl',
    n_quadrature=15,
    max_iter=100,
    tol=1e-4,
    verbose=True
)

print(f"区分度参数: {a_est}")
print(f"难度参数: {b_est}")
print(f"能力估计: {theta_est}")
```

## 支持的模型

### 单维模型
- **2PL**：二参数逻辑模型
- **GRM_**：等级反应模型

### 多维模型
- **M2PL**：多维二参数逻辑模型
- **MGRM**：多维等级反应模型

## 参数说明

### irt() 函数参数
- `response_df`: 被试作答矩阵（DataFrame）
- `model`: IRT 模型类型（'2pl', 'grm'）。
- `grm_type`: grm 估计类型（'step', 'stand'）。step为分步估计,stand为标准实现。
- `n_quadrature`: 高斯-厄米特求积点数
- `n_categories`: 项目类别数（用于等级反应模型）
- `max_iter`: 最大迭代次数
- `tol`: 收敛容忍度
- `verbose`: 是否打印详细信息

### mirt() 函数参数
- `response_df`: 被试作答矩阵（DataFrame）
- `Q`: 项目特征矩阵（numpy array）
- `method`: 估计方法（'em' 、 'mcem' 、 'saem' 、 'mcmc'）。注:em方法仅支持3维及以下
- `model`: 多维 IRT 模型类型（'m2pl' 或 'mgrm'）。
- `grm_type`: mgrm 估计类型（'step', 'stand'）。step为分步估计,stand为标准实现。
- `n_quadrature`: 求积点数（用于 EM 方法）
- `n_samples`: MCMC 样本数（用于 MCEM和MCMC 方法）
- `burn_in`: MCMC 预热期（用于 MCEM和MCMC 方法）
- `sample_interval`: MCMC 采样间隔（用于 MCEM 方法）
- `max_iter`: 最大迭代次数
- `tol`: 收敛容忍度
- `verbose`: 是否打印详细信息

## 理论背景

### 项目反应理论（IRT）
项目反应理论是现代测量理论的重要分支，广泛应用于教育测量、心理测量等领域。与经典测验理论相比，IRT 具有以下优势：

- **项目参数与被试能力分离**：项目难度和区分度不依赖于特定的被试群体
- **能力估计精确**：可以为每个被试提供个性化的能力估计和标准误
- **自适应测试支持**：为计算机自适应测试（CAT）提供理论基础
- **等值链接**：支持不同测验形式之间的分数等值

### 单维模型

#### 二参数逻辑模型（2PL）
$$P_{ij}(\theta) = \frac{1}{1 + \exp(-a_j(\theta_i - b_j))}$$

其中：
- $P_{ij}(\theta)$ 是被试 $i$ 在项目 $j$ 上答对的概率
- $a_j$ 是项目 $j$ 的区分度参数
- $b_j$ 是项目 $j$ 的难度参数
- $\theta_i$ 是被试 $i$ 的能力参数

#### 等级反应模型（GRM）
$$P_{ijk}^*(\theta) = \frac{\exp(a_j(\theta_i - b_{jk}))}{1 + \exp(a_j(\theta_i - b_{jk}))}$$

### 多维模型

#### 多维二参数逻辑模型（M2PL）
$$P_{ij}(\boldsymbol{\theta}) = \frac{1}{1 + \exp(-(\boldsymbol{a}_j^T \boldsymbol{\theta}_i + d_j))}$$

其中 $\boldsymbol{a}_j$ 是项目 $j$ 的多维区分度向量，$\boldsymbol{\theta}_i$ 是被试 $i$ 的多维能力向量，$d_j$ 是项目 $j$ 的阈值参数。

## 贡献

欢迎贡献代码！请遵循以下步骤：

1. Fork 本项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

## 开发指南

### 环境设置
```bash
git clone https://github.com/chaomowangshen/pymirt.git
cd pymirt
pip install -r requirements-dev.txt
pip install -e .
```

### 运行测试
```bash
pytest tests/
```

### 代码格式化
```bash
black src/
isort src/
```

## 许可证

本项目使用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

**MIT 许可证简介**：
- ✅ **商业使用**：可以用于商业项目
- ✅ **修改**：可以修改源代码
- ✅ **分发**：可以分发原始代码或修改后的代码
- ✅ **私人使用**：可以私人使用
- ⚠️ **责任**：作者不承担任何责任
- ⚠️ **保证**：不提供任何保证

简单来说，MIT 许可证是一个非常宽松的开源许可证，允许他人几乎可以对你的代码做任何事情，只要保留原始的版权声明即可。

## 作者

- Sheng Su - [sus473830@gmail.com](mailto:sus473830@gmail.com)

## 致谢

感谢所有为项目反应理论发展做出贡献的研究者们，特别是：
- Lord, F. M. 和 Novick, M. R. 的经典著作《Statistical Theories of Mental Test Scores》
- Hambleton, R. K. 等人在 IRT 理论和应用方面的贡献
- Embretson, S. E. 和 Reise, S. P. 在多维 IRT 方面的工作

## 引用

如果你在研究中使用了 PyMIRT，请引用：

```bibtex
@software{pymirt2025,
  author = {Sheng Su},
  title = {PyMIRT: A Python Package for Item Response Theory Parameter Estimation},
  year = {2025},
  url = {https://github.com/chaomowangshen/pymirt},
  version = {0.1.2}
}
```

## 版本历史

- **v0.1.2** (2025-07-10) - 当前版本
  - 在多维估计中增加了 SAEM 和 MCMC 方法
  - 增强了复杂模型的估计能力
  - 提高了计算效率

- **v0.1.1** (2025-07-02)
  - 支持单维和多维 IRT 模型
  - 支持 EM 和 MCEM 估计方法
  - 完整的包结构和文档
  - 初始发布版本

## 相关资源

- [项目反应理论入门教程](https://github.com/chaomowangshen/pymirt/wiki)
- [API 文档](https://chaomowangshen.github.io/pymirt)
- [示例代码](examples/)
- [常见问题解答](https://github.com/chaomowangshen/pymirt/wiki/FAQ)
