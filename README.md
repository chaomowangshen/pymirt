# PyMIRT

PyMIRT is a Python package for Item Response Theory (IRT) parameter estimation, supporting both unidimensional and multidimensional IRT models.

[English](README.md) | [中文](README_zh.md)

## Features

- **Unidimensional IRT Models**: Supports 2PL model and Graded Response Model (GRM)
- **Multidimensional IRT Models**: Supports Multidimensional 2PL (M2PL) and Multidimensional GRM (MGRM)
- **Estimation Methods**: EM algorithm, Monte Carlo EM (MCEM), Stochastic Approximation EM (SAEM), and Markov Chain Monte Carlo (MCMC)
- **Ability Estimation**: Expected A Posteriori (EAP) and MCMC estimation
- **Missing Data Handling**: Supports parameter estimation with response matrices containing missing data
- **Flexible Configuration**: Customizable quadrature points, iteration limits, convergence tolerance, etc.

## Installation

Currently requires installation from source:

```bash
git clone https://github.com/chaomowangshen/pymirt.git
cd pymirt
pip install -e .
```

Or install directly from GitHub:

```bash
pip install git+https://github.com/chaomowangshen/pymirt.git
```

## Quick Start

### Unidimensional IRT Model

```python
import pandas as pd
from pymirt import irt

# Load response data (binary format assumed)
response_df = pd.read_csv('your_response_data.csv')

# Estimate parameters using 2PL model
a_est, b_est, theta_est = irt(
    response_df=response_df,
    model='2PL',
    n_quadrature=27,
    max_iter=100,
    tol=1e-4,
    verbose=True
)

print(f"Discrimination parameters: {a_est}")
print(f"Difficulty parameters: {b_est}")
print(f"Ability estimates: {theta_est}")
```

### Multidimensional IRT Model

```python
import numpy as np
import pandas as pd
from pymirt import mirt

# Load response data and Q-matrix
response_df = pd.read_csv('your_response_data.csv')
Q = np.array([[1, 0], [1, 0], [0, 1], [0, 1]])  # Item feature matrix

# Estimate parameters using M2PL model
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

print(f"Discrimination parameters: {a_est}")
print(f"Difficulty parameters: {b_est}")
print(f"Ability estimates: {theta_est}")
```

## Supported Models

### Unidimensional Models
- **2PL**: Two-Parameter Logistic Model
- **GRM**: Graded Response Model

### Multidimensional Models
- **M2PL**: Multidimensional Two-Parameter Logistic Model
- **MGRM**: Multidimensional Graded Response Model

## Parameters Specification

### irt() Parameters
- `response_df`: Response matrix (DataFrame)
- `model`: IRT model type ('2pl', 'grm')
- `grm_type`: GRM estimation type ('step' for phased estimation, 'stand' for standard implementation)
- `n_quadrature`: Gauss-Hermite quadrature points
- `n_categories`: Number of item categories (for GRM)
- `max_iter`: Maximum iterations
- `tol`: Convergence tolerance
- `verbose`: Verbosity flag

### mirt() Parameters
- `response_df`: Response matrix (DataFrame)
- `Q`: Q-matrix (numpy array)
- `method`: Estimation method ('em', 'mcem', 'saem', 'mcmc') Note: EM method supports up to 3 dimensions
- `model`: MIRT model type ('m2pl' or 'mgrm')
- `grm_type`: MGRM estimation type ('step' for phased estimation, 'stand' for standard implementation)
- `n_quadrature`: Quadrature points (for EM)
- `n_samples`: MCMC samples (for MCEM/MCMC)
- `burn_in`: MCMC burn-in period (for MCEM/MCMC)
- `sample_interval`: MCMC sampling interval (for MCEM)
- `max_iter`: Maximum iterations
- `tol`: Convergence tolerance
- `verbose`: Verbosity flag

## Theoretical Background

### Item Response Theory (IRT)
As a fundamental component of modern measurement theory, IRT is widely used in educational and psychological measurement. Compared with Classical Test Theory, IRT offers:

- **Item Parameter Invariance**: Item parameters are independent of specific examinee groups
- **Precise Ability Estimation**: Provides individualized ability estimates with standard errors
- **CAT Support**: Theoretical foundation for Computerized Adaptive Testing
- **Equating**: Enables score equating across different test forms

### Unidimensional Models

#### Two-Parameter Logistic Model (2PL)
$$P_{ij}(\theta) = \frac{1}{1 + \exp(-a_j(\theta_i - b_j))}$$

Where:
- $P_{ij}(\theta)$ is the probability of examinee $i$ answering item $j$ correctly
- $a_j$ is the discrimination parameter
- $b_j$ is the difficulty parameter
- $\theta_i$ is the ability parameter

#### Graded Response Model (GRM)
$$P_{ijk}^*(\theta) = \frac{\exp(a_j(\theta_i - b_{jk}))}{1 + \exp(a_j(\theta_i - b_{jk}))}$$

### Multidimensional Models

#### Multidimensional 2PL (M2PL)
$$P_{ij}(\boldsymbol{\theta}) = \frac{1}{1 + \exp(-(\boldsymbol{a}_j^T \boldsymbol{\theta}_i + d_j))}$$

Where $\boldsymbol{a}_j$ is the discrimination vector, $\boldsymbol{\theta}_i$ is the ability vector, and $d_j$ is the threshold parameter.

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the project
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Development Guide

### Environment Setup
```bash
git clone https://github.com/chaomowangshen/pymirt.git
cd pymirt
pip install -r requirements-dev.txt
pip install -e .
```

### Running Tests
```bash
pytest tests/
```

### Code Formatting
```bash
black src/
isort src/
```

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

**MIT License Highlights**:
- ✅ **Commercial Use**: Allowed
- ✅ **Modification**: Permitted
- ✅ **Distribution**: Allowed for both original and modified code
- ✅ **Private Use**: Permitted
- ⚠️ **Liability**: No liability
- ⚠️ **Warranty**: No warranty

In essence, the MIT License is a permissive open-source license allowing extensive freedom with minimal restrictions.

## Authors

- Sheng Su - [sus473830@gmail.com](mailto:sus473830@gmail.com)

## Acknowledgments

We acknowledge contributions from IRT researchers, particularly:
- Foundational work by Lord, F.M. & Novick, M.R. in *Statistical Theories of Mental Test Scores*
- Contributions by Hambleton, R.K. et al. in IRT theory and applications
- Multidimensional IRT work by Embretson, S.E. & Reise, S.P.

## Citation

If using PyMIRT in research, please cite:

```bibtex
@software{pymirt2025,
  author = {Sheng Su},
  title = {PyMIRT: A Python Package for Item Response Theory Parameter Estimation},
  year = {2025},
  url = {https://github.com/chaomowangshen/pymirt},
  version = {0.1.2}
}
```

## Version History

- **v0.1.2** (2025-07-10) - Current Version
  - Added SAEM and MCMC methods for multidimensional estimation
  - Enhanced estimation capabilities for complex models
  - Improved computational efficiency

- **v0.1.1** (2025-07-02)
  - Support for uni/multidimensional IRT models
  - EM and MCEM estimation methods
  - Complete package structure and documentation
  - Initial release

## Related Resources

- [IRT Tutorial](https://github.com/chaomowangshen/pymirt/wiki)
- [API Documentation](https://chaomowangshen.github.io/pymirt)
- [Examples](examples/)
- [FAQ](https://github.com/chaomowangshen/pymirt/wiki/FAQ)
