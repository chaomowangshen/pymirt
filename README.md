# PyMIRT

PyMIRT is a Python package for Item Response Theory (IRT) parameter estimation. It supports both unidimensional and multidimensional IRT models.

English | [中文](README_zh.md)

## Features

- **Unidimensional IRT Models**: Supports 2PL model and Graded Response Model (GRM)
- **Multidimensional IRT Models**: Supports Multidimensional 2PL (M2PL) and Multidimensional Graded Response Model (MGRM)
- **Multiple Estimation Methods**: Supports EM algorithm and Monte Carlo EM (mcem) methods
- **Ability Estimation**: Supports Expected A Posteriori (EAP) and Markov Chain Monte Carlo estimation
- **Missing Data Handling**: Supports parameter estimation with response matrices containing missing data
- **Flexible Configuration**: Supports custom quadrature points, iterations, convergence tolerance and other parameters

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

# Load data (assuming binary response data)
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

# Load data and Q matrix
response_df = pd.read_csv('your_response_data.csv')
Q = np.array([[1, 0], [1, 0], [0, 1], [0, 1]])  # Item loading matrix

# Estimate parameters using multidimensional 2PL model
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
- **GRM_stand**: Standard Graded Response Model
- **GRM_step**: Stepwise Graded Response Model

### Multidimensional Models
- **M2PL**: Multidimensional Two-Parameter Logistic Model
- **MGRM_stand**: Standard Multidimensional Graded Response Model
- **MGRM_step**: Stepwise Multidimensional Graded Response Model

## Parameters

### irt() function parameters
- `response_df`: Response matrix (DataFrame)
- `model`: IRT model type ('2PL', 'GRM_stand', 'GRM_step'). GRM_stand is the standard GRM implementation, GRM_step estimates difficulty parameters step by step.
- `n_quadrature`: Number of Gauss-Hermite quadrature points
- `n_categories`: Number of categories for each item (for GRM models)
- `max_iter`: Maximum number of iterations
- `tol`: Convergence tolerance
- `verbose`: Whether to print detailed information

### mirt() function parameters
- `response_df`: Response matrix (DataFrame)
- `Q`: Item loading matrix (numpy array)
- `method`: Estimation method ('em' or 'mc' for MCEM). Note: 'em' method only supports up to 3 dimensions.
- `model`: Multidimensional IRT model type ('m2pl', 'mgrm_step', or 'mgrm_stand'). mgrm_stand is the standard mgrm implementation, grm_step estimates threshold parameters step by step.
- `n_quadrature`: Number of quadrature points (for EM method)
- `n_samples`: Number of MCMC samples (for MCEM method)
- `burn_in`: MCMC burn-in period (for MCEM method)
- `sample_interval`: MCMC sample interval (for MCEM method)
- `max_iter`: Maximum number of iterations
- `tol`: Convergence tolerance
- `verbose`: Whether to print detailed information

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the project
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

**MIT License Overview**:
- ✅ **Commercial use**: Can be used in commercial projects
- ✅ **Modification**: Can modify the source code
- ✅ **Distribution**: Can distribute original or modified code
- ✅ **Private use**: Can use privately
- ⚠️ **Liability**: Author is not liable for any damages
- ⚠️ **Warranty**: No warranty provided

## Author

- Sheng Su - [sus473830@gmail.com](mailto:sus473830@gmail.com)

## Acknowledgments

Thanks to all researchers who have contributed to the development of Item Response Theory.

## Version History

- v0.1.1 - Current version
  - Support for unidimensional and multidimensional IRT models
  - EM and MCEM estimation methods
  - Initial release
