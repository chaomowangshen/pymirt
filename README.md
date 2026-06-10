# PyMIRT

PyMIRT is a Python package for Item Response Theory (IRT) parameter estimation. It supports both unidimensional and multidimensional IRT models.

English | [中文](README_zh.md)

## Features

- **Unidimensional IRT Models**: Supports Rasch/1PL, 2PL, 3PL, and Graded Response Model (GRM)
- **Multidimensional IRT Models**: Supports Multidimensional 2PL (M2PL) and Multidimensional Graded Response Model (MGRM)
- **Multiple Estimation Methods**: Supports EM, Monte Carlo EM (MCEM), SAEM, and MCMC methods
- **Neural Backend**: Optional CEN-QB neural estimation for unidimensional 1PL/Rasch, 2PL, and GRM (`step`/`stand`) via `method='nn'`
- **Ability Estimation**: Supports Expected A Posteriori (EAP) and Markov Chain Monte Carlo estimation
- **Missing Data Handling**: Supports parameter estimation with response matrices containing missing data
- **Sparse Backend**: Optional sparse computation for high-missing response data via `use_sparse=True`
- **Object API**: Provides `IRT` and `MIRT` estimator classes with result summaries and parameter tables
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

For neural estimation, install the optional PyTorch extra:

```bash
pip install -e .[nn]
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

Neural GRM estimation supports both cumulative binary splitting and direct GRM likelihood:

```python
n_categories = [4] * response_df.shape[1]  # Scores are coded as 0, 1, 2, 3

a_est, b_est, theta_est = irt(
    response_df,
    model='grm',
    grm_type='stand',  # Use 'step' for cumulative binary splitting
    method='nn',
    n_categories=n_categories,
)
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

### Object API

The function API remains the quickest way to get `(a, b/d, theta)`. For a more organized workflow, use the estimator classes:

```python
import numpy as np
import pandas as pd
from pymirt import IRT, MIRT

response_df = pd.read_csv('your_response_data.csv')

# Unidimensional IRT
irt_result = IRT(
    model='2pl',
    method='em',
    use_sparse=True
).fit(response_df)

print(irt_result.summary())
print(irt_result.item_params().head())
print(irt_result.person_params().head())

# Multidimensional IRT
Q = np.array([[1, 0], [1, 0], [0, 1], [0, 1]])
mirt_result = MIRT(
    Q=Q,
    model='m2pl',
    method='mcmc',
    use_sparse=True,
    n_samples=300,
    burn_in=200
).fit(response_df)

print(mirt_result.summary())
print(mirt_result.item_params().head())
print(mirt_result.person_params().head())
```

Result objects provide `as_tuple()` for the original return style, plus `item_params()`, `person_params()`, and `summary()`.

## Supported Models

### Unidimensional Models
- **Rasch / 1PL**: One-Parameter Logistic Model with fixed discrimination (`a = 1`)
- **2PL**: Two-Parameter Logistic Model
- **3PL**: Three-Parameter Logistic Model with guessing parameter (`c`)
- **GRM_stand**: Standard Graded Response Model
- **GRM_step**: Stepwise Graded Response Model

### Multidimensional Models
- **M2PL**: Multidimensional Two-Parameter Logistic Model
- **MGRM_stand**: Standard Multidimensional Graded Response Model
- **MGRM_step**: Stepwise Multidimensional Graded Response Model

## Parameters

### irt() function parameters
- `response_df`: Response matrix (DataFrame)
- `model`: IRT model type ('rasch', '1pl', '2pl', '3pl', or 'grm')
- `grm_type`: GRM variant ('step' or 'stand')
- `method`: Estimation method ('em', 'mcem', 'saem', or 'mcmc')
- Rasch/1PL supports EM, MCMC, MCEM, and SAEM with fixed discrimination (`a = 1`).
- 3PL currently supports EM/EAP only; sampling methods are not implemented yet.
- `method='nn'`: Optional CEN-QB neural backend for 1PL/Rasch, 2PL, and GRM. For GRM, `grm_type='step'` uses cumulative binary pseudo-items and `grm_type='stand'` uses the direct GRM category likelihood.
- `n_quadrature`: Number of Gauss-Hermite quadrature points
- `n_categories`: Number of categories for each item (for GRM models)
- Neural GRM scores must be integer coded from `0` through `n_categories[j] - 1`.
- `n_samples`: Number of MCMC samples (for MCMC/MCEM methods)
- `burn_in`: MCMC burn-in period (for MCMC/MCEM methods)
- `sample_interval`: MCMC sample interval (for MCEM method)
- `max_iter`: Maximum number of iterations
- `tol`: Convergence tolerance
- `verbose`: Whether to print detailed information
- `use_sparse`: Whether to use the sparse response backend

### mirt() function parameters
- `response_df`: Response matrix (DataFrame)
- `Q`: Item loading matrix (numpy array)
- `method`: Estimation method ('em', 'mcem', 'saem', or 'mcmc'). Note: 'em' method only supports up to 3 dimensions.
- `model`: Multidimensional IRT model type ('m2pl' or 'mgrm')
- `grm_type`: MGRM variant ('step' or 'stand')
- `n_quadrature`: Number of quadrature points (for EM method)
- `n_samples`: Number of MCMC samples (for MCMC/MCEM methods)
- `burn_in`: MCMC burn-in period (for MCMC/MCEM methods)
- `sample_interval`: MCMC sample interval (for MCEM method)
- `max_iter`: Maximum number of iterations
- `tol`: Convergence tolerance
- `verbose`: Whether to print detailed information
- `use_sparse`: Whether to use the sparse response backend

### Object API
- `IRT(...)`: Unidimensional estimator class. Constructor arguments mirror `irt()`.
- `MIRT(Q=Q, ...)`: Multidimensional estimator class. Constructor arguments mirror `mirt()`.
- `fit(response_df)`: Estimates the model and returns an `IRTResult` or `MIRTResult`.
- `result.as_tuple()`: Returns the original tuple style.
- `result.item_params()`: Returns item parameters as a `DataFrame`.
- `result.person_params()`: Returns ability estimates as a `DataFrame`.
- `result.summary()`: Returns a lightweight summary dictionary.

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
  - EM, MCEM, SAEM, and MCMC estimation methods
  - Optional sparse backend for high-missing response data
  - Object API with IRT/MIRT estimator classes and result objects
  - Initial release
## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
