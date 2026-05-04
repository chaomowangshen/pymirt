from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from .irt_est import irt as _irt
from .mirt_est import mirt as _mirt


@dataclass
class IRTResult:
    a_: np.ndarray = field(repr=False)
    b_: Any = field(repr=False)
    theta_: np.ndarray = field(repr=False)
    model: str
    method: str
    grm_type: str
    n_categories: Optional[np.ndarray]
    use_sparse: bool
    config: Dict[str, Any]

    def as_tuple(self):
        return self.a_, self.b_, self.theta_

    def item_params(self):
        a = np.asarray(self.a_, dtype=float)
        model = self.model.lower()

        if model in {"1pl", "rasch", "2pl"}:
            return pd.DataFrame(
                {
                    "item": np.arange(a.size),
                    "a": a,
                    "b": np.asarray(self.b_, dtype=float),
                }
            )

        rows = []
        for item_id, thresholds in enumerate(self.b_):
            for threshold_id, b_value in enumerate(np.asarray(thresholds), start=1):
                rows.append(
                    {
                        "item": item_id,
                        "a": a[item_id],
                        "threshold": threshold_id,
                        "b": b_value,
                    }
                )
        return pd.DataFrame(rows, columns=["item", "a", "threshold", "b"])

    def person_params(self):
        return pd.DataFrame(
            {
                "user": np.arange(np.asarray(self.theta_).size),
                "theta": np.asarray(self.theta_, dtype=float),
            }
        )

    def summary(self):
        return {
            "model": self.model,
            "method": self.method,
            "n_users": int(np.asarray(self.theta_).size),
            "n_items": int(np.asarray(self.a_).size),
            "use_sparse": bool(self.use_sparse),
            "n_parameters": _count_scalars(self.a_, self.b_, self.theta_),
        }


@dataclass
class MIRTResult:
    a_: np.ndarray = field(repr=False)
    d_: Any = field(repr=False)
    theta_: np.ndarray = field(repr=False)
    Q: np.ndarray = field(repr=False)
    model: str
    method: str
    grm_type: str
    n_categories: Optional[np.ndarray]
    use_sparse: bool
    config: Dict[str, Any]

    def as_tuple(self):
        return self.a_, self.d_, self.theta_

    def item_params(self):
        a = _as_2d(self.a_)
        a_columns = [f"a_dim{dim + 1}" for dim in range(a.shape[1])]
        model = self.model.lower()

        if model == "m2pl":
            data = {"item": np.arange(a.shape[0])}
            for dim, column in enumerate(a_columns):
                data[column] = a[:, dim]
            data["d"] = np.asarray(self.d_, dtype=float)
            return pd.DataFrame(data)

        rows = []
        for item_id, thresholds in enumerate(self.d_):
            base = {"item": item_id}
            for dim, column in enumerate(a_columns):
                base[column] = a[item_id, dim]
            for threshold_id, d_value in enumerate(np.asarray(thresholds), start=1):
                row = base.copy()
                row["threshold"] = threshold_id
                row["d"] = d_value
                rows.append(row)
        return pd.DataFrame(rows, columns=["item"] + a_columns + ["threshold", "d"])

    def person_params(self):
        theta = _as_2d(self.theta_)
        data = {"user": np.arange(theta.shape[0])}
        for dim in range(theta.shape[1]):
            data[f"theta_dim{dim + 1}"] = theta[:, dim]
        return pd.DataFrame(data)

    def summary(self):
        theta = _as_2d(self.theta_)
        q_shape = tuple(int(value) for value in np.asarray(self.Q).shape)
        return {
            "model": self.model,
            "method": self.method,
            "n_users": int(theta.shape[0]),
            "n_items": int(np.asarray(self.a_).shape[0]),
            "use_sparse": bool(self.use_sparse),
            "n_dimensions": int(theta.shape[1]),
            "q_shape": q_shape,
            "n_parameters": _count_scalars(self.a_, self.d_, self.theta_),
        }


@dataclass
class IRT:
    model: str = "2PL"
    method: str = "em"
    grm_type: str = "step"
    n_quadrature: int = 27
    n_categories: Any = None
    n_samples: int = 100
    burn_in: int = 100
    sample_interval: int = 10
    max_iter: int = 100
    tol: float = 1e-4
    verbose: bool = False
    use_sparse: bool = False
    result_: Optional[IRTResult] = field(default=None, init=False, repr=False)

    def fit(self, response_df):
        a_est, b_est, theta_est = _irt(
            response_df=response_df,
            model=self.model,
            method=self.method,
            grm_type=self.grm_type,
            n_quadrature=self.n_quadrature,
            n_categories=self.n_categories,
            n_samples=self.n_samples,
            burn_in=self.burn_in,
            sample_interval=self.sample_interval,
            max_iter=self.max_iter,
            tol=self.tol,
            verbose=self.verbose,
            use_sparse=self.use_sparse,
        )
        self.result_ = IRTResult(
            a_=np.asarray(a_est, dtype=float),
            b_=_copy_parameter(b_est),
            theta_=np.asarray(theta_est, dtype=float),
            model=str(self.model).lower(),
            method=str(self.method).lower(),
            grm_type=str(self.grm_type).lower(),
            n_categories=_copy_optional_array(self.n_categories),
            use_sparse=bool(self.use_sparse),
            config=self._config(),
        )
        return self.result_

    def _config(self):
        return {
            "model": self.model,
            "method": self.method,
            "grm_type": self.grm_type,
            "n_quadrature": self.n_quadrature,
            "n_categories": _copy_optional_array(self.n_categories),
            "n_samples": self.n_samples,
            "burn_in": self.burn_in,
            "sample_interval": self.sample_interval,
            "max_iter": self.max_iter,
            "tol": self.tol,
            "verbose": self.verbose,
            "use_sparse": self.use_sparse,
        }


@dataclass
class MIRT:
    Q: Any
    method: str = "em"
    model: str = "m2pl"
    grm_type: str = "step"
    n_categories: Any = None
    n_quadrature: int = 15
    n_samples: int = 100
    burn_in: int = 100
    sample_interval: int = 10
    max_iter: int = 100
    tol: float = 1e-4
    verbose: bool = False
    use_sparse: bool = False
    result_: Optional[MIRTResult] = field(default=None, init=False, repr=False)

    def fit(self, response_df):
        a_est, d_est, theta_est = _mirt(
            response_df=response_df,
            Q=self.Q,
            method=self.method,
            model=self.model,
            grm_type=self.grm_type,
            n_categories=self.n_categories,
            n_quadrature=self.n_quadrature,
            n_samples=self.n_samples,
            burn_in=self.burn_in,
            sample_interval=self.sample_interval,
            max_iter=self.max_iter,
            tol=self.tol,
            verbose=self.verbose,
            use_sparse=self.use_sparse,
        )
        self.result_ = MIRTResult(
            a_=np.asarray(a_est, dtype=float),
            d_=_copy_parameter(d_est),
            theta_=np.asarray(theta_est, dtype=float),
            Q=np.asarray(self.Q).copy(),
            model=str(self.model).lower(),
            method=str(self.method).lower(),
            grm_type=str(self.grm_type).lower(),
            n_categories=_copy_optional_array(self.n_categories),
            use_sparse=bool(self.use_sparse),
            config=self._config(),
        )
        return self.result_

    def _config(self):
        return {
            "method": self.method,
            "model": self.model,
            "grm_type": self.grm_type,
            "n_categories": _copy_optional_array(self.n_categories),
            "n_quadrature": self.n_quadrature,
            "n_samples": self.n_samples,
            "burn_in": self.burn_in,
            "sample_interval": self.sample_interval,
            "max_iter": self.max_iter,
            "tol": self.tol,
            "verbose": self.verbose,
            "use_sparse": self.use_sparse,
        }


def _as_2d(value):
    array = np.asarray(value, dtype=float)
    if array.ndim == 1:
        return array.reshape(-1, 1)
    return array


def _copy_optional_array(value):
    if value is None:
        return None
    return np.asarray(value).copy()


def _copy_parameter(value):
    if isinstance(value, list):
        return [np.asarray(item, dtype=float).copy() for item in value]
    return np.asarray(value, dtype=float).copy()


def _count_scalars(*values):
    count = 0
    for value in values:
        if isinstance(value, list):
            count += sum(np.asarray(item).size for item in value)
        else:
            count += np.asarray(value).size
    return int(count)
