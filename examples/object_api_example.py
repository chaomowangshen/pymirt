import numpy as np
import pandas as pd

from pymirt import IRT, MIRT


def run_irt_object_example():
    np.random.seed(2026)
    response_df = pd.DataFrame(
        np.random.randint(0, 2, size=(60, 6)).astype(float),
        columns=[f"Item_{i + 1}" for i in range(6)],
    )

    result = IRT(
        model="2pl",
        method="em",
        n_quadrature=7,
        max_iter=3,
        tol=1e-2,
    ).fit(response_df)

    print(result.summary())
    print(result.item_params().head())
    print(result.person_params().head())


def run_mirt_object_example():
    np.random.seed(2026)
    response_df = pd.DataFrame(
        np.random.randint(0, 2, size=(60, 6)).astype(float),
        columns=[f"Item_{i + 1}" for i in range(6)],
    )
    Q = np.array(
        [
            [1, 0],
            [0, 1],
            [1, 1],
            [1, 0],
            [0, 1],
            [1, 1],
        ]
    )

    result = MIRT(
        Q=Q,
        model="m2pl",
        method="em",
        n_quadrature=3,
        max_iter=2,
        tol=1e-2,
    ).fit(response_df)

    print(result.summary())
    print(result.item_params().head())
    print(result.person_params().head())


if __name__ == "__main__":
    run_irt_object_example()
    run_mirt_object_example()
