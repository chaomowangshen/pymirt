import pytest
import numpy as np
import pandas as pd
from pymirt import irt, mirt
from pymirt.units import (
    create_irt_quadrature,
    create_mirt_quadrature,
    dataframe_to_sparse_response,
    eap_2pl,
    eap_2pl_sparse,
    eap_m2pl,
    eap_m2pl_sparse,
    pad_grm_parameters,
    compute_m2pl_log_likelihood,
    compute_m2pl_log_likelihood_item,
    compute_mgrm_log_likelihood,
    compute_mgrm_log_likelihood_item,
    compute_m2pl_user_loglik_state_sparse,
    compute_m2pl_item_loglik_state_sparse,
    compute_mgrm_user_loglik_state_sparse,
    compute_mgrm_item_loglik_state_sparse,
)


def _assert_finite(value):
    if isinstance(value, list):
        value = np.concatenate([np.asarray(v).ravel() for v in value])
    else:
        value = np.asarray(value)
    assert np.all(np.isfinite(value))


def test_irt_2pl_basic():
    """Test basic 2PL model estimation"""
    # Generate synthetic data
    np.random.seed(42)
    n_subjects = 100
    n_items = 10
    
    # Create simple response matrix
    response_data = np.random.randint(0, 2, size=(n_subjects, n_items))
    response_df = pd.DataFrame(response_data)
    
    # Test IRT estimation
    a_est, b_est, theta_est = irt(
        response_df=response_df,
        model='2PL',
        n_quadrature=15,  # Use smaller number for faster testing
        max_iter=10,      # Use fewer iterations for faster testing
        tol=1e-3,
        verbose=False
    )
    
    # Basic checks
    assert len(a_est) == n_items
    assert len(b_est) == n_items
    assert len(theta_est) == n_subjects
    assert all(a > 0 for a in a_est)  # Discrimination parameters should be positive


def test_mirt_m2pl_basic():
    """Test basic M2PL model estimation"""
    np.random.seed(42)
    n_subjects = 50
    n_items = 30
    n_dimensions = 2
    
    # Create simple response matrix
    response_data = np.random.randint(0, 2, size=(n_subjects, n_items))
    response_df = pd.DataFrame(response_data)
    
    # Create Q matrix with one row per item
    Q = np.array([[1, 0] if j % 2 == 0 else [0, 1] for j in range(n_items)])
    
    # Test MIRT estimation
    a_est, b_est, theta_est = mirt(
        response_df=response_df,
        Q=Q,
        method='em',
        model='m2pl',
        n_quadrature=7,   # Use smaller number for faster testing
        max_iter=5,       # Use fewer iterations for faster testing
        tol=1e-2,
        verbose=False
    )
    
    # Basic checks
    assert a_est.shape == (n_items, n_dimensions)
    assert len(b_est) == n_items
    assert theta_est.shape == (n_subjects, n_dimensions)


def test_irt_2pl_sparse_basic():
    """Test sparse 2PL model estimation with DataFrame input"""
    np.random.seed(7)
    n_subjects = 30
    n_items = 4
    response_data = np.random.randint(0, 2, size=(n_subjects, n_items)).astype(float)
    response_data[np.random.rand(n_subjects, n_items) < 0.25] = np.nan
    response_df = pd.DataFrame(response_data)

    a_est, b_est, theta_est = irt(
        response_df=response_df,
        model='2PL',
        n_quadrature=3,
        max_iter=1,
        tol=1e-2,
        use_sparse=True,
    )

    assert a_est.shape == (n_items,)
    assert b_est.shape == (n_items,)
    assert theta_est.shape == (n_subjects,)


def test_mirt_m2pl_sparse_basic():
    """Test sparse M2PL EM estimation with DataFrame input"""
    np.random.seed(8)
    n_subjects = 24
    n_items = 4
    n_dimensions = 2
    response_data = np.random.randint(0, 2, size=(n_subjects, n_items)).astype(float)
    response_data[np.random.rand(n_subjects, n_items) < 0.25] = np.nan
    response_df = pd.DataFrame(response_data)
    Q = np.array([[1, 0], [0, 1], [1, 1], [1, 0]])

    a_est, d_est, theta_est = mirt(
        response_df=response_df,
        Q=Q,
        method='em',
        model='m2pl',
        n_quadrature=3,
        max_iter=1,
        tol=1e-2,
        use_sparse=True,
    )

    assert a_est.shape == (n_items, n_dimensions)
    assert d_est.shape == (n_items,)
    assert theta_est.shape == (n_subjects, n_dimensions)


def test_irt_grm_sparse_basic():
    """Test sparse GRM stepwise estimation"""
    np.random.seed(9)
    n_subjects = 24
    n_items = 3
    n_categories = np.array([3, 4, 3])
    response_data = np.column_stack([
        np.random.randint(0, n_categories[j], size=n_subjects)
        for j in range(n_items)
    ]).astype(float)
    response_data[np.random.rand(n_subjects, n_items) < 0.2] = np.nan
    response_df = pd.DataFrame(response_data)

    a_est, b_est, theta_est = irt(
        response_df=response_df,
        model='grm',
        grm_type='step',
        n_categories=n_categories,
        n_quadrature=3,
        max_iter=1,
        tol=1e-2,
        use_sparse=True,
    )

    assert a_est.shape == (n_items,)
    assert len(b_est) == n_items
    assert theta_est.shape == (n_subjects,)


def test_mirt_mgrm_sparse_basic():
    """Test sparse MGRM EM estimation"""
    np.random.seed(10)
    n_subjects = 18
    n_items = 4
    n_dimensions = 2
    n_categories = np.array([3, 3, 4, 3])
    response_data = np.column_stack([
        np.random.randint(0, n_categories[j], size=n_subjects)
        for j in range(n_items)
    ]).astype(float)
    response_data[np.random.rand(n_subjects, n_items) < 0.2] = np.nan
    response_df = pd.DataFrame(response_data)
    Q = np.array([[1, 0], [0, 1], [1, 1], [1, 0]])

    a_est, d_est, theta_est = mirt(
        response_df=response_df,
        Q=Q,
        method='em',
        model='mgrm',
        grm_type='step',
        n_categories=n_categories,
        n_quadrature=3,
        max_iter=1,
        tol=1e-2,
        use_sparse=True,
    )

    assert a_est.shape == (n_items, n_dimensions)
    assert len(d_est) == n_items
    assert theta_est.shape == (n_subjects, n_dimensions)


def test_sparse_eap_matches_dense_weighting():
    """Sparse EAP should use the same quadrature weighting as dense EAP"""
    response_df = pd.DataFrame(
        [
            [1.0, 0.0, np.nan],
            [0.0, 1.0, 1.0],
            [1.0, np.nan, 0.0],
            [np.nan, 0.0, 1.0],
        ]
    )
    response = response_df.values
    mask = (~response_df.isna()).astype(int).values
    sparse_response = dataframe_to_sparse_response(response_df)

    quad_points, quad_weights = create_irt_quadrature(5)
    a_params = np.array([0.8, 1.2, 0.6])
    b_params = np.array([-0.5, 0.2, 0.9])
    dense_theta = eap_2pl(response, mask, a_params, b_params, quad_points, quad_weights)
    sparse_theta = eap_2pl_sparse(
        sparse_response, a_params, b_params, quad_points, quad_weights
    )
    np.testing.assert_allclose(sparse_theta, dense_theta, atol=1e-8)

    quad_points_nd, quad_weights_nd = create_mirt_quadrature(3, 2)
    a_params_nd = np.array([[1.0, 0.0], [0.0, 1.1], [0.7, 0.8]])
    d_params = np.array([-0.2, 0.1, 0.4])
    dense_theta_nd = eap_m2pl(
        response, mask, a_params_nd, d_params, quad_points_nd, quad_weights_nd
    )
    sparse_theta_nd = eap_m2pl_sparse(
        sparse_response, a_params_nd, d_params, quad_points_nd, quad_weights_nd
    )
    np.testing.assert_allclose(sparse_theta_nd, dense_theta_nd, atol=1e-8)


def test_sparse_state_loglik_matches_dense():
    """Sparse sampling likelihood helpers should match dense likelihood sums"""
    response_df = pd.DataFrame(
        [
            [1.0, 0.0, np.nan],
            [0.0, 1.0, 1.0],
            [1.0, np.nan, 0.0],
            [np.nan, 0.0, 1.0],
        ]
    )
    response = np.nan_to_num(response_df.values)
    mask = (~response_df.isna()).astype(int).values
    sparse_response = dataframe_to_sparse_response(response_df)

    theta = np.array([[0.1, -0.2], [0.3, 0.4], [-0.5, 0.2], [0.0, 0.1]])
    a_params = np.array([[0.8, 0.0], [0.0, 1.2], [0.7, 0.9]])
    d_params = np.array([-0.3, 0.2, 0.5])
    np.testing.assert_allclose(
        compute_m2pl_user_loglik_state_sparse(
            theta, a_params, d_params, sparse_response
        ),
        compute_m2pl_log_likelihood(theta, a_params, d_params, response, mask),
        atol=1e-8,
    )
    np.testing.assert_allclose(
        compute_m2pl_item_loglik_state_sparse(
            theta, a_params, d_params, sparse_response
        ),
        compute_m2pl_log_likelihood_item(theta, a_params, d_params, response, mask),
        atol=1e-8,
    )

    grm_df = pd.DataFrame(
        [
            [0.0, 1.0, np.nan],
            [1.0, 2.0, 0.0],
            [2.0, np.nan, 1.0],
            [np.nan, 0.0, 2.0],
        ]
    )
    grm_response = np.nan_to_num(grm_df.values)
    grm_mask = (~grm_df.isna()).astype(int).values
    grm_sparse = dataframe_to_sparse_response(grm_df)
    n_categories = np.array([3, 3, 3])
    grm_d = [np.array([0.6, -0.4]), np.array([0.5, -0.2]), np.array([0.8, -0.1])]
    d_matrix, d_mask = pad_grm_parameters(grm_d, n_categories)
    np.testing.assert_allclose(
        compute_mgrm_user_loglik_state_sparse(
            theta, a_params, grm_d, grm_sparse, n_categories
        ),
        compute_mgrm_log_likelihood(
            theta, a_params, d_matrix, grm_response, grm_mask, n_categories, d_mask=d_mask
        ),
        atol=1e-8,
    )
    np.testing.assert_allclose(
        compute_mgrm_item_loglik_state_sparse(
            theta, a_params, grm_d, grm_sparse, n_categories
        ),
        compute_mgrm_log_likelihood_item(
            theta, a_params, d_matrix, d_mask, grm_response, grm_mask, n_categories
        ),
        atol=1e-8,
    )


def test_mirt_m2pl_sparse_sampling_methods():
    """Sparse M2PL should support MCMC, MCEM, and SAEM"""
    np.random.seed(11)
    n_subjects = 12
    n_items = 4
    n_dimensions = 2
    response_data = np.random.randint(0, 2, size=(n_subjects, n_items)).astype(float)
    response_data[np.random.rand(n_subjects, n_items) < 0.2] = np.nan
    response_df = pd.DataFrame(response_data)
    Q = np.array([[1, 0], [0, 1], [1, 1], [1, 0]])

    for method in ['mcmc', 'mcem', 'saem']:
        np.random.seed(101)
        a_est, d_est, theta_est = mirt(
            response_df=response_df,
            Q=Q,
            method=method,
            model='m2pl',
            n_samples=3,
            burn_in=2,
            max_iter=1,
            tol=1e-2,
            use_sparse=True,
        )

        assert a_est.shape == (n_items, n_dimensions)
        assert d_est.shape == (n_items,)
        assert theta_est.shape == (n_subjects, n_dimensions)
        _assert_finite(a_est)
        _assert_finite(d_est)
        _assert_finite(theta_est)


def test_mirt_mgrm_sparse_sampling_methods():
    """Sparse MGRM should support MCMC, MCEM, and SAEM for both GRM variants"""
    np.random.seed(12)
    n_subjects = 10
    n_items = 4
    n_dimensions = 2
    n_categories = np.array([3, 3, 4, 3])
    response_data = np.column_stack([
        np.random.randint(0, n_categories[j], size=n_subjects)
        for j in range(n_items)
    ]).astype(float)
    response_data[np.random.rand(n_subjects, n_items) < 0.15] = np.nan
    response_df = pd.DataFrame(response_data)
    Q = np.array([[1, 0], [0, 1], [1, 1], [1, 0]])

    for grm_type in ['step', 'stand']:
        for method in ['mcmc', 'mcem', 'saem']:
            np.random.seed(202)
            a_est, d_est, theta_est = mirt(
                response_df=response_df,
                Q=Q,
                method=method,
                model='mgrm',
                grm_type=grm_type,
                n_categories=n_categories,
                n_samples=3,
                burn_in=2,
                max_iter=1,
                tol=1e-2,
                use_sparse=True,
            )

            assert a_est.shape == (n_items, n_dimensions)
            assert len(d_est) == n_items
            assert theta_est.shape == (n_subjects, n_dimensions)
            _assert_finite(a_est)
            _assert_finite(d_est)
            _assert_finite(theta_est)


def test_irt_2pl_sampling_methods():
    """Single-dimensional 2PL should support MCMC, MCEM, and SAEM"""
    np.random.seed(13)
    n_subjects = 12
    n_items = 4
    response_data = np.random.randint(0, 2, size=(n_subjects, n_items)).astype(float)
    response_data[np.random.rand(n_subjects, n_items) < 0.15] = np.nan
    response_df = pd.DataFrame(response_data)

    for method in ['mcmc', 'mcem', 'saem']:
        np.random.seed(303)
        a_est, b_est, theta_est = irt(
            response_df=response_df,
            model='2pl',
            method=method,
            n_samples=3,
            burn_in=2,
            max_iter=1,
            tol=1e-2,
        )

        assert a_est.shape == (n_items,)
        assert b_est.shape == (n_items,)
        assert theta_est.shape == (n_subjects,)
        assert np.all(a_est > 0)
        _assert_finite(a_est)
        _assert_finite(b_est)
        _assert_finite(theta_est)


def test_irt_grm_sampling_methods():
    """Single-dimensional GRM should support MCMC, MCEM, and SAEM"""
    np.random.seed(14)
    n_subjects = 10
    n_items = 3
    n_categories = np.array([3, 3, 4])
    response_data = np.column_stack([
        np.random.randint(0, n_categories[j], size=n_subjects)
        for j in range(n_items)
    ]).astype(float)
    response_data[np.random.rand(n_subjects, n_items) < 0.15] = np.nan
    response_df = pd.DataFrame(response_data)

    for grm_type in ['step', 'stand']:
        for method in ['mcmc', 'mcem', 'saem']:
            np.random.seed(404)
            a_est, b_est, theta_est = irt(
                response_df=response_df,
                model='grm',
                grm_type=grm_type,
                method=method,
                n_categories=n_categories,
                n_samples=3,
                burn_in=2,
                max_iter=1,
                tol=1e-2,
            )

            assert a_est.shape == (n_items,)
            assert len(b_est) == n_items
            assert theta_est.shape == (n_subjects,)
            assert np.all(a_est > 0)
            for item_id, thresholds in enumerate(b_est):
                assert len(thresholds) == n_categories[item_id] - 1
                assert np.all(np.diff(thresholds) >= 0)
            _assert_finite(a_est)
            _assert_finite(b_est)
            _assert_finite(theta_est)


def test_sparse_mirt_validation_errors():
    """Sparse MIRT should keep validation explicit"""
    response_df = pd.DataFrame(np.random.randint(0, 2, size=(10, 4)).astype(float))
    Q = np.array([[1, 0], [0, 1], [1, 1], [1, 0]])

    invalid_binary = response_df.copy()
    invalid_binary.iloc[0, 0] = 2
    with pytest.raises(ValueError):
        mirt(
            response_df=invalid_binary,
            Q=Q,
            method='mcmc',
            model='m2pl',
            use_sparse=True,
        )

    with pytest.raises(ValueError):
        mirt(
            response_df=response_df,
            Q=Q,
            method='mcmc',
            model='mgrm',
            use_sparse=True,
        )

    empty_item = response_df.copy()
    empty_item.iloc[:, 0] = np.nan
    with pytest.raises(ValueError):
        mirt(
            response_df=empty_item,
            Q=Q,
            method='mcmc',
            model='m2pl',
            use_sparse=True,
        )

    with pytest.raises(ValueError):
        mirt(
            response_df=response_df,
            Q=Q[:3],
            method='mcmc',
            model='m2pl',
            use_sparse=True,
        )


def test_invalid_model():
    """Test error handling for invalid model"""
    response_df = pd.DataFrame(np.random.randint(0, 2, size=(10, 5)))
    
    with pytest.raises(ValueError):
        irt(response_df=response_df, model='invalid_model')


def test_empty_dataframe():
    """Test error handling for empty dataframe"""
    empty_df = pd.DataFrame()
    
    with pytest.raises((ValueError, IndexError)):
        irt(response_df=empty_df, model='2PL')


if __name__ == "__main__":
    # Run tests when script is executed directly
    test_irt_2pl_basic()
    test_mirt_m2pl_basic()
    print("All basic tests passed!")
