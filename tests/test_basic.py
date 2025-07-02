import pytest
import numpy as np
import pandas as pd
from pymirt import irt, mirt
from pymirt.units.irt_simulate_data import simulate_2pl_data


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
    
    # Create Q matrix
    Q = np.array([
        [1, 0],
        [1, 0],
        [1, 0],
        [0, 1],
        [0, 1],
        [0, 1]
    ])
    
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
