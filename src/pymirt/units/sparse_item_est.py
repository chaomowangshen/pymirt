import time

import numpy as np
from scipy.optimize import minimize

from .sparse_data import SparseResponse
from .units import (
    create_irt_quadrature,
    create_mirt_quadrature,
    ensure_ordered_thresholds,
    grm_prob_categories,
    mgrm_prob_categories,
    pad_grm_parameters,
)


def irt_em_sparse(sparse_response, n_quadrature=27, max_iter=100, tol=1e-4, verbose=False):
    n_items = sparse_response.n_items
    a_est = np.ones(n_items)
    b_est = np.zeros(n_items)
    theta_quad, weights_quad = create_irt_quadrature(n_quadrature)
    prev_ll = -np.inf
    total_time = 0.0

    for iteration in range(max_iter):
        start_time = time.time()
        posterior = compute_2pl_posterior_sparse(
            theta_quad, a_est, b_est, sparse_response, weights_quad
        )

        a_new = a_est.copy()
        b_new = b_est.copy()
        for item_id in range(n_items):
            obs_idx = sparse_response.item_observations(item_id)
            if obs_idx.size == 0:
                continue
            users = sparse_response.user_idx[obs_idx]
            values = sparse_response.values[obs_idx]

            def objective(params):
                a_j, b_j = params
                return -compute_2pl_item_expected_ll_sparse(
                    theta_quad, a_j, b_j, users, values, posterior
                )

            res = minimize(
                objective,
                [a_est[item_id], b_est[item_id]],
                method="L-BFGS-B",
                bounds=[(0.1, None), (None, None)],
            )
            if res.success:
                a_new[item_id], b_new[item_id] = res.x

        a_est, b_est = a_new, b_new
        current_ll = compute_2pl_expected_ll_sparse(
            theta_quad, a_est, b_est, sparse_response, posterior
        )
        total_time += time.time() - start_time
        if verbose:
            print(
                f"=== sparse 2PL iter {iteration + 1}/{max_iter}, "
                f"delta_ll={current_ll - prev_ll:.6f}, elapsed={total_time:.2f}s ==="
            )
        if iteration > 0 and abs(current_ll - prev_ll) < tol:
            break
        prev_ll = current_ll

    return a_est, b_est


def estimate_b_only_sparse(
    a_params, sparse_response, n_quadrature=27, max_iter=100, tol=1e-4, verbose=False
):
    n_items = sparse_response.n_items
    b_est = np.zeros(n_items)
    theta_quad, weights_quad = create_irt_quadrature(n_quadrature)
    prev_ll = -np.inf
    total_time = 0.0

    for iteration in range(max_iter):
        start_time = time.time()
        posterior = compute_2pl_posterior_sparse(
            theta_quad, a_params, b_est, sparse_response, weights_quad
        )
        b_new = b_est.copy()
        for item_id in range(n_items):
            obs_idx = sparse_response.item_observations(item_id)
            if obs_idx.size == 0:
                continue
            users = sparse_response.user_idx[obs_idx]
            values = sparse_response.values[obs_idx]

            def objective(b_j):
                return -compute_2pl_item_expected_ll_sparse(
                    theta_quad, a_params[item_id], b_j[0], users, values, posterior
                )

            res = minimize(objective, [b_est[item_id]], method="BFGS")
            if res.success:
                b_new[item_id] = res.x[0]

        b_est = b_new
        current_ll = compute_2pl_expected_ll_sparse(
            theta_quad, a_params, b_est, sparse_response, posterior
        )
        total_time += time.time() - start_time
        if verbose:
            print(
                f"=== sparse b-only iter {iteration + 1}/{max_iter}, "
                f"delta_ll={current_ll - prev_ll:.6f} ==="
            )
        if iteration > 0 and abs(current_ll - prev_ll) < tol:
            break
        prev_ll = current_ll

    return b_est, total_time


def rasch_em_sparse(
    sparse_response, n_quadrature=27, max_iter=100, tol=1e-4, verbose=False
):
    n_items = sparse_response.n_items
    a_est = np.ones(n_items)
    b_est, _ = estimate_b_only_sparse(
        a_est,
        sparse_response,
        n_quadrature=n_quadrature,
        max_iter=max_iter,
        tol=tol,
        verbose=verbose,
    )
    return a_est, b_est


def grm_em_stepwise_sparse(
    sparse_response, n_categories, n_quadrature=27, max_iter=100, tol=1e-4, verbose=False
):
    n_categories = np.asarray(n_categories)
    max_threshold = int(np.max(n_categories) - 1)

    step1 = sparse_response.with_values((sparse_response.values >= 1).astype(float))
    a_est, b_step1 = irt_em_sparse(
        step1, n_quadrature=n_quadrature, max_iter=max_iter, tol=tol, verbose=verbose
    )
    b_est = [np.array([b_step1[j]]) for j in range(sparse_response.n_items)]

    for threshold in range(2, max_threshold + 1):
        item_mask = n_categories >= threshold + 1
        if not np.any(item_mask):
            continue
        binary_values = (sparse_response.values >= threshold).astype(float)
        subset = sparse_response.subset_items(item_mask, values=binary_values)
        b_k, _ = estimate_b_only_sparse(
            a_est[item_mask],
            subset,
            n_quadrature=n_quadrature,
            max_iter=max_iter,
            tol=tol,
            verbose=verbose,
        )
        idx = 0
        for item_id, keep in enumerate(item_mask):
            if keep:
                b_est[item_id] = np.append(b_est[item_id], b_k[idx])
                idx += 1

    return a_est, b_est


def grm_em_standard_sparse(
    sparse_response, n_categories, n_quadrature=27, max_iter=100, tol=1e-4, verbose=False
):
    n_categories = np.asarray(n_categories)
    n_items = sparse_response.n_items
    a_est = np.ones(n_items)
    b_est = [np.sort(np.random.normal(0, 1, int(k) - 1)) for k in n_categories]
    theta_quad, weights_quad = create_irt_quadrature(n_quadrature)
    prev_ll = -np.inf

    for iteration in range(max_iter):
        posterior = compute_grm_posterior_sparse(
            theta_quad, a_est, b_est, sparse_response, n_categories, weights_quad
        )

        for item_id in range(n_items):
            obs_idx = sparse_response.item_observations(item_id)
            if obs_idx.size == 0:
                continue
            users = sparse_response.user_idx[obs_idx]
            values = sparse_response.values[obs_idx].astype(int)
            k = int(n_categories[item_id])

            def objective(params):
                a_j = params[0]
                b_j = ensure_ordered_thresholds(params[1:k])
                return -compute_grm_item_expected_ll_sparse(
                    theta_quad, a_j, b_j, users, values, k, posterior
                )

            init = np.concatenate([[a_est[item_id]], b_est[item_id]])
            bounds = [(0.2, 3.0)] + [(-4.0, 4.0)] * (k - 1)
            res = minimize(objective, init, method="L-BFGS-B", bounds=bounds)
            if res.success:
                a_est[item_id] = np.clip(res.x[0], 0.2, 3.0)
                b_est[item_id] = ensure_ordered_thresholds(np.clip(res.x[1:k], -4.0, 4.0))

        current_ll = compute_grm_expected_ll_sparse(
            theta_quad, a_est, b_est, sparse_response, n_categories, posterior
        )
        if verbose:
            print(
                f"=== sparse GRM iter {iteration + 1}/{max_iter}, "
                f"delta_ll={current_ll - prev_ll:.6f} ==="
            )
        if iteration > 0 and abs(current_ll - prev_ll) < tol:
            break
        prev_ll = current_ll

    return a_est, b_est


def mirt_em_sparse(
    sparse_response, Q, n_quadrature=27, max_iter=100, tol=1e-4, verbose=False
):
    Q = np.asarray(Q)
    n_items, dim = Q.shape
    a_est = np.random.uniform(0.2, 3.0, (n_items, dim)) * Q
    d_est = np.zeros(n_items)
    theta_quad, weights_quad = create_mirt_quadrature(n_quadrature, dim)
    prev_ll = -np.inf
    total_time = 0.0

    for iteration in range(max_iter):
        start_time = time.time()
        posterior = compute_m2pl_posterior_sparse(
            theta_quad, a_est, d_est, sparse_response, weights_quad
        )
        a_new = a_est.copy()
        d_new = d_est.copy()
        for item_id in range(n_items):
            obs_idx = sparse_response.item_observations(item_id)
            if obs_idx.size == 0:
                continue
            users = sparse_response.user_idx[obs_idx]
            values = sparse_response.values[obs_idx]
            active_dims = Q[item_id] == 1
            num_a = int(np.sum(active_dims))

            def objective(params):
                a_j = np.zeros(dim)
                a_j[active_dims] = params[:num_a]
                d_j = params[num_a]
                return -compute_m2pl_item_expected_ll_sparse(
                    theta_quad, a_j, d_j, users, values, posterior
                )

            init = np.concatenate([a_est[item_id, active_dims], [d_est[item_id]]])
            bounds = [(0.1, None)] * num_a + [(None, None)]
            res = minimize(objective, init, method="L-BFGS-B", bounds=bounds)
            if res.success:
                a_new[item_id, active_dims] = res.x[:num_a]
                d_new[item_id] = res.x[num_a]

        a_est, d_est = a_new, d_new
        current_ll = compute_m2pl_expected_ll_sparse(
            theta_quad, a_est, d_est, sparse_response, posterior
        )
        total_time += time.time() - start_time
        if verbose:
            print(
                f"=== sparse M2PL iter {iteration + 1}/{max_iter}, "
                f"delta_ll={current_ll - prev_ll:.6f}, elapsed={total_time:.2f}s ==="
            )
        if iteration > 0 and abs(current_ll - prev_ll) < tol:
            break
        prev_ll = current_ll

    return a_est, d_est


def estimate_d_only_m2pl_sparse(
    a_params, sparse_response, n_quadrature=27, max_iter=100, tol=1e-4, verbose=False
):
    n_items = sparse_response.n_items
    dim = a_params.shape[1]
    d_est = np.zeros(n_items)
    theta_quad, weights_quad = create_mirt_quadrature(n_quadrature, dim)
    prev_ll = -np.inf
    total_time = 0.0

    for iteration in range(max_iter):
        start_time = time.time()
        posterior = compute_m2pl_posterior_sparse(
            theta_quad, a_params, d_est, sparse_response, weights_quad
        )
        d_new = d_est.copy()
        for item_id in range(n_items):
            obs_idx = sparse_response.item_observations(item_id)
            if obs_idx.size == 0:
                continue
            users = sparse_response.user_idx[obs_idx]
            values = sparse_response.values[obs_idx]

            def objective(d_j):
                return -compute_m2pl_item_expected_ll_sparse(
                    theta_quad, a_params[item_id], d_j[0], users, values, posterior
                )

            res = minimize(objective, [d_est[item_id]], method="L-BFGS-B")
            if res.success:
                d_new[item_id] = res.x[0]

        d_est = d_new
        current_ll = compute_m2pl_expected_ll_sparse(
            theta_quad, a_params, d_est, sparse_response, posterior
        )
        total_time += time.time() - start_time
        if verbose:
            print(
                f"=== sparse d-only iter {iteration + 1}/{max_iter}, "
                f"delta_ll={current_ll - prev_ll:.6f} ==="
            )
        if iteration > 0 and abs(current_ll - prev_ll) < tol:
            break
        prev_ll = current_ll

    return d_est, total_time


def mgrm_em_stepwise_sparse(
    sparse_response,
    Q,
    n_categories,
    n_quadrature=27,
    max_iter=100,
    tol=1e-4,
    verbose=False,
):
    Q = np.asarray(Q)
    n_categories = np.asarray(n_categories)
    max_threshold = int(np.max(n_categories) - 1)

    step1 = sparse_response.with_values((sparse_response.values >= 1).astype(float))
    a_est, d_step1 = mirt_em_sparse(
        step1,
        Q,
        n_quadrature=n_quadrature,
        max_iter=max_iter,
        tol=tol,
        verbose=verbose,
    )
    d_est = [np.array([d_step1[j]]) for j in range(sparse_response.n_items)]

    for threshold in range(2, max_threshold + 1):
        item_mask = n_categories >= threshold + 1
        if not np.any(item_mask):
            continue
        binary_values = (sparse_response.values >= threshold).astype(float)
        subset = sparse_response.subset_items(item_mask, values=binary_values)
        d_k, _ = estimate_d_only_m2pl_sparse(
            a_est[item_mask],
            subset,
            n_quadrature=n_quadrature,
            max_iter=max_iter,
            tol=tol,
            verbose=verbose,
        )
        idx = 0
        for item_id, keep in enumerate(item_mask):
            if keep:
                d_est[item_id] = np.append(d_est[item_id], d_k[idx])
                idx += 1

    d_est = [np.sort(d)[::-1] for d in d_est]
    return a_est, d_est


def mgrm_em_standard_sparse(
    sparse_response,
    Q,
    n_categories,
    n_quadrature=27,
    max_iter=100,
    tol=1e-4,
    verbose=False,
):
    Q = np.asarray(Q)
    n_categories = np.asarray(n_categories)
    n_items, dim = Q.shape
    a_est = np.random.uniform(0.2, 3.0, (n_items, dim)) * Q
    d_est = [np.sort(np.random.normal(0, 1, int(k) - 1))[::-1] for k in n_categories]
    theta_quad, weights_quad = create_mirt_quadrature(n_quadrature, dim)
    prev_ll = -np.inf

    for iteration in range(max_iter):
        posterior = compute_mgrm_posterior_sparse(
            theta_quad, a_est, d_est, sparse_response, n_categories, weights_quad
        )

        for item_id in range(n_items):
            obs_idx = sparse_response.item_observations(item_id)
            if obs_idx.size == 0:
                continue
            users = sparse_response.user_idx[obs_idx]
            values = sparse_response.values[obs_idx].astype(int)
            active_dims = Q[item_id] == 1
            num_a = int(np.sum(active_dims))
            k = int(n_categories[item_id])

            def objective(params):
                a_j = np.zeros(dim)
                a_j[active_dims] = params[:num_a]
                d_j = ensure_descending_thresholds(params[num_a:])
                return -compute_mgrm_item_expected_ll_sparse(
                    theta_quad, a_j, d_j, users, values, k, posterior
                )

            init = np.concatenate([a_est[item_id, active_dims], d_est[item_id]])
            bounds = [(0.1, None)] * num_a + [(-4.0, 4.0)] * (k - 1)
            res = minimize(objective, init, method="L-BFGS-B", bounds=bounds)
            if res.success:
                a_est[item_id, active_dims] = res.x[:num_a]
                d_est[item_id] = ensure_descending_thresholds(res.x[num_a:])

        current_ll = compute_mgrm_expected_ll_sparse(
            theta_quad, a_est, d_est, sparse_response, n_categories, posterior
        )
        if verbose:
            print(
                f"=== sparse MGRM iter {iteration + 1}/{max_iter}, "
                f"delta_ll={current_ll - prev_ll:.6f} ==="
            )
        if iteration > 0 and abs(current_ll - prev_ll) < tol:
            break
        prev_ll = current_ll

    return a_est, d_est


def eap_2pl_sparse(sparse_response, a_params, b_params, quad_points, quad_weights):
    log_lik = compute_2pl_user_loglik_sparse(
        quad_points, a_params, b_params, sparse_response
    )
    return _eap_from_loglik(log_lik, quad_points, quad_weights)


def eap_grm_sparse(
    sparse_response, a_params, b_params, n_categories, quad_points, quad_weights
):
    log_lik = compute_grm_user_loglik_sparse(
        quad_points, a_params, b_params, sparse_response, n_categories
    )
    return _eap_from_loglik(log_lik, quad_points, quad_weights)


def eap_m2pl_sparse(sparse_response, a_params, d_params, quad_points_nd, quad_weights_nd):
    log_lik = compute_m2pl_user_loglik_sparse(
        quad_points_nd, a_params, d_params, sparse_response
    )
    return _eap_from_loglik(log_lik, quad_points_nd, quad_weights_nd)


def eap_mgrm_sparse(
    sparse_response, a_params, d_params, n_categories, quad_points_nd, quad_weights_nd
):
    log_lik = compute_mgrm_user_loglik_sparse(
        quad_points_nd, a_params, d_params, sparse_response, n_categories
    )
    return _eap_from_loglik(log_lik, quad_points_nd, quad_weights_nd)


def compute_2pl_posterior_sparse(theta, a, b, sparse_response, quad_weights):
    log_lik = compute_2pl_user_loglik_sparse(theta, a, b, sparse_response)
    return _posterior_from_loglik(log_lik, quad_weights)


def compute_2pl_user_loglik_sparse(theta, a, b, sparse_response):
    p = _sigmoid(np.outer(theta, a) - np.asarray(a) * np.asarray(b))
    log_obs = _binary_log_obs(p[:, sparse_response.item_idx], sparse_response.values).T
    return _sum_by_user(log_obs, sparse_response)


def compute_2pl_expected_ll_sparse(theta, a, b, sparse_response, posterior):
    user_loglik = compute_2pl_user_loglik_sparse(theta, a, b, sparse_response)
    return float(np.sum(user_loglik * posterior))


def compute_2pl_item_expected_ll_sparse(theta, a_j, b_j, users, values, posterior):
    p = _sigmoid(a_j * (theta - b_j))
    log_obs = _binary_log_obs(p.reshape(-1, 1), values).T
    return float(np.sum(log_obs * posterior[users]))


def compute_m2pl_posterior_sparse(theta, a, d, sparse_response, quad_weights):
    log_lik = compute_m2pl_user_loglik_sparse(theta, a, d, sparse_response)
    return _posterior_from_loglik(log_lik, quad_weights)


def compute_m2pl_user_loglik_sparse(theta, a, d, sparse_response):
    a_obs = a[sparse_response.item_idx]
    logits = theta @ a_obs.T + d[sparse_response.item_idx].reshape(1, -1)
    p = _sigmoid(logits)
    log_obs = _binary_log_obs(p, sparse_response.values).T
    return _sum_by_user(log_obs, sparse_response)


def compute_m2pl_expected_ll_sparse(theta, a, d, sparse_response, posterior):
    user_loglik = compute_m2pl_user_loglik_sparse(theta, a, d, sparse_response)
    return float(np.sum(user_loglik * posterior))


def compute_m2pl_item_expected_ll_sparse(theta, a_j, d_j, users, values, posterior):
    p = _sigmoid(theta @ a_j + d_j)
    log_obs = _binary_log_obs(p.reshape(-1, 1), values).T
    return float(np.sum(log_obs * posterior[users]))


def compute_grm_posterior_sparse(
    theta, a, b, sparse_response, n_categories, quad_weights
):
    log_lik = compute_grm_user_loglik_sparse(theta, a, b, sparse_response, n_categories)
    return _posterior_from_loglik(log_lik, quad_weights)


def compute_grm_user_loglik_sparse(theta, a, b, sparse_response, n_categories):
    b_padded, b_mask = pad_grm_parameters(b, n_categories)
    probs = grm_prob_categories(theta, a, b_padded, b_mask, n_categories)
    values = sparse_response.values.astype(int)
    log_obs = np.log(probs[:, sparse_response.item_idx, values]).T
    return _sum_by_user(log_obs, sparse_response)


def compute_grm_expected_ll_sparse(
    theta, a, b, sparse_response, n_categories, posterior
):
    user_loglik = compute_grm_user_loglik_sparse(theta, a, b, sparse_response, n_categories)
    return float(np.sum(user_loglik * posterior))


def compute_grm_item_expected_ll_sparse(theta, a_j, b_j, users, values, k, posterior):
    b_j = ensure_ordered_thresholds(np.asarray(b_j))
    probs = grm_prob_categories(
        theta,
        np.array([a_j]),
        b_j.reshape(1, -1),
        np.ones((1, len(b_j)), dtype=bool),
        np.array([k]),
    )
    log_obs = np.log(probs[:, 0, values]).T
    return float(np.sum(log_obs * posterior[users]))


def compute_mgrm_posterior_sparse(
    theta, a, d, sparse_response, n_categories, quad_weights
):
    log_lik = compute_mgrm_user_loglik_sparse(
        theta, a, d, sparse_response, n_categories
    )
    return _posterior_from_loglik(log_lik, quad_weights)


def compute_mgrm_user_loglik_sparse(theta, a, d, sparse_response, n_categories):
    d_padded, d_mask = pad_grm_parameters(d, n_categories)
    probs = mgrm_prob_categories(theta, a, d_padded, d_mask, n_categories)
    values = sparse_response.values.astype(int)
    log_obs = np.log(probs[:, sparse_response.item_idx, values]).T
    return _sum_by_user(log_obs, sparse_response)


def compute_mgrm_expected_ll_sparse(
    theta, a, d, sparse_response, n_categories, posterior
):
    user_loglik = compute_mgrm_user_loglik_sparse(
        theta, a, d, sparse_response, n_categories
    )
    return float(np.sum(user_loglik * posterior))


def compute_mgrm_item_expected_ll_sparse(theta, a_j, d_j, users, values, k, posterior):
    d_j = ensure_descending_thresholds(np.asarray(d_j))
    probs = mgrm_prob_categories(
        theta,
        np.array([a_j]),
        d_j.reshape(1, -1),
        np.ones((1, len(d_j)), dtype=bool),
        np.array([k]),
    )
    log_obs = np.log(probs[:, 0, values]).T
    return float(np.sum(log_obs * posterior[users]))


def ensure_descending_thresholds(thresholds):
    if len(thresholds) <= 1:
        return np.asarray(thresholds)
    ordered = np.sort(np.asarray(thresholds).copy())[::-1]
    ordered = np.clip(ordered, -4.0, 4.0)
    for i in range(1, len(ordered)):
        if ordered[i - 1] - ordered[i] < 0.01:
            ordered[i] = ordered[i - 1] - 0.01
    return np.clip(ordered, -4.0, 4.0)


def _sum_by_user(log_obs, sparse_response):
    user_loglik = np.zeros((sparse_response.n_users, log_obs.shape[1]))
    np.add.at(user_loglik, sparse_response.user_idx, log_obs)
    return user_loglik


def _posterior_from_loglik(log_lik, weights):
    log_post = log_lik + np.log(weights).reshape(1, -1)
    max_log = np.max(log_post, axis=1, keepdims=True)
    scaled = np.exp(log_post - max_log)
    denom = np.sum(scaled, axis=1, keepdims=True)
    denom = np.where(denom < 1e-15, 1e-15, denom)
    return scaled / denom


def _eap_from_loglik(log_lik, quad_points, quad_weights):
    log_post = log_lik + np.log(quad_weights).reshape(1, -1)
    max_log = np.max(log_post, axis=1, keepdims=True)
    scaled = np.exp(log_post - max_log)
    denom = np.sum(scaled, axis=1, keepdims=True)
    denom = np.where(denom < 1e-15, 1e-15, denom)
    if quad_points.ndim == 1:
        return (scaled @ quad_points.reshape(-1, 1)).ravel() / denom.ravel()
    return (scaled @ quad_points) / denom


def _binary_log_obs(prob_by_theta, values):
    prob_by_theta = np.clip(prob_by_theta, 1e-15, 1.0 - 1e-15)
    values = values.reshape(1, -1)
    return values * np.log(prob_by_theta) + (1.0 - values) * np.log(1.0 - prob_by_theta)


def _sigmoid(logits):
    logits = np.clip(logits, -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-logits))
