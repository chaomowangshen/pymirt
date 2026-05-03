import time

import numpy as np
from scipy.optimize import minimize
from scipy.stats import multivariate_normal, norm

from .sparse_item_est import ensure_descending_thresholds
from .units import pad_grm_parameters


def compute_m2pl_user_loglik_state_sparse(theta, a, d, sparse_response):
    obs_loglik = _m2pl_observation_loglik(
        theta[sparse_response.user_idx],
        a[sparse_response.item_idx],
        d[sparse_response.item_idx],
        sparse_response.values,
    )
    return _sum_by_index(obs_loglik, sparse_response.user_idx, sparse_response.n_users)


def compute_m2pl_item_loglik_state_sparse(theta, a, d, sparse_response):
    obs_loglik = _m2pl_observation_loglik(
        theta[sparse_response.user_idx],
        a[sparse_response.item_idx],
        d[sparse_response.item_idx],
        sparse_response.values,
    )
    return _sum_by_index(obs_loglik, sparse_response.item_idx, sparse_response.n_items)


def compute_mgrm_user_loglik_state_sparse(
    theta, a, d, sparse_response, n_categories, d_mask=None
):
    d_matrix, mask = _mgrm_matrix_and_mask(d, n_categories, d_mask)
    obs_loglik = _mgrm_observation_loglik(
        theta[sparse_response.user_idx],
        a[sparse_response.item_idx],
        d_matrix[sparse_response.item_idx],
        mask[sparse_response.item_idx],
        np.asarray(n_categories)[sparse_response.item_idx],
        sparse_response.values.astype(int),
    )
    return _sum_by_index(obs_loglik, sparse_response.user_idx, sparse_response.n_users)


def compute_mgrm_item_loglik_state_sparse(
    theta, a, d, sparse_response, n_categories, d_mask=None
):
    d_matrix, mask = _mgrm_matrix_and_mask(d, n_categories, d_mask)
    obs_loglik = _mgrm_observation_loglik(
        theta[sparse_response.user_idx],
        a[sparse_response.item_idx],
        d_matrix[sparse_response.item_idx],
        mask[sparse_response.item_idx],
        np.asarray(n_categories)[sparse_response.item_idx],
        sparse_response.values.astype(int),
    )
    return _sum_by_index(obs_loglik, sparse_response.item_idx, sparse_response.n_items)


def mcmc_sampling_sparse(
    theta,
    a,
    d,
    sparse_response,
    rv,
    step_sizes,
    burn_in,
    n_samples,
    method="m2pl",
    n_categories=None,
    d_mask=None,
):
    n, dim = theta.shape
    step_sizes = np.asarray(step_sizes, dtype=float).copy()
    theta_curr = theta.copy()
    samples = np.zeros((n, n_samples, dim))
    all_accept = np.zeros((n, burn_in + n_samples), dtype=bool)

    for sample_id in range(burn_in + n_samples):
        proposals = theta_curr + np.random.normal(0, step_sizes[:, None], (n, dim))
        if method == "m2pl":
            ll_curr = compute_m2pl_user_loglik_state_sparse(
                theta_curr, a, d, sparse_response
            )
            ll_prop = compute_m2pl_user_loglik_state_sparse(
                proposals, a, d, sparse_response
            )
        elif method == "mgrm":
            ll_curr = compute_mgrm_user_loglik_state_sparse(
                theta_curr, a, d, sparse_response, n_categories, d_mask=d_mask
            )
            ll_prop = compute_mgrm_user_loglik_state_sparse(
                proposals, a, d, sparse_response, n_categories, d_mask=d_mask
            )
        else:
            raise ValueError("method must be 'm2pl' or 'mgrm'.")

        prior_curr = rv.logpdf(theta_curr)
        prior_prop = rv.logpdf(proposals)
        log_accept = np.minimum(ll_prop + prior_prop - ll_curr - prior_curr, 0.0)
        accept = np.random.rand(n) < np.exp(log_accept)
        theta_curr[accept] = proposals[accept]
        all_accept[:, sample_id] = accept
        if sample_id >= burn_in:
            samples[:, sample_id - burn_in, :] = theta_curr.copy()

    post_burn_accept = all_accept[:, burn_in:]
    accept_rates = np.mean(post_burn_accept, axis=1)
    step_sizes[accept_rates < 0.2] *= 0.9
    step_sizes[accept_rates > 0.4] *= 1.1
    step_sizes = np.clip(step_sizes, 0.05, 0.5)
    return samples, theta_curr, step_sizes, all_accept


def update_theta_parameters_sparse(
    rv, theta, a, d, sparse_response, step_size=0.2
):
    theta_curr = theta.copy()
    theta_prop = theta_curr + np.random.normal(0, step_size, size=theta_curr.shape)
    ll_curr = compute_m2pl_user_loglik_state_sparse(
        theta_curr, a, d, sparse_response
    )
    ll_prop = compute_m2pl_user_loglik_state_sparse(theta_prop, a, d, sparse_response)
    prior_curr = rv.logpdf(theta_curr)
    prior_prop = rv.logpdf(theta_prop)
    log_alpha = np.minimum((ll_prop - ll_curr) + (prior_prop - prior_curr), 0.0)
    accept_flag = np.random.rand(theta.shape[0]) < np.exp(log_alpha)
    theta_new = np.where(accept_flag[:, None], theta_prop, theta_curr)
    return theta_new, _adapt_scalar_step(step_size, np.mean(accept_flag))


def update_a_parameters_sparse(
    rv, theta, a, d, sparse_response, Q, step_size=0.2
):
    a_curr = a.copy()
    ll_curr = compute_m2pl_item_loglik_state_sparse(
        theta, a_curr, d, sparse_response
    )
    log_a_curr = np.log(np.maximum(a_curr, 1e-9))
    log_a_prop = log_a_curr.copy()
    active = Q == 1
    log_a_prop[active] += np.random.normal(0, step_size, size=np.sum(active))
    a_prop = np.exp(log_a_prop)
    a_prop[Q == 0] = 0.0
    ll_prop = compute_m2pl_item_loglik_state_sparse(theta, a_prop, d, sparse_response)
    prior_curr = np.sum(rv.logpdf(log_a_curr) * Q, axis=1)
    prior_prop = np.sum(rv.logpdf(log_a_prop) * Q, axis=1)
    log_alpha = np.minimum((ll_prop - ll_curr) + (prior_prop - prior_curr), 0.0)
    accept_flag = np.random.rand(a.shape[0]) < np.exp(log_alpha)
    a_new = np.where(accept_flag[:, None], a_prop, a_curr)
    return a_new, _adapt_scalar_step(step_size, np.mean(accept_flag))


def update_d_parameters_sparse(
    rv, theta, a, d, sparse_response, step_size=0.2
):
    d_curr = d.copy()
    ll_curr = compute_m2pl_item_loglik_state_sparse(
        theta, a, d_curr, sparse_response
    )
    d_prop = d_curr + np.random.normal(0, step_size, size=d_curr.shape)
    ll_prop = compute_m2pl_item_loglik_state_sparse(theta, a, d_prop, sparse_response)
    prior_curr = rv.logpdf(d_curr)
    prior_prop = rv.logpdf(d_prop)
    log_alpha = np.minimum((ll_prop - ll_curr) + (prior_prop - prior_curr), 0.0)
    accept_flag = np.random.rand(d.shape[0]) < np.exp(log_alpha)
    d_new = np.where(accept_flag, d_prop, d_curr)
    return d_new, _adapt_scalar_step(step_size, np.mean(accept_flag))


def update_grm_theta_parameters_sparse(
    rv, theta, a, d, d_mask, sparse_response, n_categories, step_size=0.2
):
    theta_curr = theta.copy()
    theta_prop = theta_curr + np.random.normal(0, step_size, size=theta_curr.shape)
    ll_curr = compute_mgrm_user_loglik_state_sparse(
        theta_curr, a, d, sparse_response, n_categories, d_mask=d_mask
    )
    ll_prop = compute_mgrm_user_loglik_state_sparse(
        theta_prop, a, d, sparse_response, n_categories, d_mask=d_mask
    )
    prior_curr = rv.logpdf(theta_curr)
    prior_prop = rv.logpdf(theta_prop)
    log_alpha = np.minimum((ll_prop - ll_curr) + (prior_prop - prior_curr), 0.0)
    accept_flag = np.random.rand(theta.shape[0]) < np.exp(log_alpha)
    theta_new = np.where(accept_flag[:, None], theta_prop, theta_curr)
    return theta_new, _adapt_scalar_step(step_size, np.mean(accept_flag))


def update_grm_a_parameters_sparse(
    rv, theta, a, d, d_mask, sparse_response, Q, n_categories, step_size=0.2
):
    a_curr = a.copy()
    ll_curr = compute_mgrm_item_loglik_state_sparse(
        theta, a_curr, d, sparse_response, n_categories, d_mask=d_mask
    )
    log_a_curr = np.log(np.maximum(a_curr, 1e-9))
    log_a_prop = log_a_curr.copy()
    active = Q == 1
    log_a_prop[active] += np.random.normal(0, step_size, size=np.sum(active))
    a_prop = np.exp(log_a_prop)
    a_prop[Q == 0] = 0.0
    ll_prop = compute_mgrm_item_loglik_state_sparse(
        theta, a_prop, d, sparse_response, n_categories, d_mask=d_mask
    )
    prior_curr = np.sum(rv.logpdf(log_a_curr) * Q, axis=1)
    prior_prop = np.sum(rv.logpdf(log_a_prop) * Q, axis=1)
    log_alpha = np.minimum((ll_prop - ll_curr) + (prior_prop - prior_curr), 0.0)
    accept_flag = np.random.rand(a.shape[0]) < np.exp(log_alpha)
    a_new = np.where(accept_flag[:, None], a_prop, a_curr)
    return a_new, _adapt_scalar_step(step_size, np.mean(accept_flag))


def update_grm_d_parameters_sparse(
    rv, theta, a, d, d_mask, sparse_response, n_categories, step_size=0.2
):
    d_curr = d.copy()
    delta_curr = _d_to_delta_masked(d_curr, d_mask)
    delta_prop = delta_curr + np.random.normal(0, step_size, size=delta_curr.shape) * d_mask
    d_prop = _delta_to_d_masked(delta_prop, d_mask)

    ll_curr = compute_mgrm_item_loglik_state_sparse(
        theta, a, d_curr, sparse_response, n_categories, d_mask=d_mask
    )
    ll_prop = compute_mgrm_item_loglik_state_sparse(
        theta, a, d_prop, sparse_response, n_categories, d_mask=d_mask
    )
    prior_curr = np.sum(rv.logpdf(d_curr) * d_mask, axis=1)
    prior_prop = np.sum(rv.logpdf(d_prop) * d_mask, axis=1)
    jac_curr = np.sum(delta_curr[:, 1:] * d_mask[:, 1:], axis=1)
    jac_prop = np.sum(delta_prop[:, 1:] * d_mask[:, 1:], axis=1)
    log_alpha = np.minimum(
        (ll_prop - ll_curr) + (prior_prop - prior_curr) + (jac_prop - jac_curr),
        0.0,
    )
    accept_flag = np.random.rand(d.shape[0]) < np.exp(log_alpha)
    d_new = np.where(accept_flag[:, None], d_prop, d_curr)
    return d_new, _adapt_scalar_step(step_size, np.mean(accept_flag))


def mirt_mcmc_sparse(sparse_response, Q, n_samples=3000, burn_in=2000, verbose=False):
    start_time = time.time()
    Q = np.asarray(Q)
    n_items, dim = Q.shape
    a = np.random.uniform(0.2, 3.0, (n_items, dim)) * Q
    d = np.random.normal(0, 1, n_items)
    theta = np.zeros((sparse_response.n_users, dim))
    rv_log_a = norm(loc=0, scale=np.sqrt(0.5))
    rv_d = norm(loc=0, scale=1)
    rv_theta = multivariate_normal(mean=np.zeros(dim), cov=np.eye(dim))
    samples_a = np.zeros((n_items, n_samples, dim))
    samples_d = np.zeros((n_items, n_samples))
    samples_theta = np.zeros((sparse_response.n_users, n_samples, dim))
    step_size_a = step_size_d = step_size_theta = 0.2

    for iteration in range(n_samples + burn_in):
        if verbose and iteration % 500 == 0:
            print(f"=== sparse M2PL MCMC iter {iteration + 1}/{n_samples + burn_in} ===")
        theta, step_size_theta = update_theta_parameters_sparse(
            rv_theta, theta, a, d, sparse_response, step_size=step_size_theta
        )
        a, step_size_a = update_a_parameters_sparse(
            rv_log_a, theta, a, d, sparse_response, Q, step_size=step_size_a
        )
        d, step_size_d = update_d_parameters_sparse(
            rv_d, theta, a, d, sparse_response, step_size=step_size_d
        )
        if iteration >= burn_in:
            sample_idx = iteration - burn_in
            samples_a[:, sample_idx, :] = a
            samples_d[:, sample_idx] = d
            samples_theta[:, sample_idx, :] = theta

    if verbose:
        print(f"=== sparse M2PL MCMC done, elapsed={time.time() - start_time:.2f}s ===")
    return (
        np.mean(samples_a, axis=1),
        np.mean(samples_d, axis=1),
        np.mean(samples_theta, axis=1),
    )


def mirt_mcem_sparse(
    sparse_response,
    Q,
    n_samples=300,
    burn_in=200,
    max_iter=100,
    tol=1e-4,
    sample_interval=10,
    verbose=False,
):
    Q = np.asarray(Q)
    n_items, dim = Q.shape
    a_est = np.random.uniform(0.2, 3.0, (n_items, dim)) * Q
    d_est = np.zeros(n_items)
    theta_est = np.zeros((sparse_response.n_users, dim))
    step_sizes = np.full(sparse_response.n_users, 0.2)
    rv = multivariate_normal(mean=np.zeros(dim), cov=np.eye(dim))
    prev_ll = -np.inf
    samples = None
    theta_curr = theta_est.copy()

    for iteration in range(max_iter):
        sample_this_iter = iteration < 10 or iteration % sample_interval == 0
        if sample_this_iter or samples is None:
            samples, theta_curr, step_sizes, _ = mcmc_sampling_sparse(
                theta_est,
                a_est,
                d_est,
                sparse_response,
                rv,
                step_sizes,
                burn_in,
                n_samples,
                method="m2pl",
            )
            theta_est = np.mean(samples, axis=1)

        a_new = a_est.copy()
        d_new = d_est.copy()
        sample_count = min(n_samples, 100)
        for item_id in range(n_items):
            users, values = _item_users_values(sparse_response, item_id)
            theta_item = samples[users, -sample_count:, :].reshape(-1, dim)
            values_item = np.repeat(values, sample_count)
            active_dims = Q[item_id] == 1
            a_j, d_j, success = _fit_m2pl_item_from_theta(
                theta_item,
                values_item,
                active_dims,
                a_est[item_id],
                d_est[item_id],
            )
            if success:
                a_new[item_id] = a_j
                d_new[item_id] = d_j

        a_est, d_est = a_new, d_new
        current_ll = np.sum(
            compute_m2pl_user_loglik_state_sparse(theta_curr, a_est, d_est, sparse_response)
        )
        if verbose:
            print(
                f"=== sparse M2PL MCEM iter {iteration + 1}/{max_iter}, "
                f"delta_ll={current_ll - prev_ll:.6f} ==="
            )
        if iteration > 0 and abs(current_ll - prev_ll) < tol:
            break
        prev_ll = current_ll

    return a_est, d_est, theta_est


def mirt_saem_sparse(sparse_response, Q, max_iter=100, tol=1e-4, verbose=False):
    Q = np.asarray(Q)
    n_items, dim = Q.shape
    a_est = np.random.uniform(0.2, 3.0, (n_items, dim)) * Q
    d_est = np.zeros(n_items)
    theta_curr = np.zeros((sparse_response.n_users, dim))
    step_sizes = np.full(sparse_response.n_users, 0.2)
    rv = multivariate_normal(mean=np.zeros(dim), cov=np.eye(dim))
    gamma_sequence = np.ones(max_iter)
    burn_in = 50
    n_samples = 10
    for k in range(burn_in, max_iter):
        gamma_sequence[k] = 1.0 / (k - burn_in + 1) ** 0.7
    prev_ll = -np.inf

    for iteration in range(max_iter):
        gamma = gamma_sequence[iteration]
        samples, theta_curr, step_sizes, _ = mcmc_sampling_sparse(
            theta_curr,
            a_est,
            d_est,
            sparse_response,
            rv,
            step_sizes,
            burn_in,
            n_samples,
            method="m2pl",
        )
        theta_mean = np.mean(samples, axis=1)
        a_new = a_est.copy()
        d_new = d_est.copy()
        for item_id in range(n_items):
            users, values = _item_users_values(sparse_response, item_id)
            active_dims = Q[item_id] == 1
            a_j, d_j, success = _fit_m2pl_item_from_theta(
                theta_curr[users],
                values,
                active_dims,
                a_est[item_id],
                d_est[item_id],
            )
            if success:
                a_new[item_id, active_dims] = (
                    (1 - gamma) * a_est[item_id, active_dims]
                    + gamma * a_j[active_dims]
                )
                d_new[item_id] = (1 - gamma) * d_est[item_id] + gamma * d_j
        a_est, d_est = a_new, d_new
        current_ll = np.sum(
            compute_m2pl_user_loglik_state_sparse(theta_curr, a_est, d_est, sparse_response)
        )
        if verbose:
            print(
                f"=== sparse M2PL SAEM iter {iteration + 1}/{max_iter}, "
                f"delta_ll={current_ll - prev_ll:.6f} ==="
            )
        if iteration > 0 and abs(current_ll - prev_ll) < tol:
            break
        prev_ll = current_ll

    return a_est, d_est, theta_mean


def mgrm_mcmc_standard_sparse(
    sparse_response, Q, n_categories, n_samples=3000, burn_in=2000, verbose=False
):
    Q = np.asarray(Q)
    n_categories = np.asarray(n_categories)
    n_items, dim = Q.shape
    max_k = int(np.max(n_categories) - 1)
    a_est = np.random.uniform(0.2, 3.0, (n_items, dim)) * Q
    d_est = [np.sort(np.random.normal(0, 1, int(k) - 1))[::-1] for k in n_categories]
    d_matrix, d_mask = pad_grm_parameters(d_est, n_categories)
    theta_est = np.zeros((sparse_response.n_users, dim))
    rv_log_a = norm(loc=0, scale=np.sqrt(0.5))
    rv_d = norm(loc=0, scale=1.5)
    rv_theta = multivariate_normal(mean=np.zeros(dim), cov=np.eye(dim))
    samples_a = np.zeros((n_items, n_samples, dim))
    samples_d = np.zeros((n_items, n_samples, max_k))
    samples_theta = np.zeros((sparse_response.n_users, n_samples, dim))
    step_size_a = step_size_d = step_size_theta = 0.2

    for iteration in range(n_samples + burn_in):
        if verbose and iteration % 500 == 0:
            print(f"=== sparse MGRM MCMC iter {iteration + 1}/{n_samples + burn_in} ===")
        theta_est, step_size_theta = update_grm_theta_parameters_sparse(
            rv_theta,
            theta_est,
            a_est,
            d_matrix,
            d_mask,
            sparse_response,
            n_categories,
            step_size=step_size_theta,
        )
        theta_est = theta_est - np.mean(theta_est, axis=0)
        a_est, step_size_a = update_grm_a_parameters_sparse(
            rv_log_a,
            theta_est,
            a_est,
            d_matrix,
            d_mask,
            sparse_response,
            Q,
            n_categories,
            step_size=step_size_a,
        )
        d_matrix, step_size_d = update_grm_d_parameters_sparse(
            rv_d,
            theta_est,
            a_est,
            d_matrix,
            d_mask,
            sparse_response,
            n_categories,
            step_size=step_size_d,
        )
        if iteration >= burn_in:
            sample_idx = iteration - burn_in
            samples_a[:, sample_idx, :] = a_est
            samples_d[:, sample_idx, :] = d_matrix
            samples_theta[:, sample_idx, :] = theta_est

    a_est = np.mean(samples_a, axis=1)
    d_matrix = np.mean(samples_d, axis=1)
    d_est = [d_matrix[j, : n_categories[j] - 1] for j in range(n_items)]
    theta_est = np.mean(samples_theta, axis=1)
    return a_est, d_est, theta_est


def mgrm_mcem_standard_sparse(
    sparse_response,
    Q,
    n_categories,
    n_samples=300,
    burn_in=200,
    max_iter=100,
    tol=1e-4,
    sample_interval=10,
    verbose=False,
):
    Q = np.asarray(Q)
    n_categories = np.asarray(n_categories)
    n_items, dim = Q.shape
    a_est = np.random.uniform(0.2, 3.0, (n_items, dim)) * Q
    d_est = [np.sort(np.random.normal(0, 1, int(k) - 1))[::-1] for k in n_categories]
    theta_est = np.zeros((sparse_response.n_users, dim))
    step_sizes = np.full(sparse_response.n_users, 0.2)
    rv = multivariate_normal(mean=np.zeros(dim), cov=np.eye(dim))
    prev_ll = -np.inf
    samples = None
    theta_curr = theta_est.copy()

    for iteration in range(max_iter):
        sample_this_iter = iteration < 10 or iteration % sample_interval == 0
        if sample_this_iter or samples is None:
            samples, theta_curr, step_sizes, _ = mcmc_sampling_sparse(
                theta_est,
                a_est,
                d_est,
                sparse_response,
                rv,
                step_sizes,
                burn_in,
                n_samples,
                method="mgrm",
                n_categories=n_categories,
            )
            theta_est = np.mean(samples, axis=1)

        a_new = a_est.copy()
        d_new = [d.copy() for d in d_est]
        sample_count = min(n_samples, 100)
        for item_id in range(n_items):
            users, values = _item_users_values(sparse_response, item_id)
            theta_item = samples[users, -sample_count:, :].reshape(-1, dim)
            values_item = np.repeat(values.astype(int), sample_count)
            active_dims = Q[item_id] == 1
            a_j, d_j, success = _fit_mgrm_item_from_theta(
                theta_item,
                values_item,
                active_dims,
                a_est[item_id],
                d_est[item_id],
                int(n_categories[item_id]),
            )
            if success:
                a_new[item_id] = a_j
                d_new[item_id] = d_j
        a_est, d_est = a_new, d_new
        current_ll = np.sum(
            compute_mgrm_user_loglik_state_sparse(
                theta_curr, a_est, d_est, sparse_response, n_categories
            )
        )
        if verbose:
            print(
                f"=== sparse MGRM MCEM iter {iteration + 1}/{max_iter}, "
                f"delta_ll={current_ll - prev_ll:.6f} ==="
            )
        if iteration > 0 and abs(current_ll - prev_ll) < tol:
            break
        prev_ll = current_ll

    return a_est, d_est, theta_est


def mgrm_saem_standard_sparse(
    sparse_response, Q, n_categories, max_iter=100, tol=1e-4, verbose=False
):
    Q = np.asarray(Q)
    n_categories = np.asarray(n_categories)
    n_items, dim = Q.shape
    a_est = np.random.uniform(0.2, 3.0, (n_items, dim)) * Q
    d_est = [np.sort(np.random.normal(0, 1, int(k) - 1))[::-1] for k in n_categories]
    theta_curr = np.zeros((sparse_response.n_users, dim))
    step_sizes = np.full(sparse_response.n_users, 0.2)
    rv = multivariate_normal(mean=np.zeros(dim), cov=np.eye(dim))
    burn_in = 50
    n_samples = 10
    gamma_sequence = np.ones(max_iter)
    for k in range(burn_in, max_iter):
        gamma_sequence[k] = 1.0 / (k - burn_in + 1) ** 0.7
    prev_ll = -np.inf

    for iteration in range(max_iter):
        gamma = gamma_sequence[iteration]
        samples, theta_curr, step_sizes, _ = mcmc_sampling_sparse(
            theta_curr,
            a_est,
            d_est,
            sparse_response,
            rv,
            step_sizes,
            burn_in,
            n_samples,
            method="mgrm",
            n_categories=n_categories,
        )
        theta_mean = np.mean(samples, axis=1)
        a_new = a_est.copy()
        d_new = [d.copy() for d in d_est]
        for item_id in range(n_items):
            users, values = _item_users_values(sparse_response, item_id)
            active_dims = Q[item_id] == 1
            a_j, d_j, success = _fit_mgrm_item_from_theta(
                theta_curr[users],
                values.astype(int),
                active_dims,
                a_est[item_id],
                d_est[item_id],
                int(n_categories[item_id]),
            )
            if success:
                a_new[item_id, active_dims] = (
                    (1 - gamma) * a_est[item_id, active_dims]
                    + gamma * a_j[active_dims]
                )
                d_new[item_id] = (1 - gamma) * d_est[item_id] + gamma * d_j
        a_est, d_est = a_new, d_new
        current_ll = np.sum(
            compute_mgrm_user_loglik_state_sparse(
                theta_curr, a_est, d_est, sparse_response, n_categories
            )
        )
        if verbose:
            print(
                f"=== sparse MGRM SAEM iter {iteration + 1}/{max_iter}, "
                f"delta_ll={current_ll - prev_ll:.6f} ==="
            )
        if iteration > 0 and abs(current_ll - prev_ll) < tol:
            break
        prev_ll = current_ll

    return a_est, d_est, theta_mean


def mgrm_mcmc_stepwise_sparse(
    sparse_response, Q, n_categories, n_samples=3000, burn_in=2000, verbose=False
):
    n_categories = np.asarray(n_categories)
    max_threshold = int(np.max(n_categories) - 1)
    step1 = sparse_response.with_values((sparse_response.values >= 1).astype(float))
    a_est, d_step1, theta_est = mirt_mcmc_sparse(
        step1, Q, n_samples=n_samples, burn_in=burn_in, verbose=verbose
    )
    d_est = [np.array([d_step1[j]]) for j in range(sparse_response.n_items)]

    for threshold in range(2, max_threshold + 1):
        item_mask = n_categories >= threshold + 1
        if not np.any(item_mask):
            continue
        binary_values = (sparse_response.values >= threshold).astype(float)
        subset = sparse_response.subset_items(item_mask, values=binary_values)
        d_k, _ = estimate_d_only_mcmc_sparse(
            a_est[item_mask],
            theta_est,
            subset,
            n_samples=n_samples,
            burn_in=burn_in,
            verbose=verbose,
        )
        idx = 0
        for item_id, keep in enumerate(item_mask):
            if keep:
                d_est[item_id] = np.append(d_est[item_id], d_k[idx])
                idx += 1

    d_est = [np.sort(d)[::-1] for d in d_est]
    rv = multivariate_normal(mean=np.zeros(Q.shape[1]), cov=np.eye(Q.shape[1]))
    step_sizes = np.full(sparse_response.n_users, 0.2)
    samples, _, _, _ = mcmc_sampling_sparse(
        theta_est,
        a_est,
        d_est,
        sparse_response,
        rv,
        step_sizes,
        burn_in,
        n_samples,
        method="mgrm",
        n_categories=n_categories,
    )
    return a_est, d_est, np.mean(samples, axis=1)


def mgrm_mcem_stepwise_sparse(
    sparse_response,
    Q,
    n_categories,
    n_samples=300,
    burn_in=200,
    max_iter=100,
    tol=1e-4,
    sample_interval=10,
    verbose=False,
):
    n_categories = np.asarray(n_categories)
    max_threshold = int(np.max(n_categories) - 1)
    step1 = sparse_response.with_values((sparse_response.values >= 1).astype(float))
    a_est, d_step1, theta_est = mirt_mcem_sparse(
        step1,
        Q,
        n_samples=n_samples,
        burn_in=burn_in,
        max_iter=max_iter,
        tol=tol,
        sample_interval=sample_interval,
        verbose=verbose,
    )
    d_est = [np.array([d_step1[j]]) for j in range(sparse_response.n_items)]

    for threshold in range(2, max_threshold + 1):
        item_mask = n_categories >= threshold + 1
        if not np.any(item_mask):
            continue
        binary_values = (sparse_response.values >= threshold).astype(float)
        subset = sparse_response.subset_items(item_mask, values=binary_values)
        d_k, _ = estimate_d_only_from_theta_sparse(a_est[item_mask], theta_est, subset)
        idx = 0
        for item_id, keep in enumerate(item_mask):
            if keep:
                d_est[item_id] = np.append(d_est[item_id], d_k[idx])
                idx += 1

    return a_est, [np.sort(d)[::-1] for d in d_est], theta_est


def mgrm_saem_stepwise_sparse(
    sparse_response, Q, n_categories, max_iter=100, tol=1e-4, verbose=False
):
    n_categories = np.asarray(n_categories)
    max_threshold = int(np.max(n_categories) - 1)
    step1 = sparse_response.with_values((sparse_response.values >= 1).astype(float))
    a_est, d_step1, theta_est = mirt_saem_sparse(
        step1, Q, max_iter=max_iter, tol=tol, verbose=verbose
    )
    d_est = [np.array([d_step1[j]]) for j in range(sparse_response.n_items)]

    for threshold in range(2, max_threshold + 1):
        item_mask = n_categories >= threshold + 1
        if not np.any(item_mask):
            continue
        binary_values = (sparse_response.values >= threshold).astype(float)
        subset = sparse_response.subset_items(item_mask, values=binary_values)
        d_k, _ = estimate_d_only_from_theta_sparse(a_est[item_mask], theta_est, subset)
        idx = 0
        for item_id, keep in enumerate(item_mask):
            if keep:
                d_est[item_id] = np.append(d_est[item_id], d_k[idx])
                idx += 1

    return a_est, [np.sort(d)[::-1] for d in d_est], theta_est


def irt_mcmc_sparse(sparse_response, n_samples=3000, burn_in=2000, verbose=False):
    """Estimate a single-dimensional 2PL model via the sparse M2PL sampler."""

    a_matrix, d_est, theta_matrix = mirt_mcmc_sparse(
        sparse_response,
        _unit_q(sparse_response.n_items),
        n_samples=n_samples,
        burn_in=burn_in,
        verbose=verbose,
    )
    return _m2pl_to_2pl_parameters(a_matrix, d_est, theta_matrix)


def irt_mcem_sparse(
    sparse_response,
    n_samples=300,
    burn_in=200,
    max_iter=100,
    tol=1e-4,
    sample_interval=10,
    verbose=False,
):
    """Estimate a single-dimensional 2PL model via sparse MCEM."""

    a_matrix, d_est, theta_matrix = mirt_mcem_sparse(
        sparse_response,
        _unit_q(sparse_response.n_items),
        n_samples=n_samples,
        burn_in=burn_in,
        max_iter=max_iter,
        tol=tol,
        sample_interval=sample_interval,
        verbose=verbose,
    )
    return _m2pl_to_2pl_parameters(a_matrix, d_est, theta_matrix)


def irt_saem_sparse(sparse_response, max_iter=100, tol=1e-4, verbose=False):
    """Estimate a single-dimensional 2PL model via sparse SAEM."""

    a_matrix, d_est, theta_matrix = mirt_saem_sparse(
        sparse_response,
        _unit_q(sparse_response.n_items),
        max_iter=max_iter,
        tol=tol,
        verbose=verbose,
    )
    return _m2pl_to_2pl_parameters(a_matrix, d_est, theta_matrix)


def grm_mcmc_stepwise_sparse(
    sparse_response, n_categories, n_samples=3000, burn_in=2000, verbose=False
):
    """Estimate a single-dimensional GRM stepwise model via sparse MGRM MCMC."""

    a_matrix, d_est, theta_matrix = mgrm_mcmc_stepwise_sparse(
        sparse_response,
        _unit_q(sparse_response.n_items),
        n_categories,
        n_samples=n_samples,
        burn_in=burn_in,
        verbose=verbose,
    )
    return _mgrm_to_grm_parameters(a_matrix, d_est, theta_matrix)


def grm_mcem_stepwise_sparse(
    sparse_response,
    n_categories,
    n_samples=300,
    burn_in=200,
    max_iter=100,
    tol=1e-4,
    sample_interval=10,
    verbose=False,
):
    """Estimate a single-dimensional GRM stepwise model via sparse MCEM."""

    a_matrix, d_est, theta_matrix = mgrm_mcem_stepwise_sparse(
        sparse_response,
        _unit_q(sparse_response.n_items),
        n_categories,
        n_samples=n_samples,
        burn_in=burn_in,
        max_iter=max_iter,
        tol=tol,
        sample_interval=sample_interval,
        verbose=verbose,
    )
    return _mgrm_to_grm_parameters(a_matrix, d_est, theta_matrix)


def grm_saem_stepwise_sparse(
    sparse_response, n_categories, max_iter=100, tol=1e-4, verbose=False
):
    """Estimate a single-dimensional GRM stepwise model via sparse SAEM."""

    a_matrix, d_est, theta_matrix = mgrm_saem_stepwise_sparse(
        sparse_response,
        _unit_q(sparse_response.n_items),
        n_categories,
        max_iter=max_iter,
        tol=tol,
        verbose=verbose,
    )
    return _mgrm_to_grm_parameters(a_matrix, d_est, theta_matrix)


def grm_mcmc_standard_sparse(
    sparse_response, n_categories, n_samples=3000, burn_in=2000, verbose=False
):
    """Estimate a single-dimensional GRM standard model via sparse MGRM MCMC."""

    a_matrix, d_est, theta_matrix = mgrm_mcmc_standard_sparse(
        sparse_response,
        _unit_q(sparse_response.n_items),
        n_categories,
        n_samples=n_samples,
        burn_in=burn_in,
        verbose=verbose,
    )
    return _mgrm_to_grm_parameters(a_matrix, d_est, theta_matrix)


def grm_mcem_standard_sparse(
    sparse_response,
    n_categories,
    n_samples=300,
    burn_in=200,
    max_iter=100,
    tol=1e-4,
    sample_interval=10,
    verbose=False,
):
    """Estimate a single-dimensional GRM standard model via sparse MCEM."""

    a_matrix, d_est, theta_matrix = mgrm_mcem_standard_sparse(
        sparse_response,
        _unit_q(sparse_response.n_items),
        n_categories,
        n_samples=n_samples,
        burn_in=burn_in,
        max_iter=max_iter,
        tol=tol,
        sample_interval=sample_interval,
        verbose=verbose,
    )
    return _mgrm_to_grm_parameters(a_matrix, d_est, theta_matrix)


def grm_saem_standard_sparse(
    sparse_response, n_categories, max_iter=100, tol=1e-4, verbose=False
):
    """Estimate a single-dimensional GRM standard model via sparse SAEM."""

    a_matrix, d_est, theta_matrix = mgrm_saem_standard_sparse(
        sparse_response,
        _unit_q(sparse_response.n_items),
        n_categories,
        max_iter=max_iter,
        tol=tol,
        verbose=verbose,
    )
    return _mgrm_to_grm_parameters(a_matrix, d_est, theta_matrix)


def estimate_d_only_mcmc_sparse(
    a_params, theta, sparse_response, n_samples=3000, burn_in=2000, verbose=False
):
    start_time = time.time()
    d_est = np.zeros(sparse_response.n_items)
    samples_d = np.zeros((sparse_response.n_items, n_samples))
    rv_d = norm(loc=0, scale=1)
    step_size_d = 0.2
    for iteration in range(n_samples + burn_in):
        if verbose and iteration % 500 == 0:
            print(f"=== sparse d-only MCMC iter {iteration + 1}/{n_samples + burn_in} ===")
        d_est, step_size_d = update_d_parameters_sparse(
            rv_d, theta, a_params, d_est, sparse_response, step_size=step_size_d
        )
        if iteration >= burn_in:
            samples_d[:, iteration - burn_in] = d_est
    return np.mean(samples_d, axis=1), time.time() - start_time


def estimate_d_only_from_theta_sparse(a_params, theta, sparse_response):
    d_est = np.zeros(sparse_response.n_items)
    for item_id in range(sparse_response.n_items):
        users, values = _item_users_values(sparse_response, item_id)
        res = minimize(
            lambda d_j: -np.sum(
                _m2pl_observation_loglik(
                    theta[users],
                    np.repeat(a_params[item_id][None, :], users.size, axis=0),
                    np.full(users.size, d_j[0]),
                    values,
                )
            ),
            [d_est[item_id]],
            method="L-BFGS-B",
        )
        if res.success:
            d_est[item_id] = res.x[0]
    return d_est, 0.0


def _fit_m2pl_item_from_theta(theta, values, active_dims, init_a, init_d):
    dim = init_a.shape[0]
    num_a = int(np.sum(active_dims))

    def objective(params):
        a_j = np.zeros(dim)
        a_j[active_dims] = params[:num_a]
        d_j = params[num_a]
        return -np.sum(
            _m2pl_observation_loglik(
                theta, np.repeat(a_j[None, :], values.size, axis=0), np.full(values.size, d_j), values
            )
        )

    init = np.concatenate([init_a[active_dims], [init_d]])
    bounds = [(0.1, None)] * num_a + [(None, None)]
    res = minimize(objective, init, method="L-BFGS-B", bounds=bounds)
    if not res.success:
        return init_a.copy(), init_d, False
    a_j = np.zeros(dim)
    a_j[active_dims] = res.x[:num_a]
    return a_j, res.x[num_a], True


def _fit_mgrm_item_from_theta(theta, values, active_dims, init_a, init_d, n_categories):
    dim = init_a.shape[0]
    num_a = int(np.sum(active_dims))

    def objective(params):
        a_j = np.zeros(dim)
        a_j[active_dims] = params[:num_a]
        d_j = ensure_descending_thresholds(params[num_a:])
        return -_mgrm_item_loglik(theta, a_j, d_j, values, n_categories)

    init = np.concatenate([init_a[active_dims], init_d])
    bounds = [(0.1, None)] * num_a + [(-4.0, 4.0)] * (n_categories - 1)
    res = minimize(objective, init, method="L-BFGS-B", bounds=bounds)
    if not res.success:
        return init_a.copy(), init_d.copy(), False
    a_j = np.zeros(dim)
    a_j[active_dims] = res.x[:num_a]
    d_j = ensure_descending_thresholds(res.x[num_a:])
    return a_j, d_j, True


def _item_users_values(sparse_response, item_id):
    obs_idx = sparse_response.item_observations(item_id)
    return sparse_response.user_idx[obs_idx], sparse_response.values[obs_idx]


def _unit_q(n_items):
    return np.ones((n_items, 1), dtype=int)


def _m2pl_to_2pl_parameters(a_matrix, d_est, theta_matrix):
    a_est = np.asarray(a_matrix, dtype=float)[:, 0]
    b_est = -np.asarray(d_est, dtype=float) / np.maximum(a_est, 1e-12)
    return a_est, b_est, np.asarray(theta_matrix, dtype=float)[:, 0]


def _mgrm_to_grm_parameters(a_matrix, d_est, theta_matrix):
    a_est = np.asarray(a_matrix, dtype=float)[:, 0]
    b_est = []
    for item_id, d_item in enumerate(d_est):
        b_item = -np.asarray(d_item, dtype=float) / max(a_est[item_id], 1e-12)
        b_est.append(np.sort(b_item))
    return a_est, b_est, np.asarray(theta_matrix, dtype=float)[:, 0]


def _m2pl_observation_loglik(theta_obs, a_obs, d_obs, values):
    logits = np.sum(theta_obs * a_obs, axis=1) + d_obs
    prob = _sigmoid(logits)
    prob = np.clip(prob, 1e-15, 1.0 - 1e-15)
    return values * np.log(prob) + (1.0 - values) * np.log(1.0 - prob)


def _mgrm_item_loglik(theta, a_j, d_j, values, n_categories):
    d_j = ensure_descending_thresholds(np.asarray(d_j))
    d_obs = np.repeat(d_j.reshape(1, -1), values.size, axis=0)
    d_mask = np.ones_like(d_obs, dtype=bool)
    return np.sum(
        _mgrm_observation_loglik(
            theta,
            np.repeat(a_j.reshape(1, -1), values.size, axis=0),
            d_obs,
            d_mask,
            np.full(values.size, n_categories),
            values.astype(int),
        )
    )


def _mgrm_observation_loglik(theta_obs, a_obs, d_obs, d_mask_obs, ncat_obs, values):
    max_thresholds = d_obs.shape[1]
    max_categories = int(np.max(ncat_obs))
    dot = np.sum(theta_obs * a_obs, axis=1)
    logits = np.clip(dot[:, None] + d_obs, -35.0, 35.0)
    p_star_thresholds = _sigmoid(logits) * d_mask_obs
    p_star = np.zeros((values.size, max_categories + 1))
    p_star[:, 0] = 1.0
    p_star[:, 1 : max_thresholds + 1] = p_star_thresholds
    row_idx = np.arange(values.size)
    prob = p_star[row_idx, values] - p_star[row_idx, values + 1]
    return np.log(np.clip(prob, 1e-15, 1.0 - 1e-15))


def _mgrm_matrix_and_mask(d, n_categories, d_mask=None):
    n_categories = np.asarray(n_categories)
    if d_mask is not None:
        return np.asarray(d, dtype=float), np.asarray(d_mask, dtype=bool)
    if isinstance(d, list):
        return pad_grm_parameters(d, n_categories)
    d_matrix = np.asarray(d, dtype=float)
    mask = np.zeros_like(d_matrix, dtype=bool)
    for item_id, n_cat in enumerate(n_categories):
        mask[item_id, : int(n_cat) - 1] = True
    return d_matrix, mask


def _sum_by_index(values, indexes, n_groups):
    out = np.zeros(n_groups)
    np.add.at(out, indexes, values)
    return out


def _adapt_scalar_step(step_size, accept_rate):
    if accept_rate < 0.2:
        step_size *= 0.9
    elif accept_rate > 0.4:
        step_size *= 1.1
    return float(np.clip(step_size, 0.05, 0.5))


def _d_to_delta_masked(d_matrix, d_mask):
    delta_matrix = np.zeros_like(d_matrix)
    delta_matrix[:, 0] = d_matrix[:, 0]
    increments = -np.diff(d_matrix, axis=1)
    delta_matrix[:, 1:] = np.log(np.maximum(increments, 1e-9))
    return delta_matrix * d_mask


def _delta_to_d_masked(delta_matrix, d_mask):
    d_matrix = np.zeros_like(delta_matrix)
    d_matrix[:, 0] = delta_matrix[:, 0]
    increments = -np.exp(delta_matrix[:, 1:]) * d_mask[:, 1:]
    d_matrix[:, 1:] = d_matrix[:, 0][:, None] + np.cumsum(increments, axis=1)
    return d_matrix * d_mask


def _sigmoid(logits):
    logits = np.clip(logits, -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-logits))
