import numpy as np
from .units import mgrm_prob_categories,m2pl_prob,mcmc_sampling
from scipy.stats import multivariate_normal

######################################m2pl######################################

#em方法,适用三维及以下
def eap_m2pl(res_matrix, mask_matrix,a_params, d_params, quad_points_nd, quad_weights_nd):
    """
    使用完全向量化的EAP方法为所有被试估计多维IRT能力值。
    参数:
    res_matrix (np.array): 被试作答矩阵 (num_persons, num_items)
    mask_matrix (np.array): 被试作答掩码矩阵 (num_persons, num_items)，用于指示哪些项目被作答,其中1表示作答，0表示未作答。
    a_params (np.array): 项目区分度参数 (num_items,dims)
    d_params (np.array): 项目阈值参数 (num_items)
    quad_points_nd (np.array): 高斯-厄米特求积点 (num_quad, dims)
    quad_weights_nd (np.array): 高斯-厄米特求积权重 (num_quad,)
    返回:
    np.array: 所有被试的EAP能力估计值 (num_persons, dims)
    """
    num_persons, num_items = res_matrix.shape
    res_matrix = np.nan_to_num(res_matrix, nan=0.0)  # 将NaN转换为0.0
    res_matrix=res_matrix.reshape(num_persons,1,num_items)  # (num_persons, 1, num_items)
    p_correct = m2pl_prob(quad_points_nd, a_params, d_params)# (num_quad, num_items)
    p_response = res_matrix * p_correct + (1 - res_matrix) * (1 - p_correct)# (num_persons, num_quad, num_items)
    p_response*= mask_matrix.reshape(num_persons, 1, num_items)
    log_p= np.log(p_response + 1e-9)  # 加一个小常数防止log(0)
    log_likelihood = np.sum(log_p, axis=2)
    log_likelihood_max = np.max(log_likelihood, axis=1, keepdims=True)  # (num_persons,1)
    log_likelihood_scaled = log_likelihood - log_likelihood_max  # (n_persons, num_quad)
    likelihood_scaled = np.exp(log_likelihood_scaled)  # (n_persons, num_quad)
    #(n_persons, num_quad)@ (num_quad, dims)==>(n_persons,dims)
    weighted_likelihood_scaled=likelihood_scaled*quad_weights_nd.reshape(1,-1)#  # (n_persons, num_quad)
    #(n_persons, num_quad)@(num_quad, dims)==>(n_persons, dims)
    numerator =weighted_likelihood_scaled @ quad_points_nd # (n_persons,dims)
    denominator = np.sum(weighted_likelihood_scaled, axis=1, keepdims=True)# (n_persons, 1)
    denominator = np.where(denominator < 1e-9, 1e-9, denominator)
    theta_eap = numerator / denominator
    return theta_eap




#mc方法,适用二维及以上
def mc_m2pl(res_matrix, mask_matrix, a_params, d_params, n_samples=800,burn_in=200):
    """
    使用完全向量化的MC方法为所有被试估计多维IRT能力值。
    参数:
    res_matrix (np.array): 被试作答矩阵 (num_persons, num_items)
    mask_matrix (np.array): 被试作答掩码矩阵 (num_persons, num_items)，用于指示哪些项目被作答,其中1表示作答，0表示未作答。
    a_params (np.array): 项目区分度参数 (num_items,dims)
    d_params (np.array): 项目阈值参数 (num_items)
    n_samples (int): MCMC采样数量
    burn_in (int): 烧入期
    返回:
    np.array: 所有被试的EAP能力估计值 (num_persons, dims)
    """
    n_persons= res_matrix.shape[0]
    n_items = res_matrix.shape[1]
    n_dims = a_params.shape[1]
    response = np.nan_to_num(res_matrix)
    theta_est=np.zeros((n_persons, n_dims))  # 初始化能力估计值
    step_sizes = np.full(n_persons, 0.2)
    rv = multivariate_normal(mean=np.zeros(n_dims), cov=np.eye(n_dims))
    samples, theta_curr, step_sizes, all_accept= mcmc_sampling(
        theta_est, a_params, d_params, response, mask_matrix,
          rv, step_sizes, burn_in, n_samples,method='m2pl'
          )
    # 计算每个被试的EAP估计值
    theta_est = np.mean(samples, axis=1)  # (n_persons, n_dims)
    return theta_est








######################################mgrm######################################



def eap_mgrm(res_matrix, mask_matrix, a_params, d_params_padded, d_mask, n_categories, quad_points_nd, quad_weights_nd):
    """
    使用完全向量化的EAP方法为所有被试估计多维GRM模型的能力值。
    参数:
    res_matrix: 作答矩阵，形状(n_subjects, n_items)
    mask_matrix: 掩码矩阵，形状(n_subjects, n_items)
    a_params: 区分度参数，形状(n_items,dims)
    b_params_padded: 填充后的阈值参数，形状(n_items, max_thresholds)
    b_mask: 阈值掩码，形状(n_items, max_thresholds)
    n_categories: 每个项目的类别数，形状(n_items,)
    quad_points: 积分点，形状(n_quadrature,)
    quad_weights: 积分权重，形状(n_quadrature,)
    返回:
    theta_eap: EAP估计值，形状(n_subjects,)
    """
    n_categories=np.array(n_categories)
    num_persons, num_items = res_matrix.shape
    num_total_quad = quad_points_nd.shape[0]
    res_matrix = np.nan_to_num(res_matrix, nan=0.0)  # 将NaN转换为0.0
    P_cat_quad = mgrm_prob_categories(quad_points_nd, a_params, d_params_padded, d_mask, n_categories) # Shape: (n_quad, items, max_k)
    log_P_cat = np.log(P_cat_quad)  # (n_theta, n_items, max_cat)
    resp_clipped = np.clip(res_matrix, 0, n_categories - 1).astype(int)  # (n_persons, n_items)
    theta_idx = np.arange(num_total_quad)[:, None, None]   # (num_total_quad, 1, 1)
    item_idx = np.arange(num_items)[None, None, :]  # (1, 1, n_items)
    log_p_observed = log_P_cat[theta_idx, item_idx, resp_clipped[None, :, :]]  # (n_theta, n_persons, n_items)
    mask_3d = mask_matrix.astype(bool)[None, :, :]  # (1, n_persons, n_items)
    log_p_observed = np.where(mask_3d, log_p_observed, 0.0)  # (n_theta, n_persons, n_items)
    log_likelihood = np.sum(log_p_observed, axis=2)  # (n_theta, n_persons)
    log_likelihood_max = np.max(log_likelihood, axis=0, keepdims=True)  # (1, n_persons)
    log_likelihood_scaled = log_likelihood - log_likelihood_max  # (n_theta, n_persons)
    likelihood_scaled = np.exp(log_likelihood_scaled)  # (n_theta, n_persons)
    weighted_likelihood_scaled = likelihood_scaled.T * quad_weights_nd.reshape(1, -1)#(n_persons, num_total_quad)
    # (n_persons, num_total_quad) @ (num_total_quad, dims) => (n_persons, dims)
    numerator = weighted_likelihood_scaled @ quad_points_nd  # (n_persons, dims)
    denominator = np.sum(weighted_likelihood_scaled, axis=1, keepdims=True)# (n_persons, 1)
    denominator = np.where(denominator < 1e-9, 1e-9, denominator)
    theta_eap = numerator / denominator
    return theta_eap



def mc_mgrm(res_matrix, mask_matrix, a_params, d_params,n_categories, n_samples=800,burn_in=200):
    """
    使用完全向量化的MC方法为所有被试估计多维IRT能力值。
    参数:
    res_matrix (np.array): 被试作答矩阵 (num_persons, num_items)
    mask_matrix (np.array): 被试作答掩码矩阵 (num_persons, num_items)
    a_params (np.array): 项目区分度参数 (num_items,dims)
    d_params (np.array): 项目阈值参数 (num_items)
    n_categories (list): 每个项目的类别数
    n_samples (int): MCMC有效采样数量
    burn_in (int): 烧入期
    返回:
    np.array: 所有被试的EAP能力估计值 (num_persons, dims)
    """
    n_persons= res_matrix.shape[0]
    n_items = res_matrix.shape[1]
    n_dims = a_params.shape[1]
    response = np.nan_to_num(res_matrix)
    theta_est=np.zeros((n_persons, n_dims))  # 初始化能力估计值
    step_sizes = np.full(n_persons, 0.2)
    rv = multivariate_normal(mean=np.zeros(n_dims), cov=np.eye(n_dims))
    samples, theta_curr, step_sizes, all_accept= mcmc_sampling(
        theta_est, a_params, d_params, response, mask_matrix,
          rv, step_sizes, burn_in, n_samples,method='m2pl',n_categories=n_categories
          )
    # 计算每个被试的EAP估计值
    theta_est = np.mean(samples, axis=1)  # (n_persons, n_dims)
    return theta_est