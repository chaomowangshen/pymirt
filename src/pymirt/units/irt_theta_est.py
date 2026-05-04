import numpy as np
from .units import grm_prob_categories


def prob_2pl(theta, a, b):
    """
    计算2PL模型下，给定能力theta，在项目a, b上的正确作答概率。
    支持向量化操作。
    """
    # 防止指数计算溢出
    z = np.clip(a * (theta - b), -35, 35)
    return 1 / (1 + np.exp(-z))



def eap_2pl(res_matrix, mask_matrix,a_params, b_params, quad_points, quad_weights):
    """
    使用完全向量化的EAP方法为所有被试估计单维irt能力值。
    参数:
    res_matrix (np.array): 被试作答矩阵 (num_persons, num_items)
    mask_matrix (np.array): 被试作答掩码矩阵 (num_persons, num_items)，用于指示哪些项目被作答,其中1表示作答，0表示未作答。
    a_params (np.array): 项目区分度参数 (num_items,)
    b_params (np.array): 项目难度参数 (num_items,)
    quad_points (np.array): 高斯-厄米特求积点 (num_quad,)
    quad_weights (np.array): 高斯-厄米特求积权重 (num_quad,)
    返回:
    np.array: 所有被试的EAP能力估计值 (num_persons,)
    """
    res_matrix=np.nan_to_num(res_matrix, nan=0.0)  # 将NaN转换为0.0
    num_persons, num_items = res_matrix.shape
    num_quad = quad_points.shape[0]
    # a, b: (num_items,) -> (1, 1, num_items)
    # theta_nodes: (num_quad,) -> (1, num_quad, 1)
    # res_matrix: (num_persons, num_items) -> (num_persons, 1, num_items)
    a = a_params.reshape(1, 1, -1)
    b = b_params.reshape(1, 1, -1)
    theta_nodes = quad_points.reshape(1, -1, 1)
    res = res_matrix.reshape(num_persons, 1, -1)
    # 1. 计算在每个求积点上，对每个项目的答对概率
    # p_correct shape: (1, num_quad, num_items)
    p_correct = prob_2pl(theta_nodes, a, b)
    # 2. 根据实际作答(1/0)选择P或(1-P)，得到每个被试在每个求积点对每个项目的作答概率
    # 利用广播机制，res 和 p_correct 将被扩展到 (num_persons, num_quad, num_items)
    p_response = res * p_correct + (1 - res) * (1 - p_correct)
    #应用作答掩码
    p_response *= mask_matrix.reshape(num_persons, 1, num_items)
    # 4. 取对数后，沿项目轴(axis=2)求和，得到每个被试在每个求积点上的对数似然
    # likelihood shape: (num_persons, num_quad)
    log_p= np.log(p_response + 1e-9)  # 加一个小常数防止log(0)
    log_likelihood = np.sum(log_p, axis=2)
    # 5. Log-Sum-Exp技巧: 对对数似然进行缩放，避免数值下溢
    log_likelihood_max = np.max(log_likelihood, axis=1, keepdims=True)  # (num_persons,1)
    log_likelihood_scaled = log_likelihood - log_likelihood_max  # (n_persons, num_quad)
    # 6. 计算缩放后的似然值
    likelihood_scaled = np.exp(log_likelihood_scaled)  # (n_persons, num_quad)
    # 7. 计算EAP估计的分子和分母,quad_points 和 quad_weights 需要 reshape 为 (1,num_quad) 以便广播
    numerator = np.sum(likelihood_scaled * quad_points.reshape(1,-1) * quad_weights.reshape(1,-1), axis=1)  # (n_persons,)
    denominator = np.sum(likelihood_scaled * quad_weights.reshape(1,-1), axis=1)  # (n_persons,)
    # 8. 防止除以零
    denominator = np.where(denominator < 1e-9, 1e-9, denominator)
    theta_eap = numerator / denominator
    return theta_eap




def prob_3pl(theta, a, b, c):
    z = np.clip(a * (theta - b), -35, 35)
    logistic = 1.0 / (1.0 + np.exp(-z))
    return np.clip(c + (1.0 - c) * logistic, 1e-15, 1.0 - 1e-15)


def eap_3pl(res_matrix, mask_matrix, a_params, b_params, c_params, quad_points, quad_weights):
    """
    Estimate abilities via EAP for a single-dimensional 3PL model.
    """
    res_matrix = np.nan_to_num(res_matrix, nan=0.0)
    num_persons, num_items = res_matrix.shape
    a = a_params.reshape(1, 1, -1)
    b = b_params.reshape(1, 1, -1)
    c = c_params.reshape(1, 1, -1)
    theta_nodes = quad_points.reshape(1, -1, 1)
    res = res_matrix.reshape(num_persons, 1, -1)
    p_correct = prob_3pl(theta_nodes, a, b, c)
    p_response = res * p_correct + (1 - res) * (1 - p_correct)
    p_response = np.where(mask_matrix.reshape(num_persons, 1, num_items).astype(bool), p_response, 1.0)
    log_likelihood = np.sum(np.log(p_response), axis=2)
    log_likelihood_max = np.max(log_likelihood, axis=1, keepdims=True)
    likelihood_scaled = np.exp(log_likelihood - log_likelihood_max)
    numerator = np.sum(
        likelihood_scaled * quad_points.reshape(1, -1) * quad_weights.reshape(1, -1),
        axis=1,
    )
    denominator = np.sum(likelihood_scaled * quad_weights.reshape(1, -1), axis=1)
    denominator = np.where(denominator < 1e-9, 1e-9, denominator)
    return numerator / denominator


def eap_grm(res_matrix, mask_matrix, a_params, b_params_padded, b_mask, n_categories, quad_points, quad_weights):
    """
    使用完全向量化的EAP方法为所有被试估计GRM模型的能力值。
    参数:
    res_matrix: 作答矩阵，形状(n_subjects, n_items)
    mask_matrix: 掩码矩阵，形状(n_subjects, n_items)
    a_params: 区分度参数，形状(n_items,)
    b_params_padded: 填充后的阈值参数，形状(n_items, max_thresholds)
    b_mask: 阈值掩码，形状(n_items, max_thresholds)
    n_categories: 每个项目的类别数，形状(n_items,)
    quad_points: 积分点，形状(n_quadrature,)
    quad_weights: 积分权重，形状(n_quadrature,)
    返回:
    theta_eap: EAP估计值，形状(n_subjects,)
    """
    num_persons, num_items = res_matrix.shape
    n_theta = quad_points.shape[0]  # 积分点数量
    res_matrix=np.nan_to_num(res_matrix, nan=0.0)  # 将NaN转换为0.0
    # 1. 获取所有积分点对所有项目的类别概率矩阵
    P_cat_quad = grm_prob_categories(quad_points, a_params, b_params_padded, b_mask, n_categories)  # (n_theta, n_items, max_cat)
    # 转为对数概率
    log_P_cat = np.log(P_cat_quad)  # (n_theta, n_items, max_cat)
    # 确保作答类别不越界
    resp_clipped = np.clip(res_matrix, 0, n_categories - 1).astype(int)  # (n_persons, n_items)
    # 2. 使用正确广播索引提取作答对数概率
    theta_idx = np.arange(n_theta)[:, None, None]   # (n_theta, 1, 1)
    item_idx = np.arange(num_items)[None, None, :]  # (1, 1, n_items)
    log_p_observed = log_P_cat[theta_idx, item_idx, resp_clipped[None, :, :]]  # (n_theta, n_persons, n_items)
    # 3. 应用作答掩码（未作答项目设为0）
    mask_3d = mask_matrix.astype(bool)[None, :, :]  # (1, n_persons, n_items)
    log_p_observed = np.where(mask_3d, log_p_observed, 0.0)  # (n_theta, n_persons, n_items)
    # 4. 计算对数似然（沿项目维度求和）
    log_likelihood = np.sum(log_p_observed, axis=2)  # (n_theta, n_persons)
    # 5. Log-Sum-Exp技巧: 对对数似然进行缩放，避免数值下溢
    log_likelihood_max = np.max(log_likelihood, axis=0, keepdims=True)  # (1, n_persons)
    log_likelihood_scaled = log_likelihood - log_likelihood_max  # (n_theta, n_persons)
    # 6. 计算缩放后的似然值
    likelihood_scaled = np.exp(log_likelihood_scaled)  # (n_theta, n_persons)
    # 7. 计算EAP估计的分子和分母
    numerator = np.sum(likelihood_scaled * quad_points[:, None] * quad_weights[:, None], axis=0)  # (n_persons,)
    denominator = np.sum(likelihood_scaled * quad_weights[:, None], axis=0)  # (n_persons,)
    # 8. 防止除以零
    denominator = np.where(denominator < 1e-9, 1e-9, denominator)
    theta_eap = numerator / denominator
    return theta_eap





