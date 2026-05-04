import numpy as np
import pandas as pd
from scipy.optimize import minimize
from itertools import product



################################# 单维多级计分GRM模型(GRM)相关函数 ####################
#和多维共用的填充函数
def pad_grm_parameters(b_params_list,n_categories, padding_value=0.0):
    """
    (GRM专用工具) 将GRM/MGRM的不规则b参数列表填充为规整的矩阵和掩码。
    参数：
    b_params_list : list of np.ndarray
        每个元素是一个一维数组，表示每个项目的阈值参数
    padding_value : float
        填充的值，默认为0.0
    返回：
    b_padded : np.ndarray
        填充后的二维数组，形状为 (n_items, max_thresholds)
    n_categories : np.ndarray
        每个项目的类别数，形状为 (n_items,)
    b_mask : np.ndarray
        掩码矩阵，形状为 (n_items, max_thresholds)，True表示有数据，False表示填充
    """
    num_items = len(b_params_list)
    n_categories = np.array(n_categories)
    max_thresholds = np.max(n_categories) - 1 if num_items > 0 else 0
    b_padded = np.full((num_items, max_thresholds), padding_value)
    b_mask = np.zeros_like(b_padded, dtype=bool)
    for i, b in enumerate(b_params_list):
        num_thresh = len(b)
        if num_thresh > 0:
            b_padded[i, :num_thresh] = b
            b_mask[i, :num_thresh] = True
    return b_padded, b_mask

def grm_prob_categories(theta, a, b_padded, b_mask, n_categories):
    """
    计算GRM下，给定能力theta、项目参数，得到每个等级的作答概率。
    此函数是整个模型的核心，依赖b_mask来确保计算的精确性。
    参数：
    theta : np.ndarray
        能力参数，形状为 (n_theta_points,)，一维数组
    a : np.ndarray
        区分度参数，形状为 (n_items,)，一维数组
    b_padded : np.ndarray
        填充后的阈值参数矩阵，形状为 (n_items, max_thresholds)
    b_mask : np.ndarray
        掩码矩阵，形状为 (n_items, max_thresholds)，True表示有数据，False表示填充
    n_categories : list
        每个项目的类别数，长度为 n_items 的列表
    返回：
    P_cat : np.ndarray
        每个类别的作答概率，形状为 (n_theta_points, n_items, max(n_categories))
    其中 max(n_categories) 是所有项目中最大类别数。
    例如，P_cat[i, j, k] 表示第 i 个能力点对应第 j 个项目的第 k 类别的作答概率。
    """
    n_theta_points = theta.shape[0]
    num_items, max_thresholds = b_padded.shape
    max_k = np.max(n_categories)
    #max_thresholds = max_k - 1  # 最大阈值数
    # 维度重塑以利用广播机制
    theta = theta.reshape(-1, 1, 1)
    a = a.reshape(1, -1, 1)
    b = b_padded.reshape(1, num_items, -1)
    # 1. 计算所有可能的累积概率 P*(x >= m | θ)
    logits = np.clip(a * (theta - b), -35, 35)
    p_star_m = 1 / (1 + np.exp(-logits))  # 形状为 (n_theta_points, num_items, max_thresholds)
    # 2. 应用参数掩码`b_mask`。
    # 确保由填充(padding)产生的无效阈值其对应的累积概率被强制为0。
    mask_expanded = b_mask.reshape(1, num_items, max_thresholds)
    p_star_m[~mask_expanded.repeat(n_theta_points, axis=0)] = 0.0
    # 3. 构建完整的累积概率矩阵 P_star_full
    P_star_full = np.zeros((n_theta_points, num_items, max_k + 1))
    P_star_full[:, :, 0] = 1.0 # P(U >= 0) = 1
    if max_thresholds > 0:
        P_star_full[:, :, 1:max_thresholds + 1] = p_star_m
    # 4. 计算类别概率 P(x=m|θ)
    P_cat = P_star_full[:, :, :-1] - P_star_full[:, :, 1:]
    # Clip以避免在后续log计算中出现-inf
    P_cat = np.clip(P_cat, 1e-15, 1.0-1e-15) 
    return P_cat


def compute_grm_posterior(theta, a, b, response, mask_matrix, n_categories, quad_weights):
    """
    计算单维GRM模型的后验分布
    
    参数:
    theta: 潜变量向量，形状(n_quadrature,)
    a: 区分度参数向量，形状(n_items,)
    b: 难度参数列表，每个元素是一个形状为(n_categories[j]-1,)的数组
    response: 被试作答矩阵，形状(n_subjects, n_items)
    mask_matrix: 掩码矩阵，形状(n_subjects, n_items)
    n_categories: 每个项目的类别数(计分等级)的列表，长度为n_items
    quad_weights: 积分权重，形状(n_quadrature,)
    返回:
    posterior: 后验分布，形状(n_subjects, n_quadrature)
    """
    n_categories= np.array(n_categories)
    num_persons, num_items = response.shape
    n_theta = len(theta)
    # 准备b参数（填充为统一形状）
    b_padded, b_mask = pad_grm_parameters(b,n_categories)
    # 计算所有类别在所有积分点上的概率
    P_cat = grm_prob_categories(theta, a, b_padded, b_mask, n_categories)
    # 转为对数概率
    log_P_cat = np.log(P_cat)
    # 确保作答类别不越界
    resp_clipped = np.clip(response, 0, n_categories - 1).astype(int)
    # 使用正确广播索引提取作答对数概率
    theta_idx = np.arange(n_theta)[:, None, None]   # (n_theta, 1, 1)
    item_idx = np.arange(num_items)[None, None, :]  # (1, 1, n_items)
    log_p_observed = log_P_cat[theta_idx, item_idx, resp_clipped[None, :, :]]
    # 应用作答掩码（未作答项目设为0）
    mask_3d = mask_matrix.astype(bool)[None, :, :]  # (1, n_persons, n_items)
    log_p_observed = np.where(mask_3d, log_p_observed, 0.0)
    # 计算对数似然（沿项目维度求和）
    log_likelihood = np.sum(log_p_observed, axis=2)  # (n_theta, n_persons)
    # 添加对数先验（积分权重）
    log_prior = np.log(quad_weights)[:, None]
    log_posterior = log_likelihood + log_prior
    # 计算后验概率（指数变换并归一化）
    posterior = np.exp(log_posterior)
    posterior /= np.sum(posterior, axis=0, keepdims=True)
    return posterior.T  # 转置为 (n_persons, n_theta)

    
def compute_grm_item_log_likelihood(theta, a_j, b_j, response_j, mask_matrix_j, k, posterior):
    """
    计算单维GRM模型的单个项目对数似然和,只用高斯-赫尔米特积分。
    
    参数:
    theta: 潜变量向量，形状(n_quadrature,)
    a_j: 项目j区分度参数，标量
    b_j: 项目j难度参数，形状(n_categories[j]-1,)
    response_j: 被试对项目j的作答，形状(n_subjects, 1)
    mask_matrix_j: 掩码矩阵，形状(n_subjects, 1)
    k: 项目j的类别数
    posterior: 后验分布，形状(n_subjects, n_quadrature)
    
    返回:
    ll_sum: 所有被试的对数似然和（标量）
    """
    n_subjects = response_j.shape[0]
    n_theta = len(theta)
    # 准备b参数（单个项目）
    b_j_reshaped = b_j.reshape(1, -1)  # (1, n_thresholds)
    b_mask = np.ones_like(b_j_reshaped, dtype=bool)  # (1, n_thresholds)
    n_categories_arr = np.array([k])  # 项目j的类别数
    # 计算项目j在所有积分点上的类别概率
    P_cat_j = grm_prob_categories(theta, np.array([a_j]), b_j_reshaped, b_mask, n_categories_arr)  # (n_theta, 1, k)
    # 转为对数概率
    log_P_cat_j = np.log(P_cat_j)  # (n_theta, 1, k)
    # 确保作答类别不越界（0到k-1）
    resp_clipped_j = np.clip(response_j, 0, k-1).astype(int)  # (n_subjects, 1)
    # 使用正确广播索引提取作答对数概率
    theta_idx = np.arange(n_theta)[:, None, None]   # (n_theta, 1, 1)
    item_idx = np.zeros((1, n_subjects, 1), dtype=int)  # 所有项目索引为0（只有一个项目）
    log_p_observed_j = log_P_cat_j[theta_idx, item_idx, resp_clipped_j[None, :, :]]  # (n_theta, n_subjects, 1)
    log_p_observed_j = log_p_observed_j.squeeze(axis=2)  # (n_theta, n_subjects)
    # 应用作答掩码（未作答设为0）
    mask_3d_j = mask_matrix_j.astype(bool)[None, :, :]  # (1, n_subjects, 1)
    log_p_observed_j = np.where(mask_3d_j.squeeze(axis=2), log_p_observed_j, 0.0)  # (n_theta, n_subjects)
    # 应用后验分布并计算期望
    # posterior: (n_subjects, n_theta) -> 转置为 (n_theta, n_subjects)
    expected_log_lik_j = np.sum(log_p_observed_j * posterior.T, axis=0)  # (n_subjects,)
    # 计算所有被试的对数似然和
    ll_sum = np.sum(expected_log_lik_j)
    return ll_sum
    


def compute_grm_log_likelihood(theta, a, b, response, mask_matrix, n_categories, posterior):
    """
    计算单维GRM模型的对数似然
    参数:
    theta: 潜变量向量，形状(n_quadrature,)
    a: 区分度参数向量，形状(n_items,)
    b: 难度参数列表，每个元素是一个形状为(n_categories[j]-1,)的数组
    response: 被试作答矩阵，形状(n_subjects, n_items)
    mask_matrix: 掩码矩阵，形状(n_subjects, n_items)
    n_categories: 每个项目的类别数(计分等级)的列表，长度为n_items
    posterior: 后验分布，形状(n_subjects, n_quadrature)
    返回:
    ll: 每个被试的对数似然，形状(n_subjects,)
    """
    n_categories = np.array(n_categories)
    num_persons, num_items = response.shape
    n_theta = len(theta)
    # 准备b参数
    b_padded,  b_mask = pad_grm_parameters(b,n_categories)
    # 计算所有类别的概率
    P_cat = grm_prob_categories(theta, a, b_padded, b_mask, n_categories)
    # 转为对数概率
    log_P_cat = np.log(P_cat)  # (n_theta, n_items, max_cat)
    # 确保作答类别不越界
    resp_clipped = np.clip(response, 0, n_categories - 1).astype(int)  # (n_persons, n_items)
    # 使用正确广播索引提取作答对数概率
    theta_idx = np.arange(n_theta)[:, None, None]   # (n_theta, 1, 1)
    item_idx = np.arange(num_items)[None, None, :]  # (1, 1, n_items)
    log_p_observed = log_P_cat[theta_idx, item_idx, resp_clipped[None, :, :]]  # (n_theta, n_persons, n_items)
    # 应用作答掩码（未作答项目设为0）
    mask_3d = mask_matrix.astype(bool)[None, :, :]  # (1, n_persons, n_items)
    log_p_observed = np.where(mask_3d, log_p_observed, 0.0)  # (n_theta, n_persons, n_items)
    # 计算对数似然（沿项目维度求和）: (n_theta, n_persons)
    log_likelihood = np.sum(log_p_observed, axis=2)
    # 如果没有后验分布，直接返回对数似然
    # 应用后验分布并计算期望
    # posterior: (n_persons, n_theta) -> 转置为 (n_theta, n_persons)
    expected_log_lik = np.sum(log_likelihood * posterior.T, axis=0)  # (n_persons,)
    return expected_log_lik



################多维多级计分GRM模型(MGRM)相关函数####################

def mgrm_prob_categories(theta, a, d_padded, d_mask, n_categories):
    """
    计算MGRM下，给定能力theta、项目参数，得到每个等级的作答概率。
    此函数是整个模型的核心，依赖d_mask来确保计算的精确性。
    参数：
    theta : np.ndarray
        能力参数，形状为 (n_theta_points,)，一维数组
    a : np.ndarray
        区分度参数，形状为  (n_items, n_dims)，二维数组
    d_padded : np.ndarray
        填充后的阈值参数矩阵，形状为 (n_items, max_thresholds)
    d_mask : np.ndarray
        掩码矩阵，形状为 (n_items, max_thresholds)，True表示有数据，False表示填充
    n_categories : list
        每个项目的类别数，长度为 n_items 的列表
    返回：
    P_cat : np.ndarray
        每个类别的作答概率，形状为 (n_theta_points, n_items, max(n_categories))
    其中 max(n_categories) 是所有项目中最大类别数。
    例如，P_cat[i, j, k] 表示第 i 个能力点对应第 j 个项目的第 k 类别的作答概率。
    """
    num_total_theta, num_dims = theta.shape
    num_items, max_thresholds = d_padded.shape
    max_k = np.max(n_categories)
    #max_thresholds = max_k - 1  # 最大阈值数
    # 维度重塑以利用广播机制
    dot_product = theta@a.T  # (num_total_theta, num_items)
    logits = dot_product.reshape(num_total_theta, num_items, 1) + d_padded.reshape(1, num_items, -1)#(num_total_theta, num_items, max_thresholds)
    logits = np.clip(logits, -35, 35)
    p_star_m = 1 / (1 + np.exp(-logits)) 
    # 2. 应用参数掩码`b_mask`。
    mask_expanded = d_mask.reshape(1, num_items, max_thresholds)
    p_star_m[~mask_expanded.repeat(num_total_theta, axis=0)] = 0.0
    # 3. 构建完整的累积概率矩阵 P_star_full
    P_star_full = np.zeros((num_total_theta, num_items, max_k + 1))
    P_star_full[:, :, 0] = 1.0 # P(U >= 0) = 1
    if max_thresholds > 0:
        P_star_full[:, :, 1:max_thresholds + 1] = p_star_m
    # 4. 计算类别概率 P(x=m|θ)
    P_cat = P_star_full[:, :, :-1] - P_star_full[:, :, 1:]
    # Clip以避免在后续log计算中出现-inf
    P_cat = np.clip(P_cat, 1e-15, 1.0-1e-15) 
    return P_cat








def compute_mgrm_posterior(theta, a, d, response, mask_matrix, n_categories, quad_weights):
    """
    计算多维GRM模型的后验分布
    
    参数:
    theta: 潜变量向量，形状(n_quadrature,dims)
    a: 区分度参数数组，形状(n_items,dims)
    d: 阈值列表，每个元素是一个形状为(n_categories[j]-1,)的数组
    response: 被试作答矩阵，形状(n_subjects, n_items)
    mask_matrix: 掩码矩阵，形状(n_subjects, n_items)
    n_categories: 每个项目的类别数(计分等级)的列表，长度为n_items
    quad_weights: 积分权重，形状(n_quadrature,)
    返回:
    posterior: 后验分布，形状(n_subjects, n_quadrature)
    """
    n_categories = np.array(n_categories)
    num_persons, num_items = response.shape
    n_theta = theta.shape[0]
    # 准备d参数（填充为统一形状）
    d_padded, d_mask = pad_grm_parameters(d,n_categories)
    # 计算所有类别在所有积分点上的概率
    P_cat = mgrm_prob_categories(theta, a, d_padded, d_mask, n_categories)
    # 转为对数概率
    log_P_cat = np.log(P_cat)
    # 确保作答类别不越界
    resp_clipped = np.clip(response, 0, n_categories - 1).astype(int)
    # 使用正确广播索引提取作答对数概率
    theta_idx = np.arange(n_theta)[:, None, None]   # (n_theta, 1, 1)
    item_idx = np.arange(num_items)[None, None, :]  # (1, 1, n_items)
    log_p_observed = log_P_cat[theta_idx, item_idx, resp_clipped[None, :, :]]
    # 应用作答掩码（未作答项目设为0）
    mask_3d = mask_matrix.astype(bool)[None, :, :]  # (1, n_persons, n_items)
    log_p_observed = np.where(mask_3d, log_p_observed, 0.0)
    # 计算对数似然（沿项目维度求和）
    log_likelihood = np.sum(log_p_observed, axis=2)  # (n_theta, n_persons)
    # 添加对数先验（积分权重）
    log_prior = np.log(quad_weights)[:, None]
    log_posterior = log_likelihood + log_prior
    # 计算后验概率（指数变换并归一化）
    posterior = np.exp(log_posterior)
    posterior /= np.sum(posterior, axis=0, keepdims=True)
    return posterior.T  # 转置为 (n_persons, n_theta)

    
def compute_mgrm_item_log_likelihood(theta, a_j, d_j, response_j, mask_matrix_j, k, posterior=None):
    """
    计算多维GRM模型的单个项目对数似然和
    
    参数:
    theta: 潜变量向量，形状(n_quadrature,)
    a_j: 项目j区分度参数，标量
    d_j: 项目j阈值参数，形状(n_categories[j]-1,)
    response_j: 被试对项目j的作答，形状(n_subjects, 1)
    mask_matrix_j: 掩码矩阵，形状(n_subjects, 1)
    k: 项目j的类别数
    posterior: 后验分布，形状(n_subjects, n_quadrature)
    
    返回:
    ll_sum: 所有被试的对数似然和（标量）
    """
    n_subjects = response_j.shape[0]
    n_theta = theta.shape[0]
    # 准备b参数（单个项目）
    d_j_reshaped = d_j.reshape(1, -1)  # (1, n_thresholds)
    d_mask = np.ones_like(d_j_reshaped, dtype=bool)  # (1, n_thresholds)
    n_categories_arr = np.array([k])  # 项目j的类别数
    # 计算项目j在所有积分点上的类别概率
    P_cat_j = mgrm_prob_categories(theta, np.array([a_j]), d_j_reshaped, d_mask, n_categories_arr)  # (n_theta, 1, k)
    # 转为对数概率
    log_P_cat_j = np.log(P_cat_j)  # (n_theta, 1, k)
    # 确保作答类别不越界（0到k-1）
    resp_clipped_j = np.clip(response_j, 0, k-1).astype(int)  # (n_subjects, k)
    if posterior is None:
        # 如果没有后验分布，直接计算对数似然和,mcem
        log_P_cat_j=log_P_cat_j.squeeze(axis=1)  # (n_theta, k)
        resp_clipped_j = resp_clipped_j.squeeze(1)  # (n_subjects)
        log_p_observed_j=log_P_cat_j[np.arange(n_theta),resp_clipped_j]# (n_theta, 1)
        mask_matrix_j = mask_matrix_j.squeeze(1)  # (n_subjects,)
        # 应用作答掩码
        log_p_observed_j= np.where(mask_matrix_j.astype(bool), log_p_observed_j, 0.0)  # (n_theta, n_subjects)
        # 计算所有被试的对数似然和
        ll_sum = np.sum(log_p_observed_j)
        return ll_sum
    else:
        # 使用正确广播索引提取作答对数概率,高斯-赫尔米特积分
        theta_idx = np.arange(n_theta)[:, None, None]   # (n_theta, 1, 1)
        item_idx = np.zeros((1, n_subjects, 1), dtype=int)  # 所有项目索引为0（因为只有一个项目）
        log_p_observed_j = log_P_cat_j[theta_idx, item_idx, resp_clipped_j[None, :, :]]  # (n_theta, n_subjects, 1)
        log_p_observed_j = log_p_observed_j.squeeze(axis=2)  # (n_theta, n_subjects)
        # 应用作答掩码（未作答设为0）
        mask_3d_j = mask_matrix_j.astype(bool)[None, :, :]  # (1, n_subjects, 1)
        log_p_observed_j = np.where(mask_3d_j.squeeze(axis=2), log_p_observed_j, 0.0)  # (n_theta, n_subjects)
        # 应用后验分布并计算期望
        # posterior: (n_subjects, n_theta) -> 转置为 (n_theta, n_subjects)
        expected_log_lik_j = np.sum(log_p_observed_j * posterior.T, axis=0)  # (n_subjects,)
        # 计算所有被试的对数似然和
        ll_sum = np.sum(expected_log_lik_j)
        return ll_sum
    


def compute_mgrm_log_likelihood(theta, a, d, response, mask_matrix, n_categories, posterior=None,d_mask=None):
    """
    计算多维GRM模型的对数似然
    参数:
    theta: 潜变量向量，形状(n_quadrature,)
    a: 区分度参数向量，形状(n_items,)
    d: 阈值参数列表，每个元素是一个形状为(n_categories[j]-1,)的数组
    response: 被试作答矩阵，形状(n_subjects, n_items)
    mask_matrix: 掩码矩阵，形状(n_subjects, n_items)
    n_categories: 每个项目的类别数(计分等级)的列表，长度为n_items
    posterior: 后验分布，形状(n_subjects, n_quadrature)
    d_mask: 阈值参数掩码矩阵，形状(n_items, max_thresholds)，True表示有数据，False表示填充
    返回:
    ll: 每个被试的对数似然，形状(n_subjects,)
    """
    n_categories = np.array(n_categories)
    num_persons, num_items = response.shape
    n_theta = theta.shape[0]
    if d_mask is None:
        # 如果没有提供d_mask，则自动生成
        d_padded, d_mask = pad_grm_parameters(d, n_categories)
    else:
        d_padded, d_mask = d, d_mask  # 使用提供的d_mask
    # 计算所有类别的概率
    P_cat = mgrm_prob_categories(theta, a, d_padded, d_mask, n_categories)
    # 转为对数概率
    log_P_cat = np.log(P_cat)  # (n_theta, n_items, max_cat)
    # 确保作答类别不越界
    resp_clipped = np.clip(response, 0, n_categories - 1).astype(int)  # (n_persons, n_items)
    if posterior is None:#和有后验的theta不同，这里的n_theta就是mcmc采样人数，而不是积分点数,因此不需要广播
        # 如果没有后验分布，直接计算对数似然和,mcem
        resp_clipped= resp_clipped.reshape(num_persons, num_items, 1)  # (n_persons, n_items, 1)
        log_p_observed=np.take_along_axis(log_P_cat, resp_clipped, axis=2).squeeze()  # (n_theta, n_items)
        # 应用作答掩码
        log_p_observed=np.where(mask_matrix.astype(bool), log_p_observed, 0.0)  # (n_theta,n_items)
        log_likelihood = np.sum(log_p_observed, axis=1)  # (n_theta,)
        # 返回对数似然和
        return log_likelihood
    else:
        # 使用正确广播索引提取作答对数概率
        theta_idx = np.arange(n_theta)[:, None, None]   # (n_theta, 1, 1)
        item_idx = np.arange(num_items)[None, None, :]  # (1, 1, n_items)
        log_p_observed = log_P_cat[theta_idx, item_idx, resp_clipped[None, :, :]]  # (n_theta, n_persons, n_items)
        # 应用作答掩码（未作答项目设为0）
        mask_3d = mask_matrix.astype(bool)[None, :, :]  # (1, n_persons, n_items)
        log_p_observed = np.where(mask_3d, log_p_observed, 0.0)  # (n_theta, n_persons, n_items)
        # 计算对数似然（沿项目维度求和）: (n_theta, n_persons)
        log_likelihood = np.sum(log_p_observed, axis=2)
        # 应用后验分布并计算期望
        # posterior: (n_persons, n_theta) -> 转置为 (n_theta, n_persons)
        expected_log_lik = np.sum(log_likelihood * posterior.T, axis=0)  # (n_persons,)
        return expected_log_lik
    



    
def compute_mgrm_log_likelihood_item(theta, a, d_padded,d_mask, response, mask_matrix, n_categories):
    """
    计算多维GRM模型的对数似然
    参数:
    theta: 潜变量向量，形状(n_quadrature,)
    a: 区分度参数向量，形状(n_items,)
    d: 阈值参数列表，每个元素是一个形状为(n_categories[j]-1,)的数组
    response: 被试作答矩阵，形状(n_subjects, n_items)
    mask_matrix: 掩码矩阵，形状(n_subjects, n_items)
    n_categories: 每个项目的类别数(计分等级)的列表，长度为n_items
    posterior: 后验分布，形状(n_subjects, n_quadrature)
    返回:
    ll: 每个题目的对数似然，形状(n_items,)
    """
    n_categories = np.array(n_categories)
    num_persons, num_items = response.shape
    n_theta = theta.shape[0]
    # 计算所有类别的概率
    P_cat = mgrm_prob_categories(theta, a, d_padded, d_mask, n_categories)
    # 转为对数概率
    log_P_cat = np.log(P_cat)  # (n_theta, n_items, max_cat)
    # 确保作答类别不越界
    resp_clipped = np.clip(response, 0, n_categories - 1).astype(int)  # (n_persons, n_items)
    # 如果没有后验分布，直接计算对数似然和,mcem
    resp_clipped= resp_clipped.reshape(num_persons, num_items, 1)  # (n_persons, n_items, 1)
    log_p_observed=np.take_along_axis(log_P_cat, resp_clipped, axis=2).squeeze()  # (n_theta, n_items)
    # 应用作答掩码
    log_p_observed=np.where(mask_matrix.astype(bool), log_p_observed, 0.0)  # (n_theta,n_items)
    log_likelihood = np.sum(log_p_observed, axis=0)  # (n_items,)
    # 返回对数似然和
    return log_likelihood



################################# 单维2PL模型(2PL)相关函数 #########################
def compute_2pl_prob(theta, a, b):
    """
    计算单维2PL模型下，给定能力theta、项目参数，得到每个项目的作答概率。
    参数：
    theta : np.ndarray
        潜变量向量，形状为(n_quadrature,)
    a : np.ndarray
        区分度参数向量，形状为(n_items,)
    b : np.ndarray
        难度参数向量，形状为(n_items,)
    返回：
    P_cat : np.ndarray
        作答概率矩阵，形状为(n_quadrature, n_items)
    """
    a=a.reshape(1,-1)  # [1, n_items]
    b=b.reshape(1,-1)  # [1, n_items]
    logits = a * (theta.reshape(-1, 1) - b)  # (n_quadrature, n_items)
    logits = np.clip(logits, -35, 35)  # 防止数值溢出
    P_cat = 1 / (1 + np.exp(-logits))  # (n_quadrature, n_items)
    P_cat = np.clip(P_cat, 1e-15, 1.0 - 1e-15)  # 避免在后续log计算中出现-inf
    return P_cat

def compute_2pl_posterior(theta, a, b, response, mask_matrix, quad_weights):
    """
    计算单维2PL模型的后验分布
    参数:
    theta: 潜变量向量，形状(n_quadrature,)
    a: 区分度参数向量，形状(n_items,)
    b: 难度参数向量，形状(n_items,)
    response: 被试作答矩阵，形状(n_subjects, n_items)
    mask_matrix: 掩码矩阵，形状(n_subjects, n_items)
    quad_weights: 积分权重，形状(n_quadrature,)
    返回:
    posterior: 后验分布，形状(n_subjects, n_quadrature)
    """
    p= compute_2pl_prob(theta, a, b)  # (n_quadrature, n_items)
    log_likelihood = (
        response[np.newaxis, :, :] * np.log(p[:, np.newaxis, :] + 1e-15) +
        (1 - response[np.newaxis, :, :]) * np.log(1 - p[:, np.newaxis, :] + 1e-15)
    )# (n_quadrature, n_subjects, n_items)
    log_likelihood *= mask_matrix[np.newaxis, :, :]# (n_quadrature, n_subjects, n_items)
    log_likelihood_sum = np.sum(log_likelihood, axis=2)  # (n_quadrature, n_subjects)
    posterior= np.exp(log_likelihood_sum + np.log(quad_weights)[:, None])  # (n_quadrature, n_subjects)
    posterior /= np.sum(posterior, axis=0, keepdims=True)  # 归一化
    return posterior.T  # 转置为 (n_subjects, n_quadrature)


#单维2pl只使用高斯-赫尔米特积分来计算对数似然,不需要计算每个被试的后验分布，而是返回总体对数似然值
def compute_2pl_log_likelihood(theta, a, b, response_matrix, mask_matrix, posterior=None):
    """
    计算2PL模型的完整加权对数似然值（含高斯-赫尔米特积分权重）
    参数:
    theta: 高斯-赫尔米特积分点，形状(n_quad,)
    weights: 高斯-赫尔米特积分权重，形状(n_quad,)
    a: 区分度参数向量，形状(n_items,)
    b: 难度参数向量，形状(n_items,)
    response_matrix: 作答矩阵，形状(n_examinees, n_items)
    mask_matrix: 掩码矩阵，形状(n_examinees, n_items)
    posterior : 可选的预计算后验分布 [n_quad, n_examinees]
    返回:
    expected_ll: 完整的期望对数似然值（标量）
    """
    p= compute_2pl_prob(theta, a, b)  # (n_quad, n_items)
    log_likelihood = (
        response_matrix[np.newaxis, :, :] * np.log(p[:, np.newaxis, :] + 1e-15) +
        (1 - response_matrix[np.newaxis, :, :]) * np.log(1 - p[:, np.newaxis, :] + 1e-15)
    )
    log_likelihood *= mask_matrix[np.newaxis, :, :]
    log_likelihood_sum = np.sum(log_likelihood, axis=2)# (n_quad, n_examinees)
    # 如果提供了后验分布就直接使用，否则重新计算
    # posterior: (n_examinees, n_quad) -> 转置为 (n_quad, n_examinees)
    expected_ll = np.sum(log_likelihood_sum * posterior.T)
    return expected_ll



##################################多维2PL模型(M2PL)相关函数####################
def compute_3pl_prob(theta, a, b, c):
    """
    Compute single-dimensional 3PL response probabilities.
    """
    a = np.asarray(a).reshape(1, -1)
    b = np.asarray(b).reshape(1, -1)
    c = np.asarray(c).reshape(1, -1)
    logits = np.clip(a * (np.asarray(theta).reshape(-1, 1) - b), -35, 35)
    logistic = 1.0 / (1.0 + np.exp(-logits))
    p = c + (1.0 - c) * logistic
    return np.clip(p, 1e-15, 1.0 - 1e-15)


def compute_3pl_posterior(theta, a, b, c, response, mask_matrix, quad_weights):
    p = compute_3pl_prob(theta, a, b, c)
    log_likelihood = (
        response[np.newaxis, :, :] * np.log(p[:, np.newaxis, :] + 1e-15)
        + (1 - response[np.newaxis, :, :]) * np.log(1 - p[:, np.newaxis, :] + 1e-15)
    )
    log_likelihood *= mask_matrix[np.newaxis, :, :]
    log_likelihood_sum = np.sum(log_likelihood, axis=2)
    log_post = log_likelihood_sum + np.log(quad_weights)[:, None]
    max_log = np.max(log_post, axis=0, keepdims=True)
    scaled = np.exp(log_post - max_log)
    denom = np.sum(scaled, axis=0, keepdims=True)
    denom = np.where(denom < 1e-15, 1e-15, denom)
    return (scaled / denom).T


def compute_3pl_log_likelihood(theta, a, b, c, response_matrix, mask_matrix, posterior):
    p = compute_3pl_prob(theta, a, b, c)
    log_likelihood = (
        response_matrix[np.newaxis, :, :] * np.log(p[:, np.newaxis, :] + 1e-15)
        + (1 - response_matrix[np.newaxis, :, :]) * np.log(1 - p[:, np.newaxis, :] + 1e-15)
    )
    log_likelihood *= mask_matrix[np.newaxis, :, :]
    log_likelihood_sum = np.sum(log_likelihood, axis=2)
    return float(np.sum(log_likelihood_sum * posterior.T))


def m2pl_prob(theta,a,d):
    """
    计算多维2PL模型下，给定能力theta、项目参数，得到每个等级的作答概率。
    参数：
    theta : np.ndarray
        潜变量向量，形状为(n_quadrature, n_dims)
    a : np.ndarray
        区分度参数矩阵，形状为(n_items, n_dims)
    d : np.ndarray
        难度参数矩阵，形状为(n_items,)
    返回：
    P_cat : np.ndarray
        反应概率矩阵，形状为(n_quadrature, n_items)
    """
    logits= np.dot(theta, a.T) + d.reshape(1, -1)  # (n_quadrature, n_items)
    logits = np.clip(logits, -35, 35)  # 防止数值溢出
    P_cat = 1 / (1 + np.exp(-logits))  # (n_quadrature, n_items)
    P_cat = np.clip(P_cat, 1e-15, 1.0 - 1e-15)  # 避免在后续log计算中出现-inf
    return P_cat




def compute_m2pl_posterior(theta, a, d, response, mask_matrix, quad_weights):
    """
    计算多维2pl模型的后验分布
    
    参数:
    theta: 潜变量向量，形状(n_quadrature,dims)
    a: 区分度参数数组，形状(n_items,dims)
    d: 阈值参数向量，形状(n_items,)
    response: 被试作答矩阵，形状(n_subjects, n_items)
    mask_matrix: 掩码矩阵，形状(n_subjects, n_items)
    quad_weights: 积分权重，形状(n_quadrature,)
    返回:
    posterior: 后验分布，形状(n_subjects, n_quadrature)
    """
    num_persons, num_items = response.shape
    n_theta = theta.shape[0]
    # 计算所有类别在所有积分点上的概率
    P_cat =  m2pl_prob(theta, a, d)  # (n_theta, n_items)
    p_observed=response.reshape(1, num_persons, num_items)* P_cat[:, np.newaxis, :] + \
        (1-response.reshape(1, num_persons, num_items)) * (1 - P_cat[:, np.newaxis, :])  # (n_theta, n_persons, n_items)
    # 转为对数概率
    log_p_observed = np.log(p_observed)  # (n_theta, n_persons, n_items)
    # 应用作答掩码（未作答项目设为0）
    mask_3d = mask_matrix.astype(bool)[None, :, :]  # (1, n_persons, n_items)
    log_p_observed = np.where(mask_3d, log_p_observed, 0.0)  # (n_theta, n_persons, n_items)
    # 计算对数似然（沿项目维度求和）: (n_theta, n_persons)
    log_likelihood = np.sum(log_p_observed, axis=2)  # (n_theta, n_persons)
    # 添加对数先验（积分权重）
    log_prior = np.log(quad_weights)[:, None]  # (n_theta, 1)
    log_posterior = log_likelihood + log_prior  # (n_theta,  n_persons)
    # 计算后验概率（指数变换并归一化）
    posterior = np.exp(log_posterior)
    posterior /= np.sum(posterior, axis=0, keepdims=True)  # 转置为 (n_persons, n_theta)
    return posterior.T  # 转置为 (n_persons, n_theta)
  

def compute_m2pl_item_log_likelihood(theta, a_j, d_j, response_j, mask_matrix_j, posterior=None):
    """
    计算多维2pl模型的单个项目对数似然和
    
    参数:
    theta: 潜变量向量，形状(n_quadrature,)
    a_j: 项目j区分度参数，标量
    d_j: 项目j阈值参数，标量
    response_j: 被试对项目j的作答，形状(n_subjects, 1)
    mask_matrix_j: 掩码矩阵，形状(n_subjects, 1)
    posterior: 后验分布，形状(n_subjects, n_quadrature)
    
    返回:
    ll_sum: 所有被试的对数似然和（标量）
    """
    num_persons = response_j.shape[0]
    n_theta = theta.shape[0]
    # 计算所有类别在所有积分点上的概率
    P_cat_j =  m2pl_prob(theta, a_j, d_j).reshape(-1,1)  # (n_theta, 1)
    if posterior is None:
        ll = response_j * np.log(P_cat_j) + (1 - response_j) * np.log(1 - P_cat_j)
        ll = np.where(mask_matrix_j, ll, 0)
        ll= np.sum(ll, axis=1)  # (n_persons,)
        # 如果没有后验分布，直接返回对数似然
        ll_sum= np.sum(ll)  # (标量)
        return ll_sum
    else:
        p_observed=response_j.reshape(1, num_persons, 1)* P_cat_j[:, np.newaxis, :] + \
        (1-response_j.reshape(1, num_persons, 1)) * (1 - P_cat_j[:, np.newaxis, :])  # (n_theta, n_persons, 1)
        # 转为对数概率
        log_p_observed_j = np.log(p_observed) # (n_theta, n_persons,1)
        # 应用作答掩码（未作答设为0）
        mask_3d_j = mask_matrix_j.astype(bool)[None, :]  # (1, n_persons,1)
        log_p_observed_j = np.where(mask_3d_j, log_p_observed_j, 0.0).squeeze(axis=2)  # (n_theta, n_persons)
        # 应用后验分布并计算期望
        # posterior: (n_persons, n_theta) -> 转置为 (n_theta, n_persons)
        expected_log_lik_j = np.sum(log_p_observed_j * posterior.T, axis=0)  # (n_subjects,)
        # 计算所有被试的对数似然和
        ll_sum = np.sum(expected_log_lik_j)
        return ll_sum
    


def compute_m2pl_log_likelihood(theta, a, d, response, mask_matrix, posterior=None):
    """
    计算多维2pl模型的后验分布
    
    参数:
    theta: 潜变量向量，形状(n_quadrature,)
    a: 区分度参数向量，形状(n_items,)
    d: 阈值参数向量，形状(n_items,)
    response: 被试作答矩阵，形状(n_subjects, n_items)
    mask_matrix: 掩码矩阵，形状(n_subjects, n_items)
    n_categories: 每个项目的类别数(计分等级)的列表，长度为n_items
    quad_weights: 积分权重，形状(n_quadrature,)
    返回:
    posterior: 后验分布，形状(n_subjects, n_quadrature)
    """
    num_persons, num_items = response.shape
    n_theta = theta.shape[0]
    P_cat =  m2pl_prob(theta, a, d)  # (n_theta, n_items)
    if posterior is None:
        ll = response * np.log(P_cat) + (1 - response) * np.log(1 - P_cat)
        ll = np.where(mask_matrix, ll, 0)
        ll= np.sum(ll, axis=1)  # (n_persons,)
        # 如果没有后验分布，直接返回对数似然
        return ll
    else:
        p_observed=response.reshape(1, num_persons, num_items)* P_cat[:, np.newaxis, :] + \
        (1-response.reshape(1, num_persons, num_items)) * (1 - P_cat[:, np.newaxis, :])  # (n_theta, n_persons, n_items)
        # 转为对数概率
        log_p_observed = np.log(p_observed)  # (n_theta, n_persons, n_items)
        # 应用作答掩码（未作答项目设为0）
        mask_3d = mask_matrix.astype(bool)[None, :, :]  # (1, n_persons, n_items)
        log_p_observed = np.where(mask_3d, log_p_observed, 0.0)  # (n_theta, n_persons, n_items)
        # 计算对数似然（沿项目维度求和）: (n_theta, n_persons)
        log_likelihood = np.sum(log_p_observed, axis=2)  # (n_theta, n_persons)
        # 应用后验分布并计算期望
        # posterior: (n_persons, n_theta) -> 转置为 (n_theta, n_persons)
        expected_log_lik = np.sum(log_likelihood * posterior.T, axis=0)  # (n_persons,)
        # 返回每个被试的对数似然
        return expected_log_lik
    

def compute_m2pl_log_likelihood_item(theta, a, d, response, mask_matrix):
    """
    计算m2pl模型每个项目(item)的总对数似然。
    参数:
    theta: 潜变量向量，形状(n_persons,)
    a: 区分度参数矩阵，形状(n_items, n_dims)
    d: 难度参数向量，形状(n_items,)
    response: 被试作答矩阵，形状(n_persons, n_items)
    mask_matrix: 掩码矩阵，形状(n_persons, n_items)
    返回:
    total_log_likelihood: 每个项目的总对数似然，形状为(n_items,)
    np.array: 形状为 (J,) 的向量, 每个元素是对应项目的总对数似然。
    """
    p = m2pl_prob(theta, a, d)
    p = np.clip(p, 1e-9, 1 - 1e-9) # 保证数值稳定性
    ll_matrix = response * np.log(p) + (1 - response) * np.log(1 - p)
    total_log_likelihood = np.sum(ll_matrix * mask_matrix, axis=0)  # 沿着学生轴(axis=0)求和，得到每个项目的总对数似然
    return total_log_likelihood



###############################生成积分节点和采样相关函数####################################
#采样函数

def mcmc_sampling(theta, a, d, response, mask_matrix, rv, step_sizes, burn_in, n_samples,method='m2pl',n_categories=None):
    """
    执行Metropolis-Hastings MCMC采样过程
    参数:
    theta -- 初始参数值，形状为(n, dim)
    a -- 项目区分度向量，形状为(m, dim)
    d -- 项目难度向量，形状为(m,)
    response -- 响应矩阵，形状为(n, m)
    mask_matrix -- 掩码矩阵，形状为(n, m)
    rv -- 先验分布对象（需有logpdf方法）
    step_sizes -- 初始步长，形状为(n,)
    burn_in -- 预烧期迭代次数
    n_samples -- 采样期迭代次数
    method -- 模型类型，'mgrm'或'm2pl'(多维才采样,单维使用高斯-赫尔米特积分更高效)
    
    返回:
    samples -- 采样样本，形状为(n, n_samples, dim)
    theta_curr -- 最终参数状态，形状为(n, dim)
    step_sizes -- 调整后的步长，形状为(n,)
    all_accept -- 接受矩阵，形状为(n, burn_in + n_samples)
    """
    n, dim = theta.shape
    theta_curr = theta.copy()
    
    # 初始化存储
    samples = np.zeros((n, n_samples, dim))
    all_accept = np.zeros((n, burn_in + n_samples), dtype=bool)
    
    # MCMC采样循环
    for s in range(burn_in + n_samples):
        # 生成候选样本
        proposals = theta_curr + np.random.normal(0, step_sizes[:, None], (n, dim))
        if method == 'm2pl':
            # 计算对数似然（多维2PL模型）
            ll_curr = compute_m2pl_log_likelihood(theta_curr, a, d, response, mask_matrix)
            ll_prop = compute_m2pl_log_likelihood(proposals, a, d, response, mask_matrix)
        elif method == 'mgrm':
            # 计算对数似然（多维GRM模型）
            ll_curr = compute_mgrm_log_likelihood(theta_curr, a, d, response, mask_matrix, n_categories=n_categories)
            ll_prop = compute_mgrm_log_likelihood(proposals, a, d, response, mask_matrix, n_categories=n_categories)
        # 计算先验概率
        prior_curr = rv.logpdf(theta_curr)
        prior_prop = rv.logpdf(proposals)
        # 计算接受概率
        log_accept = ll_prop + prior_prop - ll_curr - prior_curr
        log_accept = np.minimum(log_accept, 0)  # 保证概率<=1
        accept_prob = np.exp(log_accept)
        # 决定是否接受候选样本
        accept = np.random.rand(n) < accept_prob
        theta_curr[accept] = proposals[accept]
        all_accept[:, s] = accept
        # 存储采样期的样本
        if s >= burn_in:
            samples[:, s - burn_in, :] = theta_curr.copy()
    # 更新参数估计为采样期的均值
    theta = np.mean(samples, axis=1)
    # 自适应调整步长（基于采样期接受率）
    post_burn_accept = all_accept[:, burn_in:]
    accept_rates = np.mean(post_burn_accept, axis=1)
    step_sizes[accept_rates < 0.2] *= 0.9  # 接受率低则减小步长
    step_sizes[accept_rates > 0.4] *= 1.1  # 接受率高则增大步长
    step_sizes = np.clip(step_sizes, 0.05, 0.5)  # 确保步长在合理范围内
    return samples, theta_curr, step_sizes, all_accept




#节点
def create_mirt_quadrature(n_points, n_dims):
    """
    创建高斯-赫尔米特积分节点和权重
    参数:
    n_points: 积分点数
    n_dims: 潜变量维度数
    
    返回:
    theta: 积分节点，形状为(n_points, n_dims)
    quad_weights: 积分权重，形状为(n_points,)
    """
    x, w = np.polynomial.hermite.hermgauss(n_points)
    points_1d = x * np.sqrt(2)
    weights_1d = w / np.sqrt(np.pi)
    quad_points_nd = np.array(list(product(points_1d, repeat=n_dims)))
    weight_list = list(product(weights_1d, repeat=n_dims))
    quad_weights_nd = np.prod(np.array(weight_list), axis=1)
    return quad_points_nd, quad_weights_nd

def create_irt_quadrature(n_points):
    """
    创建单维高斯-赫尔米特积分节点和权重
    参数:
    n_points: 积分点数
    
    返回:
    theta: 积分节点，形状为(n_points,)
    quad_weights: 积分权重，形状为(n_points,)
    """
    x, w = np.polynomial.hermite.hermgauss(n_points)
    theta = x * np.sqrt(2)  # 标准化
    quad_weights = w / np.sqrt(np.pi)  # 权重归一化
    return theta, quad_weights


###################################GRM模型(GRM)相关函数##########################
def ensure_ordered_thresholds(thresholds):
    """确保阈值严格递增并且在有效范围内"""
    if len(thresholds) <= 1:
        return thresholds
    
    # 首先确保值在合理范围内
    ordered = np.clip(thresholds.copy(), -4.0, 4.0)
    
    # 确保严格递增
    for i in range(1, len(ordered)):
        if ordered[i] <= ordered[i-1]:
            ordered[i] = ordered[i-1] + 0.01
    
    # 再次检查上界，确保不超过4.0
    if len(ordered) > 0 and ordered[-1] > 4.0:
        # 如果最后一个值超过了4.0，则需要递减调整
        ordered[-1] = 4.0
        for i in range(len(ordered)-2, -1, -1):
            if ordered[i+1] - ordered[i] < 0.01:
                ordered[i] = ordered[i+1] - 0.01
    
    return ordered

    

##################################计算总对数似然####################################
def compute_total_log_likelihood(theta, a, d, response, mask_matrix,method='mgrm_step_gh', n_categories=None, posterior=None):
    """计算总对数似然"""
    if method == 'mgrm_step_gh' or method == 'mgrm_stand_gh':
        if n_categories is None:
            raise ValueError("多维GRM模型需要提供n_categories参数，用以指定每个项目的类别数")
        if posterior is None:
            raise ValueError("使用高斯-赫尔米特积分需要提供posterior参数")
        ll= compute_mgrm_log_likelihood(theta, a, d, response, mask_matrix, n_categories, posterior)
        return np.sum(ll)
    
    elif method == 'mgrm_step_mcmc' or method == 'mgrm_stand_mcmc':
        if n_categories is None:
            raise ValueError("多维GRM模型需要提供n_categories参数，用以指定每个项目的类别数")
        ll = compute_mgrm_log_likelihood(theta, a, d, response, mask_matrix, n_categories)
        return np.sum(ll)
    
    elif method == 'm2pl_gh':
        if posterior is None:
            raise ValueError("使用高斯-赫尔米特积分需要提供posterior参数")
        ll = compute_m2pl_log_likelihood(theta, a, d, response, mask_matrix, n_categories, posterior)
        return np.sum(ll)
    elif method == 'm2pl_mcmc':
        ll = compute_m2pl_log_likelihood(theta, a, d, response, mask_matrix, n_categories)
        return np.sum(ll)
    
    elif method == 'grm_step' or method == 'grm_stand':
        #单维默认采样高斯-赫尔米特积分，更高效
        if posterior is None:
            raise ValueError("使用高斯-赫尔米特积分需要提供posterior参数")
        ll = compute_grm_log_likelihood(theta, a, d, response, mask_matrix, n_categories, posterior)
        return np.sum(ll)
    elif method == '2pl':
        #单维默认采样高斯-赫尔米特积分，更高效
        if posterior is None:
            raise ValueError("使用高斯-赫尔米特积分需要提供posterior参数")
        ll = compute_2pl_log_likelihood(theta, a, d, response, mask_matrix, posterior)
        return ll


