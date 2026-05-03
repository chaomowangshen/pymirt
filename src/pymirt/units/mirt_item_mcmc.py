import numpy as np
import scipy.stats as stats
from .units import (
    m2pl_prob,compute_m2pl_log_likelihood_item, compute_m2pl_log_likelihood,pad_grm_parameters,
    compute_mgrm_log_likelihood_item,compute_mgrm_log_likelihood
                    )





def update_a_parameters(rv, theta, a, d, response, mask_matrix, Q, step_size=0.2):
    """
    使用MCMC方法更新M2PL模型的区分度参数。
    参数：
        rv: 区分度参数的先验分布（如 Normal(0, 0.5)）
        theta: 能力矩阵，形状为 (n, dim)
        a: 区分度参数矩阵，形状为 (items, dim)
        d: 阈值参数矩阵，形状为 (n, k)
        response: 响应变量,形状为 (n, items)
        mask_matrix: 掩码矩阵，用于指示哪些响应是有效的，形状同response
        Q: Q矩阵
        step_size: 步长大小
    返回：
        a_new: 更新后的区分度参数矩阵，形状为 (items, dim)
        step_size: 更新后的步长大小
    """
    a_curr = a.copy()
    ll_curr = compute_m2pl_log_likelihood_item(theta, a_curr, d, response, mask_matrix)
    
    # 1. 转换到对数空间（注意：rv 现在是 log(a) 的先验！）
    log_a_curr = np.log(np.maximum(a_curr, 1e-9))  # 保护log(0)
    log_a_prop = log_a_curr.copy()
    
    # 2. 仅在活动维度（Q=1）上添加噪声
    a_mask = Q == 1
    log_a_prop[a_mask] += np.random.normal(0, step_size, size=np.sum(a_mask))
    
    # 3. 转换回原始尺度
    a_prop = np.exp(log_a_prop)
    a_prop[Q == 0] = 0  # 确保非活动维度为0
    
    # 4. 计算似然（在原始a空间）
    ll_prop = compute_m2pl_log_likelihood_item(theta, a_prop, d, response, mask_matrix)
    
    # 5. 计算先验（直接在 log(a) 空间计算！）
    def calculate_log_prior(log_a_matrix, Q_matrix, dist):
        log_p = dist.logpdf(log_a_matrix) * Q_matrix  # dist 是 log(a) 的先验（如 Normal(0, 0.5)）
        return np.sum(log_p, axis=1)
    
    prior_curr = calculate_log_prior(log_a_curr, Q, rv)
    prior_prop = calculate_log_prior(log_a_prop, Q, rv)
    
    # 6. 接受概率（无需雅可比校正！）
    log_alpha = (ll_prop - ll_curr) + (prior_prop - prior_curr)
    accept_prob = np.exp(np.minimum(log_alpha, 0))
    accept_flag = np.random.rand(a.shape[0]) < accept_prob
    
    # 7. 更新参数和步长
    a_new = np.where(accept_flag[:, None], a_prop, a_curr)
    accept_rate = np.mean(accept_flag)
    
    # 动态调整步长
    if accept_rate < 0.2:
        step_size *= 0.9
    elif accept_rate > 0.4:
        step_size *= 1.1
    step_size = np.clip(step_size, 0.05, 0.5)
    
    return a_new, step_size







def update_d_parameters(rv,theta, a, d, response, mask_matrix, step_size=0.2):
    """
    使用MCMC方法更新M2PL模型的阈值参数。
    参数：
        rv: 阈值参数的先验分布（如 Normal(0, 1.5)）
        theta: 能力矩阵，形状为 (n, dim)
        a: 区分度参数矩阵，形状为 (items, dim)
        d: 阈值参数矩阵，形状为 (n, k)，k=np.max(n_categories)-1
        response: 响应变量,形状为 (n, items)
        mask_matrix: 掩码矩阵，用于指示哪些响应是有效的，形状同response
        step_size: 步长大小
    返回：
        d_new: 更新后的阈值参数矩阵，形状为 (n, k)
        step_size: 更新后的步长大小
    """
    d_curr = d.copy()
    ll_curr = compute_m2pl_log_likelihood_item(theta, a, d_curr, response, mask_matrix)
    # 1. 在原始尺度上直接生成提议参数
    d_prop = d_curr + np.random.normal(0, step_size, size=d_curr.shape)
    # 2. 计算提议参数的似然 
    ll_prop = compute_m2pl_log_likelihood_item(theta, a, d_prop, response, mask_matrix)
    
    # 3. 计算先验 
    prior_curr = rv.logpdf(d_curr.flatten())
    prior_prop = rv.logpdf(d_prop.flatten())

    # 4. 计算接受概率 
    log_alpha = (ll_prop - ll_curr) + (prior_prop - prior_curr)
    
    # 5. 更新参数和对数似然
    accept_prob = np.exp(np.minimum(log_alpha, 0))
    accept_flag = np.random.rand(d.shape[0]) < accept_prob
    accept_rates = np.mean(accept_flag)
    
    d_new = np.where(accept_flag, d_prop.flatten(), d_curr.flatten()).reshape(d.shape)
    
    # 6. 更新似然,避免下一步重新计算
    #ll_new = np.where(accept_flag, ll_prop, ll_curr)

    # 7. 更新step_size
    if accept_rates < 0.2:
        step_size *= 0.9
    elif accept_rates > 0.4:
        step_size *= 1.1
    step_size = np.clip(step_size, 0.05, 0.5)  # 限制step_size在合理范围内

    return d_new, step_size



def update_theta_parameters(rv,theta, a, d, response, mask_matrix,  step_size=0.2):
    """
    使用MCMC方法更新M2PL模型的能力参数。
    参数：
        rv: 能力参数的先验分布（如 Normal(0, 1.5)）
        theta: 能力矩阵，形状为 (n, dim)
        a: 区分度参数矩阵，形状为 (items, dim)
        d: 阈值参数矩阵，形状为 (n, k)
        response: 响应变量,形状为 (n, items)
        mask_matrix: 掩码矩阵，用于指示哪些响应是有效的，形状同response
        step_size: 步长大小
    返回：
        theta_new: 更新后的能力参数矩阵，形状为 (n, dim)
        step_size: 更新后的步长大小
    """
    n, dim = theta.shape
    theta_curr = theta.copy()
    ll_curr= compute_m2pl_log_likelihood(theta_curr, a, d, response, mask_matrix)
    # 1. 在原始尺度上生成提议参数
    theta_prop = theta_curr + np.random.normal(0, step_size, size=theta_curr.shape)
    
    # 2. 计算提议参数的似然
    ll_prop = compute_m2pl_log_likelihood(theta_prop, a, d, response, mask_matrix)
    
    # 3. 计算先验
    prior_curr = rv.logpdf(theta_curr)
    prior_prop = rv.logpdf(theta_prop)

    # 4. 计算接受概率
    log_alpha = (ll_prop - ll_curr) + (prior_prop - prior_curr)
    
    # 5. 更新参数和对数似然
    accept_prob = np.exp(np.minimum(log_alpha, 0))#(n,)
    accept_flag = np.random.rand(n) < accept_prob
    accept_rates = np.mean(accept_flag)
    
    theta_new = np.where(accept_flag[:, None], theta_prop, theta_curr)

    #6.更新step_size
    if accept_rates < 0.2:
        step_size *= 0.9
    elif accept_rates > 0.4:
        step_size *= 1.1
    step_size = np.clip(step_size, 0.05, 0.5)  # 限制step_size在合理范围内
    
    # 根据接受与否，决定新的对数似然状态
    return theta_new, step_size








def d_to_delta_masked(d_matrix, d_mask):
    """
    将有序的d（降序）转换为无约束的delta，并支持掩码。
    参数：
        d_matrix: 有序的d矩阵，形状为 (n, k)，其中 n 是样本数，k 是类别数。
        d_mask: 掩码矩阵，形状为 (n, k)，指示哪些d是有效的（True表示有效，False表示无效）。
    返回：
        delta_matrix: 无约束的delta矩阵，形状为 (n, k)。
    """
    delta_matrix = np.zeros_like(d_matrix)
    delta_matrix[:, 0] = d_matrix[:, 0]  #保留d1
    increments = -np.diff(d_matrix, axis=1)  #  对减量的相反数取对数
    delta_matrix[:, 1:] = np.log(np.maximum(increments, 1e-9))  
    delta_matrix=delta_matrix * d_mask#应用掩码，去除不存在的d
    return delta_matrix

def delta_to_d_masked(delta_matrix, d_mask):
    """
    将无约束的delta转换为有序的d（降序），并支持掩码。
    参数：
        delta_matrix: 无约束的delta矩阵，形状为 (n, k)，其中 n 是样本数，k 是类别数。
        d_mask: 掩码矩阵，形状为 (n, k)，指示哪些d是有效的（True表示有效，False表示无效）。
    返回：
        d_matrix: 有序的d矩阵，形状为 (n, k)。
    """
    d_matrix = np.zeros_like(delta_matrix)
    d_matrix[:, 0] = delta_matrix[:, 0]  #保留d1
    increments = -np.exp(delta_matrix[:, 1:]) * d_mask[:, 1:]
    cumulative_increments = np.cumsum(increments, axis=1)
    d_matrix[:, 1:] = d_matrix[:, 0][:, None] + cumulative_increments
    d_matrix=d_matrix * d_mask
    return d_matrix




def update_grm_a_parameters(rv, theta, a, d,d_mask, response, mask_matrix, Q,n_categories,step_size=0.2):
    """
    使用mcmc方法更新mgrm模型的区分度参数
    参数：
        rv: 区分度参数的先验分布（如 Normal(0, 0.5)）
        theta: 能力矩阵，形状为 (n, dim)
        a: 区分度参数矩阵，形状为 (items, dim)
        d: 阈值参数矩阵，形状为 (n, k)
        d_mask: 掩码矩阵，指示哪些d是有效的（True表示有效，False表示无效）
        response: 响应变量,形状为 (n, items)
        mask_matrix: 掩码矩阵，用于指示哪些响应是有效的，形状同response
        Q: Q矩阵
        n_categories: 类别数列表
        step_size: 步长大小
    返回：
        a_new: 更新后的区分度参数矩阵，形状为 (items, dim)
        step_size: 更新后的步长大小
    """
    a_curr = a.copy()
    log_a_curr = np.log(np.maximum(a_curr, 1e-9))  # 保护log(0)
    log_a_prop = log_a_curr.copy()
    a_mask = Q == 1
    log_a_prop[a_mask] += np.random.normal(0, step_size, size=np.sum(a_mask))
    a_prop = np.exp(log_a_prop)
    a_prop[Q == 0] = 0  # 确保非活动维度为0

    ll_curr = compute_mgrm_log_likelihood_item(theta, a_curr, d, d_mask, response, mask_matrix,n_categories)
    ll_prop = compute_mgrm_log_likelihood_item(theta, a_prop, d, d_mask, response, mask_matrix,n_categories)
    def calculate_log_prior(log_a_matrix, Q_matrix, dist):
        log_p = dist.logpdf(log_a_matrix) * Q_matrix  # dist 是 log(a) 的先验（如 Normal(0, 0.5)）
        return np.sum(log_p, axis=1)
    prior_curr = calculate_log_prior(log_a_curr, Q, rv)
    prior_prop = calculate_log_prior(log_a_prop, Q, rv)
    log_alpha = (ll_prop - ll_curr) + (prior_prop - prior_curr)

    accept_prob = np.exp(np.minimum(log_alpha, 0))
    accept_flag = np.random.rand(a.shape[0]) < accept_prob
    accept_rate = np.mean(accept_flag)

    a_new = np.where(accept_flag[:, None], a_prop, a_curr)
    # 动态调整步长
    if accept_rate < 0.2:
        step_size *= 0.9
    elif accept_rate > 0.4:
        step_size *= 1.1
    step_size = np.clip(step_size, 0.05, 0.5)
    return a_new, step_size




def update_grm_d_parameters(rv,theta, a, d,d_mask, response, mask_matrix,n_categories, step_size=0.2):
    """
    使用mcmc方法更新mgrm模型的阈值参数
    参数：
        rv: 阈值参数的先验分布（如 Normal(0, 1.5)）
        theta: 能力矩阵，形状为 (n, dim)
        a: 区分度参数矩阵，形状为 (items, dim)
        d: 阈值参数矩阵，形状为 (n, k)，k=np.max(n_categories)-1
        d_mask: 掩码矩阵，指示哪些d是有效的（True表示有效，False表示无效）
        response: 响应变量,形状为 (n, items)
        mask_matrix: 掩码矩阵，用于指示哪些响应是有效的，形状同response
        n_categories: 类别数列表
        step_size: 步长大小
    返回：
        d_new: 更新后的阈值参数矩阵，形状为 (n, k)
        step_size: 更新后的步长大小

    """
    d_curr = d.copy()
    delta_curr = d_to_delta_masked(d_curr, d_mask)
    noise = np.random.normal(0, step_size, size=delta_curr.shape)
    delta_prop = delta_curr + noise * d_mask
    d_prop = delta_to_d_masked(delta_prop, d_mask)

    ll_curr = compute_mgrm_log_likelihood_item(theta, a, d_curr,d_mask, response, mask_matrix, n_categories)
    ll_prop = compute_mgrm_log_likelihood_item(theta, a, d_prop,d_mask, response, mask_matrix, n_categories)
    prior_curr = np.sum(rv.logpdf(d_curr) * d_mask, axis=1)
    prior_prop = np.sum(rv.logpdf(d_prop) * d_mask, axis=1)
    log_jacobian_curr = np.sum(delta_curr[:, 1:] * d_mask[:, 1:], axis=1)
    log_jacobian_prop = np.sum(delta_prop[:, 1:] * d_mask[:, 1:], axis=1)
    log_alpha = (ll_prop - ll_curr) + (prior_prop - prior_curr) + (log_jacobian_prop - log_jacobian_curr)
    
    accept_prob = np.exp(np.minimum(log_alpha, 0))
    accept_flag = np.random.rand(d.shape[0]) < accept_prob
    accept_rates = np.mean(accept_flag)
    d_new = np.where(accept_flag[:,None], d_prop, d_curr)
    if accept_rates < 0.2:
        step_size *= 0.9
    elif accept_rates > 0.4:
        step_size *= 1.1
    step_size = np.clip(step_size, 0.05, 0.5)   
    return d_new, step_size




def update_grm_theta_parameters(rv,theta, a, d,d_mask, response, mask_matrix,n_categories, step_size=0.2):
    """
    使用mcmc方法更新mgrm模型的能力参数
    参数：
        rv: 能力参数的先验分布（如 Normal(0, 1.5)）
        theta: 能力矩阵，形状为 (n, dim)
        a: 区分度参数矩阵，形状为 (items, dim)
        d: 阈值参数矩阵，形状为 (n, k)，k=np.max(n_categories)-1
        d_mask: 掩码矩阵，指示哪些d是有效的（True表示有效，False表示无效）
        response: 响应变量,形状为 (n, items)
        mask_matrix: 掩码矩阵，用于指示哪些响应是有效的，形状同response
        n_categories: 类别数列表
        step_size: 步长大小
    返回：
        theta_new: 更新后的能力参数矩阵，形状为 (n, dim)
        step_size: 更新后的步长大小
    """
    n, dim = theta.shape
    theta_curr = theta.copy()
    # 1. 在原始尺度上生成提议参数
    theta_prop = theta_curr + np.random.normal(0, step_size, size=theta_curr.shape)
    
    ll_curr = compute_mgrm_log_likelihood(theta_curr, a, d, response, mask_matrix, n_categories,d_mask=d_mask)
    ll_prop = compute_mgrm_log_likelihood(theta_prop, a, d, response, mask_matrix, n_categories,d_mask=d_mask)
    prior_curr = rv.logpdf(theta_curr)
    prior_prop = rv.logpdf(theta_prop)
    log_alpha = (ll_prop - ll_curr) + (prior_prop - prior_curr)
    
    accept_prob = np.exp(np.minimum(log_alpha, 0))#(n,)
    accept_flag = np.random.rand(n) < accept_prob
    accept_rates = np.mean(accept_flag)
    
    theta_new = np.where(accept_flag[:, None], theta_prop, theta_curr)
    if accept_rates < 0.2:
        step_size *= 0.9
    elif accept_rates > 0.4:
        step_size *= 1.1
    step_size = np.clip(step_size, 0.05, 0.5)  
    return theta_new, step_size




    


    
