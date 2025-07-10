from scipy.optimize import minimize
from scipy.stats import multivariate_normal,norm
import numpy as np
import time
from .units import (create_mirt_quadrature,pad_grm_parameters,mgrm_prob_categories,compute_mgrm_posterior,compute_mgrm_item_log_likelihood,
                    compute_mgrm_log_likelihood,ensure_ordered_thresholds,compute_m2pl_log_likelihood_item,
                    m2pl_prob,compute_m2pl_posterior,compute_m2pl_item_log_likelihood,compute_m2pl_log_likelihood,mcmc_sampling,
                    update_a_parameters,update_d_parameters,update_theta_parameters,update_grm_d_parameters,
                    update_grm_a_parameters,update_grm_theta_parameters
                    )


#################################多维2PL#################################

#######高斯-赫尔米特积分实现#######
def mirt_em(response, mask_matrix,Q,n_quadrature=27, max_iter=100, tol=1e-4,  verbose=False):
    '''
    多维IRT的EM算法，使用高斯-赫尔米特积分。适用于3维及以下,3维也很比较慢
    参数:
    - response: 被试的响应矩阵，形状为 (n, items)
    - mask_matrix: 掩码矩阵，形状为 (n, items)，1表示有数据，0表示缺失
    - Q: 题目特征矩阵，形状为 (items, dim)
    - n_quadrature: 高斯-赫尔米特积分点数，默认为27
    - max_iter: 最大迭代次数
    - tol: 收敛阈值
    - verbose: 是否打印迭代信息
    返回:
    - a_est: 估计的区分度参数，形状为 (items, dim)
    - d_est: 估计的难度参数，形状为 (items, )
    '''
    n, items = response.shape
    dim = Q.shape[1]
    response = np.nan_to_num(response)  # 将NaN替换为0
    #初始化参数
    a_est = np.random.uniform(0.2, 3, (items, dim)) * Q
    d_est = np.zeros(items)
    theta_quad, weights_quad = create_mirt_quadrature(n_quadrature, dim)
    prev_ll= -np.inf
    total_time = 0
    for iteration in range(max_iter):
        if verbose:
            print(f"=== 迭代 {iteration + 1}/{max_iter} ===")
        start_time = time.time()
        # E步：计算后验分布
        posterior = compute_m2pl_posterior(theta_quad, a_est, d_est, response, mask_matrix,weights_quad)
        # M步：更新参数
        a_new=np.zeros_like(a_est)
        d_new=np.zeros_like(d_est)
        for j in range(items):
            mask_j=mask_matrix[:,j].reshape(-1,1)
            response_j=response[:,j].reshape(-1,1)
            active_dims = Q[j] == 1
            num_a = np.sum(active_dims)
            def negative_log_likelihood(params):
                a_params = params[:num_a]
                d_j = params[num_a]
                a_j = np.zeros(dim)
                a_j[active_dims] = a_params
                expected_ll=compute_m2pl_item_log_likelihood(theta_quad, a_j, d_j, response_j, mask_j, posterior)
                return -expected_ll
            init_params =np.concatenate([a_est[j, active_dims], [d_est[j]]])
            bounds = [(0.1, None)] * num_a+ [(None, None)]
            result = minimize(negative_log_likelihood, init_params, bounds=bounds, method='L-BFGS-B')
            if result.success:
                a_new[j, active_dims] = result.x[:-1]
                d_new[j] = result.x[-1]
            else:
                if verbose:
                    print(f"题目 {j} 参数优化失败,采用上次估计值")
                a_new[j, active_dims] = a_est[j, active_dims]
                d_new[j] = d_est[j]
        end_time = time.time()
        a_est, d_est = a_new, d_new
        #收敛检查
        current_ll =np.sum(compute_m2pl_log_likelihood(theta_quad, a_est, d_est, response, mask_matrix, posterior))
        delta_ll = current_ll - prev_ll
        total_time += (end_time - start_time)
        if verbose:
            print(f"对数似然变化: {delta_ll:.6f}, 耗时: {end_time - start_time:.2f}秒")
        if iteration > 0 and abs(current_ll - prev_ll) < tol:
            if verbose:
                print(f"*** 收敛于迭代 {iteration + 1} ***, 总耗时: {total_time:.2f}秒")
            break
        prev_ll = current_ll
    return a_est, d_est




###############MCEM实现#############
def mirt_mcem(response, mask_matrix,Q,n_samples=300,burn_in=200, max_iter=100, tol=1e-4, sample_interval=10, verbose=False):
    '''
    多维IRT的MCEM算法，使用马尔科夫链蒙特卡洛方法。适用于多维IRT模型
    该方法通过采样后验分布来估计参数，适用于高维和复杂模型。
    参数:
    - response: 被试的响应矩阵，形状为 (n, items)
    - mask_matrix: 掩码矩阵，形状为 (n, items)，1表示有数据，0表示缺失
    - Q: 题目特征矩阵，形状为 (items, dim)
    - n_samples: MCMC有效采样次数
    - burn_in: 烧入期，前多少次采样不用于估计
    - max_iter: 最大迭代次数
    - tol: 收敛阈值
    - sample_interval: 每隔多少次迭代采样一次
    - verbose: 是否打印迭代信息
    返回:
    - a_est: 估计的区分度参数，形状为 (items, dim)
    - d_est: 估计的难度参数，形状为 (items, )
    - theta_est: 估计的能力参数，形状为 (n, dim)
    '''
    n, items = response.shape
    dim = Q.shape[1]
    response = np.nan_to_num(response)  # 将NaN替换为0
    #初始化参数
    a_est = np.random.uniform(0.2, 3, (items, dim)) * Q
    d_est = np.zeros(items)
    theta_est=np.zeros((n, dim))
    step_sizes = np.full(n, 0.2)
    rv = multivariate_normal(mean=np.zeros(dim), cov=np.eye(dim))
    prev_ll= -np.inf
    total_time = 0
    for iteration in range(max_iter):
        if verbose:
            print(f"=== 迭代 {iteration + 1}/{max_iter} ===")
        start_time = time.time()
        # 采样频率控制
        sample_this_iter = (iteration < 10) or (iteration % sample_interval == 0)
        if sample_this_iter:
            samples, theta_curr, step_sizes, all_accept = mcmc_sampling(theta_est, a_est, d_est, response, 
                                                                        mask_matrix, rv, step_sizes, burn_in, n_samples,method='m2pl')
            theta_est = np.mean(samples, axis=1)
        a_new = np.zeros_like(a_est)
        d_new = np.zeros_like(d_est)
        for j in range(items):
            if n_samples>100:
                theta_flat = samples[:,-100:,:,].reshape(-1, dim)  # [n * 100, dim]
                resp_flat = np.repeat(response[:, j], 100).reshape(-1,1)   # [n * 100,1]
                mask_flat = np.repeat(mask_matrix[:, j], 100).reshape(-1,1)  # [n * 100,1]
            else:
                theta_flat = samples.reshape(-1, dim)  # [n * n_samples, dim]
                resp_flat = np.repeat(response[:, j], n_samples).reshape(-1,1)   # [n * n_samples,1]
                mask_flat = np.repeat(mask_matrix[:, j], n_samples).reshape(-1,1)  # [n * n_samples]
            active_dims = Q[j] == 1
            num_a = np.sum(active_dims)
            def negative_log_likelihood(params):
                a_params = params[:num_a]
                d_j = params[num_a]
                a_j = np.zeros(dim)
                a_j[active_dims] = a_params
                expected_ll=compute_m2pl_item_log_likelihood(theta_flat, a_j, d_j, resp_flat, mask_flat)
                return -expected_ll
            init_params =np.concatenate([a_est[j, active_dims], [d_est[j]]])
            bounds = [(0.1, None)] * num_a + [(None, None)]
            result = minimize(negative_log_likelihood, init_params, bounds=bounds, method='L-BFGS-B')
            if result.success:
                a_new[j, active_dims] = result.x[:-1]
                d_new[j] = result.x[-1]
            else:
                if verbose:
                    print(f"题目 {j} 参数优化失败,采用上次估计值")
                a_new[j, active_dims] = a_est[j, active_dims]
                d_new[j] = d_est[j]
        end_time = time.time()
        a_est, d_est = a_new, d_new
        #收敛检查
        current_ll = np.sum(compute_m2pl_log_likelihood(theta_curr, a_est, d_est, response, mask_matrix))
        delta_ll = current_ll - prev_ll
        total_time += (end_time - start_time)
        if verbose:
            print(f"对数似然变化: {delta_ll:.6f}, 耗时: {end_time - start_time:.2f}秒")
        if iteration > 0 and abs(current_ll - prev_ll) < tol:
            if verbose:
                print(f"*** 收敛于迭代 {iteration + 1} ***, 总耗时: {total_time:.2f}秒")
            break
        prev_ll = current_ll
    if n_samples+ burn_in < 2000:
        samples, theta_curr, step_sizes, all_accept = mcmc_sampling(theta_est, a_est, d_est, response, 
                                                                        mask_matrix, rv, step_sizes, 1000, 1000,method='m2pl')
        theta_est = np.mean(samples, axis=1)
    return a_est, d_est,theta_est



#######SAEM实现#######
def mirt_saem(response, mask_matrix,Q, max_iter=100, tol=1e-4, verbose=False):
    '''
    多维IRT的SAEM算法，使用随机游走方法。适用于多维IRT模型
    该方法通过采样后验分布来估计参数，适用于高维和复杂模型。
    参数:
    - response: 被试的响应矩阵，形状为 (n, items)
    - mask_matrix: 掩码矩阵，形状为 (n, items)，1表示有数据，0表示缺失
    - Q: 题目特征矩阵，形状为 (items, dim)
    - max_iter: 最大迭代次数
    - tol: 收敛阈值
    - verbose: 是否打印迭代信息
    返回:
    - a_est: 估计的区分度参数，形状为 (items, dim)
    - d_est: 估计的难度参数，形状为 (items, )
    - theta_est: 估计的能力参数，形状为 (n, dim)
    '''
    n, items = response.shape
    dim = Q.shape[1]
    response = np.nan_to_num(response)  # 将NaN替换为0
    #初始化参数
    a_est = np.random.uniform(0.2, 3, (items, dim)) * Q
    d_est = np.zeros(items)
    theta_curr=np.zeros((n, dim))
    step_sizes = np.full(n, 0.2)
    burn_in=50
    n_samples=10
    gamma_sequence = np.ones(max_iter)
    for k in range(burn_in, max_iter):
        # 在预热期后，步长开始递减
        gamma_sequence[k] = 1.0 / (k - burn_in + 1)**0.7
    rv = multivariate_normal(mean=np.zeros(dim), cov=np.eye(dim))
    prev_ll= -np.inf
    total_time = 0
    for iteration in range(max_iter):
        start_time = time.time()
        gamma = gamma_sequence[iteration]
        if verbose:
            print(f"=== 迭代 {iteration + 1}/{max_iter} ===")
        samples, theta_curr, step_sizes, all_accept = mcmc_sampling(theta_curr, a_est, d_est, response, 
                                                                        mask_matrix, rv, step_sizes, burn_in, n_samples,method='m2pl')

        a_new = a_est.copy()
        d_new = d_est.copy()
        for j in range(items):
            active_dims = Q[j] == 1
            num_a = np.sum(active_dims)
            response_j=response[:,j].reshape(-1,1)
            mask_j=mask_matrix[:,j].reshape(-1,1)
            def negative_log_likelihood(params):
                a_params = params[:num_a]
                d_j = params[num_a]
                a_j = np.zeros(dim)
                a_j[active_dims] = a_params
                expected_ll=compute_m2pl_item_log_likelihood(theta_curr, a_j, d_j, response_j, mask_j)
                return -expected_ll
            init_params =np.concatenate([a_est[j, active_dims], [d_est[j]]])
            bounds = [(0.1, None)] * num_a + [(None, None)]
            result = minimize(negative_log_likelihood, init_params, bounds=bounds, method='L-BFGS-B')
            if result.success:
                a_new_target = result.x[:-1]
                d_new_target = result.x[-1]
                current_a=a_new[j,active_dims]
                a_new[j,active_dims]=(1-gamma)*current_a+gamma*a_new_target
                current_d = d_new[j]
                d_new[j] = (1 - gamma) * current_d + gamma * d_new_target
            else:
                if verbose:
                    print(f"题目 {j} 参数优化失败,采用上次估计值")
                a_new[j, active_dims] = a_est[j, active_dims]
                d_new[j] = d_est[j]
        end_time = time.time()
        a_est, d_est = a_new, d_new
        #收敛检查
        current_ll = np.sum(compute_m2pl_log_likelihood(theta_curr, a_est, d_est, response, mask_matrix))
        delta_ll = current_ll - prev_ll
        total_time += (end_time - start_time)
        if verbose:
            print(f"对数似然变化: {delta_ll:.6f}, 耗时: {end_time - start_time:.2f}秒")
        prev_ll = current_ll
    start_time = time.time()
    samples, theta_curr, step_sizes, all_accept = mcmc_sampling(theta_curr, a_est, d_est, response, 
                                                                        mask_matrix, rv, step_sizes, 1000, 1000,method='m2pl')
    theta_est = np.mean(samples, axis=1)
    end_time = time.time()
    total_time += (end_time - start_time)
    if verbose:
        print(f"=== SAEM方法估计完成, 总耗时: {total_time:.2f}秒 ===")
    return a_est, d_est,theta_est



################MCMC实现#############
def mirt_mcmc(response, mask_matrix,Q,n_samples=3000,burn_in=2000, verbose=False):
    '''
    多维IRT的MCMC算法，使用马尔科夫链蒙特卡洛方法。适用于多维IRT模型
    该方法通过采样后验分布来估计参数，适用于高维和复杂模型。
    参数:
    - response: 被试的响应矩阵，形状为 (n, items)
    - mask_matrix: 掩码矩阵，形状为 (n, items)，1表示有数据，0表示缺失
    - Q: 题目特征矩阵，形状为 (items, dim)
    - n_samples: MCMC有效采样次数
    - burn_in: 烧入期，前多少次采样不用于估计
    - verbose: 是否打印迭代信息
    返回:
    - a_est: 估计的区分度参数，形状为 (items, dim)
    - d_est: 估计的难度参数，形状为 (items, )
    - theta_est: 估计的能力参数，形状为 (n, dim)
    '''
    time_start = time.time()
    n, items = response.shape
    dim = Q.shape[1]
    response = np.nan_to_num(response)  # 将NaN替换为0
    #初始化参数
    a= np.random.uniform(0.2, 3, (items, dim)) * Q
    d= np.random.normal(0, 1, items)
    theta = np.zeros((n, dim))
    rv_log_a=norm(loc=0, scale=np.sqrt(0.5))
    rv_d=norm(loc=0, scale=1)
    rv_theta=multivariate_normal(mean=np.zeros(dim), cov=np.eye(dim))
    n_samples_a=np.zeros((items,n_samples,dim))
    n_samples_d=np.zeros((items,n_samples))
    n_samples_theta=np.zeros((n,n_samples,dim))
    step_size_a=step_size_d=step_size_theta=0.2
    for iteration in range(n_samples + burn_in):
        if verbose and iteration % 500 == 0:
            print(f"=== 迭代 {iteration + 1}/{n_samples + burn_in} ===")
        theta,step_size_theta = update_theta_parameters(rv_theta,theta,a, d, response, mask_matrix, step_size=step_size_theta)
        a,step_size_a = update_a_parameters(rv_log_a,theta, a, d, response, mask_matrix, Q, step_size=step_size_a)
        d,step_size_d = update_d_parameters(rv_d,theta, a, d, response, mask_matrix, step_size=step_size_d)
        if iteration >= burn_in:
            idx = iteration - burn_in
            n_samples_a[:, idx, :] = a.copy()
            n_samples_d[:, idx] = d.copy()
            n_samples_theta[:, idx, :] = theta.copy()
    # 计算后验均值
    a_est = np.mean(n_samples_a, axis=1)
    d_est = np.mean(n_samples_d, axis=1)
    theta_est = np.mean(n_samples_theta, axis=1)
    time_end = time.time()
    total_time = time_end - time_start
    if verbose:
        print(f"=== MCMC采样完成, 共{n_samples}次有效采样, 烧入期{burn_in}次, 总耗时: {total_time:.2f}秒 ===")
    return a_est, d_est, theta_est
        
        
        











######################################## 多维GRM分布实现########################################


###########高斯-赫尔米特积分实现###########



def estimate_d_only_gh(a_params,response, mask_matrix,Q,n_quadrature=27, max_iter=100, tol=1e-4,verbose=False):
    '''
    固定区分度参数a，仅估计阈值参数d
    参数:
    - a_params: 区分度参数矩阵，形状为 (items, dim)
    - response: 被试的响应矩阵，形状为 (n, items)
    - mask_matrix: 掩码矩阵，形状为 (n, items)，1表示有数据，0表示缺失
    - Q: 题目特征矩阵，形状为 (items, dim)
    - n_quadrature: 高斯-赫尔米特积分点数，默认为27
    - max_iter: 最大迭代次数
    - tol: 收敛阈值
    - verbose: 是否打印迭代信息
    返回:
    - d_est: 估计的阈值参数，形状为（items,).注：该items为参加这一阶运算的题目，不一定等于总体题目
    - total_time: 总耗时
    '''
    n,items= response.shape
    dim = Q.shape[1]
    response = np.nan_to_num(response)  # 将NaN替换为0
    #初始化参数
    d_est = np.zeros(items)
    theta_quad, weights_quad = create_mirt_quadrature(n_quadrature, dim)
    prev_ll = -np.inf
    total_time = 0
    for iteration in range(max_iter):
        if verbose:
            print(f"=== 迭代 {iteration + 1}/{max_iter} ===")
        start_time = time.time()
        # E步：计算后验分布
        posterior = compute_m2pl_posterior(theta_quad, a_params, d_est, response, mask_matrix, weights_quad)
        # M步：更新参数
        d_new = np.zeros_like(d_est)
        for j in range(items):
            mask_j = mask_matrix[:, j].reshape(-1, 1)
            response_j = response[:, j].reshape(-1, 1)
            def negative_log_likelihood(d_j):
                expected_ll = compute_m2pl_item_log_likelihood(theta_quad, a_params[j], d_j, response_j, mask_j, posterior)
                return -expected_ll
            result = minimize(negative_log_likelihood, d_est[j], bounds=[(None, None)], method='L-BFGS-B')
            if result.success:
                d_new[j] = result.x[0]
            else:
                d_new[j] = d_est[j]
        end_time = time.time()
        d_est = d_new
        #收敛检查
        current_ll = np.sum(compute_m2pl_log_likelihood(theta_quad, a_params, d_est, response, mask_matrix, posterior))
        delta_ll = current_ll - prev_ll
        total_time += (end_time - start_time)
        if verbose:
            print(f"对数似然变化: {delta_ll:.6f}, 耗时: {end_time - start_time:.2f}秒")
        if iteration > 0 and abs(current_ll - prev_ll) < tol:
            print(f"*** 收敛于迭代 {iteration + 1} ***, 总耗时: {total_time:.2f}秒")
            break
        prev_ll = current_ll
    return d_est,total_time


def mgrm_em_stepwise(response, mask_matrix,Q, n_categories,n_quadrature=27, max_iter=100, tol=1e-4,  verbose=True):
    '''
    多维IRT的EM算法，使用高斯-赫尔米特积分。适用于3维及以下
    参数:
    - response: 被试的响应矩阵，形状为 (n, items)
    - mask_matrix: 掩码矩阵，形状为 (n, items)，1表示有数据，0表示缺失
    - Q: 题目特征矩阵，形状为 (items, dim)
    - n_categories: 每个项目的类别数列表，长度为 items
    - n_quadrature: 高斯-赫尔米特积分点数，默认为27
    - max_iter: 最大迭代次数
    - tol: 收敛阈值
    - verbose: 是否打印迭代信息
    返回:
    - a_est: 估计的区分度参数，形状为 (items, dim)
    - d_est: 估计的难度参数，估计的难度参数(每个项目有K-1个阈值)
    '''
    n, items = response.shape
    dim = Q.shape[1]
    response = np.nan_to_num(response)  # 将NaN替换为0
    max_threshold = np.max(n_categories) - 1  # 最大阈值数
    total_time=0.0
    start_time= time.time()
    # 步骤1: 所有题目参与，估计a和d1
    if verbose:
        print(f"=== 处理第 1 阶段阈值 ===")
    step1_matrix = (response >= 1).astype(int)
    a_est, d_step1 = mirt_em(step1_matrix, mask_matrix, Q, n_quadrature, max_iter, tol,verbose=verbose)
    # 初始化阈值存储
    d_est = [np.array([d_step1[j]]) for j in range(items)]
    total_time += (time.time() - start_time)
    if verbose:
        print(f"=== 阶段 1 完成,共{max_threshold}阶段, 耗时: {total_time:.2f}秒 ===")
    #处理高阶阈值，K>2的题目
    for k in range(2, max_threshold + 1):
        if verbose:
            print(f"=== 处理第 {k} 阶段阈值 ===")
        # 筛选题目：只保留计分等级≥k+1的题目（避免为0-1计分题添加额外阈值）
        item_mask = [categ >= k+1 for categ in n_categories]
        if not any(item_mask):  # 没有题目满足条件
            continue
        k_matrix = response[:, item_mask]
        k_matrix = (k_matrix >= k).astype(int)
        k_mask = mask_matrix[:, item_mask]
        # 固定a值，仅估计b_k
        a_subset = a_est[item_mask]
        d_k,time_d = estimate_d_only_gh(a_subset, k_matrix,k_mask,Q, n_quadrature, max_iter, tol, verbose=verbose)
        total_time += time_d
        # 只将估计的d_k分配给实际参与此步估计的题目
        idx = 0
        for j in range(items):
            if item_mask[j]:
                d_est[j] = np.append(d_est[j], d_k[idx])
                idx += 1
        if verbose:
            print(f"=== 阶段 {k} 完成,共{max_threshold}阶段, 耗时: {time_d:.2f}秒 ===")
    if verbose:
        print(f"=== 所有阶段完成, 共{max_threshold}阶段, 总耗时: {total_time:.2f}秒 ===")
    return a_est, d_est



###########MCEM实现###########


def estimate_d_only_mcem(a_params,theta,response, mask_matrix,Q,n_samples=300,burn_in=200, max_iter=100, tol=1e-4, sample_interval=10, verbose=False):
    '''
    固定区分度参数a，仅估计阈值参数d
    参数:
    - a_params: 区分度参数矩阵，形状为 (items, dim)
    - theta: 被试的能力参数矩阵，形状为 (n, dim)
    - response: 被试的响应矩阵，形状为 (n, items)
    - mask_matrix: 掩码矩阵，形状为 (n, items)，1表示有数据，0表示缺失
    - Q: 题目特征矩阵，形状为 (items, dim)
    - n_samples: MCMC有效采样次数
    - burn_in: 烧入期，前多少次采样不用于估计
    - max_iter: 最大迭代次数
    - tol: 收敛阈值
    - sample_interval: 每隔多少次迭代采样一次
    - verbose: 是否打印迭代信息
    返回:
    - d_est: 估计的难度参数，形状为 (items, )
    - total_time: 总耗时
    '''
    n,items= response.shape
    dim = Q.shape[1]
    response = np.nan_to_num(response)  # 将NaN替换为0
    d_est = np.zeros(items)
    theta_est=theta
    step_sizes = np.full(n, 0.2)
    rv = multivariate_normal(mean=np.zeros(dim), cov=np.eye(dim))
    prev_ll = -np.inf
    total_time_d = 0.0
    for iteration in range(max_iter):
        if verbose:
            print(f"=== 迭代 {iteration + 1}/{max_iter} ===")
        start_time = time.time()
        # 采样频率控制
        sample_this_iter = (iteration < 10) or (iteration % sample_interval == 0)
        if sample_this_iter:
            samples, theta_curr, step_sizes, all_accept = mcmc_sampling(theta_est, a_params, d_est, response, 
                                                                        mask_matrix, rv, step_sizes, burn_in, n_samples,method='m2pl')
            theta = np.mean(samples, axis=1)
        d_new = np.zeros_like(d_est)
        for j in range(items):
            if n_samples>100:
                theta_flat = samples[:,-100:,:,].reshape(-1, dim)  # [n * 100, dim]
                resp_flat = np.repeat(response[:, j], 100).reshape(-1,1)   # [n * 100,1]
                mask_flat = np.repeat(mask_matrix[:, j], 100).reshape(-1,1)  # [n * 100,1]
            else:
                theta_flat = samples.reshape(-1, dim)  # [n * n_samples, dim]
                resp_flat = np.repeat(response[:, j], n_samples).reshape(-1,1)   # [n * n_samples,1]
                mask_flat = np.repeat(mask_matrix[:, j], n_samples).reshape(-1,1)  # [n * n_samples]
            def negative_log_likelihood(d_j):
                expected_ll = compute_m2pl_item_log_likelihood(theta_flat, a_params[j], d_j, resp_flat, mask_flat)
                return -expected_ll
            result = minimize(negative_log_likelihood, d_est[j], bounds=[(None, None)], method='L-BFGS-B')
            if result.success:
                d_new[j] = result.x[0]
            else:
                d_new[j] = d_est[j]
        end_time = time.time()
        d_est = d_new
        #收敛检查
        current_ll = np.sum(compute_m2pl_log_likelihood(theta_curr, a_params, d_est, response, mask_matrix))
        delta_ll = current_ll - prev_ll
        total_time_d += (end_time - start_time)
        if verbose:
            print(f"对数似然变化: {delta_ll:.6f}, 耗时: {end_time - start_time:.2f}秒")
        if iteration > 0 and abs(current_ll - prev_ll) < tol:
            print(f"*** 收敛于迭代 {iteration + 1} ***, 总耗时: {total_time_d:.2f}秒")
            break
        prev_ll = current_ll
    return d_est, total_time_d




def mgrm_mcem_stepwise(response, mask_matrix,Q,n_categories,n_samples=300,burn_in=200, max_iter=100, tol=1e-4, sample_interval=10, verbose=False):  
    '''
    多维GRM的MCEM算法，使用马尔科夫链蒙特卡洛方法。适用于3维及以下
    参数:
    - response: 被试的响应矩阵，形状为 (n, items)
    - mask_matrix: 掩码矩阵，形状为 (n, items)，1表示有数据，0表示缺失
    - Q: 题目特征矩阵，形状为 (items, dim)
    - n_categories: 每个项目的类别数列表，长度为 items
    - n_samples: MCMC有效采样次数
    - burn_in: 烧入期，前多少次采样不用于估计
    - max_iter: 最大迭代次数
    - tol: 收敛阈值
    - sample_interval: 每隔多少次迭代采样一次
    - verbose: 是否打印迭代信息
    返回:
    - a_est: 估计的区分度参数，形状为 (items, dim)
    - d_est: 估计的难度参数，每个元素是一个形状为(n_categories[j]-1,)的数组
    - theta_est: 估计的能力参数，形状为 (n, dim)
    '''
    n, items = response.shape
    dim = Q.shape[1]
    response = np.nan_to_num(response)  # 将NaN替换为0
    max_threshold = np.max(n_categories) - 1  # 最大阈值数
    total_time=0.0
    start_time= time.time()
    # 步骤1: 所有题目参与，估计a和d1
    if verbose:
        print(f"=== 处理第 1 阶段阈值 ===")
    step1_matrix = (response >= 1).astype(int)
    a_est, d_step1,theta_est = mirt_mcem(step1_matrix, mask_matrix, Q, n_samples, burn_in, max_iter, tol, sample_interval,verbose=verbose)
    # 初始化阈值存储
    d_est = [np.array([d_step1[j]]) for j in range(items)]
    total_time += (time.time() - start_time)
    if verbose:
        print(f"=== 阶段 1 完成,共{max_threshold}阶段, 耗时: {total_time:.2f}秒 ===")
    #处理高阶阈值，K>2的题目
    for k in range(2, max_threshold + 1):
        if verbose:
            print(f"=== 处理第 {k} 阶段阈值 ===")
        # 筛选题目：只保留计分等级≥k+1的题目（避免为0-1计分题添加额外阈值）
        item_mask = [categ >= k+1 for categ in n_categories]
        if not any(item_mask):  # 没有题目满足条件
            continue
        k_matrix = response[:, item_mask]
        k_matrix = (k_matrix >= k).astype(int)
        k_mask = mask_matrix[:, item_mask]
        # 固定a值，仅估计d_k
        a_subset = a_est[item_mask]
        d_k, time_d = estimate_d_only_mcem(a_subset,theta_est,k_matrix, k_mask, Q, n_samples, burn_in, max_iter, tol, sample_interval, verbose=verbose)
        total_time += time_d
        # 只将估计的d_k分配给实际参与此步估计的题目
        idx = 0
        for j in range(items):
            if item_mask[j]:
                d_est[j] = np.append(d_est[j], d_k[idx])
                idx += 1
        if verbose:
            print(f"=== 阶段 {k} 完成,共{max_threshold}阶段, 耗时: {time_d:.2f}秒 ===")
    d_est=[np.sort(d)[::-1] for d in d_est]  # 确保每个题目的阈值按升序排列
    #利用所有计算的参数对能力进行估计
    start_time = time.time()
    rv= multivariate_normal(mean=np.zeros(dim), cov=np.eye(dim))
    step_sizes = np.full(n, 0.2)
    if n_samples + burn_in < 2000:
        samples, theta_curr, step_sizes, all_accept = mcmc_sampling(theta_est, a_est, d_est, response, 
                                                                        mask_matrix, rv, step_sizes, 1000, 1000,method='mgrm',n_categories=n_categories)
        theta_est = np.mean(samples, axis=1)
    else:
        samples, theta_curr, step_sizes, all_accept = mcmc_sampling(theta_est, a_est, d_est, response, 
                                                                        mask_matrix, rv, step_sizes, n_samples, burn_in,method='mgrm',n_categories=n_categories)
        theta_est = np.mean(samples, axis=1)
    end_time = time.time()
    total_time += (end_time - start_time)
    if verbose:
        print(f"=== 所有阶段完成, 共{max_threshold}阶段, 总耗时: {total_time:.2f}秒 ===")
    return a_est, d_est,theta_est




##########SAEM实现##########
def estimate_d_only_saem(a_params,theta,response, mask_matrix,Q, max_iter=100, tol=1e-4, verbose=False):
    '''
    固定区分度参数a，仅估计阈值参数d
    参数:
    - a_params: 区分度参数矩阵，形状为 (items, dim)
    - theta: 被试的能力参数矩阵，形状为 (n, dim)
    - response: 被试的响应矩阵，形状为 (n, items)
    - mask_matrix: 掩码矩阵，形状为 (n, items)，1表示有数据，0表示缺失
    - Q: 题目特征矩阵，形状为 (items, dim)
    - max_iter: 最大迭代次数
    - tol: 收敛阈值
    - verbose: 是否打印迭代信息
    返回:
    - d_est: 估计的难度参数，形状为 (items, )
    - total_time: 总耗时
    '''
    n,items= response.shape
    dim = Q.shape[1]
    response = np.nan_to_num(response)  # 将NaN替换为0
    d_est = np.zeros(items)
    theta_curr=theta
    step_sizes = np.full(n, 0.2)
    burn_in=50
    n_samples=10
    gamma_sequence = np.ones(max_iter)
    for k in range(burn_in, max_iter):
        # 在预热期后，步长开始递减
        gamma_sequence[k] = 1.0 / (k - burn_in + 1)**0.7
    rv = multivariate_normal(mean=np.zeros(dim), cov=np.eye(dim))
    prev_ll = -np.inf
    total_time_d = 0.0
    for iteration in range(max_iter):
        start_time = time.time()
        gamma = gamma_sequence[iteration]
        if verbose:
            print(f"=== 迭代 {iteration + 1}/{max_iter} ===")
        samples, theta_curr, step_sizes, all_accept = mcmc_sampling(theta_curr, a_params, d_est, response, 
                                                                        mask_matrix, rv, step_sizes, burn_in, n_samples,method='m2pl')
        d_new = d_est.copy()
        for j in range(items):
            response_j = response[:, j].reshape(-1, 1)
            mask_j = mask_matrix[:, j].reshape(-1, 1)
            def negative_log_likelihood(d_j):
                expected_ll = compute_m2pl_item_log_likelihood(theta_curr, a_params[j], d_j, response_j, mask_j)
                return -expected_ll
            result = minimize(negative_log_likelihood, d_est[j], bounds=[(None, None)], method='L-BFGS-B')
            if result.success:
                d_new_target = result.x[0]
                current_d = d_new[j]
                d_new[j]=(1 - gamma) * current_d + gamma * d_new_target
            else:
                if verbose:
                    print(f"题目 {j} 参数优化失败,采用上次估计值")
                d_new[j] = d_est[j]
        end_time = time.time()
        d_est = d_new
        #收敛检查
        current_ll = np.sum(compute_m2pl_log_likelihood(theta_curr, a_params, d_est, response, mask_matrix))
        delta_ll = current_ll - prev_ll
        total_time_d += (end_time - start_time)
        if verbose:
            print(f"对数似然变化: {delta_ll:.6f}, 耗时: {end_time - start_time:.2f}秒")
        prev_ll = current_ll
    return d_est, total_time_d



def mgrm_saem_stepwise(response, mask_matrix,Q,n_categories, max_iter=100, tol=1e-4, verbose=False):  
    '''
    多维GRM的SAEM算法，使用随机游走方法。分步估计每个题目的阈值参数
    参数:
    - response: 被试的响应矩阵，形状为 (n, items)
    - mask_matrix: 掩码矩阵，形状为 (n, items)，1表示有数据，0表示缺失
    - Q: 题目特征矩阵，形状为 (items, dim)
    - n_categories: 每个项目的类别数列表，长度为 items
    - max_iter: 最大迭代次数
    - tol: 收敛阈值
    - verbose: 是否打印迭代信息
    返回:
    - a_est: 估计的区分度参数，形状为 (items, dim)
    - d_est: 估计的难度参数，每个元素是一个形状为(n_categories[j]-1,)的数组
    - theta_est: 估计的能力参数，形状为 (n, dim)
    '''
    n, items = response.shape
    dim = Q.shape[1]
    response = np.nan_to_num(response)  # 将NaN替换为0
    max_threshold = np.max(n_categories) - 1  # 最大阈值数
    total_time=0.0
    start_time= time.time()
    # 步骤1: 所有题目参与，估计a和d1
    if verbose:
        print(f"=== 处理第 1 阶段阈值 ===")
    step1_matrix = (response >= 1).astype(int)
    a_est, d_step1,theta_est = mirt_saem(step1_matrix, mask_matrix, Q,  max_iter, tol,verbose=verbose)
    # 初始化阈值存储
    d_est = [np.array([d_step1[j]]) for j in range(items)]
    total_time += (time.time() - start_time)
    if verbose:
        print(f"=== 阶段 1 完成,共{max_threshold}阶段, 耗时: {total_time:.2f}秒 ===")
    #处理高阶阈值，K>2的题目
    for k in range(2, max_threshold + 1):
        if verbose:
            print(f"=== 处理第 {k} 阶段阈值 ===")
        # 筛选题目：只保留计分等级≥k+1的题目（避免为0-1计分题添加额外阈值）
        item_mask = [categ >= k+1 for categ in n_categories]
        if not any(item_mask):  # 没有题目满足条件
            continue
        k_matrix = response[:, item_mask]
        k_matrix = (k_matrix >= k).astype(int)
        k_mask = mask_matrix[:, item_mask]
        # 固定a值，仅估计d_k
        a_subset = a_est[item_mask]
        d_k, time_d = estimate_d_only_saem(a_subset,theta_est,k_matrix, k_mask, Q,  max_iter, tol,  verbose=verbose)
        total_time += time_d
        # 只将估计的d_k分配给实际参与此步估计的题目
        idx = 0
        for j in range(items):
            if item_mask[j]:
                d_est[j] = np.append(d_est[j], d_k[idx])
                idx += 1
        if verbose:
            print(f"=== 阶段 {k} 完成,共{max_threshold}阶段, 耗时: {time_d:.2f}秒 ===")
    d_est=[np.sort(d)[::-1] for d in d_est]  # 确保每个题目的阈值按升序排列
    #利用所有计算的参数对能力进行估计
    start_time = time.time()
    rv= multivariate_normal(mean=np.zeros(dim), cov=np.eye(dim))
    step_sizes = np.full(n, 0.2)
    samples, theta_curr, step_sizes, all_accept = mcmc_sampling(theta_est, a_est, d_est, response, 
                                                                        mask_matrix, rv, step_sizes, 1000, 1000,method='mgrm',n_categories=n_categories)
    theta_est = np.mean(samples, axis=1)
    end_time = time.time()
    total_time += (end_time - start_time)
    if verbose:
        print(f"=== 所有阶段完成, 共{max_threshold}阶段, 总耗时: {total_time:.2f}秒 ===")
    return a_est, d_est,theta_est



#######MCMC实现###########


def estimate_d_only_mcmc(a_params,theta,response, mask_matrix,Q,n_samples=300,burn_in=200, verbose=False):
    '''
    固定区分度参数a，仅估计阈值参数d
    参数:
    - a_params: 区分度参数矩阵，形状为 (items, dim)
    - theta: 被试的能力参数矩阵，形状为 (n, dim)
    - response: 被试的响应矩阵，形状为 (n, items)
    - mask_matrix: 掩码矩阵，形状为 (n, items)，1表示有数据，0表示缺失
    - Q: 题目特征矩阵，形状为 (items, dim)
    - n_samples: MCMC有效采样次数
    - burn_in: 烧入期，前多少次采样不用于估计
    - verbose: 是否打印迭代信息
    返回:
    - d_est: 估计的难度参数，形状为 (items, )
    - total_time: 总耗时
    '''
    start_time = time.time()
    n,items= response.shape
    dim = Q.shape[1]
    response = np.nan_to_num(response)  # 将NaN替换为0
    a_est= a_params.copy()
    d_est = np.zeros(items)
    theta_est=theta
    n_samples_d=np.zeros((items,n_samples))
    step_size_d=0.2
    rv_d=norm(loc=0, scale=1)
    for iteration in range(n_samples + burn_in):
        if verbose and iteration % 500 == 0:
            print(f"=== 迭代 {iteration + 1}/{n_samples + burn_in} ===")
        d_est,step_size_d = update_d_parameters(rv_d,theta_est, a_est, d_est, response, mask_matrix, step_size=step_size_d)
        if iteration >= burn_in:
            idx = iteration - burn_in
            n_samples_d[:, idx] = d_est.copy()
    # 计算后验均值
    d_est = np.mean(n_samples_d, axis=1)
    end_time = time.time()
    total_time = end_time - start_time
    if verbose:
        print(f"=== MCMC采样完成, 共{n_samples}次有效采样, 烧入期{burn_in}次, 总耗时: {total_time:.2f}秒 ===")
    return d_est, total_time

       

def mgrm_mcmc_stepwise(response, mask_matrix,Q,n_categories,n_samples=3000,burn_in=2000, verbose=False):  
    '''
    多维GRM的MCMC算法，使用马尔科夫链蒙特卡洛方法。适用于多维
    参数:
    - response: 被试的响应矩阵，形状为 (n, items)
    - mask_matrix: 掩码矩阵，形状为 (n, items)，1表示有数据，0表示缺失
    - Q: 题目特征矩阵，形状为 (items, dim)
    - n_categories: 每个项目的类别数列表，长度为 items
    - n_samples: MCMC有效采样次数
    - burn_in: 烧入期，前多少次采样不用于估计
    - verbose: 是否打印迭代信息
    返回:
    - a_est: 估计的区分度参数，形状为 (items, dim)
    - d_est: 估计的难度参数，每个元素是一个形状为(n_categories[j]-1,)的数组
    - theta_est: 估计的能力参数，形状为 (n, dim)
    '''
    n, items = response.shape
    dim = Q.shape[1]
    response = np.nan_to_num(response)  # 将NaN替换为0
    max_threshold = np.max(n_categories) - 1  # 最大阈值数
    total_time=0.0
    start_time= time.time()
    # 步骤1: 所有题目参与，估计a和d1
    if verbose:
        print(f"=== 处理第 1 阶段阈值 ===")
    step1_matrix = (response >= 1).astype(int)
    a_est, d_step1,theta_est = mirt_mcmc(step1_matrix, mask_matrix, Q, n_samples, burn_in,verbose=verbose)
    # 初始化阈值存储
    d_est = [np.array([d_step1[j]]) for j in range(items)]
    total_time += (time.time() - start_time)
    if verbose:
        print(f"=== 阶段 1 完成,共{max_threshold}阶段, 耗时: {total_time:.2f}秒 ===")
    #处理高阶阈值，K>2的题目
    for k in range(2, max_threshold + 1):
        if verbose:
            print(f"=== 处理第 {k} 阶段阈值 ===")
        # 筛选题目：只保留计分等级≥k+1的题目（避免为0-1计分题添加额外阈值）
        item_mask = [categ >= k+1 for categ in n_categories]
        if not any(item_mask):  # 没有题目满足条件
            continue
        k_matrix = response[:, item_mask]
        k_matrix = (k_matrix >= k).astype(int)
        k_mask = mask_matrix[:, item_mask]
        # 固定a值，仅估计d_k
        a_subset = a_est[item_mask]
        d_k, time_d = estimate_d_only_mcmc(a_subset,theta_est,k_matrix, k_mask, Q, n_samples, burn_in, verbose=verbose)
        total_time += time_d
        # 只将估计的d_k分配给实际参与此步估计的题目
        idx = 0
        for j in range(items):
            if item_mask[j]:
                d_est[j] = np.append(d_est[j], d_k[idx])
                idx += 1
        if verbose:
            print(f"=== 阶段 {k} 完成,共{max_threshold}阶段, 耗时: {time_d:.2f}秒 ===")
    d_est=[np.sort(d)[::-1] for d in d_est]  # 确保每个题目的阈值按升序排列
    #利用所有计算的参数对能力进行估计
    start_time = time.time()
    rv= multivariate_normal(mean=np.zeros(dim), cov=np.eye(dim))
    step_sizes = np.full(n, 0.2)
    if n_samples + burn_in < 2000:
        samples, theta_curr, step_sizes, all_accept = mcmc_sampling(theta_est, a_est, d_est, response, 
                                                                        mask_matrix, rv, step_sizes, 1000, 1000,method='mgrm',n_categories=n_categories)
        theta_est = np.mean(samples, axis=1)
    else:
        samples, theta_curr, step_sizes, all_accept = mcmc_sampling(theta_est, a_est, d_est, response, 
                                                                        mask_matrix, rv, step_sizes, n_samples, burn_in,method='mgrm',n_categories=n_categories)
        theta_est = np.mean(samples, axis=1)
    if verbose:
        print(f"=== 所有阶段完成, 共{max_threshold}阶段, 总耗时: {total_time:.2f}秒 ===")
    return a_est, d_est,theta_est














######################################### 多维GRM标准实现########################################

   
def mgrm_em_standard(response, mask_matrix, Q, n_categories, n_quadrature=27, max_iter=100, tol=1e-4, verbose=False):
    '''
    多维GRM的EM算法，使用高斯-赫尔米特积分。适用于3维及以下
    参数:
    - response: 被试的响应矩阵，形状为 (n, items)
    - mask_matrix: 掩码矩阵，形状为 (n, items)，1表示有数据，0表示缺失
    - Q: 题目特征矩阵，形状为 (items, dim)
    - n_categories: 每个项目的类别数列表，长度为 items
    - n_quadrature: 高斯-赫尔米特积分点数，默认为27
    - max_iter: 最大迭代次数
    - tol: 收敛阈值
    - verbose: 是否打印迭代信息
    返回:
    - a_est: 估计的区分度参数，形状为 (items, dim)
    - d_est: 估计的难度参数列表，每个元素是一个形状为(n_categories[j]-1,)的数组
    '''
    n, items = response.shape
    dim = Q.shape[1]
    response = np.nan_to_num(response)  # 将NaN替换为0
    #初始化参数
    a_est = np.random.uniform(0.2, 3, (items, dim)) * Q
    d_est = [np.sort(np.random.normal(0,1,k-1))[::-1] for k in n_categories]  # 确保每个题目的阈值按降序排列
    
    theta_quad, weights_quad = create_mirt_quadrature(n_quadrature, dim)
    
    prev_ll = -np.inf
    total_time = 0
    
    for iteration in range(max_iter):
        if verbose:
            print(f"=== 迭代 {iteration + 1}/{max_iter} ===")
        start_time = time.time()
        # E步：计算后验分布
        posterior = compute_mgrm_posterior(theta_quad, a_est, d_est, response, mask_matrix, n_categories, weights_quad)
        # M步：更新参数
        a_new = np.zeros_like(a_est)
        d_new = [np.zeros_like(d) for d in d_est]
        for j in range(items):
            mask_j = mask_matrix[:, j].reshape(-1, 1)
            response_j = response[:, j].reshape(-1, 1)
            active_dims = Q[j] == 1
            num_a = np.sum(active_dims)
            k= n_categories[j]
            def negative_log_likelihood(params):
                a_params = params[:num_a]
                d_j = params[num_a:]
                a_j = np.zeros(dim)
                a_j[active_dims] = a_params
                expected_ll = compute_mgrm_item_log_likelihood(theta_quad, a_j, d_j, response_j, mask_j,k, posterior)
                return -expected_ll
            
            init_params = np.concatenate([a_est[j, active_dims], d_est[j]])
            bounds = [(0.1, None)] * num_a + [(None, None)] * (k-1)
            result = minimize(negative_log_likelihood, init_params, bounds=bounds, method='L-BFGS-B')
            if result.success:
                a_new[j, active_dims] = result.x[:num_a]
                d_new[j] = result.x[-(n_categories[j] - 1):]
            else:
                if verbose:
                    print(f"题目 {j} 参数优化失败,采用上次估计值")
                a_new[j, active_dims] = a_est[j, active_dims]
                d_new[j] = d_est[j]
        end_time = time.time()
        a_est, d_est = a_new, d_new
        #收敛检查
        current_ll = np.sum(compute_mgrm_log_likelihood(theta_quad, a_est, d_est, response, mask_matrix, n_categories, posterior))
        delta_ll = current_ll - prev_ll
        total_time += (end_time - start_time)
        if verbose:
            print(f"对数似然变化: {delta_ll:.6f}, 耗时: {end_time - start_time:.2f}秒")
        if iteration > 0 and abs(current_ll - prev_ll) < tol:
            if verbose:
                print(f"*** 收敛于迭代 {iteration + 1} ***, 总耗时: {total_time:.2f}秒")
            break
        prev_ll = current_ll
    return a_est, d_est





def mgrm_mcem_standard(response, mask_matrix, Q, n_categories, n_samples=300, burn_in=200, max_iter=100, tol=1e-4, sample_interval=10, verbose=False):
    '''
    多维GRM的MCEM算法，使用马尔科夫链蒙特卡洛方法。适用于3维及以上
    参数:
    - response: 被试的响应矩阵，形状为 (n, items)
    - mask_matrix: 掩码矩阵，形状为 (n, items)，1表示有数据，0表示缺失
    - Q: 题目特征矩阵，形状为 (items, dim)
    - n_categories: 每个项目的类别数列表，长度为 items
    - n_samples: MCMC有效采样次数
    - burn_in: 烧入期，前多少次采样不用于估计
    - max_iter: 最大迭代次数
    - tol: 收敛阈值
    - sample_interval: 每隔多少次迭代采样一次
    - verbose: 是否打印迭代信息
    返回:
    - a_est: 估计的区分度参数，形状为 (items, dim)
    - d_est: 估计的难度参数列表，每个元素是一个形状为(n_categories[j]-1,)的数组
    - theta_est: 估计的能力参数，形状为 (n, dim)
    '''
    n, items = response.shape
    dim = Q.shape[1]
    response = np.nan_to_num(response)  # 将NaN替换为0
    #初始化参数
    a_est = np.random.uniform(0.2, 3, (items, dim)) * Q
    d_est = [np.sort(np.random.normal(0,1,k-1))[::-1] for k in n_categories]  # 确保每个题目的阈值按降序排列
    theta_est = np.zeros((n, dim))
    step_sizes = np.full(n, 0.2)
    rv = multivariate_normal(mean=np.zeros(dim), cov=np.eye(dim))
    prev_ll = -np.inf
    total_time = 0
    for iteration in range(max_iter):
        if verbose:
            print(f"=== 迭代 {iteration + 1}/{max_iter} ===")
        start_time =time.time()
        # 采样频率控制
        sample_this_iter = (iteration < 10) or (iteration % sample_interval == 0)
        if sample_this_iter:
            #print(f"采样第{iteration + 1}次")
            samples, theta_curr, step_sizes, all_accept = mcmc_sampling(theta_est, a_est, d_est, response, 
                                                                        mask_matrix, rv, step_sizes, burn_in, n_samples,method='mgrm',n_categories=n_categories)
            theta_est = np.mean(samples, axis=1)
        a_new = np.zeros_like(a_est)
        d_new = [np.zeros_like(d) for d in d_est]
        for j in range(items):
            if n_samples>100:
                theta_flat = samples[:,-100:,:,].reshape(-1, dim)  # [n * 100, dim]
                resp_flat = np.repeat(response[:, j], 100).reshape(-1,1)   # [n * 100,1]
                mask_flat = np.repeat(mask_matrix[:, j], 100).reshape(-1,1)  # [n * 100,1]
            else:
                theta_flat = samples.reshape(-1, dim)  # [n * n_samples, dim]
                resp_flat = np.repeat(response[:, j], n_samples).reshape(-1,1)   # [n * n_samples,1]
                mask_flat = np.repeat(mask_matrix[:, j], n_samples).reshape(-1,1)  # [n * n_samples]
            active_dims = Q[j] == 1
            num_a = np.sum(active_dims)
            k = n_categories[j]
            def negative_log_likelihood(params):
                a_params = params[:num_a]
                d_j = params[num_a:]
                a_j = np.zeros(dim)
                a_j[active_dims] = a_params
                expected_ll = compute_mgrm_item_log_likelihood(theta_flat, a_j, d_j, resp_flat, mask_flat, k)
                return -expected_ll
            init_params = np.concatenate([a_est[j, active_dims], d_est[j]])
            bounds = [(0.1, None)] * num_a + [(None, None)] * (k - 1)
            result = minimize(negative_log_likelihood, init_params, bounds=bounds, method='L-BFGS-B')
            if result.success:
                a_new[j, active_dims] = result.x[:num_a]
                d_new[j] = result.x[num_a:]
            else:
                if verbose:
                    print(f"题目 {j} 参数优化失败,采用上次估计值")
                a_new[j, active_dims] = a_est[j, active_dims]
                d_new[j] = d_est[j]
        end_time = time.time()
        a_est, d_est = a_new, d_new
        #收敛检查
        current_ll = np.sum(compute_mgrm_log_likelihood(theta_curr, a_est, d_est, response, mask_matrix, n_categories))
        delta_ll = current_ll - prev_ll
        total_time += (end_time - start_time)
        if verbose:
            print(f"对数似然变化: {delta_ll:.6f}, 耗时: {end_time - start_time:.2f}秒")
        if iteration > 0 and abs(current_ll - prev_ll) < tol:
            if verbose:
                print(f"*** 收敛于迭代 {iteration + 1} ***")
            break
        prev_ll = current_ll

    #利用多采样的参数对能力进行估计
    start_time = time.time()
    if n_samples + burn_in < 2000:
        samples, theta_curr, step_sizes, all_accept = mcmc_sampling(theta_est, a_est, d_est, response, 
                                                                        mask_matrix, rv, step_sizes, 1000, 1000,method='mgrm',n_categories=n_categories)
        theta_est = np.mean(samples, axis=1)
    return a_est, d_est, theta_est




def mgrm_saem_standard(response, mask_matrix, Q, n_categories,  max_iter=100, tol=1e-4,  verbose=False):
    '''
    多维IRT的SAEM算法，使用随机游走方法。适用于多维IRT模型
    该方法通过采样后验分布来估计参数，适用于高维和复杂模型。
    参数:
    - response: 被试的响应矩阵，形状为 (n, items)
    - mask_matrix: 掩码矩阵，形状为 (n, items)，1表示有数据，0表示缺失
    - Q: 题目特征矩阵，形状为 (items, dim)
    - n_categories: 每个项目的类别数列表，长度为 items
    - max_iter: 最大迭代次数
    - tol: 收敛阈值
    - verbose: 是否打印迭代信息
    返回:
    - a_est: 估计的区分度参数，形状为 (items, dim)
    - d_est: 估计的难度参数，形状为 (items, )
    - theta_est: 估计的能力参数，形状为 (n, dim)
    '''
    n, items = response.shape
    dim = Q.shape[1]
    response = np.nan_to_num(response)  # 将NaN替换为0
    #初始化参数
    a_est = np.random.uniform(0.2, 3, (items, dim)) * Q
    d_est = [np.sort(np.random.normal(0,1,k-1))[::-1] for k in n_categories]
    theta_curr=np.zeros((n, dim))
    step_sizes = np.full(n, 0.2)
    burn_in=50
    n_samples=10
    gamma_sequence = np.ones(max_iter)
    for k in range(burn_in, max_iter):
        # 在预热期后，步长开始递减
        gamma_sequence[k] = 1.0 / (k - burn_in + 1)**0.7
    rv = multivariate_normal(mean=np.zeros(dim), cov=np.eye(dim))
    prev_ll= -np.inf
    total_time = 0
    for iteration in range(max_iter):
        start_time = time.time()
        gamma = gamma_sequence[iteration]
        if verbose:
            print(f"=== 迭代 {iteration + 1}/{max_iter} ===")
        samples, theta_curr, step_sizes, all_accept = mcmc_sampling(theta_curr, a_est, d_est, response, 
                                                                        mask_matrix, rv, step_sizes, burn_in, n_samples,method='mgrm',n_categories=n_categories)
        a_new = a_est.copy()
        d_new = d_est.copy()
        for j in range(items):
            active_dims = Q[j] == 1
            num_a = np.sum(active_dims)
            k = n_categories[j]
            response_j=response[:,j].reshape(-1,1)
            mask_j=mask_matrix[:,j].reshape(-1,1)
            def negative_log_likelihood(params):
                a_params = params[:num_a]
                d_j = params[num_a:]
                a_j = np.zeros(dim)
                a_j[active_dims] = a_params
                expected_ll = compute_mgrm_item_log_likelihood(theta_curr, a_j, d_j, response_j, mask_j, k)
                return -expected_ll
            init_params = np.concatenate([a_est[j, active_dims], d_est[j]])
            bounds = [(0.1, None)] * num_a + [(None, None)] * (k - 1)
            result = minimize(negative_log_likelihood, init_params, bounds=bounds, method='L-BFGS-B')
            if result.success:
                a_new_target = result.x[:num_a]
                d_new_target = result.x[num_a:]
                current_a=a_new[j,active_dims]
                a_new[j,active_dims]=(1-gamma)*current_a+gamma*a_new_target
                current_d = d_new[j]
                d_new[j] = (1 - gamma) * current_d + gamma * d_new_target
            else:
                if verbose:
                    print(f"题目 {j} 参数优化失败,采用上次估计值")
                a_new[j, active_dims] = a_est[j, active_dims]
                d_new[j] = d_est[j]
        end_time = time.time()
        a_est, d_est = a_new, d_new
        #收敛检查
        current_ll = np.sum(compute_mgrm_log_likelihood(theta_curr, a_est, d_est, response, mask_matrix, n_categories))
        delta_ll = current_ll - prev_ll
        total_time += (end_time - start_time)
        if verbose:
            print(f"对数似然变化: {delta_ll:.6f}, 耗时: {end_time - start_time:.2f}秒")
        prev_ll = current_ll
    start_time = time.time()
    samples, theta_curr, step_sizes, all_accept = mcmc_sampling(theta_curr, a_est, d_est, response, 
                                                                        mask_matrix, rv, step_sizes, 1000, 1000,method='mgrm',n_categories=n_categories)
    theta_est = np.mean(samples, axis=1)
    end_time = time.time()
    total_time += (end_time - start_time)
    if verbose:
        print(f"=== SAEM方法估计完成, 总耗时: {total_time:.2f}秒 ===")
    return a_est, d_est,theta_est




def mgrm_mcmc_standard(response, mask_matrix, Q, n_categories, n_samples=3000, burn_in=2000, verbose=False):
    '''
    多维GRM的MCMC算法，使用马尔科夫链蒙特卡洛方法。适用于多维GRM模型
    参数:
    - response: 被试的响应矩阵，形状为 (n, items)
    - mask_matrix: 掩码矩阵，形状为 (n, items)，1表示有数据，0表示缺失
    - Q: 题目特征矩阵，形状为 (items, dim)
    - n_categories: 每个项目的类别数列表，长度为 items
    - n_samples: MCMC有效采样次数
    - burn_in: 烧入期，前多少次采样不用于估计
    - verbose: 是否打印迭代信息
    返回:
    - a_est: 估计的区分度参数，形状为 (items, dim)
    - d_est: 估计的难度参数列表，每个元素是一个形状为(n_categories[j]-1,)的数组
    - theta_est: 估计的能力参数，形状为 (n, dim)
    '''
    start_time = time.time()
    n, items = response.shape
    dim = Q.shape[1]
    response = np.nan_to_num(response)  # 将NaN替换为0
    max_k= np.max(n_categories)-1  # 最大阈值数
    #初始化参数
    a_est = np.random.uniform(0.2, 3, (items, dim)) * Q
    d_est = [np.sort(np.random.normal(0,1,k-1))[::-1] for k in n_categories]  # 确保每个题目的阈值按降序排列
    theta_est = np.zeros((n, dim))
    rv_log_a=norm(loc=0, scale=np.sqrt(0.5))
    rv_d=norm(loc=0, scale=1.5)
    rv_theta=multivariate_normal(mean=np.zeros(dim), cov=np.eye(dim))
    n_samples_a=np.zeros((items,n_samples,dim))
    n_samples_d=np.zeros((items,n_samples,max_k))
    n_samples_theta=np.zeros((n,n_samples,dim))
    step_size_a=step_size_d=step_size_theta=0.2
    d_matrix,d_mask=pad_grm_parameters(d_est,n_categories)
    for iteration in range(n_samples + burn_in):
        if verbose and iteration % 500 == 0:
            print(f"=== 迭代 {iteration + 1}/{n_samples+burn_in} ===")
        theta_est,step_size_theta = update_grm_theta_parameters(rv_theta,theta_est,a_est, d_matrix, d_mask, response, mask_matrix,n_categories, step_size=step_size_theta)
        theta_est = theta_est - np.mean(theta_est, axis=0)
        a_est,step_size_a = update_grm_a_parameters(rv_log_a,theta_est, a_est, d_matrix, d_mask, response, mask_matrix, Q,n_categories, step_size=step_size_a)
        d_matrix,step_size_d = update_grm_d_parameters(rv_d,theta_est, a_est, d_matrix, d_mask, response, mask_matrix, n_categories, step_size=step_size_d)
        if iteration >= burn_in:
            idx= iteration - burn_in
            n_samples_a[:, idx, :] = a_est.copy()
            n_samples_d[:, idx, :] = d_matrix.copy()
            n_samples_theta[:, idx, :] = theta_est.copy()
    # 计算后验均值
    a_est = np.mean(n_samples_a, axis=1)
    d_est= np.mean(n_samples_d, axis=1)
    d_est=[d_est[j][:n_categories[j]-1] for j in range(items)]
    theta_est = np.mean(n_samples_theta, axis=1)
    return a_est, d_est, theta_est





    






    