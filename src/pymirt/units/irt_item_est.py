from scipy.optimize import minimize
import numpy as np
import time
from .units import (create_irt_quadrature,compute_2pl_posterior,compute_2pl_log_likelihood,
                    compute_grm_posterior,compute_grm_item_log_likelihood,
                    compute_grm_log_likelihood,ensure_ordered_thresholds)




################################## 单维2PL##################################
def irt_em(response_matrix, mask_matrix, n_quadrature=27, max_iter=100, tol=1e-4,verbose=False):
    """
    单维项目反应理论（IRT，0-1计分）项目参数估计，使用期望最大化（EM）算法处理缺失数据。
    参数:
    - response_matrix: 考生的反应矩阵，形状为 [n_examinees, n_items]，其中每个元素为0或1，表示考生对项目的反应。
    - mask_matrix: 掩码矩阵，形状为 [n_examinees, n_items]，其中1表示该项目有反应，0表示缺失。
    - n_quadrature: 高斯-赫尔米特积分的点数，默认为27。
    - max_iter: 最大迭代次数，默认为100。
    - tol: 收敛阈值，默认为1e-4。
    - verbose: 是否可视化结果，默认为False。
    返回:
    - a_est: 估计的项目难度参数，形状为 [n_items]。
    - b_est: 估计的项目区分度参数，形状为 [n_items]。
    """
    n_examinees, n_items = response_matrix.shape
    a_est = np.ones(n_items)
    b_est = np.zeros(n_items)
    response_matrix = np.nan_to_num(response_matrix)  # NaN替换为0，方便计算
    # 设置高斯-赫尔米特积分
    theta_quad, weights_quad = create_irt_quadrature(n_quadrature)
    prev_ll = -np.inf
    total_time = 0  # 总耗时
    for iteration in range(max_iter):
        if verbose:
            print(f"=== 迭代 {iteration + 1}/{max_iter} ===")
        #记录迭代开始时间
        start_time = time.time()
        # === E步：计算后验分布 ===
        posterior = compute_2pl_posterior(
            theta_quad, a_est, b_est,
            response_matrix, mask_matrix, weights_quad
        )
        # === M步：逐项目优化 ===
        a_new = np.zeros(n_items)
        b_new = np.zeros(n_items)
        for j in range(n_items):
            mask_j = mask_matrix[:, j].reshape(-1,1)# [n_examinees, 1]
            response_j = response_matrix[:, j].reshape(-1,1)  # [n_examinees, 1]
            def neg_log_likelihood(params):
                a_j, b_j = params
                expected_ll=compute_2pl_log_likelihood(
                    theta_quad, a_j, b_j,
                    response_j, mask_j, posterior
                )
                return -expected_ll  # 最小化负对数似然
            # Constrain discrimination to the positive range required by 2PL.
            res = minimize(
                neg_log_likelihood,
                [a_est[j], b_est[j]],
                method='L-BFGS-B',
                bounds=[(0.1, None), (None, None)],
            )
            a_new[j], b_new[j] = res.x
        # 记录每次迭代的时间
        end_time = time.time()
        a_est, b_est = a_new, b_new
        # === 收敛检查 ===
        current_ll = compute_2pl_log_likelihood(
            theta_quad,  a_est, b_est,
            response_matrix, mask_matrix, posterior
        )
        delta_ll = current_ll - prev_ll
        total_time += (end_time - start_time)
        if verbose:
            print(f"对数似然变化: {delta_ll:.6f}, 耗时: {end_time - start_time:.2f}秒")
        if iteration > 0 and abs(current_ll - prev_ll) < tol:
            if verbose:
                print(f"*** 收敛于迭代 {iteration + 1} ***, 总耗时: {total_time:.2f}秒")
            break
        prev_ll = current_ll
    return a_est, b_est




#####################################单维GRM分布实现#####################################
def estimate_b_only(a_params, response_matrix,mask_matrix, n_quadrature=27, max_iter=100, tol=1e-4,verbose=False):
    """
    固定区分度参数a，仅估计难度参数b
    参数:
    - a_params : numpy数组, 形状(n_items,)
        固定的区分度参数
    - response_matrix : numpy数组, 形状(n_examinees, n_items)
        被试作答矩阵(0/1)
    - mask_matrix : numpy数组, 形状(n_examinees, n_items)
        掩码矩阵，1表示有数据，0表示缺失
    - n_quadrature : int, 默认27
        高斯-埃尔米特积分点数
    - max_iter : int, 默认100
        最大迭代次数
    - tol : float, 默认1e-4
        收敛阈值
    - verbose : bool, 默认False
    返回:
    - b_est : numpy数组, 形状(n_items,)
        估计的难度参数
    """
    n_examinees, n_items = response_matrix.shape
    # 初始化b参数
    b_est = np.zeros(n_items)
    # 设置高斯-埃尔米特积分点
    theta_quad, weights_quad = create_irt_quadrature(n_quadrature)
    prev_ll = -np.inf
    total_time = 0  # 总耗时
    for iteration in range(max_iter):
        time_start = time.time()  # 记录迭代开始时间
        if verbose:
            print(f"=== 迭代 {iteration + 1}/{max_iter} ===")
        # E步: 计算后验分布
        posterior = compute_2pl_posterior(
            theta_quad, a_params, b_est,
            response_matrix, mask_matrix, weights_quad
        )

        # M步: 更新b参数
        b_new = np.zeros(n_items)
        for j in range(n_items):
            # 目标函数: 条件期望对数似然
            response_j = response_matrix[:, j].reshape(-1, 1)  # [n_examinees, 1]
            mask_j = mask_matrix[:, j].reshape(-1, 1)  # [n_examinees, 1]
            def neg_log_likelihood(b_j):
                expected_ll = compute_2pl_log_likelihood(
                    theta_quad, a_params[j], b_j,
                    response_j, mask_j, posterior
                )
                return -expected_ll
            res = minimize(neg_log_likelihood, b_est[j], method='BFGS')
            b_new[j] = res.x[0]
        # 检查收敛
        b_est = b_new
        current_ll = compute_2pl_log_likelihood(
            theta_quad, a_params, b_est,
            response_matrix, mask_matrix, posterior
        )
        delta_ll = current_ll - prev_ll
        total_time += (time.time() - time_start)
        if verbose:
            print(f"对数似然变化: {delta_ll:.6f}, 耗时: {time.time() - time_start:.2f}秒")
        if iteration > 0 and abs(delta_ll) < tol:
            break
    return b_est,total_time



def grm_em_stepwise(response_matrix,mask_matrix, n_categories, n_quadrature=27, max_iter=100, tol=1e-4, verbose=False):
    """
    分步估计多级计分IRT参数
    参数:
    response_matrix : numpy数组, 形状(n_examinees, n_items)
        被试作答矩阵(多级计分)
    mask_matrix : numpy数组, 形状(n_examinees, n_items)
        掩码矩阵，1表示有数据，0表示缺失
    n_categories : list, 每个项目的类别数(计分等级)
    n_quadrature : int, 默认27
        高斯-埃尔米特积分点数
    max_iter : int, 默认100
        最大迭代次数
    tol : float, 默认1e-4
        收敛阈值
    verbose : bool, 默认False
        是否可视化结果
    返回:
    a_est : numpy数组, 形状(n_items,)
        估计的区分度参数
    b_est : list of arrays, 每个数组包含一个项目的难度阈值
        估计的难度参数(每个项目有K-1个阈值)
    """
    response_matrix = np.nan_to_num(response_matrix)  # NaN替换为0，方便计算
    n_examinees, n_items = response_matrix.shape
    max_threshold = max(n_categories) - 1
    total_time = 0  # 总耗时
    time_start = time.time()
    # 步骤1: 所有题目参与，估计a和b1
    if verbose:
        print(f"=== 处理第 1 阶段阈值 ===")
    step1_matrix = (response_matrix >= 1).astype(int)
    a_est, b_step1 = irt_em(step1_matrix,mask_matrix, n_quadrature, max_iter, tol, verbose=verbose)
    # 初始化阈值存储
    b_est = [np.array([b_step1[j]]) for j in range(n_items)]
    total_time += (time.time() - time_start)
    if verbose:
        print(f"=== 阶段 1 完成,共{max_threshold}阶段, 耗时: {total_time:.2f}秒 ===")
    # 处理高阶阈值 (k>=2)
    for k in range(2, max_threshold + 1):
        if verbose:
            print(f"=== 处理第 {k} 阶段阈值 ===")
        # 筛选题目：只保留计分等级≥k+1的题目（避免为0-1计分题添加额外阈值）
        item_mask = [categ >= k+1 for categ in n_categories]
        if not any(item_mask):  # 没有题目满足条件
            continue
        k_matrix = response_matrix[:, item_mask]
        k_matrix = (k_matrix >= k).astype(int)
        k_mask = mask_matrix[:, item_mask]
        # 固定a值，仅估计b_k
        a_subset = a_est[item_mask]
        b_k,time_b = estimate_b_only(a_subset, k_matrix,k_mask, n_quadrature, max_iter, tol, verbose=verbose)
        total_time += time_b
        # 只将估计的b_k分配给实际参与此步估计的题目
        idx = 0
        for j in range(n_items):
            if item_mask[j]:
                b_est[j] = np.append(b_est[j], b_k[idx])
                idx += 1
        if verbose:
            print(f"=== 阶段 {k} 完成,共{max_threshold}阶段, 耗时: {time_b:.2f}秒 ===")
    # 确保每个项目的阈值是升序的
    #b_est = [np.sort(b) for b in b_est]
    if verbose:
        print(f"=== 所有阶段完成, 共{max_threshold}阶段, 总耗时: {total_time:.2f}秒 ===")
    return a_est, b_est





##############################单维GRM标准实现##################################
def grm_em_standard(response,mask_matrix,n_categories,n_quadrature=27,max_iter=100, tol=1e-4, verbose=False):
    """
    单维GRM的标准EM极大似然估计，支持缺失数据
    参数：
    response : np.ndarray
        响应矩阵，形状为 (n_subjects, n_items)，缺失值用NaN表示
    mask_matrix : np.ndarray
        掩码矩阵，形状为 (n_subjects, n_items)，1=有数据，0=缺失
    n_categories : list
        每个项目的类别数，长度为 n_items 的单维数组
    n_quadrature : int
        高斯-赫尔米特积分点数
    max_iter : int
        最大迭代次数
    tol : float
        收敛容忍度
    verbose : bool
        是否可视化迭代过程
    返回：
    a_est : np.ndarray
        估计的区分度参数，形状为 (n_items,)
    b_est : np.ndarray
        估计的难度参数，列表长度为 n_items，每个元素是一个单维数组，包含每个类别的难度
    """
    n_examinees, n_items = response.shape
    a_est = np.ones(n_items)
    b_est=[np.sort(np.random.normal(0, 1, k-1)) for k in n_categories]
    response = np.nan_to_num(response)  # NaN替换为0
    # 设置高斯-赫尔米特积分
    theta_quad, weights_quad = create_irt_quadrature(n_quadrature)
    prev_ll = -np.inf
    total_time = 0  # 总耗时
    for iteration in range(max_iter):
        if verbose:
            print(f"=== 迭代 {iteration + 1}/{max_iter} ===")
        #记录迭代开始时间
        start_time = time.time()
        # === E步：计算后验分布 ===
        posterior = compute_grm_posterior(
            theta_quad, a_est, b_est, response, mask_matrix, n_categories,weights_quad
        )
        # === M步：逐项目优化 === 
        for j in range(n_items):
            mask_j = mask_matrix[:, j].reshape(-1,1)# [n_examinees, 1]
            response_j = response[:, j].reshape(-1,1)  # [n_examinees, 1]
            k= n_categories[j]  # 当前项目的类别数
            def neg_log_likelihood(params):
                a_j = params[0]
                b_j = params[1:k]  # k-1个阈值
                expected_ll=compute_grm_item_log_likelihood(
                    theta_quad, a_j, b_j, response_j, mask_j, k, posterior
                )
                return -expected_ll  # 最小化负对数似然
            init = np.concatenate([[a_est[j]], b_est[j]])
            bounds = [(0.2, 3.0)]
            # 为阈值设置边界，确保升序且不会有下界大于上界的问题
            for i in range(k-1):
                if i == 0:
                    bounds.append((-4.0, 4.0))
                else:
                # 确保阈值递增，同时确保下界不超过上界
                    lower = max(b_est[j][i-1] + 0.01, -3.99)  # 最小间隔0.01，且下界在有效范围内
                    bounds.append((lower, 4.0))
            # 使用BFGS优化更新参数
            options = {'maxiter': 100, 'disp': False}  # 增加最大迭代次数
            res = minimize(neg_log_likelihood, init, method='L-BFGS-B', bounds=bounds, options=options)
            if res.success:
                a_est[j] = np.clip(res.x[0], 0.2, 3.0)  # 更新区分度参数
                b_est[j] = ensure_ordered_thresholds(np.clip(res.x[1:],-4.0,4.0)) # 更新阈值
            else:
                if verbose:
                    print(f"项目 {j} 的参数优化失败,采用上次估计值")      
        # 记录每次迭代的时间
        end_time = time.time()
        # === 收敛检查 ===
        current_ll = np.sum(compute_grm_log_likelihood(
            theta_quad, a_est, b_est, response, mask_matrix, n_categories, posterior))
        delta_ll = current_ll - prev_ll
        total_time += (end_time - start_time)
        if verbose:
            print(f"对数似然变化: {delta_ll:.6f}, 耗时: {end_time - start_time:.2f}秒")
        if iteration > 0 and abs(current_ll - prev_ll) < tol:
            if verbose:
                print(f"*** 收敛于迭代 {iteration + 1} ***, 总耗时: {total_time:.2f}秒")
            break
        prev_ll = current_ll
    return a_est, b_est




                    
                    



