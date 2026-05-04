import numpy as np




#######################2pl######################
def generate_2pl_data(n=1000, items=20, missing_rate=0.2,seed=None):
    """
    生成含缺失值的单维2PL测试数据
    参数:
    n: 被试数量
    items: 项目数量
    missing_rate: 缺失数据比例
    seed: 随机种子，默认为None
    
    返回:
    response_missing: 含缺失值的响应矩阵，形状(n_subjects, n_items)
    mask_matrix: 掩码矩阵，形状(n_subjects, n_items)
    a_true: 真实的区分度参数，形状(n_items,)
    b_true: 真实的难度参数，形状(n_items,)
    theta: 真实的潜在特质参数，形状(n_subjects,)
    """
    # 生成真实参数
    a_true = np.random.uniform(0.5, 3.0, items)
    b_true = np.random.normal(0, 1.0, items)
    theta = np.random.normal(0, 1, n)
    # 计算正确作答概率
    logits = a_true.reshape(1,-1)*(theta.reshape(-1,1)-b_true.reshape(1,-1))
    p_correct = 1 / (1 + np.exp(-np.clip(logits, -35, 35)))
    # 生成响应
    response = (np.random.rand(n, items) < p_correct).astype(int)
    # 添加缺失值
    mask = np.random.choice([0, 1], size=(n, items), p=[missing_rate, 1-missing_rate])
    mask_matrix = mask.astype(float)
    response_missing = response.copy().astype(float)
    response_missing[mask == 0] = np.nan
    return response_missing, mask_matrix, a_true, b_true, theta


def generate_rasch_data(n=1000, items=20, missing_rate=0.2, seed=None):
    """
    Generate binary Rasch/1PL response data with missing values.
    """
    if seed is not None:
        np.random.seed(seed)

    a_true = np.ones(items)
    b_true = np.random.normal(0, 1.0, items)
    theta = np.random.normal(0, 1, n)
    logits = theta.reshape(-1, 1) - b_true.reshape(1, -1)
    p_correct = 1 / (1 + np.exp(-np.clip(logits, -35, 35)))
    response = (np.random.rand(n, items) < p_correct).astype(int)
    mask = np.random.choice([0, 1], size=(n, items), p=[missing_rate, 1 - missing_rate])
    mask_matrix = mask.astype(float)
    response_missing = response.copy().astype(float)
    response_missing[mask == 0] = np.nan
    return response_missing, mask_matrix, a_true, b_true, theta


generate_1pl_data = generate_rasch_data


def generate_3pl_data(n=1000, items=20, missing_rate=0.2, seed=None):
    """
    Generate binary 3PL response data with missing values.
    """
    if seed is not None:
        np.random.seed(seed)

    a_true = np.random.uniform(0.5, 2.5, items)
    b_true = np.random.normal(0, 1.0, items)
    c_true = np.random.uniform(0.05, 0.25, items)
    theta = np.random.normal(0, 1, n)
    logits = a_true.reshape(1, -1) * (theta.reshape(-1, 1) - b_true.reshape(1, -1))
    logistic = 1 / (1 + np.exp(-np.clip(logits, -35, 35)))
    p_correct = c_true.reshape(1, -1) + (1 - c_true.reshape(1, -1)) * logistic
    response = (np.random.rand(n, items) < p_correct).astype(int)
    mask = np.random.choice([0, 1], size=(n, items), p=[missing_rate, 1 - missing_rate])
    mask_matrix = mask.astype(float)
    response_missing = response.copy().astype(float)
    response_missing[mask == 0] = np.nan
    return response_missing, mask_matrix, a_true, b_true, c_true, theta



def generate_m2pl_data(n=1000, items=20, dim=3, missing_rate=0.2, seed=None):
    """
    生成含缺失值的多维2PL测试数据
    
    参数:
    n: 被试数量
    items: 项目数量
    dim: 能力维度数
    missing_rate: 缺失数据比例
    seed: 随机种子，默认为None
    
    返回:
    response_missing: 含缺失值的响应矩阵，形状(n_subjects, n_items)
    mask_matrix: 掩码矩阵，形状(n_subjects, n_items)
    Q:Q矩阵，需包含1个单位矩阵
    a_true: 真实的区分度参数矩阵，形状(items, dims)
    d_true: 真实的阈值参数向量，形状(items,)
    theta_true: 真实的潜在特质参数，形状(n_subjects, dims)
    """
    # 创建Q矩阵
    Q = np.zeros((items, dim), dtype=int)
    for j in range(items):
        num_dims = np.random.randint(1, dim+1)
        dims = np.random.choice(dim, size=num_dims, replace=False)
        for d in dims:
            Q[j, d] = 1
        if j < dim:
            Q[j] = np.zeros(dim)
            Q[j, j] = 1
    # 生成真实参数
    a_true = np.random.uniform(0.2, 3, (items, dim)) * Q
    d_true = np.random.normal(0, 1, items)
    # 打乱题目顺序
    indices = np.arange(items)
    np.random.shuffle(indices)
    Q = Q[indices]
    a_true = a_true[indices]
    d_true = d_true[indices]
    theta = np.random.multivariate_normal(np.zeros(dim), np.eye(dim), n)
    logits = theta @ a_true.T+d_true
    p = 1 / (1 + np.exp(-logits))
    response = (np.random.rand(n, items) < p).astype(float)
    mask = np.random.choice([0, 1], size=(n, items), p=[missing_rate, 1-missing_rate])
    mask_matrix = mask.astype(float)
    response[mask == 0] = np.nan
    return response, mask_matrix, Q, a_true, d_true,theta


#########################grm#################################


def generate_grm_data(n=1000, items=20,max_k=6, n_categories=None, missing_rate=0.2, seed=None):
    """
    生成含缺失值的单维GRM测试数据
    参数:
    n: 被试数量
    items: 项目数量
    n_categories: 每个项目的类别数列表，如果为None则自动生成
    max_k: 项目的最大类别数
    missing_rate: 缺失数据比例
    seed: 随机种子，默认为None
    
    返回:
    response_missing: 含缺失值的响应矩阵，形状(n_subjects, n_items)
    mask_matrix: 掩码矩阵，形状(n_subjects, n_items)
    a_true: 真实的区分度参数，形状(n_items,)
    b_true: 真实的难度参数列表，每个元素是一个形状为(n_categories[j]-1,)的数组
    n_categories: 每个项目的类别数列表，长度为n_items
    theta: 真实的潜在特质参数，形状(n_subjects,)
    """
    # 如果未指定类别数，设定默认值（前一半是二分类，后一半是多分类）
    if n_categories is None:
        n_categories = np.random.randint(2, max_k, size=items)
    
    # 生成真实参数
    a_true = np.random.uniform(0.2, 3, items)
    b_true = []
    min_gap = 0.2  # 最小间隔,保证有一定数量的作答
    for j in range(items):
        k = n_categories[j]
        if k > 1:
            while True:  # 确保阈值有足够间隔
                thresholds = np.sort(np.random.normal(0, 1, k-1))
                adjusted = True
                
                # 检查并调整阈值间隔
                for i in range(len(thresholds)-1):
                    if thresholds[i+1] - thresholds[i] < min_gap:
                        required_adjustment = min_gap - (thresholds[i+1] - thresholds[i])
                        thresholds[i+1:] += required_adjustment
                        adjusted = False
                        break
                
                if adjusted:
                    break
            b_true.append(thresholds)
        else:
            b_true.append(np.array([]))
    # 生成能力与响应
    theta = np.random.normal(0, 1, n)
    response = np.zeros((n, items), dtype=int)
    for j in range(items):
        k = n_categories[j]
        # 计算累积概率
        P_star = np.zeros((n, k+1))
        P_star[:, 0] = 1.0
        P_star[:, k] = 0.0
        for m in range(1, k):
            if m-1 < len(b_true[j]):
                diff = a_true[j] * (theta - b_true[j][m-1])
                diff_clip = np.clip(diff, -30.0, 30.0)  # 数值稳定性
                P_star[:, m] = 1.0 / (1.0 + np.exp(-diff_clip))
        # 计算类别概率
        P_cat = P_star[:, :-1] - P_star[:, 1:]
        P_cat = np.clip(P_cat, 1e-15, 1.0-1e-15)
        P_cat /= P_cat.sum(axis=1, keepdims=True)
        rand_vals = np.random.rand(n, 1)
        cum_probs = np.cumsum(P_cat, axis=1)
        response[:, j] = np.sum(cum_probs<rand_vals,axis=1)
    # 添加缺失值
    mask = np.random.choice([0, 1], size=(n, items), p=[missing_rate, 1-missing_rate])
    mask_matrix = mask.astype(float)
    response_missing = response.copy().astype(float)
    response_missing[mask == 0] = np.nan
    return response_missing, mask_matrix, a_true, b_true, theta, n_categories





def generate_mgrm_data(n=1000, items=20, dim=3,max_k=6, n_categories=None, missing_rate=0.2, seed=None):
    """
    生成含缺失值的多维GRM测试数据
    
    参数:
    n: 被试数量
    items: 项目数量
    dim: 能力维度数
    max_k:项目的最大类别数
    n_categories: 每个项目的类别数列表，如果为None则自动生成
    missing_rate: 缺失数据比例
    seed: 随机种子，默认为None
    
    返回:
    response_missing: 含缺失值的响应矩阵，形状(n_subjects, n_items)
    mask_matrix: 掩码矩阵，形状(n_subjects, n_items)
    a_true: 真实的区分度参数矩阵，形状(items, dims)
    d_true: 真实的阈值参数列表，每个元素是一个形状为(n_categories[j]-1,)的数组
    theta_true: 真实的潜在特质参数，形状(n_subjects, dims)
    n_categories: 每个项目的类别数列表，长度为items
    """
    # 如果未指定类别数，设定默认值
    if n_categories is None:
        n_categories = np.random.randint(2, max_k, size=items)

    # 创建Q矩阵
    Q = np.zeros((items, dim), dtype=int)
    for j in range(items):
        num_dims = np.random.randint(1, dim+1)
        dims = np.random.choice(dim, size=num_dims, replace=False)
        for d in dims:
            Q[j, d] = 1
        if j < dim:
            Q[j] = np.zeros(dim)
            Q[j, j] = 1
    a_true = np.random.uniform(0.2, 3, (items, dim)) * Q
    d_true = []
    min_gap = 0.2  # 最小间隔
    for j in range(items):
        k = n_categories[j]
        if k > 1:
            while True:  # 确保阈值有足够间隔
                thresholds = np.sort(np.random.normal(0, 1.2, k-1))[::-1]  # 从大到小排序
                adjusted = True
                # 检查并调整阈值间隔
                for i in range(len(thresholds)-1):
                    if thresholds[i] - thresholds[i+1] < min_gap:  # 注意: 阈值从大到小
                        required_adjustment = min_gap - (thresholds[i] - thresholds[i+1])
                        thresholds[i+1:] -= required_adjustment
                        adjusted = False
                        break
                if adjusted:
                    break
            d_true.append(thresholds)
        else:
            d_true.append(np.array([]))
    # 生成能力
    theta_true = np.random.multivariate_normal(np.zeros(dim), np.eye(dim), n)
    response = np.zeros((n, items), dtype=int)
    
    # 计算累积概率
    for j in range(items):
        k = n_categories[j]
        P_star = np.zeros((n, k+1))
        P_star[:, 0] = 1.0
        P_star[:, k] = 0.0
        for m in range(1, k):
            if m-1 < len(d_true[j]):
                diff = theta_true@a_true[j].reshape(-1,1)+d_true[j][m-1]
                diff_clip = np.clip(diff, -30.0, 30.0).flatten()  # 数值稳定性
                P_star[:, m] = 1.0 / (1.0 + np.exp(-diff_clip))
        # 计算类别概率
        P_cat = P_star[:, :-1] - P_star[:, 1:]
        P_cat = np.clip(P_cat, 1e-15, 1.0-1e-15)
        P_cat /= P_cat.sum(axis=1, keepdims=True)
        rand_vals = np.random.rand(n, 1)
        cum_probs = np.cumsum(P_cat, axis=1)
        response[:, j] = np.sum(cum_probs<rand_vals,axis=1)
    # 添加缺失值
    mask = np.random.choice([0, 1], size=(n, items), p=[missing_rate, 1-missing_rate])
    mask_matrix = mask.astype(float)
    response_missing = response.copy().astype(float)
    response_missing[mask == 0] = np.nan
    return response_missing,mask_matrix,Q,a_true, d_true, theta_true, n_categories

