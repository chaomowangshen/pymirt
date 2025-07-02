def contains_identity_matrix(Q):
    """
    判断Q矩阵是否包含单位矩阵
    参数:
        Q: 二维numpy数组或列表的列表，形状为(items×dim)
    返回:
        bool: 如果包含单位矩阵返回True，否则返回False
    """
    dim = len(Q[0])  # 获取维度数
    found_dims = set()
    
    for item in Q:
        # 计算该项目在多少个维度上有载荷
        non_zero = [i for i, val in enumerate(item) if val != 0]
        if len(non_zero) == 1:  # 只在一个维度上有载荷
            dim_idx = non_zero[0]
            found_dims.add(dim_idx)
    
    # 检查是否所有维度都被覆盖
    return len(found_dims) == dim



def analyze_data_matrix(df,n_categories=None,verbose=False):
    """
    分析数据框的缺失值和类别分布以及是否和传入的n_categories一致
    参数：
        df (pd.DataFrame): 输入数据框
        n_categories (list或None): 每列预期的类别数量，默认为None
        verbose (bool): 是否打印详细信息，默认为False
    返回：
        issues_found (bool): 是否发现问题
        missing_flag (bool): 是否存在缺失值
        issues_list (list): 发现的问题列表
        missing_list (list): 缺失值相关信息列表
        response (np.array): 响应矩阵
        missing_mask (np.array): 缺失值掩码矩阵
    """
    # 检查列名是否有效
    if any(not isinstance(col, str) for col in df.columns):
        if verbose:
            print("警告：检测到非字符串列名，使用默认列标识符")
        df.columns = [f"列_{i+1}" for i in range(len(df.columns))]
    issues_found = False
    missing_flag= False
    issues_list = []
    missing_list=[]
    # 1. 制作缺失值掩码矩阵 (非缺失1，缺失0)
    response= df.values
    missing_mask = (~df.isna()).astype(int).values
    # 2. 分析每列分布
    for i,col_name in enumerate(df.columns):
        col_data = df[col_name]
        non_missing_data = col_data.dropna()
        if non_missing_data.empty:
            missing_flag = True
            raise ValueError(f"列 '{col_name}' 中所有值都缺失，请检查数据。")
        # 统计各类计数
        value_counts = non_missing_data.value_counts()
        #统计类别数量是否和传入的n_categories一致
        if n_categories is not None and len(value_counts) != n_categories[i]:
            issues_found = True
            issues_list.append((f"  列 '{col_name}' 的类别数量 {len(value_counts)} 与预期的 {n_categories[i]} 不一致。\n"))
        min_count = value_counts.min()
        if min_count < 10:
            issues_found = True
            for category, count in value_counts.items():
                if count < 5:
                    display_category = int(category) if category == int(category) else category
                    missing_list.append((f" 列 '{col_name}' 中类别 '{display_category}' 的计数为 {count} (小于10)。\n"))
    if not issues_found:
        if verbose:
            print("数据分析完成，未发现问题。")
    return issues_found,missing_flag, issues_list,missing_list, response,missing_mask