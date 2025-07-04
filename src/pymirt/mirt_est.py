from .units import (
    contains_identity_matrix,analyze_data_matrix,mirt_em_gh,mirt_em_mc,pad_grm_parameters,
    mgrm_em_gh_stepwise,mgrm_em_mc_stepwise,mgrm_em_gh_standard,mgrm_em_mc_standard,create_mirt_quadrature,
    eap_m2pl,eap_mgrm
    )
import numpy as np
import pandas as pd



def mirt(
        response_df: pd.DataFrame,
        Q:np.array,
        method='em',
        model='m2pl',
        n_categories=None,
        n_quadrature: int = 15,
        n_samples: int = 100,
        burn_in: int = 100,
        sample_interval: int = 10,
        max_iter: int = 100,
        tol: float = 1e-4,
        verbose: bool = False
        ):
    '''
    根据反应矩阵，使用IRT模型估计被试的能力值和项目参数。
    参数:
    response_df (pd.DataFrame): 被试作答矩阵，行表示被试，列表示项目。
    Q (np.array): 项目特征矩阵，形状为 (n_items, n_dimensions)，每行表示一个项目在各维度上的特征。
    method (str): IRT估计方法，支持'em'、'mcem'。
    model (str): IRT模型类型，支持'm2pl'、'mgrm_step'、'mgrm_stand'。
    n_categories (list,np.array or None): 每个项目的类别数，若为None则表示2PL模型。
    n_quadrature (int): 高斯-厄米特求积点数，默认为15。仅在method为'em'时有效。
    n_samples (int): MCMC有效采样次数，默认为100。仅在method为'mcem'时有效。
    burn_in (int): MCMC烧入期，默认为100。仅在method为'mcem'时有效。
    sample_interval (int): MCMC采样间隔，默认为10。开始10此次均采样，此后每隔sample_interval次采样一次。仅在method为'mcem'时有效。
    max_iter (int): 最大迭代次数，默认为100。
    tol (float): 收敛容忍度，默认为1e-4。
    verbose (bool): 是否打印详细信息，默认为False。
    返回:
    a_est (np.array): 项目区分度参数估计值(items,dim)。
    d_est (np.array)或List: 项目阈值参数或列表，列表中为单维数组。
    theta_est (np.array): 被试的能力估计值(n_subjects,dim)。
    '''
    if model.lower() == 'm2pl':
    # 检查排除 NA 后是否仅含 0 或 1（兼容整数和浮点数）
        valid_values = {0, 1, 0.0, 1.0}
        stacked = response_df.stack()  # 将数据转为单列（自动排除 NA）
        if not stacked.isin(valid_values).all():
            raise ValueError("M2PL模型要求数据（排除缺失值后）必须为二元作答（0或1），请检查数据。")
    Q= np.array(Q)
    if Q.ndim != 2:
        raise ValueError("Q矩阵必须为二维数组，请检查Q矩阵。")

    #查看method和model是否符合要求
    if method not in ['em', 'mcem']:
        raise ValueError(
            f"method参数必须为'em'或'mcem'，当前方法为{method}。\n"
        )
    if model not in ['m2pl', 'mgrm_step',  'mgrm_stand']:
        raise ValueError(
            f"model参数必须为'm2pl', 'mgrm_step',  'mgrm_stand'，当前模型为{model}。\n"
        )
    #检查n_categories是否为None或list或单维np.array
    if n_categories is not None:
        if isinstance(n_categories, (list, np.ndarray)):
            n_categories = np.array(n_categories)
            if n_categories.ndim != 1:
                raise ValueError("n_categories必须为一维数组或列表，请检查数据。")
        else:
            raise ValueError("n_categories必须为None、列表或一维numpy数组，请检查数据。")
    dim= Q.shape[1]  # 获取维度数
    # 检查Q是否为二维数组
    #检查Q是否包含单位矩阵
    if not contains_identity_matrix(Q):
        raise ValueError("Q矩阵必须包含单位矩阵，请检查Q矩阵。")
    #检查Q矩阵是否与response_matrix的列数一致
    if response_df.shape[1] != Q.shape[0]:
        raise ValueError(
            f"Q矩阵的行数({Q.shape[0]})与响应矩阵的列数({response_matrix.shape[1]})不一致，请检查数据。"
        )
    # 分析数据矩阵
    if n_categories is not None:
        issues_found, missing_flag, issues_list,missing_list, response_matrix, mask_matrix = analyze_data_matrix(response_df, n_categories, verbose=verbose)
    else:
        issues_found, missing_flag, issues_list,missing_list, response_matrix, mask_matrix = analyze_data_matrix(response_df, verbose=verbose)
    if issues_found:
        raise ValueError(
            "数据分析发现问题:\n" +
            "\n".join(issues_list) +
            "请检查数据并修正后再进行MIRT估计。"
        )
    if missing_flag and model in ['mgrm_stand_gh','mgrm_stand_mc']:
        print(
            "警告: 数据中部分作答数量过少，采用标准grm模型处理可能影响MIRT估计结果。\n" +
            "\n".join(missing_list) +
            "请考虑采用mgrm_step_gh或mgrm_step_gh模型进行估计。"
        )

    #检查模型
    if dim>3 and method != 'mcem':
        raise ValueError(
            f"3维以上仅支持采用mcem方法进行估计，请从新选择方法。\n"
        )
    elif 2<=dim<4 and method=='em':
        if n_quadrature > 30:
            print(
                "警告: 采用高斯-厄米特求积法时，节点数过大可能导致计算开销过大。\n" +
                "建议将节点数(n_quadrature)设置为30或更小。当前n_quadrature为{n_quadrature}。\n"
            )
        elif n_quadrature < 10:
            print(
                "警告: 采用高斯-厄米特求积法时，节点数过小可能导致估计不准确。\n" +
                "建议将节点数(n_quadrature)设置为10或更大。当前n_quadrature为{n_quadrature}。\n")
        elif n_quadrature < 1:
            raise ValueError(
                f"n_quadrature必须大于等于1，当前n_quadrature为{n_quadrature}。\n"
            )
        elif n_quadrature > 50:
            raise ValueError(
                f"n_quadrature必须小于等于50，当前n_quadrature为{n_quadrature}。\n"
            )
    elif dim==1 and method=='mcem':
        raise ValueError(
            f"单维IRT模型仅支持采用em方法进行估计。\n"
        )

    #根据method和model传入数据，估计参数
    if method == 'em':
        quad_points_nd, quad_weights_nd = create_mirt_quadrature(n_quadrature, dim)
        if model == 'm2pl':
            a_est, d_est = mirt_em_gh(
                response_matrix, mask_matrix, Q, 
                n_quadrature=n_quadrature,max_iter=max_iter, tol=tol, verbose=verbose
            )
            theta_est = eap_m2pl(
                response_matrix, mask_matrix, a_est, d_est,quad_points_nd, quad_weights_nd
                )
        elif model == 'mgrm_step':
            a_est, d_est = mgrm_em_gh_stepwise(
                response_matrix, mask_matrix, Q, n_categories,
                n_quadrature=n_quadrature, max_iter=max_iter, tol=tol, verbose=verbose
            )
            d_params_padded, d_mask = pad_grm_parameters(d_est, n_categories, padding_value=0.0)
            theta_est = eap_mgrm(
                response_matrix, mask_matrix, a_est, d_params_padded, d_mask,
                n_categories, quad_points_nd, quad_weights_nd
            )
        elif model == 'mgrm_stand':
            a_est, d_est = mgrm_em_gh_standard(
                response_matrix, mask_matrix, Q, n_categories,
                n_quadrature=n_quadrature, max_iter=max_iter, tol=tol, verbose=verbose
            )
            d_params_padded, d_mask = pad_grm_parameters(d_est, n_categories, padding_value=0.0)
            theta_est = eap_mgrm(
                response_matrix, mask_matrix, a_est, d_params_padded, d_mask,
                n_categories, quad_points_nd, quad_weights_nd
            )
    elif method == 'mcem':
        if model == 'm2pl':
            a_est, d_est, theta_est = mirt_em_mc(
                response_matrix, mask_matrix, Q,
                n_samples=n_samples, burn_in=burn_in, sample_interval=sample_interval,
                max_iter=max_iter, tol=tol, verbose=verbose
            )
        elif model == 'mgrm_step':
            a_est, d_est, theta_est = mgrm_em_mc_stepwise(
                response_matrix, mask_matrix, Q, n_categories,
                n_samples=n_samples, burn_in=burn_in, sample_interval=sample_interval,
                max_iter=max_iter, tol=tol, verbose=verbose
            )
        elif model == 'mgrm_stand':
            a_est, d_est, theta_est = mgrm_em_mc_standard(
                response_matrix, mask_matrix, Q, n_categories,
                n_samples=n_samples, burn_in=burn_in, sample_interval=sample_interval,
                max_iter=max_iter, tol=tol, verbose=verbose
            )
    return a_est, d_est, theta_est





    


