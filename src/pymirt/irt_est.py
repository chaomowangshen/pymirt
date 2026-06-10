from .units import (
    analyze_data_matrix, irt_em, rasch_em, irt_3pl_em, grm_em_stepwise, grm_em_standard,
    eap_2pl, eap_3pl, eap_grm, pad_grm_parameters, create_irt_quadrature,
    dataframe_to_sparse_response, analyze_sparse_response,
    irt_em_sparse, rasch_em_sparse, irt_3pl_em_sparse, grm_em_stepwise_sparse, grm_em_standard_sparse,
    eap_2pl_sparse, eap_3pl_sparse, eap_grm_sparse,
    irt_mcem_sparse, irt_mcmc_sparse, irt_saem_sparse,
    rasch_mcem_sparse, rasch_mcmc_sparse, rasch_saem_sparse,
    grm_mcem_stepwise_sparse, grm_mcem_standard_sparse,
    grm_mcmc_stepwise_sparse, grm_mcmc_standard_sparse,
    grm_saem_stepwise_sparse, grm_saem_standard_sparse,
    neural_irt_est, neural_grm_est,
)
import numpy as np
import pandas as pd


def irt(
        response_df: pd.DataFrame,
        model='2PL',
        method='em',
        grm_type='step',
        n_quadrature: int = 27,
        n_categories=None,
        n_samples: int = 100,
        burn_in: int = 100,
        sample_interval: int = 10,
        max_iter: int = 100,
        tol: float = 1e-4,
        verbose: bool = False,
        use_sparse: bool = False,
        nn_config=None,
        _return_backend_model: bool = False,
        ):
    
    '''
    根据反应矩阵，使用IRT模型估计被试的能力值和项目参数。单维只采用高斯-厄米特求积法进行估计，没有method方法。
    参数:
    response_df (pd.DataFrame): 被试作答矩阵，行表示被试，列表示项目。
    model (str): IRT模型类型，支持'2pl'、'grm'，默认为'2pl'。
    grm_type (str): 多级评分模型类型，支持'step'、'stand'，仅在model为'grm'时有效，默认为'step'。
    n_quadrature (int): 高斯-厄米特求积点数，默认为27。
    n_categories (list or None): 项目类别数，若为None则表示2PL模型。
    max_iter (int): 最大迭代次数，默认为100。
    tol (float): 收敛容忍度，默认为1e-4。
    verbose (bool): 是否打印详细信息，默认为False。
    返回:
    a_est (np.array): 项目区分度参数估计值。
    b_est (np.array)或List: 项目难度参数,2pl时为单维数组，grm时为列表，列表中为单维数组。
    theta_est (np.array): 被试的能力估计值。
    '''
    model = model.lower()
    method = method.lower()
    grm_type = grm_type.lower()
    binary_models = {'1pl', 'rasch', '2pl', '3pl'}
    #检查2pl模型的数据是否符合要求，即是否为二元数据
    if model in binary_models:
        # 检查排除 NA 后是否仅含 0 或 1（兼容整数和浮点数）
        valid_values = {0, 1, 0.0, 1.0}
        stacked = response_df.stack()  # 将数据转为单列（自动排除 NA）
        if not stacked.isin(valid_values).all():
            raise ValueError("二分IRT模型要求数据（排除缺失值后）必须为二元作答（0或1），请检查数据。")
        
    if model not in ['1pl', 'rasch', '2pl', '3pl', 'grm']:
        raise ValueError(
            f"model参数必须为'1pl', 'rasch', '2pl', '3pl', 'grm',当前模型为{model}。\n"
        )
    if method not in ['em', 'mcem', 'saem', 'mcmc', 'nn']:
        raise ValueError(
            f"method must be one of 'em', 'mcem', 'saem', 'mcmc', or 'nn'; got {method}."
        )
    if grm_type not in ['step', 'stand']:
        raise ValueError(
            f"grm_type参数必须为'step'或'stand',当前类型为{grm_type}。\n"
        )

    if n_categories is None and model == 'grm':
        raise ValueError(
            "当model为'grm'时，n_categories参数不能为空，请传入每个项目的类别数。\n"
        )
    if model == 'grm':
        if isinstance(n_categories, (list, np.ndarray)):
            n_categories = np.array(n_categories)
            if n_categories.ndim != 1:
                raise ValueError("n_categories必须为一维数组或列表，请检查数据。")
        else:
            raise ValueError("n_categories必须为None、列表或一维numpy数组，请检查数据。")
    if method == 'nn':
        if model in {'1pl', 'rasch', '2pl'}:
            a_est, b_est, theta_est, backend_model = neural_irt_est(
                response_df,
                model=model,
                nn_config=nn_config,
                verbose=verbose,
            )
        elif model == 'grm':
            a_est, b_est, theta_est, backend_model = neural_grm_est(
                response_df,
                n_categories=n_categories,
                grm_type=grm_type,
                nn_config=nn_config,
                verbose=verbose,
            )
        else:
            raise NotImplementedError(
                "Neural IRT currently supports only 1PL/Rasch, 2PL, and GRM."
            )
        if _return_backend_model:
            return a_est, b_est, theta_est, backend_model
        return a_est, b_est, theta_est

    if method != 'em':
        if model == '3pl':
            raise NotImplementedError("3PL sampling methods are not implemented yet.")
        sparse_response = dataframe_to_sparse_response(response_df)
        analyze_sparse_response(
            sparse_response,
            n_categories=n_categories if model == 'grm' else None,
            binary=(model in binary_models),
        )

        if model in {'1pl', 'rasch'}:
            if method == 'mcmc':
                return rasch_mcmc_sparse(
                    sparse_response,
                    n_samples=n_samples,
                    burn_in=burn_in,
                    verbose=verbose,
                )
            if method == 'mcem':
                return rasch_mcem_sparse(
                    sparse_response,
                    n_samples=n_samples,
                    burn_in=burn_in,
                    max_iter=max_iter,
                    tol=tol,
                    sample_interval=sample_interval,
                    verbose=verbose,
                )
            return rasch_saem_sparse(
                sparse_response,
                max_iter=max_iter,
                tol=tol,
                verbose=verbose,
            )

        if model == '2pl':
            if method == 'mcmc':
                return irt_mcmc_sparse(
                    sparse_response,
                    n_samples=n_samples,
                    burn_in=burn_in,
                    verbose=verbose,
                )
            if method == 'mcem':
                return irt_mcem_sparse(
                    sparse_response,
                    n_samples=n_samples,
                    burn_in=burn_in,
                    max_iter=max_iter,
                    tol=tol,
                    sample_interval=sample_interval,
                    verbose=verbose,
                )
            return irt_saem_sparse(
                sparse_response,
                max_iter=max_iter,
                tol=tol,
                verbose=verbose,
            )

        if grm_type == 'step':
            if method == 'mcmc':
                return grm_mcmc_stepwise_sparse(
                    sparse_response,
                    n_categories,
                    n_samples=n_samples,
                    burn_in=burn_in,
                    verbose=verbose,
                )
            if method == 'mcem':
                return grm_mcem_stepwise_sparse(
                    sparse_response,
                    n_categories,
                    n_samples=n_samples,
                    burn_in=burn_in,
                    max_iter=max_iter,
                    tol=tol,
                    sample_interval=sample_interval,
                    verbose=verbose,
                )
            return grm_saem_stepwise_sparse(
                sparse_response,
                n_categories,
                max_iter=max_iter,
                tol=tol,
                verbose=verbose,
            )

        if method == 'mcmc':
            return grm_mcmc_standard_sparse(
                sparse_response,
                n_categories,
                n_samples=n_samples,
                burn_in=burn_in,
                verbose=verbose,
            )
        if method == 'mcem':
            return grm_mcem_standard_sparse(
                sparse_response,
                n_categories,
                n_samples=n_samples,
                burn_in=burn_in,
                max_iter=max_iter,
                tol=tol,
                sample_interval=sample_interval,
                verbose=verbose,
            )
        return grm_saem_standard_sparse(
            sparse_response,
            n_categories,
            max_iter=max_iter,
            tol=tol,
            verbose=verbose,
        )

    if use_sparse:
        if n_quadrature < 1:
            raise ValueError("n_quadrature must be greater than 0.")
        elif n_quadrature > 100:
            print("Warning: n_quadrature is large and may be slow.")

        sparse_response = dataframe_to_sparse_response(response_df)
        analyze_sparse_response(
            sparse_response,
            n_categories=n_categories if model == 'grm' else None,
            binary=(model in binary_models),
        )
        quad_points, quad_weights = create_irt_quadrature(n_quadrature)

        if model in {'1pl', 'rasch'}:
            a_est, b_est = rasch_em_sparse(
                sparse_response,
                n_quadrature=n_quadrature,
                max_iter=max_iter,
                tol=tol,
                verbose=verbose,
            )
            theta_est = eap_2pl_sparse(
                sparse_response, a_est, b_est, quad_points, quad_weights
            )
        elif model == '2pl':
            a_est, b_est = irt_em_sparse(
                sparse_response,
                n_quadrature=n_quadrature,
                max_iter=max_iter,
                tol=tol,
                verbose=verbose,
            )
            theta_est = eap_2pl_sparse(
                sparse_response, a_est, b_est, quad_points, quad_weights
            )
        elif model == '3pl':
            a_est, b_est, c_est = irt_3pl_em_sparse(
                sparse_response,
                n_quadrature=n_quadrature,
                max_iter=max_iter,
                tol=tol,
                verbose=verbose,
            )
            theta_est = eap_3pl_sparse(
                sparse_response, a_est, b_est, c_est, quad_points, quad_weights
            )
            return a_est, b_est, c_est, theta_est
        elif model == 'grm':
            if grm_type == 'step':
                a_est, b_est = grm_em_stepwise_sparse(
                    sparse_response,
                    n_categories,
                    n_quadrature=n_quadrature,
                    max_iter=max_iter,
                    tol=tol,
                    verbose=verbose,
                )
            elif grm_type == 'stand':
                a_est, b_est = grm_em_standard_sparse(
                    sparse_response,
                    n_categories,
                    n_quadrature=n_quadrature,
                    max_iter=max_iter,
                    tol=tol,
                    verbose=verbose,
                )
            theta_est = eap_grm_sparse(
                sparse_response, a_est, b_est, n_categories, quad_points, quad_weights
            )
        return a_est, b_est, theta_est

    # 分析数据矩阵
    if n_categories is not None:
        issues_found, missing_flag, issues_list,missing_list, response_matrix, mask_matrix = analyze_data_matrix(response_df, n_categories, verbose=verbose)
    else:
        issues_found, missing_flag, issues_list,missing_list, response_matrix, mask_matrix = analyze_data_matrix(response_df, verbose=verbose)
    if issues_found:
        raise ValueError(
            "数据分析发现问题:\n" +
            "\n".join(issues_list) +
            "请检查数据并修正后再进行IRT估计。"
        )
    if missing_flag and model=='grm' and grm_type == 'stand':
        print(
            "警告: 数据中部分作答数量过少，采用标准grm模型处理可能影响IRT估计结果。\n" +
            "\n".join(missing_list) +
            "请考虑采用grm_type为'step'进行估计。"
        )
    if n_quadrature < 1:
        raise ValueError("n_quadrature必须大于0，请检查参数。")
    elif n_quadrature > 100:
        print("警告: n_quadrature过大，可能导致计算时间过长，请确认是否需要。")

    #根据method和model传入数据，估计参数
    quad_points,quad_weights=create_irt_quadrature(n_quadrature)
    if model in {'1pl', 'rasch'}:
        a_est, b_est=rasch_em(response_matrix,mask_matrix,n_quadrature,max_iter=max_iter, tol=tol, verbose=verbose)
        theta_est = eap_2pl(response_matrix,mask_matrix, a_est, b_est,quad_points,quad_weights)
    elif model == '2pl':
        a_est, b_est=irt_em(response_matrix,mask_matrix,n_quadrature,max_iter=max_iter, tol=tol, verbose=verbose)
        theta_est = eap_2pl(response_matrix,mask_matrix, a_est, b_est,quad_points,quad_weights)
    elif model == '3pl':
        a_est, b_est, c_est=irt_3pl_em(response_matrix,mask_matrix,n_quadrature,max_iter=max_iter, tol=tol, verbose=verbose)
        theta_est = eap_3pl(response_matrix,mask_matrix, a_est, b_est, c_est,quad_points,quad_weights)
        return a_est, b_est, c_est, theta_est
    elif model == 'grm':
        if grm_type == 'step':
            a_est, b_est = grm_em_stepwise(response_matrix, mask_matrix, n_categories, n_quadrature, max_iter=max_iter, tol=tol, verbose=verbose)
            b_params_padded, b_mask=pad_grm_parameters(b_est,n_categories, padding_value=0.0)
            theta_est = eap_grm(response_matrix, mask_matrix, a_est, b_params_padded,b_mask, n_categories, quad_points,quad_weights)
        elif grm_type == 'stand':
            a_est, b_est = grm_em_standard(response_matrix, mask_matrix, n_categories, n_quadrature, max_iter=max_iter, tol=tol, verbose=verbose)
            b_params_padded, b_mask=pad_grm_parameters(b_est,n_categories, padding_value=0.0)
            theta_est = eap_grm(response_matrix, mask_matrix, a_est, b_params_padded,b_mask, n_categories, quad_points,quad_weights)
        else:
            raise ValueError(f"不支持的GRM类型: {grm_type}，请从 'step', 'stand' 中选择。")
    else:
        raise ValueError(f"不支持的IRT模型: {model}，请从 '1pl', 'rasch', '2pl', '3pl', 'grm'中选择。")
    return a_est, b_est, theta_est

    
