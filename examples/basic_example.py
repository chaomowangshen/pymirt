"""
PyMIRT 基础使用示例

这个示例展示了如何使用 PyMIRT 进行基本的 IRT 参数估计。
作者: Sheng Su
项目地址: https://github.com/chaomowangshen/pymirt
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pymirt import irt, mirt
from pymirt.units.irt_simulate_data import generate_2pl_data, generate_grm_data, generate_m2pl_data, generate_mgrm_data

# 设置随机种子以确保结果可重现
seed=2025
##注：多维模型的参数输出flatten是为了方便绘制图形，实际使用不需要也不应该flatten
def example_irt_2pl():
    """单维 2PL 模型示例"""
    print("=" * 50)
    print("单维 2PL 模型示例")
    print("=" * 50)
    
    # 生成模拟数据
    n_subjects = 1000
    n_items = 30
    
    # 真实参数和作答数据
    response_matrix, mask_matrix, a_true, b_true, theta_true=generate_2pl_data(n_subjects, n_items, missing_rate=0.2,seed=seed)
    response_df = pd.DataFrame(response_matrix, 
                              columns=[f'Item_{i+1}' for i in range(n_items)])
    print(f"数据维度: {response_df.shape}")
    # 使用 PyMIRT 估计参数
    print("\n开始参数估计...")
    a_est, b_est, theta_est = irt(
        response_df=response_df,
        model='2pl',
        n_quadrature=27,
        max_iter=100,
        tol=1e-3,
        verbose=True
    )
    # 显示结果
    print(f"\n估计完成!")
    print(f"区分度参数 (前5个): {a_est[:5]}")
    print(f"难度参数 (前5个): {b_est[:5]}")
    print(f"能力估计 (前5个): {theta_est[:5]}")
    # 计算估计精度
    a_mae = np.mean(np.abs(a_true - a_est))
    b_mae = np.mean(np.abs(b_true - b_est))
    theta_mae = np.mean(np.abs(theta_true - theta_est))
    a_mse = np.mean((a_true - a_est) ** 2)
    b_mse = np.mean((b_true - b_est) ** 2)
    theta_mse = np.mean((theta_true - theta_est) ** 2)
    a_rmse = np.sqrt(a_mse)
    b_rmse = np.sqrt(b_mse)
    theta_rmse = np.sqrt(theta_mse)
    a_bias = np.mean(a_est - a_true)
    b_bias = np.mean(b_est - b_true)
    theta_bias = np.mean(theta_est - theta_true)
    print(f'\n估计精度:')
    print(f"区分度参数 MAE: {a_mae:.4f}, MSE: {a_mse:.4f}, RMSE: {a_rmse:.4f}, Bias: {a_bias:.4f}")
    print(f"难度参数 MAE: {b_mae:.4f}, MSE: {b_mse:.4f}, RMSE: {b_rmse:.4f}, Bias: {b_bias:.4f}")
    print(f"能力参数 MAE: {theta_mae:.4f}, MSE: {theta_mse:.4f}, RMSE: {theta_rmse:.4f}, Bias: {theta_bias:.4f}")
    return a_true, b_true, theta_true, a_est, b_est, theta_est



def example_grm_step():
    """单维 GRM 模型分步估计示例"""
    print("\n" + "=" * 50)
    print("单维 GRM 模型分步估计示例")
    print("=" * 50)
    
    # 生成模拟数据
    n_subjects = 1000
    n_items = 30
    
    # 真实参数和作答数据
    response_matrix, mask_matrix, a_true, b_true, theta_true, n_categories=generate_grm_data(n_subjects, n_items, max_k=5,seed=seed)
    
    response_df = pd.DataFrame(response_matrix,
                              columns=[f'Item_{i+1}' for i in range(n_items)])
    print(f"数据维度: {response_df.shape}")
    
    # 使用 PyMIRT 估计参数
    print("\n开始参数估计...")
    a_est, b_est, theta_est = irt(
        response_df=response_df,
        model='grm',
        grm_type='step',
        n_categories=n_categories,
        n_quadrature=27,
        max_iter=100,
        tol=1e-3,
        verbose=True
    )
    # 显示结果
    print(f"\n估计完成!")
    print(f"区分度参数 (前5个): {a_est[:5]}")
    print(f"难度参数 (前5个): {b_est[:5]}")
    print(f"能力估计 (前5个): {theta_est[:5]}")
    b_est_flat=np.concatenate(b_est, axis=0)
    b_true_flat=np.concatenate(b_true, axis=0)
    # 计算估计精度
    a_mae = np.mean(np.abs(a_true - a_est))
    b_mae = np.mean(np.abs(b_true_flat - b_est_flat))
    theta_mae = np.mean(np.abs(theta_true - theta_est))
    a_mse = np.mean((a_true - a_est) ** 2)
    b_mse = np.mean((b_true_flat - b_est_flat) ** 2)
    theta_mse = np.mean((theta_true - theta_est) ** 2)
    a_rmse = np.sqrt(a_mse)
    b_rmse = np.sqrt(b_mse)
    theta_rmse = np.sqrt(theta_mse)
    a_bias = np.mean(a_est - a_true)
    b_bias = np.mean(b_est_flat - b_true_flat)
    theta_bias = np.mean(theta_est - theta_true)
    print(f'\n估计精度:')
    print(f"区分度参数 MAE: {a_mae:.4f}, MSE: {a_mse:.4f}, RMSE: {a_rmse:.4f}, Bias: {a_bias:.4f}")
    print(f"难度参数 MAE: {b_mae:.4f}, MSE: {b_mse:.4f}, RMSE: {b_rmse:.4f}, Bias: {b_bias:.4f}")
    print(f"能力参数 MAE: {theta_mae:.4f}, MSE: {theta_mse:.4f}, RMSE: {theta_rmse:.4f}, Bias: {theta_bias:.4f}")
    return a_true, b_true_flat, theta_true, a_est, b_est_flat, theta_est

def example_grm_stand():
    """单维 GRM 模型标准估计示例"""
    print("\n" + "=" * 50)
    print("单维 GRM 标准模型示例")
    print("=" * 50)
    
    # 生成模拟数据
    n_subjects = 1000
    n_items = 30
    
    # 真实参数和作答数据
    response_matrix, mask_matrix, a_true, b_true, theta_true, n_categories=generate_grm_data(n_subjects, n_items, max_k=5,seed=seed)
    
    response_df = pd.DataFrame(response_matrix,
                              columns=[f'Item_{i+1}' for i in range(n_items)])
    print(f"数据维度: {response_df.shape}")
    
    # 使用 PyMIRT 估计参数
    print("\n开始参数估计...")
    a_est, b_est, theta_est = irt(
        response_df=response_df,
        model='grm',
        grm_type='stand',
        n_categories=n_categories,
        n_quadrature=27,
        max_iter=100,
        tol=1e-3,
        verbose=True
    )
    # 显示结果
    print(f"\n估计完成!")
    print(f"区分度参数 (前5个): {a_est[:5]}")
    print(f"难度参数 (前5个): {b_est[:5]}")
    print(f"能力估计 (前5个): {theta_est[:5]}")
    b_est_flat=np.concatenate(b_est, axis=0)
    b_true_flat=np.concatenate(b_true, axis=0)
    # 计算估计精度
    a_mae = np.mean(np.abs(a_true - a_est))
    b_mae = np.mean(np.abs(b_true_flat - b_est_flat))
    theta_mae = np.mean(np.abs(theta_true - theta_est))
    a_mse = np.mean((a_true - a_est) ** 2)
    b_mse = np.mean((b_true_flat - b_est_flat) ** 2)
    theta_mse = np.mean((theta_true - theta_est) ** 2)
    a_rmse = np.sqrt(a_mse)
    b_rmse = np.sqrt(b_mse)
    theta_rmse = np.sqrt(theta_mse)
    a_bias = np.mean(a_est - a_true)
    b_bias = np.mean(b_est_flat - b_true_flat)
    theta_bias = np.mean(theta_est - theta_true)
    print(f'\n估计精度:')
    print(f"区分度参数 MAE: {a_mae:.4f}, MSE: {a_mse:.4f}, RMSE: {a_rmse:.4f}, Bias: {a_bias:.4f}")
    print(f"难度参数 MAE: {b_mae:.4f}, MSE: {b_mse:.4f}, RMSE: {b_rmse:.4f}, Bias: {b_bias:.4f}")
    print(f"能力参数 MAE: {theta_mae:.4f}, MSE: {theta_mse:.4f}, RMSE: {theta_rmse:.4f}, Bias: {theta_bias:.4f}")
    return a_true, b_true_flat, theta_true, a_est, b_est_flat, theta_est




def example_m2pl_em():
    """多维 2PL 模型em方法示例"""
    print("\n" + "=" * 50)
    print("多维 2PL 模型em示例")
    print("=" * 50)
    
    # 生成模拟数据
    n_subjects = 1000
    n_items = 30
    n_dimensions = 2
    response_matrix, mask_matrix, Q, a_true, d_true,theta_true=generate_m2pl_data(n_subjects, n_items, n_dimensions, missing_rate=0.2,seed=seed)
    response_df = pd.DataFrame(response_matrix,
                              columns=[f'Item_{i+1}' for i in range(n_items)])
    
    print(f"数据维度: {response_df.shape}")
    print(f"Q 矩阵形状: {Q.shape}")
    
    # 使用 PyMIRT 估计参数
    print("\n开始多维参数估计...")
    a_est, d_est, theta_est = mirt(
        response_df=response_df,
        Q=Q,
        method='em',
        model='m2pl',
        n_quadrature=10,
        max_iter=50,
        tol=1e-3,
        verbose=True
    )
    
    # 显示结果
    print(f"\n估计完成!")
    # 计算估计精度（仅对非零元素）
    mask = Q.astype(bool)
    a_true_flat = a_true[mask].flatten()
    a_est_flat = a_est[mask].flatten()
    theta_true_flat=theta_true.flatten()
    theta_est_flat=theta_est.flatten()
    a_mae = np.mean(np.abs(a_true_flat - a_est_flat))
    d_mae = np.mean(np.abs(d_true - d_est))
    theta_mae = np.mean(np.abs(theta_true - theta_est))
    a_mse = np.mean((a_true_flat - a_est_flat) ** 2)
    d_mse = np.mean((d_true - d_est) ** 2)
    theta_mse = np.mean((theta_true - theta_est) ** 2)
    a_rmse = np.sqrt(a_mse)
    d_rmse = np.sqrt(d_mse)
    theta_rmse = np.sqrt(theta_mse)
    a_bias = np.mean(a_est_flat - a_true_flat)
    d_bias = np.mean(d_est - d_true)
    theta_bias = np.mean(theta_est - theta_true)
    print(f'\n估计精度:')
    print(f"区分度参数 MAE: {a_mae:.4f}, MSE: {a_mse:.4f}, RMSE: {a_rmse:.4f}, Bias: {a_bias:.4f}")
    print(f"难度参数 MAE: {d_mae:.4f}, MSE: {d_mse:.4f}, RMSE: {d_rmse:.4f}, Bias: {d_bias:.4f}")
    print(f"能力参数 MAE: {theta_mae:.4f}, MSE: {theta_mse:.4f}, RMSE: {theta_rmse:.4f}, Bias: {theta_bias:.4f}")
    return  a_true_flat, d_true, theta_true_flat, a_est_flat, d_est, theta_est_flat


def example_m2pl_mcem():
    """多维 2PL 模型mcem方法示例"""
    print("\n" + "=" * 50)
    print("多维 2PL 模型mcem示例")
    print("=" * 50)
    
    # 生成模拟数据
    n_subjects = 1000
    n_items = 30
    n_dimensions = 2
    response_matrix, mask_matrix, Q, a_true, d_true,theta_true=generate_m2pl_data(n_subjects, n_items, n_dimensions, missing_rate=0.2,seed=seed)
    response_df = pd.DataFrame(response_matrix,
                              columns=[f'Item_{i+1}' for i in range(n_items)])
    
    print(f"数据维度: {response_df.shape}")
    print(f"Q 矩阵形状: {Q.shape}")
    
    # 使用 PyMIRT 估计参数
    print("\n开始多维参数估计...")
    a_est, d_est, theta_est = mirt(
        response_df=response_df,
        Q=Q,
        method='mcem',
        model='m2pl',
        n_samples=100,
        burn_in=100,
        sample_interval=10,
        max_iter=50,
        tol=1e-4,
        verbose=True
    )
    
    # 显示结果
    print(f"\n估计完成!")
    # 计算估计精度（仅对非零元素）
    mask = Q.astype(bool)
    a_true_flat = a_true[mask].flatten()
    a_est_flat = a_est[mask].flatten()
    theta_true_flat=theta_true.flatten()
    theta_est_flat=theta_est.flatten()
    a_mae = np.mean(np.abs(a_true_flat - a_est_flat))
    d_mae = np.mean(np.abs(d_true - d_est))
    theta_mae = np.mean(np.abs(theta_true - theta_est))
    a_mse = np.mean((a_true_flat - a_est_flat) ** 2)
    d_mse = np.mean((d_true - d_est) ** 2)
    theta_mse = np.mean((theta_true - theta_est) ** 2)
    a_rmse = np.sqrt(a_mse)
    d_rmse = np.sqrt(d_mse)
    theta_rmse = np.sqrt(theta_mse)
    a_bias = np.mean(a_est_flat - a_true_flat)
    d_bias = np.mean(d_est - d_true)
    theta_bias = np.mean(theta_est - theta_true)
    print(f'\n估计精度:')
    print(f"区分度参数 MAE: {a_mae:.4f}, MSE: {a_mse:.4f}, RMSE: {a_rmse:.4f}, Bias: {a_bias:.4f}")
    print(f"难度参数 MAE: {d_mae:.4f}, MSE: {d_mse:.4f}, RMSE: {d_rmse:.4f}, Bias: {d_bias:.4f}")
    print(f"能力参数 MAE: {theta_mae:.4f}, MSE: {theta_mse:.4f}, RMSE: {theta_rmse:.4f}, Bias: {theta_bias:.4f}")
    return  a_true_flat, d_true, theta_true_flat, a_est_flat, d_est, theta_est_flat


def example_m2pl_saem():
    """多维 2PL 模型saem方法示例"""
    print("\n" + "=" * 50)
    print("多维 2PL 模型saem示例")
    print("=" * 50)
    
    # 生成模拟数据
    n_subjects = 1000
    n_items = 30
    n_dimensions = 2
    response_matrix, mask_matrix, Q, a_true, d_true,theta_true=generate_m2pl_data(n_subjects, n_items, n_dimensions, missing_rate=0.2,seed=seed)
    response_df = pd.DataFrame(response_matrix,
                              columns=[f'Item_{i+1}' for i in range(n_items)])
    
    print(f"数据维度: {response_df.shape}")
    print(f"Q 矩阵形状: {Q.shape}")
    
    # 使用 PyMIRT 估计参数
    print("\n开始多维参数估计...")
    a_est, d_est, theta_est = mirt(
        response_df=response_df,
        Q=Q,
        method='saem',
        model='m2pl',
        max_iter=100,
        tol=1e-4,
        verbose=True
    )
    
    # 显示结果
    print(f"\n估计完成!")
    # 计算估计精度（仅对非零元素）
    mask = Q.astype(bool)
    a_true_flat = a_true[mask].flatten()
    a_est_flat = a_est[mask].flatten()
    theta_true_flat=theta_true.flatten()
    theta_est_flat=theta_est.flatten()
    a_mae = np.mean(np.abs(a_true_flat - a_est_flat))
    d_mae = np.mean(np.abs(d_true - d_est))
    theta_mae = np.mean(np.abs(theta_true - theta_est))
    a_mse = np.mean((a_true_flat - a_est_flat) ** 2)
    d_mse = np.mean((d_true - d_est) ** 2)
    theta_mse = np.mean((theta_true - theta_est) ** 2)
    a_rmse = np.sqrt(a_mse)
    d_rmse = np.sqrt(d_mse)
    theta_rmse = np.sqrt(theta_mse)
    a_bias = np.mean(a_est_flat - a_true_flat)
    d_bias = np.mean(d_est - d_true)
    theta_bias = np.mean(theta_est - theta_true)
    print(f'\n估计精度:')
    print(f"区分度参数 MAE: {a_mae:.4f}, MSE: {a_mse:.4f}, RMSE: {a_rmse:.4f}, Bias: {a_bias:.4f}")
    print(f"难度参数 MAE: {d_mae:.4f}, MSE: {d_mse:.4f}, RMSE: {d_rmse:.4f}, Bias: {d_bias:.4f}")
    print(f"能力参数 MAE: {theta_mae:.4f}, MSE: {theta_mse:.4f}, RMSE: {theta_rmse:.4f}, Bias: {theta_bias:.4f}")
    return  a_true_flat, d_true, theta_true_flat, a_est_flat, d_est, theta_est_flat



def example_m2pl_mcmc():
    """多维 2PL 模型mcmc方法示例"""
    print("\n" + "=" * 50)
    print("多维 2PL 模型mcmc示例")
    print("=" * 50)
    
    # 生成模拟数据
    n_subjects = 1000
    n_items = 30
    n_dimensions = 2
    response_matrix, mask_matrix, Q, a_true, d_true,theta_true=generate_m2pl_data(n_subjects, n_items, n_dimensions, missing_rate=0.2,seed=seed)
    response_df = pd.DataFrame(response_matrix,
                              columns=[f'Item_{i+1}' for i in range(n_items)])
    
    print(f"数据维度: {response_df.shape}")
    print(f"Q 矩阵形状: {Q.shape}")
    
    # 使用 PyMIRT 估计参数
    print("\n开始多维参数估计...")
    a_est, d_est, theta_est = mirt(
        response_df=response_df,
        Q=Q,
        method='mcmc',
        model='m2pl',
        n_samples=3000,
        burn_in=2000,
        verbose=True
    )
    
    # 显示结果
    print(f"\n估计完成!")
    # 计算估计精度（仅对非零元素）
    mask = Q.astype(bool)
    a_true_flat = a_true[mask].flatten()
    a_est_flat = a_est[mask].flatten()
    theta_true_flat=theta_true.flatten()
    theta_est_flat=theta_est.flatten()
    a_mae = np.mean(np.abs(a_true_flat - a_est_flat))
    d_mae = np.mean(np.abs(d_true - d_est))
    theta_mae = np.mean(np.abs(theta_true - theta_est))
    a_mse = np.mean((a_true_flat - a_est_flat) ** 2)
    d_mse = np.mean((d_true - d_est) ** 2)
    theta_mse = np.mean((theta_true - theta_est) ** 2)
    a_rmse = np.sqrt(a_mse)
    d_rmse = np.sqrt(d_mse)
    theta_rmse = np.sqrt(theta_mse)
    a_bias = np.mean(a_est_flat - a_true_flat)
    d_bias = np.mean(d_est - d_true)
    theta_bias = np.mean(theta_est - theta_true)
    print(f'\n估计精度:')
    print(f"区分度参数 MAE: {a_mae:.4f}, MSE: {a_mse:.4f}, RMSE: {a_rmse:.4f}, Bias: {a_bias:.4f}")
    print(f"难度参数 MAE: {d_mae:.4f}, MSE: {d_mse:.4f}, RMSE: {d_rmse:.4f}, Bias: {d_bias:.4f}")
    print(f"能力参数 MAE: {theta_mae:.4f}, MSE: {theta_mse:.4f}, RMSE: {theta_rmse:.4f}, Bias: {theta_bias:.4f}")
    return  a_true_flat, d_true, theta_true_flat, a_est_flat, d_est, theta_est_flat





def example_mgrm_step_em():
    """多维 mgrm_step 模型em方法示例"""
    print("\n" + "=" * 50)
    print("多维 mgrm模型em方法分步计算示例")
    print("=" * 50)
    # 生成模拟数据
    n_subjects = 1000
    n_items = 30
    n_dimensions = 2
    response_matrix, mask_matrix, Q, a_true, d_true,theta_true,n_categories=generate_mgrm_data(n_subjects, n_items, n_dimensions, missing_rate=0.2)
    response_df = pd.DataFrame(response_matrix,columns=[f'Item_{i+1}' for i in range(n_items)])
    print(f"数据维度: {response_df.shape}")
    print(f"Q 矩阵形状: {Q.shape}")
    # 使用 PyMIRT 估计参数
    print("\n开始多维参数估计...")
    a_est, d_est, theta_est = mirt(
        response_df=response_df,
        Q=Q,
        method='em',
        model='mgrm',
        grm_type='step',
        n_categories=n_categories,
        n_quadrature=10,
        max_iter=50,
        tol=1e-3,
        verbose=True
    )
    # 显示结果
    print(f"\n估计完成!")
    mask = Q.astype(bool)
    a_true_flat = a_true[mask].flatten()
    a_est_flat = a_est[mask].flatten()
    d_est_flat = np.concatenate(d_est, axis=0)
    d_true_flat = np.concatenate(d_true, axis=0)
    theta_true_flat=theta_true.flatten()
    theta_est_flat=theta_est.flatten()
    a_mae = np.mean(np.abs(a_true_flat - a_est_flat))
    d_mae = np.mean(np.abs(d_true_flat - d_est_flat))
    theta_mae = np.mean(np.abs(theta_true - theta_est))
    a_mse = np.mean((a_true_flat - a_est_flat) ** 2)
    d_mse = np.mean((d_true_flat - d_est_flat) ** 2)
    theta_mse = np.mean((theta_true - theta_est) ** 2)
    a_rmse = np.sqrt(a_mse)
    d_rmse = np.sqrt(d_mse)
    theta_rmse = np.sqrt(theta_mse)
    a_bias = np.mean(a_est_flat - a_true_flat)
    d_bias = np.mean(d_est_flat - d_true_flat)
    theta_bias = np.mean(theta_est - theta_true)
    print(f'\n估计精度:')
    print(f"区分度参数 MAE: {a_mae:.4f}, MSE: {a_mse:.4f}, RMSE: {a_rmse:.4f}, Bias: {a_bias:.4f}")
    print(f"难度参数 MAE: {d_mae:.4f}, MSE: {d_mse:.4f}, RMSE: {d_rmse:.4f}, Bias: {d_bias:.4f}")
    print(f"能力参数 MAE: {theta_mae:.4f}, MSE: {theta_mse:.4f}, RMSE: {theta_rmse:.4f}, Bias: {theta_bias:.4f}")
    return a_true_flat, d_true_flat, theta_true_flat, a_est_flat, d_est_flat, theta_est_flat




def example_mgrm_step_mcem():
    """多维 mgrm_step 模型mcem方法示例"""
    print("\n" + "=" * 50)
    print("多维 mgrm模型mcem方法分步计算示例")
    print("=" * 50)
    # 生成模拟数据
    n_subjects = 1000
    n_items = 30
    n_dimensions = 2
    response_matrix, mask_matrix, Q, a_true, d_true,theta_true,n_categories=generate_mgrm_data(n_subjects, n_items, n_dimensions, missing_rate=0.2)
    response_df = pd.DataFrame(response_matrix,columns=[f'Item_{i+1}' for i in range(n_items)])
    print(f"数据维度: {response_df.shape}")
    print(f"Q 矩阵形状: {Q.shape}")
    # 使用 PyMIRT 估计参数
    print("\n开始多维参数估计...")
    a_est, d_est, theta_est = mirt(
        response_df=response_df,
        Q=Q,
        method='mcem',
        model='mgrm',
        grm_type='step',
        n_categories=n_categories,
        n_samples=100,
        burn_in=100,
        sample_interval=10,
        max_iter=50,
        tol=1e-4,
        verbose=True
    )
    # 显示结果
    print(f"\n估计完成!")
    mask = Q.astype(bool)
    a_true_flat = a_true[mask].flatten()
    a_est_flat = a_est[mask].flatten()
    d_est_flat = np.concatenate(d_est, axis=0)
    d_true_flat = np.concatenate(d_true, axis=0)
    theta_true_flat=theta_true.flatten()
    theta_est_flat=theta_est.flatten()
    a_mae = np.mean(np.abs(a_true_flat - a_est_flat))
    d_mae = np.mean(np.abs(d_true_flat - d_est_flat))
    theta_mae = np.mean(np.abs(theta_true - theta_est))
    a_mse = np.mean((a_true_flat - a_est_flat) ** 2)
    d_mse = np.mean((d_true_flat - d_est_flat) ** 2)
    theta_mse = np.mean((theta_true - theta_est) ** 2)
    a_rmse = np.sqrt(a_mse)
    d_rmse = np.sqrt(d_mse)
    theta_rmse = np.sqrt(theta_mse)
    a_bias = np.mean(a_est_flat - a_true_flat)
    d_bias = np.mean(d_est_flat - d_true_flat)
    theta_bias = np.mean(theta_est - theta_true)
    print(f'\n估计精度:')
    print(f"区分度参数 MAE: {a_mae:.4f}, MSE: {a_mse:.4f}, RMSE: {a_rmse:.4f}, Bias: {a_bias:.4f}")
    print(f"难度参数 MAE: {d_mae:.4f}, MSE: {d_mse:.4f}, RMSE: {d_rmse:.4f}, Bias: {d_bias:.4f}")
    print(f"能力参数 MAE: {theta_mae:.4f}, MSE: {theta_mse:.4f}, RMSE: {theta_rmse:.4f}, Bias: {theta_bias:.4f}")
    return a_true_flat, d_true_flat, theta_true_flat, a_est_flat, d_est_flat, theta_est_flat


def example_mgrm_step_saem():
    """多维 mgrm_step 模型saem方法示例"""
    print("\n" + "=" * 50)
    print("多维 mgrm模型saem方法分步计算示例")
    print("=" * 50)
    # 生成模拟数据
    n_subjects = 1000
    n_items = 30
    n_dimensions = 2
    response_matrix, mask_matrix, Q, a_true, d_true,theta_true,n_categories=generate_mgrm_data(n_subjects, n_items, n_dimensions, missing_rate=0.2)
    response_df = pd.DataFrame(response_matrix,columns=[f'Item_{i+1}' for i in range(n_items)])
    print(f"数据维度: {response_df.shape}")
    print(f"Q 矩阵形状: {Q.shape}")
    # 使用 PyMIRT 估计参数
    print("\n开始多维参数估计...")
    a_est, d_est, theta_est = mirt(
        response_df=response_df,
        Q=Q,
        method='saem',
        model='mgrm',
        grm_type='step',
        n_categories=n_categories,
        max_iter=100,
        tol=1e-4,
        verbose=True
    )
    # 显示结果
    print(f"\n估计完成!")
    mask = Q.astype(bool)
    a_true_flat = a_true[mask].flatten()
    a_est_flat = a_est[mask].flatten()
    d_est_flat = np.concatenate(d_est, axis=0)
    d_true_flat = np.concatenate(d_true, axis=0)
    theta_true_flat=theta_true.flatten()
    theta_est_flat=theta_est.flatten()
    a_mae = np.mean(np.abs(a_true_flat - a_est_flat))
    d_mae = np.mean(np.abs(d_true_flat - d_est_flat))
    theta_mae = np.mean(np.abs(theta_true - theta_est))
    a_mse = np.mean((a_true_flat - a_est_flat) ** 2)
    d_mse = np.mean((d_true_flat - d_est_flat) ** 2)
    theta_mse = np.mean((theta_true - theta_est) ** 2)
    a_rmse = np.sqrt(a_mse)
    d_rmse = np.sqrt(d_mse)
    theta_rmse = np.sqrt(theta_mse)
    a_bias = np.mean(a_est_flat - a_true_flat)
    d_bias = np.mean(d_est_flat - d_true_flat)
    theta_bias = np.mean(theta_est - theta_true)
    print(f'\n估计精度:')
    print(f"区分度参数 MAE: {a_mae:.4f}, MSE: {a_mse:.4f}, RMSE: {a_rmse:.4f}, Bias: {a_bias:.4f}")
    print(f"难度参数 MAE: {d_mae:.4f}, MSE: {d_mse:.4f}, RMSE: {d_rmse:.4f}, Bias: {d_bias:.4f}")
    print(f"能力参数 MAE: {theta_mae:.4f}, MSE: {theta_mse:.4f}, RMSE: {theta_rmse:.4f}, Bias: {theta_bias:.4f}")
    return a_true_flat, d_true_flat, theta_true_flat, a_est_flat, d_est_flat, theta_est_flat




def example_mgrm_step_mcmc():
    """多维 mgrm_step 模型mcmc方法示例"""
    print("\n" + "=" * 50)
    print("多维 mgrm模型mcmc方法分步计算示例")
    print("=" * 50)
    # 生成模拟数据
    n_subjects = 1000
    n_items = 30
    n_dimensions = 2
    response_matrix, mask_matrix, Q, a_true, d_true,theta_true,n_categories=generate_mgrm_data(n_subjects, n_items, n_dimensions, missing_rate=0.2)
    response_df = pd.DataFrame(response_matrix,columns=[f'Item_{i+1}' for i in range(n_items)])
    print(f"数据维度: {response_df.shape}")
    print(f"Q 矩阵形状: {Q.shape}")
    # 使用 PyMIRT 估计参数
    print("\n开始多维参数估计...")
    a_est, d_est, theta_est = mirt(
        response_df=response_df,
        Q=Q,
        method='mcmc',
        model='mgrm',
        grm_type='step',
        n_categories=n_categories,
        n_samples=3000,
        burn_in=2000,
        verbose=True
    )
    # 显示结果
    print(f"\n估计完成!")
    mask = Q.astype(bool)
    a_true_flat = a_true[mask].flatten()
    a_est_flat = a_est[mask].flatten()
    d_est_flat = np.concatenate(d_est, axis=0)
    d_true_flat = np.concatenate(d_true, axis=0)
    theta_true_flat=theta_true.flatten()
    theta_est_flat=theta_est.flatten()
    a_mae = np.mean(np.abs(a_true_flat - a_est_flat))
    d_mae = np.mean(np.abs(d_true_flat - d_est_flat))
    theta_mae = np.mean(np.abs(theta_true - theta_est))
    a_mse = np.mean((a_true_flat - a_est_flat) ** 2)
    d_mse = np.mean((d_true_flat - d_est_flat) ** 2)
    theta_mse = np.mean((theta_true - theta_est) ** 2)
    a_rmse = np.sqrt(a_mse)
    d_rmse = np.sqrt(d_mse)
    theta_rmse = np.sqrt(theta_mse)
    a_bias = np.mean(a_est_flat - a_true_flat)
    d_bias = np.mean(d_est_flat - d_true_flat)
    theta_bias = np.mean(theta_est - theta_true)
    print(f'\n估计精度:')
    print(f"区分度参数 MAE: {a_mae:.4f}, MSE: {a_mse:.4f}, RMSE: {a_rmse:.4f}, Bias: {a_bias:.4f}")
    print(f"难度参数 MAE: {d_mae:.4f}, MSE: {d_mse:.4f}, RMSE: {d_rmse:.4f}, Bias: {d_bias:.4f}")
    print(f"能力参数 MAE: {theta_mae:.4f}, MSE: {theta_mse:.4f}, RMSE: {theta_rmse:.4f}, Bias: {theta_bias:.4f}")
    return a_true_flat, d_true_flat, theta_true_flat, a_est_flat, d_est_flat, theta_est_flat



    

def example_mgrm_stand_em():
    """多维 mgrm_stand 模型em方法示例"""
    print("\n" + "=" * 50)
    print("多维 mgrm 模型em方法示例")
    print("=" * 50)
    # 生成模拟数据
    n_subjects = 1000
    n_items = 30
    n_dimensions = 2
    response_matrix, mask_matrix, Q, a_true, d_true,theta_true,n_categories=generate_mgrm_data(n_subjects, n_items, n_dimensions, missing_rate=0.2)
    response_df = pd.DataFrame(response_matrix,columns=[f'Item_{i+1}' for i in range(n_items)])
    print(f"数据维度: {response_df.shape}")
    print(f"Q 矩阵形状: {Q.shape}")
    # 使用 PyMIRT 估计参数
    print("\n开始多维参数估计...")
    a_est, d_est, theta_est = mirt(
        response_df=response_df,
        Q=Q,
        method='em',
        model='mgrm',
        grm_type='stand',
        n_categories=n_categories,
        n_quadrature=10,
        max_iter=50,
        tol=1e-3,
        verbose=True
    )
    # 显示结果
    print(f"\n估计完成!")
    mask = Q.astype(bool)
    a_true_flat = a_true[mask].flatten()
    a_est_flat = a_est[mask].flatten()
    d_est_flat = np.concatenate(d_est, axis=0)
    d_true_flat = np.concatenate(d_true, axis=0)
    theta_true_flat=theta_true.flatten()
    theta_est_flat=theta_est.flatten()
    a_mae = np.mean(np.abs(a_true_flat - a_est_flat))
    d_mae = np.mean(np.abs(d_true_flat - d_est_flat))
    theta_mae = np.mean(np.abs(theta_true - theta_est))
    a_mse = np.mean((a_true_flat - a_est_flat) ** 2)
    d_mse = np.mean((d_true_flat - d_est_flat) ** 2)
    theta_mse = np.mean((theta_true - theta_est) ** 2)
    a_rmse = np.sqrt(a_mse)
    d_rmse = np.sqrt(d_mse)
    theta_rmse = np.sqrt(theta_mse)
    a_bias = np.mean(a_est_flat - a_true_flat)
    d_bias = np.mean(d_est_flat - d_true_flat)
    theta_bias = np.mean(theta_est - theta_true)
    print(f'\n估计精度:')
    print(f"区分度参数 MAE: {a_mae:.4f}, MSE: {a_mse:.4f}, RMSE: {a_rmse:.4f}, Bias: {a_bias:.4f}")
    print(f"难度参数 MAE: {d_mae:.4f}, MSE: {d_mse:.4f}, RMSE: {d_rmse:.4f}, Bias: {d_bias:.4f}")
    print(f"能力参数 MAE: {theta_mae:.4f}, MSE: {theta_mse:.4f}, RMSE: {theta_rmse:.4f}, Bias: {theta_bias:.4f}")
    return a_true_flat, d_true_flat, theta_true_flat, a_est_flat, d_est_flat, theta_est_flat




def example_mgrm_stand_mcem():
    """多维 mgrm 模型mcem方法示例"""
    print("\n" + "=" * 50)
    print("多维 mgrm模型mcem方法示例")
    print("=" * 50)
    # 生成模拟数据
    n_subjects = 1000
    n_items = 30
    n_dimensions = 2
    response_matrix, mask_matrix, Q, a_true, d_true,theta_true,n_categories=generate_mgrm_data(n_subjects, n_items, n_dimensions, missing_rate=0.2)
    response_df = pd.DataFrame(response_matrix,columns=[f'Item_{i+1}' for i in range(n_items)])
    print(f"数据维度: {response_df.shape}")
    print(f"Q 矩阵形状: {Q.shape}")
    # 使用 PyMIRT 估计参数
    print("\n开始多维参数估计...")
    a_est, d_est, theta_est = mirt(
        response_df=response_df,
        Q=Q,
        method='mcem',
        model='mgrm',
        grm_type='stand',
        n_categories=n_categories,
        n_samples=100,
        burn_in=100,
        sample_interval=10,
        max_iter=50,
        tol=1e-4,
        verbose=True
    )
    # 显示结果
    print(f"\n估计完成!")
    mask = Q.astype(bool)
    a_true_flat = a_true[mask].flatten()
    a_est_flat = a_est[mask].flatten()
    d_est_flat = np.concatenate(d_est, axis=0)
    d_true_flat = np.concatenate(d_true, axis=0)
    theta_true_flat=theta_true.flatten()
    theta_est_flat=theta_est.flatten()
    a_mae = np.mean(np.abs(a_true_flat - a_est_flat))
    d_mae = np.mean(np.abs(d_true_flat - d_est_flat))
    theta_mae = np.mean(np.abs(theta_true - theta_est))
    a_mse = np.mean((a_true_flat - a_est_flat) ** 2)
    d_mse = np.mean((d_true_flat - d_est_flat) ** 2)
    theta_mse = np.mean((theta_true - theta_est) ** 2)
    a_rmse = np.sqrt(a_mse)
    d_rmse = np.sqrt(d_mse)
    theta_rmse = np.sqrt(theta_mse)
    a_bias = np.mean(a_est_flat - a_true_flat)
    d_bias = np.mean(d_est_flat - d_true_flat)
    theta_bias = np.mean(theta_est - theta_true)
    print(f'\n估计精度:')
    print(f"区分度参数 MAE: {a_mae:.4f}, MSE: {a_mse:.4f}, RMSE: {a_rmse:.4f}, Bias: {a_bias:.4f}")
    print(f"难度参数 MAE: {d_mae:.4f}, MSE: {d_mse:.4f}, RMSE: {d_rmse:.4f}, Bias: {d_bias:.4f}")
    print(f"能力参数 MAE: {theta_mae:.4f}, MSE: {theta_mse:.4f}, RMSE: {theta_rmse:.4f}, Bias: {theta_bias:.4f}")
    return a_true_flat, d_true_flat, theta_true_flat, a_est_flat, d_est_flat, theta_est_flat



def example_mgrm_stand_saem():
    """多维 mgrm_stand 模型saem方法示例"""
    print("\n" + "=" * 50)
    print("多维 mgrm模型saem法示例")
    print("=" * 50)
    # 生成模拟数据
    n_subjects = 1000
    n_items = 30
    n_dimensions = 2
    response_matrix, mask_matrix, Q, a_true, d_true,theta_true,n_categories=generate_mgrm_data(n_subjects, n_items, n_dimensions, missing_rate=0.2)
    response_df = pd.DataFrame(response_matrix,columns=[f'Item_{i+1}' for i in range(n_items)])
    print(f"数据维度: {response_df.shape}")
    print(f"Q 矩阵形状: {Q.shape}")
    # 使用 PyMIRT 估计参数
    print("\n开始多维参数估计...")
    a_est, d_est, theta_est = mirt(
        response_df=response_df,
        Q=Q,
        method='saem',
        model='mgrm',
        grm_type='stand',
        n_categories=n_categories,
        max_iter=100,
        tol=1e-4,
        verbose=True
    )
    # 显示结果
    print(f"\n估计完成!")
    mask = Q.astype(bool)
    a_true_flat = a_true[mask].flatten()
    a_est_flat = a_est[mask].flatten()
    d_est_flat = np.concatenate(d_est, axis=0)
    d_true_flat = np.concatenate(d_true, axis=0)
    theta_true_flat=theta_true.flatten()
    theta_est_flat=theta_est.flatten()
    a_mae = np.mean(np.abs(a_true_flat - a_est_flat))
    d_mae = np.mean(np.abs(d_true_flat - d_est_flat))
    theta_mae = np.mean(np.abs(theta_true - theta_est))
    a_mse = np.mean((a_true_flat - a_est_flat) ** 2)
    d_mse = np.mean((d_true_flat - d_est_flat) ** 2)
    theta_mse = np.mean((theta_true - theta_est) ** 2)
    a_rmse = np.sqrt(a_mse)
    d_rmse = np.sqrt(d_mse)
    theta_rmse = np.sqrt(theta_mse)
    a_bias = np.mean(a_est_flat - a_true_flat)
    d_bias = np.mean(d_est_flat - d_true_flat)
    theta_bias = np.mean(theta_est - theta_true)
    print(f'\n估计精度:')
    print(f"区分度参数 MAE: {a_mae:.4f}, MSE: {a_mse:.4f}, RMSE: {a_rmse:.4f}, Bias: {a_bias:.4f}")
    print(f"难度参数 MAE: {d_mae:.4f}, MSE: {d_mse:.4f}, RMSE: {d_rmse:.4f}, Bias: {d_bias:.4f}")
    print(f"能力参数 MAE: {theta_mae:.4f}, MSE: {theta_mse:.4f}, RMSE: {theta_rmse:.4f}, Bias: {theta_bias:.4f}")
    return a_true_flat, d_true_flat, theta_true_flat, a_est_flat, d_est_flat, theta_est_flat



def example_mgrm_stand_mcmc():
    """多维 mgrm_stand 模型mcmc方法示例"""
    print("\n" + "=" * 50)
    print("多维 mgrm模型mcmc方法示例")
    print("=" * 50)
    # 生成模拟数据
    n_subjects = 1000
    n_items = 30
    n_dimensions = 2
    response_matrix, mask_matrix, Q, a_true, d_true,theta_true,n_categories=generate_mgrm_data(n_subjects, n_items, n_dimensions, missing_rate=0.2)
    response_df = pd.DataFrame(response_matrix,columns=[f'Item_{i+1}' for i in range(n_items)])
    print(f"数据维度: {response_df.shape}")
    print(f"Q 矩阵形状: {Q.shape}")
    # 使用 PyMIRT 估计参数
    print("\n开始多维参数估计...")
    a_est, d_est, theta_est = mirt(
        response_df=response_df,
        Q=Q,
        method='mcmc',
        model='mgrm',
        grm_type='stand',
        n_categories=n_categories,
        n_samples=3000,
        burn_in=2000,
        verbose=True
    )
    # 显示结果
    print(f"\n估计完成!")
    mask = Q.astype(bool)
    a_true_flat = a_true[mask].flatten()
    a_est_flat = a_est[mask].flatten()
    d_est_flat = np.concatenate(d_est, axis=0)
    d_true_flat = np.concatenate(d_true, axis=0)
    theta_true_flat=theta_true.flatten()
    theta_est_flat=theta_est.flatten()
    a_mae = np.mean(np.abs(a_true_flat - a_est_flat))
    d_mae = np.mean(np.abs(d_true_flat - d_est_flat))
    theta_mae = np.mean(np.abs(theta_true - theta_est))
    a_mse = np.mean((a_true_flat - a_est_flat) ** 2)
    d_mse = np.mean((d_true_flat - d_est_flat) ** 2)
    theta_mse = np.mean((theta_true - theta_est) ** 2)
    a_rmse = np.sqrt(a_mse)
    d_rmse = np.sqrt(d_mse)
    theta_rmse = np.sqrt(theta_mse)
    a_bias = np.mean(a_est_flat - a_true_flat)
    d_bias = np.mean(d_est_flat - d_true_flat)
    theta_bias = np.mean(theta_est - theta_true)
    print(f'\n估计精度:')
    print(f"区分度参数 MAE: {a_mae:.4f}, MSE: {a_mse:.4f}, RMSE: {a_rmse:.4f}, Bias: {a_bias:.4f}")
    print(f"难度参数 MAE: {d_mae:.4f}, MSE: {d_mse:.4f}, RMSE: {d_rmse:.4f}, Bias: {d_bias:.4f}")
    print(f"能力参数 MAE: {theta_mae:.4f}, MSE: {theta_mse:.4f}, RMSE: {theta_rmse:.4f}, Bias: {theta_bias:.4f}")
    return a_true_flat, d_true_flat, theta_true_flat, a_est_flat, d_est_flat, theta_est_flat














def plot_results(true_params, est_params, param_name,model='2pl'):
    """绘制参数估计结果对比图"""
    try:
        plt.figure(figsize=(8, 6))
        plt.scatter(true_params, est_params, alpha=0.7)
        plt.plot([min(true_params), max(true_params)], 
                [min(true_params), max(true_params)], 'r--', label='Perfect estimation')
        plt.xlabel(f'True {param_name}')
        plt.ylabel(f'Estimated {param_name}')
        plt.title(f'{model}/{param_name} Parameter Recovery')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 计算相关系数
        corr = np.corrcoef(true_params, est_params)[0, 1]
        plt.text(0.05, 0.95, f'r = {corr:.3f}', transform=plt.gca().transAxes, 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        plt.tight_layout()
        plt.show()
    except ImportError:
        print("Matplotlib 未安装，跳过绘图")


if __name__ == "__main__":
    print("PyMIRT 使用示例")
    print("欢迎使用 PyMIRT - 项目反应理论参数估计包！\n")
    
    # 运行单维0-1计分示例
    true_a, true_b, true_theta, a_est, b_est, theta_est = example_irt_2pl()
    # 绘制结果（如果有 matplotlib）
    plot_results(true_a, a_est, 'Discrimination', model='2PL')
    plot_results(true_b, b_est, 'Difficulty', model='2PL')
    plot_results(true_theta, theta_est, 'Ability', model='2PL')
    print("\n" + "=" * 50)


    # 运行单维分布多级计分计分示例
    true_a, true_b, true_theta, a_est, b_est, theta_est = example_grm_step()
    # 绘制结果（如果有 matplotlib）
    plot_results(true_a, a_est, 'Discrimination', model='GRM Step')
    plot_results(true_b, b_est, 'Difficulty', model='GRM Step')
    plot_results(true_theta, theta_est, 'Ability', model='GRM Step')
    print("\n" + "=" * 50)

    # 运行单维标准多级计分计分示例
    true_a, true_b, true_theta, a_est, b_est, theta_est = example_grm_stand()
    # 绘制结果（如果有 matplotlib）
    plot_results(true_a, a_est, 'Discrimination', model='GRM Stand')
    plot_results(true_b, b_est, 'Difficulty', model='GRM Stand')
    plot_results(true_theta, theta_est, 'Ability', model='GRM Stand')
    print("\n" + "=" * 50)


    
    # 运行em算法多维2pl示例
    true_a, true_d, true_theta, a_est, d_est, theta_est = example_m2pl_em()
    # 绘制结果（如果有 matplotlib）
    plot_results(true_a, a_est, 'Discrimination', model='M2PL EM')
    plot_results(true_d, d_est, 'Difficulty', model='M2PL EM')
    plot_results(true_theta, theta_est, 'Ability', model='M2PL EM')
    print("\n" + "=" * 50)

    # 运行mcem算法多维2pl示例
    true_a, true_d, true_theta, a_est, d_est, theta_est = example_m2pl_mcem()
    # 绘制结果（如果有 matplotlib）
    plot_results(true_a, a_est, 'Discrimination', model='M2PL MCEM')
    plot_results(true_d, d_est, 'Difficulty', model='M2PL MCEM')
    plot_results(true_theta, theta_est, 'Ability', model='M2PL MCEM')
    print("\n" + "=" * 50)

    # 运行saem算法多维2pl示例
    true_a, true_d, true_theta, a_est, d_est, theta_est = example_m2pl_saem()
    # 绘制结果（如果有 matplotlib）
    plot_results(true_a, a_est, 'Discrimination', model='M2PL SAEM')
    plot_results(true_d, d_est, 'Difficulty', model='M2PL SAEM')
    plot_results(true_theta, theta_est, 'Ability', model='M2PL SAEM')
    print("\n" + "=" * 50)


    #运行mcmc算法多维2pl示例
    true_a, true_d, true_theta, a_est, d_est, theta_est = example_m2pl_mcmc()
    # 绘制结果（如果有 matplotlib）
    plot_results(true_a, a_est, 'Discrimination', model='M2PL MCMC')
    plot_results(true_d, d_est, 'Difficulty', model='M2PL MCMC')
    plot_results(true_theta, theta_est, 'Ability', model='M2PL MCMC')
    print("\n" + "=" * 50)




    # 运行em算法多维mgrm_step示例
    true_a, true_d, true_theta, a_est, d_est, theta_est = example_mgrm_step_em()
    # 绘制结果（如果有 matplotlib）
    plot_results(true_a, a_est, 'Discrimination', model='MGRM Step EM')
    plot_results(true_d, d_est, 'Difficulty', model='MGRM Step EM')
    plot_results(true_theta, theta_est, 'Ability', model='MGRM Step EM')
    print("\n" + "=" * 50)


    # 运行mcem算法多维mgrm_step示例
    true_a, true_d, true_theta, a_est, d_est, theta_est = example_mgrm_step_mcem()
    # 绘制结果（如果有 matplotlib）
    plot_results(true_a, a_est, 'Discrimination', model='MGRM Step MCEM')
    plot_results(true_d, d_est, 'Difficulty', model='MGRM Step MCEM')
    plot_results(true_theta, theta_est, 'Ability', model='MGRM Step MCEM')
    print("\n" + "=" * 50)

    #运行saem算法多维mgrm_step示例
    true_a, true_d, true_theta, a_est, d_est, theta_est = example_mgrm_step_saem()
    # 绘制结果（如果有 matplotlib）
    plot_results(true_a, a_est, 'Discrimination', model='MGRM Step SAEM')
    plot_results(true_d, d_est, 'Difficulty', model='MGRM Step SAEM')
    plot_results(true_theta, theta_est, 'Ability', model='MGRM Step SAEM')
    print("\n" + "=" * 50)

    # 运行mcmc算法多维mgrm_step示例
    true_a, true_d, true_theta, a_est, d_est, theta_est = example_mgrm_step_mcmc()
    # 绘制结果（如果有 matplotlib）
    plot_results(true_a, a_est, 'Discrimination', model='MGRM Step MCMC')
    plot_results(true_d, d_est, 'Difficulty', model='MGRM Step MCMC')
    plot_results(true_theta, theta_est, 'Ability', model='MGRM Step MCMC')
    print("\n" + "=" * 50)





    # 运行em算法多维mgrm_stand示例
    true_a, true_d, true_theta, a_est, d_est, theta_est = example_mgrm_stand_em()
    # 绘制结果（如果有 matplotlib）
    plot_results(true_a, a_est, 'Discrimination', model='MGRM Stand EM')
    plot_results(true_d, d_est, 'Difficulty', model='MGRM Stand EM')
    plot_results(true_theta, theta_est, 'Ability', model='MGRM Stand EM')
    print("\n" + "=" * 50)


    # 运行mcem算法多维mgrm_stand示例
    true_a, true_d, true_theta, a_est, d_est, theta_est = example_mgrm_stand_mcem()
    # 绘制结果（如果有 matplotlib）
    plot_results(true_a, a_est, 'Discrimination', model='MGRM Stand MCEM')
    plot_results(true_d, d_est, 'Difficulty', model='MGRM Stand MCEM')
    plot_results(true_theta, theta_est, 'Ability', model='MGRM Stand MCEM')


    # 运行saem算法多维mgrm_stand示例
    true_a, true_d, true_theta, a_est, d_est, theta_est = example_mgrm_stand_saem()
    # 绘制结果（如果有 matplotlib）
    plot_results(true_a, a_est, 'Discrimination', model='MGRM Stand SAEM')
    plot_results(true_d, d_est, 'Difficulty', model='MGRM Stand SAEM')
    plot_results(true_theta, theta_est, 'Ability', model='MGRM Stand SAEM')


    # 运行mcmc算法多维mgrm_stand示例
    true_a, true_d, true_theta, a_est, d_est, theta_est = example_mgrm_stand_mcmc()
    # 绘制结果（如果有 matplotlib）
    plot_results(true_a, a_est, 'Discrimination', model='MGRM Stand MCMC')
    plot_results(true_d, d_est, 'Difficulty', model='MGRM Stand MCMC')
    plot_results(true_theta, theta_est, 'Ability', model='MGRM Stand MCMC')


    print("\n" + "=" * 50)
    print("示例运行完成！")
    print("=" * 50)

