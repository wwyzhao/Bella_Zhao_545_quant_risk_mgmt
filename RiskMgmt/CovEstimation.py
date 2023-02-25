# Module for estimating covariance, and exponentially weighted covariance, correlation
import numpy as np
import pandas as pd

# Generate a correlation matrix and variance vector 2 ways
def pearson_cor_var_cov(data):
    """calculate Standard Pearson correlation, covariance and covariance

    Args:
        data (pandas.df): dataframe input matrix (daily prices)
        
    Returns:
        np.array: Standard Pearson correlation, variance and covariance
    """
    cor_P = np.array(data.corr())
    var_P = np.array(data.var())
    m = cor_P.shape[0]
    var_reshape = var_P.reshape((m, 1))
    cov_P = cor_P * np.sqrt(var_reshape) * np.sqrt(var_reshape.T)
    
    print("Standard Pearson correlation")
    print(cor_P)
    print("Standard Pearson variance")
    print(var_P)
    print("Standard Pearson covariance")
    print(cov_P)
    
    return cor_P, var_P, cov_P
    
def exp_w_cor_var_cov(df, lambd):
    """calculate Exponentially Weighted correlation, variance and covariance

    Args:
        data (pandas.df): dataframe input matrix (daily prices)s
        lambd (lambda): exp parameter

    Returns:
        np.array: Exponentially Weighted correlation, variance and covariance
    """
    n, m = df.shape
    weights = np.zeros(n)
    for i in range(n):  # calculate weights array reversely
        weights[i] = (1 - lambd) * lambd ** (i - 1)
    weights = weights / sum(weights)
    weights = np.flip(weights)  # reverse the weights array from n-1 to 0

    df = np.array(df)[:, 1:].astype(np.float32)
    df_n = df - np.mean(df, axis=0, keepdims=True)
    df_w = weights.reshape((n, 1)) * df_n
    cov = df_w.T @ df_n
    var = np.diagonal(cov)
    var_reshape = var.reshape((m - 1, 1))
    cor = cov / np.sqrt(var_reshape) / np.sqrt(var_reshape.T)

    print("Exponentially Weighted correlation")
    print(cor)
    print("Exponentially Weighted variance")
    print(var)
    print("Exponentially Weighted covariance")
    print(cov)
    
    return cor, var, cov