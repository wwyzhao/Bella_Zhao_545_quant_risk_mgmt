# module of simulations
import numpy as np
import pandas as pd
import scipy.stats as st

# Cholesky Factorization for doing direct simulation
def chol_psd(a):
    """Cholesky Factorization for getting Cholesky root

    Args:
        a (np.array): PSD matrix

    Returns:
        np.array: Cholesky root
    """
    n = a.shape[0]
    root = np.zeros([n, n])
    for j in range(n):
        s = 0.0
        if j > 0:
            s = root[j, :j].T @ root[j, :j]

        temp = a[j, j] - s
        if -1e-8 <= temp <= 0:
            temp = 0.0
        root[j, j] = np.sqrt(temp)

        if abs(root[j, j]) <= 1e-6:
            root[j, j+1:] = 0.0
        else:
            ir = 1 / (root[j, j]) # avoid denominator 0
            for i in range(j+1, n):
                s = root[i, :j].T @ root[j, :j]
                root[i, j] = (a[i, j] - s) * ir
    return root


def simulate_direct(cov, nsim):
    """Direct matrix simulation

    Args:
        cov (np.array): covariance needed to be simulated
        nsim (int): times of simulation

    Returns:
        np.array: simulation matrix
    """
    m = cov.shape[0]
    root = chol_psd(cov)
    z = np.random.normal(size=(m, nsim)) 
    z = (root @ z)
    # z = np.random.multivariate_normal(np.zeros(m), cov, nsim).T
    return z

def simulate_pca(cov, nsim, explain):
    """PCA matrxi simulation

    Args:
        cov (np.array): covariance needed to be simulated
        nsim (int): times of simulation
        explain (float): percentage of PCA explanation

    Returns:
        np.array: simulation matrix
    """
    vals, vecs = np.linalg.eigh(cov)
    m = len(vals)
    for i in range(m):
        if vals[i] < 1e-8:
            vals[i] = 0.0
    vals = np.real(vals)

    vals = np.flip(vals)
    vecs = np.fliplr(vecs)
    ev_sum = sum(vals)
    var_explained = vals / ev_sum
    cum_var_explained = np.cumsum(var_explained)

    # get the index of the eigenvalue just reach the explained percentage
    n_cum = np.where(cum_var_explained >= explain)[0][0]
    eigen_value = vals[:n_cum+1]
    eigen_vector = vecs[:, :n_cum+1]

    B = eigen_vector @ np.diag(np.sqrt(eigen_value))
    m = eigen_value.shape[0]
    r = np.random.normal(size=(m, nsim))
    s = B @ r
    
    return s

def copula_t(rt, nsim = 1000):
    #remove the mean
    n = rt.shape[1]
    rt = (rt - rt.mean()).astype(np.float64)

    # calculate each T parameters, CDF for each column (Fit T Models to the returns)
    paras = [st.t.fit(rt[col]) for col in rt.columns]
    print("paras of fitted T distribution")
    print(paras)
    cdf = st.t.cdf(rt, *zip(*paras))
    cdf = pd.DataFrame(data=cdf, index=rt.index, columns=rt.columns)
    
    # convert from multivariate normal to simulations of a T distribution (Gaussian Copula)
    corr = cdf.corr(method = "spearman")
    sim_t = pd.DataFrame(st.norm.cdf(simulate_pca(corr, nsim, 1)))
    sim_df = [st.t.ppf(sim_t[col], *zip(*paras)) for col in sim_t.columns]
    sim_df = pd.DataFrame(data=sim_df, columns=rt.columns)

    return sim_df