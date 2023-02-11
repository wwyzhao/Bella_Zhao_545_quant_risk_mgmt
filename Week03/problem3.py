import pandas as pd
import numpy as np
import time
from numpy.linalg import norm
import matplotlib.pyplot as plt

def exp_w_cor_var(df, lambd):
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

    return cor, var

def cal_cov(cor, var):
    m = cor.shape[0]
    var_reshape = var.reshape((m, 1))
    return cor * np.sqrt(var_reshape) * np.sqrt(var_reshape.T)

def simulate_direct(cov, nsim):
    m = cov.shape[0]
    z = np.random.multivariate_normal(np.zeros(m), cov, nsim).T
    return z

def simulate_pca(cov, nsim, explain):
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

if __name__ == '__main__':

    df = pd.read_csv("DailyReturn.csv")

    # Generate a correlation matrix and variance vector 2 ways
    cor_P = np.array(df.corr())
    var_P = np.array(df.var())
    cor_w, var_w = exp_w_cor_var(df, 0.97)
    print("Generate a correlation matrix and variance vector using Standard Pearson")
    print(cor_P)
    print(var_P)
    print("Generate a correlation matrix and variance vector using Exponentially weighted λ = 0.97")
    print(cor_w)
    print(var_w)

    # Combine these to form 4 different covariance matrices
    cov_P_P = cal_cov(cor_P, var_P)
    cov_P_EW = cal_cov(cor_P, var_w)
    cov_EW_P = cal_cov(cor_w, var_P)
    cov_EW_EW = cal_cov(cor_w, var_w)

    # Pearson correlation + variance
    f_norm_1 = []
    t_1 = []
    start = time.time()
    P_P_d = simulate_direct(cov_P_P, 25000)
    end = time.time()
    d = np.cov(P_P_d) - cov_P_P
    f_norm = norm(d, 'fro')
    f_norm_1.append(f_norm)
    t_1.append(end - start)
    print("Pearson correlation + variance, Direct Simulation")
    print("Frobenius Norm: ", f_norm, "Runtime: ", f"{end - start:8f}")
    start = time.time()
    P_P_100 = simulate_pca(cov_P_P, 25000, 0.9999)
    end = time.time()
    d = np.cov(P_P_100) - cov_P_P
    f_norm = norm(d, 'fro')
    f_norm_1.append(f_norm)
    t_1.append(end - start)
    print("Pearson correlation + variance, PCA 100% explained")
    print("Frobenius Norm: ", f_norm, "Runtime: ", f"{end - start:8f}")
    start = time.time()
    P_P_75 = simulate_pca(cov_P_P, 25000, 0.75)
    end = time.time()
    d = np.cov(P_P_75) - cov_P_P
    f_norm = norm(d, 'fro')
    f_norm_1.append(f_norm)
    t_1.append(end - start)
    print("Pearson correlation + variance, PCA 75% explained")
    print("Frobenius Norm: ", f_norm, "Runtime: ", f"{end - start:8f}")
    start = time.time()
    P_P_50 = simulate_pca(cov_P_P, 25000, 0.5)
    end = time.time()
    d = np.cov(P_P_50) - cov_P_P
    f_norm = norm(d, 'fro')
    f_norm_1.append(f_norm)
    t_1.append(end - start)
    print("Pearson correlation + variance, PCA 50% explained")
    print("Frobenius Norm: ", f_norm, "Runtime: ", f"{end - start:8f}")
    print()
    x = ["Direct", "PCA 100%", "PCA 75%", "PCA 50%"]
    plt.cla()
    fig, ax1 = plt.subplots()
    ax1.plot(x, f_norm_1, label="Frobenius Norm", color="orange")
    plt.legend(loc=1)
    ax2 = ax1.twinx()
    ax2.plot(x, t_1, label="Runtime")
    plt.legend(loc=2)
    plt.title("Pearson correlation + variance")
    fig.tight_layout()
    plt.savefig("plots/problem3_Pearson_Pearson.png")


    # Pearson correlation + EW variance
    f_norm_2 = []
    t_2 = []
    start = time.time()
    P_EW_d = simulate_direct(cov_P_EW, 25000)
    end = time.time()
    d = np.cov(P_EW_d) - cov_P_EW
    f_norm = norm(d, 'fro')
    f_norm_2.append(f_norm)
    t_2.append(end - start)
    print("Pearson correlation + EW variance, Direct Simulation")
    print("Frobenius Norm: ", f_norm, "Runtime: ", f"{end - start:8f}")
    start = time.time()
    P_EW_100 = simulate_pca(cov_P_EW, 25000, 0.9999)
    end = time.time()
    d = np.cov(P_EW_100) - cov_P_EW
    f_norm = norm(d, 'fro')
    f_norm_2.append(f_norm)
    t_2.append(end - start)
    print("Pearson correlation + EW variance, PCA 100% explained")
    print("Frobenius Norm: ", f_norm, "Runtime: ", f"{end - start:8f}")
    start = time.time()
    P_EW_75 = simulate_pca(cov_P_EW, 25000, 0.75)
    end = time.time()
    d = np.cov(P_EW_75) - cov_P_EW
    f_norm = norm(d, 'fro')
    f_norm_2.append(f_norm)
    t_2.append(end - start)
    print("Pearson correlation + EW variance, PCA 75% explained")
    print("Frobenius Norm: ", f_norm, "Runtime: ", f"{end - start:8f}")
    start = time.time()
    P_EW_50 = simulate_pca(cov_P_EW, 25000, 0.5)
    end = time.time()
    d = np.cov(P_EW_50) - cov_P_EW
    f_norm = norm(d, 'fro')
    f_norm_2.append(f_norm)
    t_2.append(end - start)
    print("Pearson correlation + EW variance, PCA 50% explained")
    print("Frobenius Norm: ", f_norm, "Runtime: ", f"{end - start:8f}")
    print()
    plt.cla()
    fig, ax1 = plt.subplots()
    ax1.plot(x, f_norm_2, label="Frobenius Norm", color="orange")
    plt.legend(loc=1)
    ax2 = ax1.twinx()
    ax2.plot(x, t_2, label="Runtime")
    plt.legend(loc=2)
    plt.title("Pearson correlation + EW variance")
    fig.tight_layout()
    plt.savefig("plots/problem3_Pearson_EW.png")


    # EW correlation + variance
    f_norm_3 = []
    t_3 = []
    start = time.time()
    EW_P_d = simulate_direct(cov_EW_P, 25000)
    end = time.time()
    d = np.cov(EW_P_d) - cov_EW_P
    f_norm = norm(d, 'fro')
    f_norm_3.append(f_norm)
    t_3.append(end - start)
    print("EW correlation + variance, Direct Simulation")
    print("Frobenius Norm: ", f_norm, "Runtime: ", f"{end - start:8f}")
    start = time.time()
    EW_P_100 = simulate_pca(cov_EW_P, 25000, 0.9999)
    end = time.time()
    d = np.cov(EW_P_100) - cov_EW_P
    f_norm = norm(d, 'fro')
    f_norm_3.append(f_norm)
    t_3.append(end - start)
    print("EW correlation + variance, PCA 100% explained")
    print("Frobenius Norm: ", f_norm, "Runtime: ", f"{end - start:8f}")
    start = time.time()
    EW_P_75 = simulate_pca(cov_EW_P, 25000, 0.75)
    end = time.time()
    d = np.cov(EW_P_75) - cov_EW_P
    f_norm = norm(d, 'fro')
    f_norm_3.append(f_norm)
    t_3.append(end - start)
    print("EW correlation + variance, PCA 75% explained")
    print("Frobenius Norm: ", f_norm, "Runtime: ", f"{end - start:8f}")
    start = time.time()
    EW_P_50 = simulate_pca(cov_EW_P, 25000, 0.5)
    end = time.time()
    d = np.cov(EW_P_50) - cov_EW_P
    f_norm = norm(d, 'fro')
    f_norm_3.append(f_norm)
    t_3.append(end - start)
    print("EW correlation + variance, PCA 50% explained")
    print("Frobenius Norm: ", f_norm, "Runtime: ", f"{end - start:8f}")
    print()
    plt.cla()
    fig, ax1 = plt.subplots()
    ax1.plot(x, f_norm_3, label="Frobenius Norm", color="orange")
    plt.legend(loc=1)
    ax2 = ax1.twinx()
    ax2.plot(x, t_3, label="Runtime")
    plt.legend(loc=2)
    plt.title("EW correlation + Pearson variance")
    fig.tight_layout()
    plt.savefig("plots/problem3_EW_Pearson.png")

    # Pearson correlation + variance
    f_norm_4 = []
    t_4 = []
    start = time.time()
    EW_EW_d = simulate_direct(cov_EW_EW, 25000)
    end = time.time()
    d = np.cov(EW_EW_d) - cov_EW_EW
    f_norm = norm(d, 'fro')
    f_norm_4.append(f_norm)
    t_4.append(end - start)
    print("EW correlation + EW variance, Direct Simulation")
    print("Frobenius Norm: ", f_norm, "Runtime: ", f"{end - start:8f}")
    start = time.time()
    EW_EW_100 = simulate_pca(cov_EW_EW, 25000, 0.9999)
    end = time.time()
    d = np.cov(EW_EW_100) - cov_EW_EW
    f_norm = norm(d, 'fro')
    f_norm_4.append(f_norm)
    t_4.append(end - start)
    print("EW correlation + EW variance, PCA 100% explained")
    print("Frobenius Norm: ", f_norm, "Runtime: ", f"{end - start:8f}")
    start = time.time()
    EW_EW_75 = simulate_pca(cov_EW_EW, 25000, 0.75)
    end = time.time()
    d = np.cov(EW_EW_75) - cov_EW_EW
    f_norm = norm(d, 'fro')
    f_norm_4.append(f_norm)
    t_4.append(end - start)
    print("EW correlation + EW variance, PCA 75% explained")
    print("Frobenius Norm: ", f_norm, "Runtime: ", f"{end - start:8f}")
    start = time.time()
    EW_EW_50 = simulate_pca(cov_EW_EW, 25000, 0.5)
    end = time.time()
    d = np.cov(EW_EW_50) - cov_EW_EW
    f_norm = norm(d, 'fro')
    f_norm_4.append(f_norm)
    t_4.append(end - start)
    print("EW correlation + EW variance, PCA 50% explained")
    print("Frobenius Norm: ", f_norm, "Runtime: ", f"{end - start:8f}")
    print()
    plt.cla()
    fig, ax1 = plt.subplots()
    ax1.plot(x, f_norm_4, label="Frobenius Norm", color="orange")
    plt.legend(loc=1)
    ax2 = ax1.twinx()
    ax2.plot(x, t_4, label="Runtime")
    plt.legend(loc=2)
    plt.title("EW correlation + Pearson variance")
    fig.tight_layout()
    plt.savefig("plots/problem3_EW_EW.png")






