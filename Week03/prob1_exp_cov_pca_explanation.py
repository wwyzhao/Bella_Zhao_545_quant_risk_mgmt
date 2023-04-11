import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def exp_w_cov(df, lambd):
    n, m = df.shape
    weights = np.zeros(n)
    for i in range(n): # calculate weights array reversely
        weights[i] = (1 - lambd) * lambd ** (i - 1)
    weights = weights / sum(weights)
    weights = np.flip(weights) # reverse the weights array from n-1 to 0

    df = np.array(df)[:, 1:].astype(np.float32)
    df_n = df - np.mean(df, axis=0, keepdims=True)
    df_w = weights.reshape((n,1)) * df_n
    cov = df_w.T @ df_n
    return cov

def pca(df, lambd):
    cov = exp_w_cov(df, lambd)

    # calculate sorted eigen value for a symmetric cov matrix
    eigen_value = np.linalg.eigh(np.array(cov, dtype=float))[0]
    for i in range(len(eigen_value)):
        if eigen_value[i] < 1e-8:
            eigen_value[i] = 0.0
    eigen_value = np.real(eigen_value)

    eigen_value = np.flip(eigen_value)
    ev_sum = sum(eigen_value)
    var_explained = eigen_value / ev_sum
    cum_var_explained = np.cumsum(var_explained)

    return cum_var_explained


if __name__ == '__main__':
    df = pd.read_csv("DailyReturn.csv")

    cum_1 = pca(df, 0.2)
    cum_2 = pca(df, 0.4)
    cum_3 = pca(df, 0.6)
    cum_4 = pca(df, 0.8)
    cum_5 = pca(df, 0.9)
    cum_6 = pca(df, 0.95)
    cum_7 = pca(df, 0.97)
    cum_8 = pca(df, 0.99)

    x = np.linspace(0, 101, 101)
    plt.cla()
    plt.plot(x, cum_1, label="λ = 0.2")
    plt.plot(x, cum_2, label="λ = 0.4")
    plt.plot(x, cum_3, label="λ = 0.6")
    plt.plot(x, cum_4, label="λ = 0.8")
    plt.plot(x, cum_5, label="λ = 0.9")
    plt.plot(x, cum_6, label="λ = 0.95")
    plt.plot(x, cum_7, label="λ = 0.97")
    plt.plot(x, cum_8, label="λ = 0.99")
    plt.legend()
    plt.title("PCA Explained")
    plt.savefig("plots/problem1_pca_explained.png")
    # plt.show()



