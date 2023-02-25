import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt

# Cholesky that assumes PSD matrix
def chol_psd(a):
    m = a.shape[0]
    root = np.zeros([m, m])
    for j in range(m):
        s = 0.0
        if j > 0:
            s = root[j, :j-1].T @ root[j, :j-1]

        temp = a[j, j] - s
        if -1e-8 <= temp <= 0:
            temp = 0.0
        root[j, j] = np.sqrt(temp)

        if abs(root[j, j]) <= 1e-6:
            root[j, j+1:m] = 0.0
        else:
            ir = 1 / (root[j, j]) # avoid denominator 0
            for i in range(j+1, m):
                s = root[i, :j-1].T @ root[j, :j-1]
                root[i, j] = (a[i, j] - s) * ir
    return root

# Near PSD Matrix
def near_psd(a):
    epsilon = 0.0
    m = a.shape[0]
    invSD = np.array([])
    out = a.copy()

    if sum(np.diag(out) == 1) != m:
        invSD = np.diag(1 / np.sqrt(np.diag(out)))
        out = invSD @ out @ invSD

    vals, vecs = np.linalg.eigh(out)
    vals = np.maximum(vals, epsilon)
    T = 1 / ((vecs * vecs) @ vals)
    T = np.diag(np.sqrt(T))
    l = np.diag(np.sqrt(vals))
    B = T @ vecs @ l
    out = B @ B.T

    if invSD.size != 0:
        invSD = np.diag(1 / np.diag(invSD))
        out = invSD @ out @ invSD
    return out

# higham PSD Matrix
def get_norm(A, Y, W):
    W_half = np.sqrt(W)
    A_w = W_half * (A - Y) * W_half
    F_Norm = np.sum(A_w ** 2)
    return F_Norm

def get_PS(R, W):
    W_half = np.sqrt(W)
    R_w = W_half * R * W_half
    vals, vecs = np.linalg.eigh(R_w)
    R_w = (vecs * np.maximum(vals, 0)) @ vecs.T
    PS = 1 / W_half * R_w * 1 / W_half
    return PS

def get_PU(X):
    # assuming W is diagonal
    PU = X
    np.fill_diagonal(PU, 1)
    return PU

def higham_psd(a):
    max_iter = 1000
    tol = 1e-8

    m = a.shape[0]
    W = np.ones(m) # identical matrix
    delta_s = np.zeros(a.shape)
    Y = a.copy()
    gama = np.inf
    gama_pre = 0 # help to remember gama of the previous iteration

    iter = 0
    while abs(gama - gama_pre) > tol:
        if iter > max_iter:
            print(f"No higham PSD found in {max_iter} iterations")
            return a
        if np.linalg.eigh(Y)[0][0] > -1e-8:
            print(f"The smallest eigenvalue is non-negative, given matrix is PSD after {iter} iterations")
            return Y
        gama_pre = gama

        R = Y - delta_s
        X = get_PS(R, W)
        delta_s = X - R
        Y = get_PU(X)
        gama = get_norm(a, Y, W)
        iter += 1
    return Y

def get_nonpsd(n):
    sigma = np.zeros([n, n]) + 0.9
    for i in range(n):
        sigma[i, i] = 1.0
    sigma[0, 1] = 0.7357
    sigma[1, 0] = 0.7357
    return sigma

def compare_near_higham(n):
    sigma = get_nonpsd(n)
    s1 = time.time()
    psd_near = near_psd(sigma)
    e1 = time.time()
    t_near = e1 - s1
    f_norm_near = np.sum((sigma - psd_near) ** 2)

    s2 = time.time()
    psd_higham = higham_psd(sigma)
    e2 = time.time()
    t_higham = e2 - s2
    f_norm_higham = np.sum((sigma - psd_higham) ** 2)

    print(f"n = {n}, \n Runtime: near_psd = {t_near:.8f} seconds, higham = {t_higham:.8f} seconds. \n Frobenius Norm: near_psd = {f_norm_near}, higham = {f_norm_higham}")
    return [t_near, f_norm_near, t_higham, f_norm_higham]

if __name__ == '__main__':

    sigma = get_nonpsd(5)

    # cholesky algorithm test using a slightly non-psd correlation matrix that is 5x5
    print("Test Cholesky Algorithm using a slightly non-psd correlation matrix that is 5x5")
    print("This is the given matrix A")
    print(sigma)
    s1 = time.time()
    L = chol_psd(sigma)
    e1 = time.time()
    t1 = e1 - s1
    print(f"This is Cholesky root L, runtime: {t1:.8f} seconds.")
    print(L)
    a = L @ L.T
    print("Test LL', equals matrix A")
    print(a)
    print()

    # near psd test using a non-psd correlation matrix that is 5x5
    print("Test Near PSD Algorithm using a non-psd correlation matrix that is 5x5")
    print("This is the given matrix C")
    print(sigma)
    s2 = time.time()
    res_near = near_psd(sigma)
    e2 = time.time()
    t2 = e2 - s2
    print(f"This is the near_psd result, runtime: {t2:.8f} seconds.")
    print(res_near)
    if np.linalg.eigh(sigma)[0][0] < -1e-8:
        print("The given matrix is not psd.")
    else:
        print("The given matrix is psd.")
    if np.linalg.eigh(res_near)[0][0] < -1e-8:
        print("The near_psd matrix is not psd.")
    else:
        print("The near_psd matrix is psd.")
    print()

    # higham psd test using a non-psd correlation matrix that is 5x5
    sigma = get_nonpsd(5)
    print("Test Higham PSD Algorithm using a non-psd correlation matrix that is 5x5")
    print("This is the given matrix C")
    print(sigma)
    s3 = time.time()
    res_higham = higham_psd(sigma)
    e3 = time.time()
    t3 = e3 - s3
    print(f"This is the higham_psd result, runtime: {t3:.8f} seconds.")
    print(res_higham)
    if np.linalg.eigh(sigma)[0][0] < -1e-8:
        print("The given matrix is not psd.")
    else:
        print("The given matrix is psd.")
    if np.linalg.eigh(res_higham)[0][0] < -1e-6:
        print("The higham_psd matrix is not psd.")
    else:
        print("The higham_psd matrix is psd.")
    print()

    n_list = [50, 100, 200, 500, 800, 1000]
    t_near = []
    f_norm_near = []
    t_higham = []
    f_norm_higham = []
    for n in n_list:
        res = compare_near_higham(n)
        t_near.append(res[0])
        f_norm_near.append(res[1])
        t_higham.append(res[2])
        f_norm_higham.append(res[3])

    plt.cla()
    plt.plot(n_list, t_near, label="near_psd")
    plt.plot(n_list, t_higham, label="higham_psd")
    plt.legend(loc=1)
    plt.title("Runtime comparison of near_psd and higham_psd")
    plt.savefig("plots/problem2_runtime_near_higham.png")

    plt.cla()
    plt.plot(n_list, f_norm_near, label="near_psd")
    plt.plot(n_list, f_norm_higham, label="higham_psd")
    plt.legend(loc=1)
    plt.title("Frobenius Norm comparison of near_psd and higham_psd")
    plt.savefig("plots/problem2_Frobenius_Norm_near_higham.png")


