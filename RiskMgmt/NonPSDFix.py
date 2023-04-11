# module for converting Non PSD covariance matrix to PSD matrix, and decide whether a matrix is PSD
import numpy as np

def near_psd(a):
    """Near PSD method for converting Non PSD covariance matrix to PSD matrix

    Args:
        a (np.array): Non PSD Matrix

    Returns:
        np.array: PSD Matrix
    """
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
    """Higham PSD method for converting Non PSD covariance matrix to PSD matrix

    Args:
        a (np.array): Non PSD covariance matrix

    Returns:
        np.array: PSD matrix
    """
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
    """get an nxn Non PSD matrix

    Args:
        n (int): dimension of Non PSD matrix

    Returns:
        np.array: an nxn Non PSD matrix
    """
    sigma = np.zeros([n, n]) + 0.9
    for i in range(n):
        sigma[i, i] = 1.0
    sigma[0, 1] = 0.7357
    sigma[1, 0] = 0.7357
    return sigma

def get_psd(n):
    """get an nxn PSD matrix

    Args:
        n (int): dimension of PSD matrix

    Returns:
        np.array: an nxn PSD matrix
    """
    sigma = np.zeros([n, n]) + 1
    for i in range(n):
        sigma[i, i] = 0.9
    return sigma

def is_PSD(a):
    """Decide whether a covariance matrix is PSD, which cannot be used by Cholesky method

    Args:
        a (np.array): matrix

    Returns:
        bool: PSD return True; otherwise return False
    """
    if np.linalg.eigh(a)[0][0] >= -1e-6:
        return True
    else:
        return False