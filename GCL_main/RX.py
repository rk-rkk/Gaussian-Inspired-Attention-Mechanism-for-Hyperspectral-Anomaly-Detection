import numpy as np


def RX(X):
    if is_3d_array(X):
        w, h, b = X.shape
        X = np.reshape(X, [w * h, b])
        X = np.transpose(X)

    N, M = X.shape

    X_mean = np.mean(X, axis=1).reshape(-1, 1)

    X = X - np.tile(X_mean, M)

    Sigma = np.dot(X, X.T) / M

    Sigma_inv = np.linalg.inv(Sigma)

    D = np.zeros(M)

    for m in range(M):
        D[m] = np.dot(np.dot(X[:, m].T, Sigma_inv), X[:, m])
    return D


def is_3d_array(data):
    if isinstance(data, np.ndarray) and data.ndim == 3:
        return True
    else:
        return False
