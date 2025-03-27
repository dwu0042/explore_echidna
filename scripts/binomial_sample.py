import numpy as np
from numba import njit, prange


@njit(parallel=True)
def bino_1(N, P):
    # Initialize an empty matrix to store the samples
    X = np.empty_like(N)

    # Iterate over each element of N and P in parallel
    for i in prange(N.shape[0]):
        for j in prange(N.shape[1]):
            X[i, j] = np.random.binomial(N[i, j], P[i, j])

    return X

@njit(parallel=True)
def _bino_2(N, P):
    X = np.empty_like(N)

    # here, assume N, P flat
    for i in prange(N.shape[0]):
        if N[i] == 0:
            continue
        X[i] = np.random.binomial(N[i], P[i])
    
    return X

def bino_2(N, P):
    return np.reshape(_bino_2(N.flatten(), P.flatten()), N.shape)


@njit(parallel=True)
def bino_3(N, P):
    # Initialize an empty matrix to store the samples
    X = np.empty_like(N)

    # Iterate over each element of N and P in parallel
    for i in prange(N.shape[0]):
        for j in prange(N.shape[1]):
            if N[i, j] == 0 or P[i,j] == 0:
                continue
            X[i, j] = np.random.binomial(N[i, j], P[i, j])

    return X