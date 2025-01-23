import numpy as np
from scipy import linalg
from scipy.special import factorial


def without(Q, index=-1):
    return np.delete(np.delete(Q, index, axis=0), index, axis=1)


def conform_compacted_array(hitting_arr, idx):
    N = np.shape(hitting_arr)[0]
    full_arr = np.zeros(N + 1)
    full_arr[:idx] = hitting_arr[:idx]
    full_arr[idx + 1 :] = hitting_arr[idx:]
    return full_arr


def stitch_hitting_arrays(hitting_arrs):
    return np.vstack(
        [conform_compacted_array(arr, idx) for idx, arr in enumerate(hitting_arrs)]
    ).T


def compute_all_hitting_times(Q, n=1):

    return stitch_hitting_arrays(
        [compute_hitting_time_one_row(Q, index=idx, n=n) for idx in range(Q.shape[0])]
    )


def compute_hitting_time_one_row(Q, index=-1, n=1):

    QA = without(Q, index)
    lambdas, coefs = compute_coefficients_and_decay_rates(QA)
    return blocked_compute(coefs, -lambdas, n)


def blocked_compute(cs, lambdas, n):
    """
    Computes the integral Int_0^∞ [u(t)]^n dt where u(t) = sum(c_i * exp(-lambda_i * t)).

    Parameters:
        c (array-like): Coefficients c_i (supports matrix C, where rows of the matrix are instances of c_i)
        lambdas (array-like): Exponential decay rates lambda_i.
        n (int): Positive integer power of [u(t)].

    Returns:
        float: Value of the integral.
    """
    z, m = cs.shape

    if len(lambdas) != m:
        raise ValueError("c and lambdas must have the same length")

    # Generate all combinations of k_1, k_2, ..., k_m such that sum(k) = n
    # Create a grid of all possible combinations of k values
    indices = np.indices((n + 1,) * m).reshape(m, -1).T
    valid_combinations = indices[np.sum(indices, axis=1) == n]

    # Compute multinomial coefficients
    multinomial_coeffs = factorial(n) / np.prod(factorial(valid_combinations), axis=1)

    # Compute the terms for each combination
    indv_coefs = np.prod(
        cs[:, np.newaxis, :] ** valid_combinations[np.newaxis, :, :], axis=2
    )
    indv_exp_integral = np.sum(lambdas * valid_combinations, axis=1)
    terms = multinomial_coeffs * indv_coefs / indv_exp_integral
    # Sum the terms to compute the integral
    integral = np.sum(terms, axis=1)
    return integral


def compute_coefficients_and_decay_rates(Q):
    """
    Compute the coefficients of the solutions to ODE
    u' = Qu
    where u = sum_i c_i exp(lambda_i t)

    We know that if Q = VDV^-1
    then
    u = V exp(Dt) V^-1 u(0)

    We have u(0) = 1 for the survival function
    """
    Ls, Vs = linalg.eig(Q)
    Us = linalg.inv(Vs)
    cs = Vs * Us.sum(axis=1)[np.newaxis, :]
    return np.real(Ls), cs
