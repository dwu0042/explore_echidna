"""Hitting time and probability computations, as presented in Norris. (1997) Markov chains"""

from typing import Iterable
import numpy as np
from numpy.typing import NDArray
import igraph as ig
from scipy import optimize, sparse


def Q_mat_naive(G: ig.Graph):
    """Extracts the generator matrix of a movement graph"""

    A = np.array(G.get_adjacency().data)
    return _Q_mat_naive(A)

def _Q_mat_naive(A: np.ndarray):
    rowsums = A.sum(axis=1)
    Q = A - np.diag(rowsums)

    return Q

def Q_mat(G: ig.Graph, p_null: np.ndarray):
    """Extracts the generator matrix for a transfer graph with probability of transfer"""

    Z = np.array(G.get_adjacency().data)
    return _Q_mat(Z, p_null)

def _Q_mat(Z: np.ndarray, p_null: np.ndarray):
    rowsums = Z.sum(axis=1)
    true_rowsums = rowsums / (1 - p_null)

    # model by adding nulls as zero-th row
    A = np.zeros([n+1 for n in Z.shape])
    A[1:,1:] = Z
    A[1:,0] = p_null * true_rowsums

    Q = A - np.diag(np.concatenate([[0], true_rowsums]))

    return Q

def _Q_mat_sparse(A: sparse.sparray):

    rowsums = A.sum(axis=1)
    Q = A.copy()
    Q.setdiag(-rowsums)

    return Q

def min_nonnegative_soln(A: np.ndarray, b: np.ndarray, full=False):
    """Generic wrapper for solving an ill-conditioned Ax=b linear equation for minimal non-neg x
    
    We wrap the use of scipy.optimize.linprog to solve the linear program:

    min 1.x
    s/t 
        Ax = b
         x > 0
    """

    # check shape constraints
    M, N = A.shape
    k, = b.shape
    assert k == N

    # cost vector
    c = np.ones(N)
    # induced linear constraint
    A_eq = A
    b_eq = b
    # non-negativtity constraint
    bounds = (0, None)

    soln = optimize.linprog(
        c=c,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=bounds,
    )

    if not soln.success or full:
        return soln

    return soln.x

def build_hitting_time_problem(Q: NDArray, target_set: Iterable):

    A = -Q.copy()
    b = np.ones(Q.shape[0])
    
    for i in target_set:
        A[i, :] = 0
        A[i, i] = 1
        b[i] = 0

    return A, b


def solve_hitting_time(Q: NDArray, target_set: Iterable):

    A, b = build_hitting_time_problem(Q, target_set)
    return min_nonnegative_soln(A, b)

def build_hitting_prob_problem(Q: np.ndarray, target_set: Iterable):

    A = Q.copy()
    b = np.zeros(Q.shape[0])

    for i in target_set:
        A[i, :] = 0
        A[i, i] = 1
        b[i] = 1

    return A, b

def solve_hitting_prob(Q: np.ndarray, target_set: Iterable):

    A, b = build_hitting_prob_problem(Q, target_set)
    return min_nonnegative_soln(A, b)