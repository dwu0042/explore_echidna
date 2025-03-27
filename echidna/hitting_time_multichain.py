"""
This is an implementation of the computation of expected minimal hitting time over N identically parameterised continuous-time Markov chains {M_i(λ,Q)} 
"""

from . import hitting_markov as mkv  # single chain computations
import numpy as np
from scipy import linalg
from scipy import integrate

from typing import Iterable

def without(Q, index=-1):
    return np.delete(np.delete(Q, index, axis=0), index, axis=1)

def construct_surival_ode(Q, index=-1):
    """Implement the problem u'(t) = Q_{A^cA^c} u(t)"""
    # remove row and column
    Qcc = without(Q, index) 

    def backward_equation(t, y):
        return (Qcc @ y[:, np.newaxis]).flatten()

    u_0 = np.ones(Qcc.shape[0])

    return backward_equation, u_0


def determine_laguerre_roots_and_weights(precision=30):
    return np.polynomial.laguerre.laggauss(precision)


def solve_survival_ode(Q, index=-1, locations=None, endpoint=200):

    ode_fun, ode_y0 = construct_surival_ode(Q, index=index)

    if locations is not None:
        tspan = [0, max(locations) * 1.05]
    else:
        tspan = [0, endpoint]

    return integrate.solve_ivp(
        fun=ode_fun,
        t_span=tspan,
        y0=ode_y0,
        t_eval=locations,
    )


def compute_multichain_hitting_time(Nchains, Q, index=-1, precision=30, scaling=1.0):
    roots, weights = determine_laguerre_roots_and_weights(precision)
    gl_weights = weights * np.exp(roots)

    survival_sol = solve_survival_ode(Q, index=index, locations=(roots * scaling))

    return scaling * np.dot(gl_weights, survival_sol.y.T**Nchains)


def conform_compacted_array(hitting_arr, idx):
    N = np.shape(hitting_arr)[0]
    full_arr = np.zeros(N + 1)
    full_arr[:idx] = hitting_arr[:idx]
    full_arr[idx + 1 :] = hitting_arr[idx:]
    return full_arr


def stitch_hitting_arrays(hitting_arrs: Iterable):
    return np.vstack(
        [conform_compacted_array(arr, idx) for idx, arr in enumerate(hitting_arrs)]
    ).T

def compute_subset_multichain_hittings(Nchains, Q, index_subset=slice(None), precision=30, scaling=1.0):
    if isinstance(index_subset, slice):
        index_range = range(*index_subset.indices(Q.shape[0]))
    else:
        index_range = index_subset
    return stitch_hitting_arrays(
        [
            compute_multichain_hitting_time(
                Nchains,
                Q, index=idx,
                precision=precision,
                scaling=scaling,
            )[index_subset]
            for idx in index_range 
        ]
    )

def compute_all_multichain_hittings(Nchains, Q, precision=30, scaling=1.0):
    return stitch_hitting_arrays(
        [
            compute_multichain_hitting_time(
                Nchains, 
                Q, index=idx, 
                precision=precision, 
                scaling=scaling
            )
            for idx in range(Q.shape[0])
        ]
    )

