"""
This is an implementation of the computation of expected minimal hitting time over N identically parameterised continuous-time Markov chains {M_i(λ,Q)} 
"""
import hitting_markov as mkv  # single chain computations
import numpy as np
from scipy import linalg
from scipy import integrate

def construct_surival_ode(Q, index=-1):
    """Implement the problem u'(t) = Q_{A^cA^c} u(t)"""
    # remove row and column
    Qcc = np.delete(np.delete(Q, index, axis=0), index, axis=1)

    def backward_equation(t, y):
        return (Qcc @ y[:,np.newaxis]).flatten()
    
    u_0 = np.ones(Qcc.shape[0])

    return backward_equation, u_0


"""
STRATEGY

-> Set up the integration scheme using Gauss-Laguerre quadrature.
    - requires the setup of Laguerre polynomials to solve for roots.
        - can use np.polynomial.laguerre
-> Use scipy.integrate.solve_ivp to get u(t) at the given roots
-> Use w_i * (u(t) * Exp(t)) to solve the integral and return hitting times.
"""

def determine_laguerre_roots_and_weights(precision=30):
    return np.polynomial.laguerre.laggauss(precision)

def solve_survival_ode(Nchains, Q, index=-1, precision=30):
    roots, weights = determine_laguerre_roots_and_weights(precision)
    gl_weights = weights * np.exp(roots)

    survival_ode_fun, survival_ode_y0 = construct_surival_ode(Q, index=index)

    tspan = [0, max(roots)*1.1]

    survival_sol = integrate.solve_ivp(
        fun=survival_ode_fun,
        t_span=tspan,
        y0=survival_ode_y0,
        t_eval=roots
    )

    return np.dot(gl_weights, survival_sol.y.T ** Nchains)

