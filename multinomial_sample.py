import numpy as np
from numba import njit
from scipy import sparse


@njit()
def _mn_sparse(input_vector, prob_matrix_data, prob_matrix_indices, prob_matrix_indptr, output_size):
    # Initialize result vector
    result = np.zeros(output_size, dtype=np.int64)  
    
    for i,v in enumerate(input_vector):
        # Skip rows with zero values (micro-optimisation, could also save headache with v=0 edge case not being handled)
        if v == 0:
            continue  

        row_start = prob_matrix_indptr[i]
        row_end = prob_matrix_indptr[i+1]
        probs = prob_matrix_data[row_start:row_end]

        samples = np.random.multinomial(v, probs)
        result[prob_matrix_indices[row_start:row_end]] += samples

    return result

def multinomial_sample_sparse(trials, prob_matrix: sparse.csr_array):
    """Performs multinomial random variate sampling when the probability matrix is 2D and sparse
    
    Arguments
    ---
    trials: <np.ndarray[int]> vector of number of trials, corresponds to rows of prob_matrix
    prob_matrix: <scipy.sparse.csr_array> 2D sparse matrix of probabilities. Normalised so that each row sums to 1.

    Notes
    ---
    - Outcomes are collapsed over the trails. If the shape of the probability matrix is NxQ, the output is a vector of length of Q.
    - This simply wraps a numba jit-ted function that requires the components of the sparse matrix to be split 
    """
    return _mn_sparse(
        trials,
        prob_matrix.data,
        prob_matrix.indices,
        prob_matrix.indptr,
        prob_matrix.shape[1],
    )