import time
import numpy as np

import hitting_time_multichain as hmch
import hitting_markov as hmkv

def tictoc(tic):
    toc = time.perf_counter()
    return toc, toc-tic

sample_graph_Q = np.array([
    [-2, 1, 1],
    [1, -1, 0],
    [2, 1, -3],
], dtype=np.float64)

hittings = []
times = []
for NN in range(1, 10):
    tic = time.perf_counter()
    hittings.append([
        hmch.solve_survival_ode(NN, sample_graph_Q, index=ii)
        for ii in range(sample_graph_Q.shape[0])
    ])
    _, toc = tictoc(tic)
    times.append(toc)

# print(times)
# print(hittings)

hit_arr = np.vstack([np.hstack(x) for x in hittings])

print(hit_arr)

