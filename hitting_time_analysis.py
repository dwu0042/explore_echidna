import numpy as np
import igraph as ig
from matplotlib import pyplot as plt, colors
from scipy import sparse

import network_conversion as ntc
import hitting_markov as hit


### STANDARD STATIC MODEL

G = ig.Graph.Read("./concordant_networks/shuf_static_network.graphml")

graph_map = {int(k):v for k,v in zip(G.vs['name'], G.vs['id'])}
sorted_graph_map = {k: graph_map[k] for k in sorted(graph_map.keys())}
graph_ordering = ntc.Ordering(sorted_graph_map)
N = len(graph_ordering.sizes)

## we use the naive construction, since the null set (never return) version creates 
## results that misrepresent the amount of time it takes to move through the system
## We also use the internal function in order to directly pass the transition matrix 
## constructed, as opposed to the non-scaled weighted adjacency matrix
T = ntc.transition_matrix_from_graph(G, 
                                     ordering=graph_ordering, 
                                     global_scaling=(10*365), 
                                     ordering_key='name', 
                                     adjacency_attribute='weight'
                                    )
Q = hit._Q_mat_naive(T)

## I wish I could pseed this up, but I think we need the absorption
hitting_times = [
    hit.solve_hitting_time(Q, [i]) for i in range(N)
]

hitting_time_arr = np.hstack([h.reshape((-1, 1)) for h in hitting_times])

plt.matshow(hitting_time_arr, norm=colors.LogNorm())
plt.colorbar(label='hitting time (days)')
plt.xlabel('to')
plt.ylabel('from')
plt.savefig("zstatic_no_home_time.png", dpi=360)

### STATIC MODEL WITH DELAYED RETURN

Z = ig.Graph.Read("./concordant_networks/trimmed_static_base_1_threshold_4.graphml")

zgraph_map = {int(k):v for k,v in zip(Z.vs['node'], Z.vs['id'])}
zsorted_graph_map = {k: zgraph_map[k] for k in sorted(zgraph_map.keys())}
zgraph_ordering = ntc.Ordering(zsorted_graph_map)
zN = len(zgraph_ordering.sizes)

T_direct = ntc.transition_matrix_from_graph(Z, 
                                            ordering=zgraph_ordering, 
                                            global_scaling=(10*365), 
                                            ordering_key='node', 
                                            adjacency_attribute='direct_weight'
                                           )
T_indirect = ntc.transition_matrix_from_graph(Z, 
                                            ordering=zgraph_ordering, 
                                            global_scaling=(10*365), 
                                            ordering_key='node', 
                                            adjacency_attribute='indirect_weight'
                                           )
T_return = ntc.transition_matrix_from_graph(Z, 
                                            ordering=zgraph_ordering, 
                                            ordering_key='node', 
                                            adjacency_attribute='link_time'
                                           )

# build expanded state transition matrix
# this matrix is usually too large to be an explicit array in memory e.g.
# T = np.zeros((zN+zN**2, zN+zN**2))
# so we will construct a sparse matrix
# to prevent degeneracy (rank-deficiency), we will have one row per node and one row per edge

# extract sparse matrix constructor data for direct (node-node) components
T_direct_sparse = sparse.coo_array(T_direct)
T_xs = list(T_direct_sparse.row)
T_ys = list(T_direct_sparse.col)
T_data = list(T_direct_sparse.data)

MACHINE_EPS = 1e-7

# indirect transfers
idx = zN - 1
edge_table = dict()
Tix, Tiy = [mat.flatten() for mat in np.indices(T_indirect.shape)]
for (x,y,v) in zip(Tix, Tiy, T_indirect.flatten()):
    if np.abs(v) > MACHINE_EPS:
        idx += 1
        edge_table[(x, y)] = idx
        T_xs.append(x)
        T_ys.append(idx)
        T_data.append(v)


# indirect return, map link times to number of observations
Trx, Try = [mat.flatten() for mat in np.indices(T_indirect.shape)]
for x,y,v in zip(Trx, Try, T_return.flatten()):
    if np.abs(v) > MACHINE_EPS:
        T_xs.append(edge_table[(x,y)])
        T_ys.append(y)
        T_data.append(1/v)

T = sparse.coo_array((T_data, (T_xs, T_ys)))

R = hit._Q_mat_sparse(T).tocsr()

zhitting_times = [
    hit.solve_hitting_time(R, [i]) for i in range(zN)
]

zhitting_time_arr = np.hstack([h.reshape((-1, 1))[:zN,:] for h in zhitting_times])

plt.matshow(zhitting_time_arr, norm=colors.LogNorm())
plt.colorbar(label='hitting time (days)')
plt.xlabel('to')
plt.ylabel('from')
plt.savefig("z_z_static_no_home_time.png", dpi=360)

zresidence_times = -1/R.diagonal()
plt.figure()
plt.hist(np.log10(zresidence_times), bins=31)
plt.yscale('log')
plt.xlabel('$log_{10}$ residence time (log-days)')
plt.savefig('zstatic_no_home_residence.png', dpi=360)
