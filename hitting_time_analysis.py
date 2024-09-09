import numpy as np
import igraph as ig
from matplotlib import pyplot as plt, colors

import network_conversion as ntc
import hitting_markov as hit

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
plt.xlabel('from')
plt.ylabel('to')
plt.savefig("zstatic_no_home_time.png")