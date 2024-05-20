import numpy as np
from scipy import sparse
import igraph as ig
from typing import Mapping

class Ordering():
    def __init__(self, order_mapping: Mapping):
        """Constructs an ordering of objectes, based on a mapping of values that can be ordered"""
        self.order = np.asanyarray(sorted(order_mapping.keys()))
        self.sizes = np.asanyarray([order_mapping[k] for k in self.order])

    @property
    def hospitals(self):
        return self.order

    def __iter__(self):
        for item in self.order:
            yield item

class TemporalNetworkConverter():
    def __init__(self, network: ig.Graph, ordering: Ordering, weight: str|None=None):
        """Extracts transition matrix from a network"""
        
        times = {k: i for i,k in enumerate(sorted(network.vs['time']))}
        NT = len(times)
        times_by_order = sorted(times.keys())
        DT = times_by_order[1] - times_by_order[0]  # assumes regular time grid
        locs = {k: i for i, k in enumerate(ordering)}
        NLOC = len(locs)


        txn_matrix_data, txn_matrix_i, txn_matrix_j = [], [], []

        # structure is blocks where we have all locs at time 0, then all locs at time 1 etc
        for node in network.vs:
            nd_idx = NLOC * times[node["time"]] + locs[node["loc"]]
            # nd_idx = (locs[node['loc']], times[node['time']])

            for edge in node.out_edges():
                if edge.target == node.index:
                    continue
                else:
                    other = network.vs[edge.target]
                    # target_idx = (locs[other['loc']], times[other['time']])
                    target_idx = NLOC * times[other["time"]] + locs[other["loc"]]
                    out_num = edge[weight] / ordering.sizes[locs[node["loc"]]] / DT
                    txn_matrix_data.append(out_num)
                    txn_matrix_i.append(nd_idx)
                    txn_matrix_j.append(target_idx)

        transition_matrix = sparse.coo_array(
            (txn_matrix_data, (txn_matrix_i, txn_matrix_j)),
            shape=(NLOC * NT, NLOC * NT),
        )

        self.A = transition_matrix.tocsr()
        self.DT = DT
        self.NLOC = NLOC
        self.NT = NT
    
    @property
    def DIMENSIONS(self):
        return {
            'NLOC': self.NLOC,
            'NT': self.NT
        }

    def map_parameters(self, parameters):
        """converts parameters based on the internal transition matrix"""
        pass