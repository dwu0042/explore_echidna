import numpy as np
from scipy import sparse
import igraph as ig
from typing import Mapping, Hashable, Iterable, Sequence
from util import Iden

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

            for edge in node.out_edges():
                if edge.target == node.index:
                    continue
                else:
                    other = network.vs[edge.target]
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
    
        raise NotImplemented


    @property
    def DIMENSIONS(self):
        return {
            'NLOC': self.NLOC,
            'NT': self.NT
        }

    def map_parameters(self, parameters):
        """converts parameters based on the internal transition matrix"""
        # the required parameters?
        # prob_final_stay -> removal rates
        # beta [transmission] -> remains unchanged
        
        mapped_parameters = dict()

        mapped_parameters['beta'] = parameters['beta']

        movements_out = self.A.sum(axis=1)
        M_mat = movements_out.reshape((self.NLOC, self.NT), order="F")
        p = parameters['prob_final_stay'].reshape((-1, 1))
        # here we use that M_mat = gamma * (1-p)
        mapped_parameters['gamma'] = M_mat / (1 - p)

        mapped_parameters['transition_matrix'] = (self.A / movements_out[:, np.newaxis]).tocsr()

        return mapped_parameters


def transition_matrix_from_graph(
    graph: ig.Graph,
    ordering: Mapping|Sequence = None,
    scaling_per_node: Sequence = None,
    global_scaling: float = 1,
    ordering_key: Hashable = None,
    adjacency_attribute: Hashable = None,
    matrix_size: int = None,
):
    """Given a Graph, generates the associated transition matrix

    Allows for arbitrary ordering given by"""

    if ordering is None:
        ordering = Iden()  # implicit identitiy mapping
    if ordering_key is None:
        graph_order_base = graph.vs.indices
    else:
        graph_order_base = graph.vs[ordering_key]

    graph_paste_order = [ordering[int(k)] for k in graph_order_base]
    A = graph.get_adjacency(attribute=adjacency_attribute)
    if matrix_size is None:
        MAXN = len(graph_paste_order)
    else:
        MAXN = matrix_size
    if scaling_per_node is None:
        scaling_per_node = np.ones((MAXN, 1))
    ordered_adj = np.zeros((MAXN, MAXN))
    # fill the matrix with correclty pivoted A data
    # np.ix_ creates the mesh indices to do the correct index mapping
    ordered_adj[np.ix_(graph_paste_order, graph_paste_order)] = np.array(A.data)
    transition_matrix = (
        ordered_adj / np.reshape(scaling_per_node, ((-1, 1))) / global_scaling
    )
    return transition_matrix


class SnapshotWithHomeConverter():
    def __init__(self, network_directory: ig.Graph, ordering: Ordering, weight: str|None=None):

        self.ordering = ordering


    def parse_transition_matrix_from_graph(
            self,
            graph: ig.Graph,
            duration: float,
            adj_key: str='weight',
    ):
        return transition_matrix_from_graph(
            graph=graph,
            ordering=self.ordering.order,
            scaling_per_node=self.ordering.sizes,
            global_scaling=duration,
            ordering_key="name",
            adjacency_attribute=adj_key,
            matrix_size=max(self.ordering.sizes) + 1
        )