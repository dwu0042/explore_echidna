import numpy as np
import numba as nb
from scipy import sparse
import igraph as ig
from typing import Mapping, Hashable, Sequence
from functools import lru_cache
from pathlib import Path
from util import Iden, nparr_find, NotFound
import examine_transfers_for_sizes as esz

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

    @lru_cache()
    def __getitem__(self, key):
        """A reverse search of the order list"""
        value = nparr_find(self.order, int(key))
        if value is None:
            raise NotFound
        # value is a 1D tuple; we should return the interior item
        return value[0]

    @classmethod
    def from_file(cls, file_name):
        size_mapping = esz.quick_read(file_name)
        return cls(size_mapping)


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
        mapped_parameters['prob_final_stay'] = parameters['prob_final_stay']

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

    ADJTYPES = ("direct", "out", "in")
    ADJMAP = {
        "direct": "weight",
        "out": "departures",
        "in": "arrivals",
    }

    def __init__(self, 
                 network_snaphots: Mapping[int, ig.Graph], 
                 ordering: Ordering, 
                 trunc: int=26):

        self.ordering = ordering

        self.snapshot_times = sorted(network_snaphots.keys())
        self.snapshot_durations = np.diff(self.snapshot_times)

        # construct raw transition matrices that are to be parsed into weighting matrices
        self.raw_transition_matrices = {
            adjtype: [
                self.parse_transition_matrix_from_graph(
                    network_snaphots[snapshot_time], 
                    duration=snapshot_duration, 
                    adj_key=self.ADJMAP[adjtype]
                )
                for snapshot_time, snapshot_duration in zip(
                    self.snapshot_times, self.snapshot_durations
                )
            ]
            for adjtype in self.ADJTYPES
        }

        self.outwards_weighting_matrices = None
        self.inwards_weighting_matrices = None
        self.outwards_flow_weight = None
        self.compute_weighting_matrices(trunc=trunc)

    @classmethod
    def from_directory(cls, directory: str|Path, *args, **kwargs):

        snapshots = cls.load_snapshots(directory)
        return cls(snapshots, *args, **kwargs)

    @staticmethod
    def load_snapshots(rootpath: str|Path) -> Mapping[int, ig.Graph]:
        snapshots = dict()
        root = Path(rootpath)
        for graphpath in root.glob(f"*.graphml"):
            # safe for windows?
            name = int(graphpath.stem)
            with open(graphpath, "r") as graph_file:
                snapshots[name] = ig.Graph.Read_GraphML(graph_file)
        return snapshots

    @staticmethod
    @nb.njit()
    def power_kernel(steps, power=-0.6, trunc=26):
        steps = max(steps, 1) # bound below by 1
        steps = min(steps, trunc) # bound above by trunc
        t = np.arange(steps, 0, -1, dtype=nb.float64)
        attributions = t**power
        normalising_constant = np.sum(attributions)
        return attributions / normalising_constant

    def parse_transition_matrix_from_graph(
        self,
        graph: ig.Graph,
        duration: float,
        adj_key: str='weight',
    ):
        return transition_matrix_from_graph(
            graph=graph,
            ordering=self.ordering,
            scaling_per_node=self.ordering.sizes,
            global_scaling=duration,
            ordering_key="name",
            adjacency_attribute=adj_key,
            matrix_size=len(self.ordering.order)
        )

    def compute_weighting_matrices(self, trunc=26):

        out_weights, out_sums = self.compute_outwards_weighting_matrices_and_sum()
        in_weights = self.compute_inwards_weighting_matrices(trunc=trunc)

        self.outwards_weighting_matrices = out_weights
        self.inwards_weighting_matrices = in_weights
        self.outwards_flow_weight = out_sums

    def compute_inwards_weighting_matrices(self, trunc=26):
        inwards_weighting_matrices = []
        for i, E in enumerate(self.raw_transition_matrices["in"]):
            if i == 0:
                # special case at t=0
                inwards_weighting_matrices.append(E)  # should be all zeros
                continue
            powker = self.power_kernel(i, trunc=trunc)
            i_start = np.clip(i - trunc, 0, i)
            U_out = np.sum(
                self.raw_transition_matrices["out"][i_start:i]
                * powker[:, np.newaxis, np.newaxis],
                axis=0,
            )
            E_idx = np.where(E)
            E_transformed = np.zeros_like(E)
            E_transformed[E_idx] = E[E_idx] / U_out[E_idx]
            inwards_weighting_matrices.append(E_transformed)

        return inwards_weighting_matrices
    
    def compute_outwards_weighting_matrices_and_sum(self):
        outwards_weighting_matrices = []
        outwards_weighting_sum = []
        for D, U in zip(
            self.raw_transition_matrices["direct"], self.raw_transition_matrices["out"]
        ):
            Sig = (D + U).sum(axis=1).reshape((-1, 1))
            outwards_weighting_sum.append(Sig)
            Dp = D / Sig
            Up = U / Sig
            dense_out_matrix = np.hstack([np.nan_to_num(Dp), np.nan_to_num(Up)])
            sparse_out_matrix = sparse.csr_array(dense_out_matrix)
            sparse_out_matrix.eliminate_zeros()
            outwards_weighting_matrices.append(sparse_out_matrix)

        return outwards_weighting_matrices, outwards_weighting_sum

    def map_parameters(self, parameters):
        mapped_parameters = dict()

        mapped_parameters['beta'] = parameters['beta']

        p = np.asanyarray(parameters['prob_final_stay']).reshape((-1, 1))
        mapped_parameters['prob_final_stay'] = p

        mapped_parameters['gamma'] = [S / (1 - p) for S in self.outwards_flow_weight]

        mapped_parameters['transition_matrix_out'] = self.outwards_weighting_matrices
        mapped_parameters['transition_matrix_in'] = self.inwards_weighting_matrices

        return mapped_parameters
