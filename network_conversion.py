import abc
from os import PathLike
import random
import numpy as np
from numpy import typing as npt
import numba as nb
from scipy import sparse
import polars as pl
import igraph as ig
from typing import Mapping, Hashable, Sequence, Any, SupportsFloat
from functools import lru_cache
from pathlib import Path

import graph_importer as gim
from util import Iden, nparr_find, NotFound, SupportsGet
import examine_transfers_for_sizes as esz


class Ordering:
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

    def __len__(self):
        return len(self.order)

    @lru_cache()
    def __getitem__(self, key):
        """A reverse search of the order list"""
        value = np.nonzero(self.order == key)[0]
        # check for not found
        if len(value) == 0:
            raise NotFound(key)
        return value.item()

    @classmethod
    def from_file(cls, file_name: PathLike):
        size_mapping = esz.quick_read(file_name)
        return cls(size_mapping)

    def conform(self, other: Mapping):
        return [other.get(x) for x in self]

    def __call__(self, key):
        """returns the size of the object"""
        idx = self[key]
        return self.sizes[idx]
    
    def todf(self):
        """Return the dataframe that corresponds to the ordering
        Has columns 'hospital', 'index', and 'size'
        """
        return pl.from_dict({
            'hospital': self.order,
            'index': np.arange(len(self.order)),
            'size': self.sizes
        })


class ColumnDict(dict):
    @classmethod
    def from_file(cls, file: PathLike, key_name, value_name):
        df = pl.read_csv(file)
        return cls(
            zip(
                df.select(key_name).to_series().to_list(),
                df.select(value_name).to_series().to_list(),
            )
        )

    @classmethod
    def from_prob_final_file(cls, file):
        return cls.from_file(file, "loc", "final_stay")

    def organise_by(self, ordering: Ordering) -> npt.NDArray:
        output = ordering.conform(self)
        return np.array(output)


def transition_matrix_from_graph(
    graph: ig.Graph,
    ordering: SupportsGet | None = None,
    scaling_per_node: Sequence | None = None,
    global_scaling: float = 1,
    ordering_key: Hashable | None = None,
    adjacency_attribute: Hashable = None,
    matrix_size: int | None = None,
):
    """Given a Graph, generates the associated transition matrix

    Allows for arbitrary ordering given by"""

    if ordering is None:
        ordering = Iden()  # implicit identity mapping
    if ordering_key is None:
        graph_order_base = graph.vs.indices
    else:
        graph_order_base = graph.vs[ordering_key]

    graph_paste_order = [ordering[k] for k in graph_order_base]
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


class Converter(abc.ABC):
    @abc.abstractmethod
    def __init__(self, network: ig.Graph, ordering: Ordering):
        pass

    @abc.abstractmethod
    def map_parameters(self, parameters: Mapping) -> Mapping:
        pass


class TemporalNetworkConverter(Converter):
    def __init__(
        self, network: ig.Graph, ordering: Ordering, weight: str='weight',
    ):
        """Extracts transition matrix from a network"""
        self.ordering = ordering

        times = {k: i for i, k in enumerate(sorted(set(network.vs["time"])))}
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

    @property
    def DIMENSIONS(self):
        return {"NLOC": self.NLOC, "NT": self.NT}
    
    @classmethod
    def from_file(cls, network_filepath: str | PathLike, **kwargs):
        graph = gim.make_graph(network_filepath)
        return cls(network=graph, **kwargs)

    def map_parameters(self, parameters):
        """converts parameters based on the internal transition matrix"""
        # the required parameters?
        # prob_final_stay -> removal rates
        # beta [transmission] -> remains unchanged

        mapped_parameters = dict()

        mapped_parameters["beta"] = parameters["beta"]
        mapped_parameters["prob_final_stay"] = parameters["prob_final_stay"].reshape((-1, 1))

        movements_out = self.A.sum(axis=1)
        M_mat = movements_out.reshape((self.NLOC, self.NT), order="F")
        # here we use that M_mat = gamma * (1-p)
        mapped_parameters["gamma"] = M_mat / (1 - mapped_parameters["prob_final_stay"])

        mapped_parameters["transition_matrix"] = (
            self.A / movements_out[:, np.newaxis]
        ).tocsr()

        return mapped_parameters
    
    def delay(self, n: int):
        """Reformats internal attributes so that the starting time is delayed by n time steps"""

        self.NT -= n

        nidx = int(n * self.NLOC)
        self.A = self.A[nidx:, nidx:] # CSR sparse -> CSR sparse


class SnapshotWithHomeConverter(Converter):
    ADJTYPES = ("direct", "out", "in")
    ADJMAP = {
        "direct": "weight",
        "out": "departures",
        "in": "arrivals",
    }

    def __init__(
        self,
        network_snapshots: Mapping[int, ig.Graph],
        ordering: Ordering,
        shuffle_snapshots=False,
        infer_durations=False,
    ):
        self.ordering = ordering

        self.snapshot_times = sorted(network_snapshots.keys())
        if infer_durations:
            base_snapshot_durations = np.diff(self.snapshot_times)
            base_duration_map = {k:d for k,d in zip(self.snapshot_times, base_snapshot_durations)}

        if shuffle_snapshots:
            random.shuffle(self.snapshot_times)

        # infer the duration scaling for each snapshot
        if infer_durations:
            self.snapshot_durations = [base_duration_map.get(k, 1) for k in self.snapshot_times]
        else:
            self.snapshot_durations = [network_snapshots[k]['duration'] for k in self.snapshot_times]
        

        # construct raw transition matrices that are to be parsed into weighti ng matrices
        self.raw_transition_matrices = {
            adjtype: [
                self.parse_transition_matrix_from_graph(
                    network_snapshots[snapshot_time],
                    duration=snapshot_duration,
                    adj_key=self.ADJMAP[adjtype],
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
        self.compute_weighting_matrices()

    @classmethod
    def from_directory(cls, directory: str | Path, *args, **kwargs):
        snapshots = cls.load_snapshots(directory)
        return cls(snapshots, *args, **kwargs)

    @staticmethod
    def load_snapshots(rootpath: str | Path) -> Mapping[int, ig.Graph]:
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
        steps = max(steps, 1)  # bound below by 1
        steps = min(steps, trunc)  # bound above by trunc
        t = np.arange(steps, 0, -1, dtype=nb.float64)
        attributions = t**power
        normalising_constant = np.sum(attributions)
        return attributions / normalising_constant

    def parse_transition_matrix_from_graph(
        self,
        graph: ig.Graph,
        duration: float,
        adj_key: str = "weight",
    ):
        return transition_matrix_from_graph(
            graph=graph,
            ordering=self.ordering,
            scaling_per_node=self.ordering.sizes,
            global_scaling=duration,
            ordering_key="name",
            adjacency_attribute=adj_key,
            matrix_size=len(self.ordering.order),
        )

    def compute_weighting_matrices(self):
        out_weights, out_sums = self.compute_outwards_weighting_matrices_and_sum()
        in_weights = self.compute_inwards_weighting_matrices()

        self.outwards_weighting_matrices = out_weights
        self.inwards_weighting_matrices = in_weights
        self.outwards_flow_weight = out_sums

    def compute_inwards_weighting_matrices(self):
        inwards_weighting_matrices = []
        at_home = 0
        for departs, arrives in zip(
            self.raw_transition_matrices["out"],
            self.raw_transition_matrices["in"],
        ):
            at_home += departs
            arrive_places = np.where(arrives)
            inwards_mat = np.zeros_like(arrives)
            inwards_mat[arrive_places] = arrives[arrive_places] / at_home[arrive_places]
            inwards_weighting_matrices.append(np.clip(inwards_mat, 0, 1))

            at_home -= arrives

        return inwards_weighting_matrices

    def compute_outwards_weighting_matrices_and_sum(self):
        outwards_weighting_matrices = []
        outwards_weighting_sum = []
        for direct, departs in zip(
            self.raw_transition_matrices["direct"], self.raw_transition_matrices["out"]
        ):
            # purge direct self-loops as they are non-physical
            diagidx = np.diag_indices_from(direct)
            direct[diagidx] = 0.0

            leaves = (direct + departs).sum(axis=1).reshape((-1, 1))
            outwards_weighting_sum.append(leaves)
            direct_prop = direct / leaves
            departs_prop = departs / leaves
            dense_out_matrix = np.hstack(
                [np.nan_to_num(direct_prop), np.nan_to_num(departs_prop)]
            )
            sparse_out_matrix = sparse.csr_array(dense_out_matrix)
            sparse_out_matrix.eliminate_zeros()
            outwards_weighting_matrices.append(sparse_out_matrix)

        return outwards_weighting_matrices, outwards_weighting_sum

    def map_parameters(self, parameters):
        mapped_parameters = dict()

        mapped_parameters["beta"] = parameters["beta"]

        p = np.asanyarray(parameters["prob_final_stay"]).reshape((-1, 1))
        mapped_parameters["prob_final_stay"] = p

        mapped_parameters["gamma"] = [S / (1 - p) for S in self.outwards_flow_weight]

        mapped_parameters["transition_matrix_out"] = self.outwards_weighting_matrices
        mapped_parameters["transition_matrix_in"] = self.inwards_weighting_matrices

        return mapped_parameters


class StaticConverter(Converter):
    ADJMAP = {
        "direct": "direct_weight",
        "indirect": "indirect_weight",
    }

    def __init__(
        self, network: ig.Graph, ordering: Ordering, time_span: SupportsFloat | None = None, purge_selfloops=False
    ):
        self.ordering = ordering

        if time_span is not None:
            self.time_span = time_span
        elif "time_span" in network.attributes():
            self.time_span = network["time_span"]
        else:
            raise ValueError("No time span provided")

        self.raw_transition_matrices = {
            adjtype: transition_matrix_from_graph(
                graph=network,
                ordering=self.ordering,
                scaling_per_node=self.ordering.sizes,
                global_scaling=self.time_span,
                ordering_key="node",
                adjacency_attribute=adjkey,
                matrix_size=len(self.ordering.order),
            )
            for adjtype, adjkey in self.ADJMAP.items()
        }
        self.raw_transition_matrices["link_time"] = transition_matrix_from_graph(
            graph=network,
            ordering=self.ordering,
            ordering_key="node",
            adjacency_attribute="link_time",
        )

        self.outwards_flow_weight = None
        self.outwards_weighting_matrix = None
        self.inwards_weighting_matrix = None
        self.compute_weighting_matrices(purge_selfloops=purge_selfloops)

    @classmethod
    def from_file(
        cls, file: PathLike, ordering: Ordering, *args, **kwargs
    ):
        network = ig.Graph.Read(file)
        return cls(network, ordering, *args, **kwargs)

    def compute_weighting_matrices(self, purge_selfloops=False):
        # compute outgoing matrix

        direct = self.raw_transition_matrices["direct"]
        departs = self.raw_transition_matrices["indirect"]

        if purge_selfloops:
            for matrix in (direct, departs):
                diag_idx = np.diag_indices_from(matrix)
                # these should be soft refs, and thus update
                matrix[diag_idx] = 0.0

        leaves = (direct + departs).sum(axis=1).reshape((-1, 1))
        direct_prop = direct / leaves
        departs_prop = departs / leaves
        dense_out_matrix = np.hstack(
            [np.nan_to_num(direct_prop), np.nan_to_num(departs_prop)]
        )

        sparse_out_matrix = sparse.csr_array(dense_out_matrix)
        sparse_out_matrix.eliminate_zeros()

        self.outwards_weighting_matrix = sparse_out_matrix
        self.outwards_flow_weight = leaves

        # use link time for incoming matrix

        link_time = self.raw_transition_matrices["link_time"]
        link_rate = link_time**-1
        link_rate[~np.isfinite(link_rate)] = 0.0

        self.inwards_weighting_matrix = link_rate

    def map_parameters(self, parameters: Mapping) -> Mapping:
        mapped_parameters = dict()

        mapped_parameters["beta"] = parameters["beta"]

        p = np.asanyarray(parameters["prob_final_stay"]).reshape((-1, 1))
        mapped_parameters["prob_final_stay"] = p

        mapped_parameters["gamma"] = self.outwards_flow_weight / (1 - p)

        mapped_parameters["transition_matrix_out"] = self.outwards_weighting_matrix
        mapped_parameters["transition_matrix_in"] = self.inwards_weighting_matrix

        return mapped_parameters


class NaiveStaticConverter(Converter):

    def __init__(
        self, network: ig.Graph, ordering: Ordering, time_span: SupportsFloat | None=None
    ):
        self.ordering = ordering

        if time_span is not None:
            self.time_span = time_span
        elif "time_span" in network.attributes():
            self.time_span = network["time_span"]
        else:
            raise ValueError("No time span provided")
        
        self.transition_matrix =  transition_matrix_from_graph(
            graph=network,
            ordering=self.ordering,
            scaling_per_node=self.ordering.sizes,
            global_scaling=self.time_span,
            ordering_key="name",
            adjacency_attribute="weight",
            matrix_size=len(self.ordering.order),
        )

    @classmethod
    def from_file(
        cls, file: PathLike, ordering: Ordering, *args, **kwargs
    ):
        network = ig.Graph.Read(file)
        return cls(network, ordering, *args, **kwargs)

    def map_parameters(self, parameters: Mapping) -> Mapping[str, Any]:
        mapped_parameters = dict()

        mapped_parameters["beta"] = parameters["beta"]

        p = np.asanyarray(parameters["prob_final_stay"]).reshape((-1, 1))
        mapped_parameters["prob_final_stay"] = p

        observed_departure_rate = self.transition_matrix.sum(axis=1).reshape((-1, 1))
        mapped_parameters["gamma"] = observed_departure_rate / (1 - p)

        mapped_parameters["transition_matrix"] = self.transition_matrix

        return mapped_parameters

