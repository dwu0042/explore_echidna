"""High dim metapopulation SIS model for AMR infection"""

"""Discrete time simulation"""
"""for the purpose of determining if temporality matters via simulaiton study"""
import numpy as np
import igraph as ig
import glob
import pathlib
from scipy import sparse
from util import Iden, BlackHole
from numba_sample import (
    multinomial_sample_sparse_collapsed,
    multinomial_sparse_full,
    truncated_poisson,
)
from typing import Mapping, Iterable, Hashable

_EPS = 1e-14


class Simulation:
    def __init__(
        self,
        hospital_sizes,
        transition_matrix,
        prob_final,
        parameters,
        dt=1.0,
        remove_diagonal=True,
        make_removal=True,
    ):
        """Initialises a stochastic simulation of an SIS metapopulation model
        Assumes that the number of individuals in each metapopulation patch (hospital) is constant and known
        Implements mass-action infection, exponential recovery and movement according to a transition matrix.
        Parameters are :
            beta: rate of infection
            gamma: rate of recovery
            dissociation: rate of movement by transition matrix
        Simulation proceeds in constant time steps, defined by the argument dt
        The transition matrix can be provided with diagonal entries, and those diagonal entries will be zeroed if remove_diagonal is set to true (default True)
        """
        self.state = np.zeros((len(hospital_sizes), 1))
        self.N = np.array(hospital_sizes).reshape((-1, 1))
        if remove_diagonal:
            transition_matrix -= np.diag(np.diag(transition_matrix))
        self.PP = transition_matrix
        if make_removal:
            self.removal = (
                self.PP.sum(axis=1).reshape((-1, 1))
                / (1 - prob_final.reshape((-1, 1)))
                * prob_final.reshape((-1, 1))
            )
        self.parameters = parameters
        self.dt = dt
        self._history = [self.state]
        self.ts = [0.0]

    @property
    def history(self):
        return np.hstack(self._history)

    def reset(self):
        """Reset the state of the simulation"""
        self.state = self._history[0]
        self._history = [self.state]
        self.ts = [0.0]

    def seed(self, n_seedings=1, seed_value=1, wipe=True, rng_seed=None):
        """Sets the initial condition, by seeding a number of hospitals with some number of infected each
        Does not perform sanity checking on the number of individuals to infect with each seeding event
        If wipe is True, calls the .reset method
        The rng seed can be specified.
        """
        # seed will wipe the memory by default
        self.rng = np.random.default_rng(rng_seed)

        if wipe:
            self.reset()

        for _ in range(n_seedings):
            hospital = int(self.rng.uniform(0, len(self.state)))
            self.state[hospital, 0] += seed_value

    def step(self):
        """Performs a single time step of simulation
        Returns the state at the end of the time step"""
        beta, gamma, dissoc = self.parameters
        # S = self.state[::2,:]
        # I = self.state[1::2,:]
        I = self.state

        # we note we get _rates_
        # I will model as Poisson, w/ fixed rates wrt the start of the step
        n_inf = self.rng.poisson(beta * (self.N - I) * I / self.N * self.dt)
        n_rec = self.rng.poisson(self.removal * I * self.dt)
        # here we take the transition matrix and need to ensure that poisson() gets +ve rates
        # in theory, the transition matrix should be all +ve; the computational tradeoff is minimal
        mov_I = self.PP * I
        M_mov_I = self.rng.poisson(dissoc * np.abs(mov_I) * self.dt) * np.sign(mov_I)
        # A_ij is from i to j; col sum gives incoming, row sum gives outgoing
        n_mov_I = (M_mov_I.sum(axis=0) - M_mov_I.sum(axis=1)).reshape(I.shape)
        # clipping values at zero to prevent pathological behaviour
        # this might leak ppl out of/into the system if we are not careful, but it'll be quite small
        I_new = np.clip(I + n_inf - n_rec + n_mov_I, 0, self.N)

        return I_new

    def simulate(self, until=100, nostop=False):
        """Performs simualtion by repeated stepping until the specified time
        Can terminate early if the system has no more infected individuals"""
        for ti in range(int(until / self.dt)):
            self.state = self.step()
            self._history.append(self.state)
            self.ts.append(self.ts[-1] + self.dt)
            if np.sum(self.state) < 1 and not nostop:
                print(f"Early termination: {self.ts[-1] = }")
                break


class SimulationODE(Simulation):
    def step(self):
        beta, gamma, dissoc = self.parameters
        I = self.state
        dI = beta * (self.N - I) * I / self.N - gamma * I + dissoc * self.PP @ I
        I_new = np.clip(I + self.dt * dI, 0, self.N)

        return I_new


class TemporalNetworkSimulation(Simulation):
    """Simulation on a temporal network
    Has a convenience method to map a temporal network loaded in with loc and time attrs on nodes to a transition matrix (snapshot representation)
    Performs stepping for the temporal network assuming that dynamics clump at the start of the snapshot time, allows for edges that move both loc and time
    """

    def __init__(
        self,
        hospital_size_mapping,
        network,
        prob_final_stay,
        parameters,
        dt=1.0,
        track_movers=False,
    ):
        # step size is determined by the temporal network
        transition_matrix, net_dt, dimensions = self.map_network_to_transition_matrix(
            hospital_size_mapping, network
        )
        self.DIMENSIONS = dimensions
        self.DT = net_dt
        self.prob_final_stay = np.array(
            [prob_final_stay[hosp] for hosp, _ in sorted(hospital_size_mapping.items())]
        ).reshape((-1, 1))
        hospital_sizes = np.asanyarray(
            [v for _, v in sorted(hospital_size_mapping.items())]
        )
        super().__init__(
            hospital_sizes,
            transition_matrix,
            self.prob_final_stay,
            parameters,
            dt=dt,
            remove_diagonal=False,
            make_removal=False,
        )
        self.time_travellers = np.zeros(
            (self.DIMENSIONS["NLOC"], self.DIMENSIONS["NT"])
        )
        self.compute_removal_rates()
        if track_movers:
            self.mover_out = []
            self.mover_in = []
        else:
            self.mover_out = BlackHole()
            self.mover_in = BlackHole()

    @staticmethod
    def map_network_to_transition_matrix(
        hospital_size_mapping: dict[int, int], network: ig.Graph
    ):
        # setup an all leave situation
        times = {k: i for i, k in enumerate(sorted(set(network.vs["time"])))}
        NT = len(times)
        times_by_order = sorted(times.keys())
        DT = times_by_order[1] - times_by_order[0]  # assumes regular time grid
        locs = {k: i for i, k in enumerate(sorted(hospital_size_mapping))}
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
                    out_num = edge["weight"] / hospital_size_mapping[node["loc"]] / DT
                    txn_matrix_data.append(out_num)
                    txn_matrix_i.append(nd_idx)
                    txn_matrix_j.append(target_idx)

        transition_matrix = sparse.coo_array(
            (txn_matrix_data, (txn_matrix_i, txn_matrix_j)),
            shape=(NLOC * NT, NLOC * NT),
        )

        return transition_matrix.tocsr(), DT, {"NLOC": NLOC, "NT": NT}

    def compute_removal_rates(self):
        NLOC = self.DIMENSIONS["NLOC"]
        NT = self.DIMENSIONS["NT"]
        movements_out = self.PP.sum(axis=1)
        M_mat = movements_out.reshape((NLOC, NT), order="F")
        p = self.prob_final_stay.reshape((-1, 1))
        # here we use that M_mat = gamma * (1-p)
        gamma = M_mat / (1 - p)
        self.removal_rate = gamma

        # now we normalise the movement matrix so we can use in the multinomial later
        self.PP = (self.PP / movements_out[:, np.newaxis]).tocsr()

    def step(self):
        """
        Simulate a single step of reactions over the timestep size of self.dt

        Subprocesses:
            1. Move individuals from home to the hospital
            2. Simulate the infection process inside the hospital
            3. Move individuals out of hospital
                a. Remove individuals that permanently leave the system
                b. Partition the individuals that readmit at another hospital
                    i. Select and immediately move individuals that have immediate transfers
                    ii. Retain individuals that have indirect transfers
        """
        beta, *gamma = self.parameters  # transmission, recovery, discharge
        NLOC, NT = self.DIMENSIONS["NLOC"], self.DIMENSIONS["NT"]
        N = self.N

        # these are discrete steps that match the temporal network
        # jitter up by small number for computer epsilon
        t = self.ts[-1]
        tidx = int(t / self.DT + 1e-8)
        XTHIS = tidx * NLOC
        XNEXT = (tidx + 1) * NLOC

        # new_state = np.array(self.state)

        # substep 1. reintroduce time travellers
        travellers = self.time_travellers[:, tidx : tidx + 1]
        next_time_boundary = (tidx + 1) * self.DT
        remaining_time = next_time_boundary - t
        rel_time = np.clip(self.dt / remaining_time, 0, 1)
        movers = self.rng.binomial(travellers.astype(np.int64), rel_time)
        movers = np.clip(movers, 0, travellers)
        self.time_travellers[:, tidx : tidx + 1] = travellers - movers
        new_state = np.clip(self.state + movers, 0, N)
        self.mover_in.append(new_state - self.state)

        # substep 2: do infection/mass action discharge
        I = self.state
        n_inf = self.rng.poisson(beta * (N - I) * I / N * self.dt).astype("int64")
        n_out = self.rng.poisson(self.removal_rate[:, tidx : tidx + 1] * I * self.dt)
        # truncated poisson: cannot have more people leave than are present
        n_out = np.clip(n_out, 0, I).astype("int64")

        new_state += n_inf - n_out

        # substep 3a: split out people that never return
        n_removed = self.rng.binomial(n_out, self.prob_final_stay)
        n_retained = n_out - n_removed
        self.mover_out.append(n_retained)

        # substep 3b: partition individuals that transfer
        M = self.PP[XTHIS:XNEXT, XTHIS:]
        n_out_collapsed = multinomial_sample_sparse_collapsed(n_retained.flatten(), M)
        indirect_transfers = n_out_collapsed[NLOC:]

        # substep 3bi: move the direct transfers
        direct_transfers = n_out_collapsed[:NLOC]
        new_state += direct_transfers.reshape(*new_state.shape)

        # substep 3bii: store the indirect transfers
        indirect_movers_influx = indirect_transfers.reshape((NLOC, -1), order="F")
        self.time_travellers[:, tidx + 1 :] += indirect_movers_influx

        # substep 4: truncate the number of infected in each location
        new_state = np.clip(new_state, 0, N)

        return new_state

    def seed(self, n_seedings=1, seed_value=1, wipe=True, rng_seed=None):
        # seed will wipe the memory by default
        self.rng = np.random.default_rng(rng_seed)

        if wipe:
            self.reset()

        # we want to be a bit more careful about where we seed
        # we want this node to be connected at t=0
        NLOC = self.DIMENSIONS["NLOC"]
        valid_hospitals = list(set(self.PP[:NLOC, :].nonzero()[0]))

        for hospital in self.rng.choice(valid_hospitals, n_seedings, replace=False):
            self.state[hospital, 0] += seed_value

    def reset(self):
        super().reset()
        self.time_travellers = np.zeros_like(self.time_travellers)
        self.mover_out = type(self.mover_out)()
        self.mover_in = type(self.mover_in)()

    def delay(self, n: int):
        """Post-initailisation, starts the simulation from the n-th time index"""
        NLOC = self.DIMENSIONS["NLOC"]
        NIDX = int(n * NLOC)

        self.DIMENSIONS["NT"] -= n
        self.state = self.state[NIDX:] #?????
        self.time_travellers = self.time_travellers[:, n:]
        self.removal_rate = self.removal_rate[:, n:]
        self.PP = self.PP[NIDX:, NIDX:]

    # @property
    # def history(self):
    #     bins = np.arange(0, self.ts[-1]+1, self.DT)
    #     # map things so that we capture the correct boundaries, and also capture the initial condition
    #     tidxs = np.clip(np.digitize(self.ts, bins, right=True) - 1, 0, None)
    #     L = self.DIMENSIONS['NLOC']
    #     history = np.hstack([self._history[i][tidx*L:(tidx+1)*L,:] for i, tidx in enumerate(tidxs)])
    #     return history


class SnapshotNetworkSimulation(Simulation):
    """Extends Simulation in order to allow for swapping out of the transition matrix (PP)"""

    def __init__(
        self, hospital_size_mapping, snapshots, prob_final, parameters, dt=1.0
    ):
        self.hospital_ordering = self.order_hospital_size_mapping(hospital_size_mapping)
        self.hospital_lookup = {v: k for k, v in self.hospital_ordering.items()}
        self.hospital_sizes = [
            hospital_size_mapping[i] for i in self.hospital_lookup.values()
        ]
        self.snapshot_times = sorted(snapshots.keys())
        self.snapshot_durations = np.diff(self.snapshot_times)
        self.prob_final_arr = np.array(
            [prob_final[hosp] for hosp, _ in sorted(hospital_size_mapping.items())]
        ).reshape((-1, 1))
        self.transition_matrices = [
            self.make_transition_matrix_from_graph(snapshots[snapkey], duration)
            for snapkey, duration in zip(self.snapshot_times, self.snapshot_durations)
        ]
        super().__init__(
            self.hospital_sizes,
            self.transition_matrices[0],
            self.prob_final_arr,
            parameters,
            dt=dt,
        )
        self.current_index = 0

    @staticmethod
    def order_hospital_size_mapping(hospital_size_mapping):
        return {int(k): i for i, k in enumerate(sorted(hospital_size_mapping.keys()))}

    def step(self):
        step_res = super().step()
        if self.ts[-1] >= self.snapshot_times[self.current_index + 1]:
            # swap in new transition matrix
            self.current_index += 1
            self.PP = self.transition_matrices[self.current_index]
            self.removal = (
                self.PP.sum(axis=1).reshape((-1, 1))
                / (1 - self.prob_final_arr)
                * self.prob_final_arr
            )
        return step_res

    def make_transition_matrix_from_graph(self, graph: ig.Graph, duration: float):
        return transition_matrix_from_graph(
            graph=graph,
            ordering=self.hospital_ordering,
            scaling_per_node=self.hospital_sizes,
            global_scaling=duration,
            ordering_key="name",
            adjacency_attribute="weight",
            matrix_size=max(self.hospital_ordering.values()) + 1,
        )


class SnapshotNoveauSimulation(Simulation):
    _adjacency_key = {
        "direct": "weight",
        "out": "departures",
        "in": "arrivals",
    }

    @staticmethod
    def power_kernel(steps, power=-0.6, trunc=26):
        t = np.arange(np.clip(steps, 1, trunc), 0, -1)
        s = t**power
        c = 1 / np.sum(s)
        return c * s

    def __init__(
        self,
        hospital_size_mapping,
        snapshots,
        prob_final,
        parameters,
        dt=1.0,
        trunc=26,
        track_movers=False,
    ):
        self.hospital_ordering = self.order_hospital_size_mapping(hospital_size_mapping)
        self.NHOSP = len(self.hospital_ordering)
        self.hospital_lookup = [hospital for hospital in self.hospital_ordering.keys()]
        self.hospital_sizes = [
            hospital_size_mapping[hosp] for hosp in self.hospital_lookup
        ]
        self.snapshot_times = sorted(snapshots.keys())
        self.snapshot_durations = np.diff(self.snapshot_times)
        self.raw_transition_matrices = {
            adjtype: [
                self.make_transition_matrix_from_graph(
                    snapshots[snapkey], duration, adj_key=self._adjacency_key[adjtype]
                )
                for snapkey, duration in zip(
                    self.snapshot_times, self.snapshot_durations
                )
            ]
            for adjtype in ("direct", "out", "in")
        }
        self.prob_final = np.array(
            [prob_final[hosp] for hosp in self.hospital_lookup]
        ).reshape((-1, 1))
        self.transition_matrices = dict()
        # adjust weighting on the 'direct', 'out', 'in' transition matrices to reflect correct prob tree
        self.transform_out_transition_matrix()
        self.transform_in_transition_matrix(trunc=trunc)
        if track_movers:
            self.mover_in = []
            self.mover_out = []
        else:
            self.mover_in = BlackHole()
            self.mover_out = BlackHole()

        super().__init__(
            self.hospital_sizes,
            None,
            None,
            parameters,
            dt=dt,
            remove_diagonal=False,
            make_removal=False,
        )
        self.current_index = 0
        self.shadow_state = np.zeros((self.NHOSP, self.NHOSP), dtype=np.int64)
        self.transient_shadow = np.zeros_like(self.shadow_state, dtype=np.int64)

    def transform_out_transition_matrix(self):
        self.transition_matrices["out"] = []
        self.leave_rate = []
        for D, U in zip(
            self.raw_transition_matrices["direct"], self.raw_transition_matrices["out"]
        ):
            Sig = (D + U).sum(axis=1).reshape((-1, 1))
            self.leave_rate.append(Sig / (1 - self.prob_final).reshape((-1, 1)))
            Dp = D / Sig
            Up = U / Sig
            dense_out_matrix = np.hstack([np.nan_to_num(Dp), np.nan_to_num(Up)])
            sparse_out_matrix = sparse.csr_array(dense_out_matrix)
            sparse_out_matrix.eliminate_zeros()
            self.transition_matrices["out"].append(sparse_out_matrix)

    def transform_in_transition_matrix(self, trunc=26):
        self.transition_matrices["in"] = []
        for i, E in enumerate(self.raw_transition_matrices["in"]):
            if i == 0:
                # special case at t=0
                self.transition_matrices["in"].append(E)  # should be all zeros
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
            self.transition_matrices["in"].append(E_transformed)

    def reset(self):
        super().reset()
        self.shadow_state = np.zeros((self.NHOSP, self.NHOSP), dtype=np.int64)
        self.transient_shadow = np.zeros_like(self.shadow_state, dtype=np.int64)
        self.current_index = 0
        self.mover_out = type(self.mover_out)()
        self.mover_in = type(self.mover_in)()

    @staticmethod
    def order_hospital_size_mapping(hospital_size_mapping):
        return {int(k): i for i, k in enumerate(sorted(hospital_size_mapping.keys()))}

    def step(self):
        beta, *_ = self.parameters

        I = self.state
        X = self.shadow_state
        Z = self.transient_shadow
        NARROW = I.shape
        WIDE = X.shape

        # we note we get _rates_
        # I will model as Poisson, w/ fixed rates wrt the start of the step
        n_inf = self.rng.poisson(beta * (self.N - I) * I / self.N * self.dt)
        n_out = np.clip(
            self.rng.poisson(self.leave_rate[self.current_index] * I * self.dt), 0, I
        ).astype("int64")

        n_abandon = self.rng.binomial(n_out, self.prob_final)
        n_remain = n_out - n_abandon
        self.mover_out.append(n_remain)

        # move_to = self.rng.multinomial(n_remain.flatten(), self.transition_matrices['out'][self.current_index])
        move_to = multinomial_sparse_full(
            n_remain.flatten(), self.transition_matrices["out"][self.current_index]
        )
        direct_move_to = move_to[:, : self.NHOSP].sum(axis=0).reshape(NARROW)
        indirect_move_to = move_to[:, self.NHOSP :].reshape(WIDE)

        indirect_return_rate = self.transition_matrices["in"][self.current_index] * X
        indirect_returns_raw = truncated_poisson(indirect_return_rate * self.dt, X)
        indirect_returns = indirect_returns_raw.sum(axis=1).reshape(NARROW)
        self.mover_in.append(indirect_returns)

        I_new = np.clip(
            I + n_inf - n_out + direct_move_to + indirect_returns, 0, self.N
        )
        self.shadow_state = X - indirect_returns_raw
        self.transient_shadow = Z + indirect_move_to

        if self.ts[-1] >= self.snapshot_times[self.current_index + 1]:
            # swap in new transition matrix
            self.current_index += 1
            self.shadow_state += self.transient_shadow
            self.transient_shadow = np.zeros_like(self.shadow_state)

        return I_new

    def make_transition_matrix_from_graph(
        self, graph: ig.Graph, duration: float, adj_key: str = "weight"
    ):
        return transition_matrix_from_graph(
            graph=graph,
            ordering=self.hospital_ordering,
            scaling_per_node=self.hospital_sizes,
            global_scaling=duration,
            ordering_key="name",
            adjacency_attribute=adj_key,
            matrix_size=max(self.hospital_ordering.values()) + 1,
        )

    @staticmethod
    def load_snapshots(rootpath):
        snapshots = dict()
        for graph in glob.glob(f"{rootpath}/*.graphml"):
            # safe for windows?
            name = int(pathlib.Path(graph).stem)
            with open(graph, "r") as graph_file:
                snapshots[name] = ig.Graph.Read_GraphML(graph_file)
        return snapshots


def load_snapshots(rootpath):
    snapshots = dict()
    for graph in glob.glob(f"{rootpath}/*.graphml"):
        # safe for windows?
        name = int(pathlib.Path(graph).stem)
        with open(graph, "r") as graph_file:
            snapshots[name] = ig.Graph.Read_GraphML(graph_file)
    return snapshots


def transition_matrix_from_graph(
    graph: ig.Graph,
    ordering: Mapping = None,
    scaling_per_node: Iterable = None,
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


def sample_poisson_on_sparse(sparse_rates, rng=None):
    """We want to sample poisson-distributed values, with mean values given by the values of a sparse array"""
    if rng is None:
        rng = np.random.default_rng()
    # convert to COO format
    sparse_rates = sparse_rates.tocoo()
    # pull sample based on the saprse matrix non-zero values
    sample = rng.poisson(sparse_rates.data)
    # generate a new sparse matrix based on the sampled values
    sample_sparse = sparse.coo_array(
        (sample, (sparse_rates.row, sparse_rates.col)),
        shape=sparse_rates.shape,
        dtype=sparse_rates.dtype,
    )

    return sample_sparse


#############################################################################################################


def stoch_sim_test():
    sim = Simulation(
        [100, 300, 200],
        np.array(
            [
                [
                    0,
                    0.1,
                    0.05,
                ],
                [0.01, 0, 0.1],
                [0.04, 0.1, 0],
            ]
        ),
        [2, 1.6, 0.2],
        dt=0.1,
    )
    sim.seed(1)
    sim.simulate(20)


def temporal_test():
    import graph_importer as gim

    sim = TemporalNetworkSimulation(
        [200, 300, 100], gim.make_graph("tiny_temporal_network.lgl"), [20.0, 16.0]
    )
    sim.seed(1)
    sim.simulate(2)


def full_snapshot_sim():
    import graph_importer as gim
    import examine_transfers_for_sizes as esz

    G = gim.make_graph("./concordant_networks/temponet_14_365.lgl")
    rough_hosp_size = esz.rough_hospital_size(G)
    snapshots = SnapshotNetworkSimulation.load_snapshots("./conc_tempo_14_snapshots/")
    sim = SnapshotNetworkSimulation(rough_hosp_size, snapshots, [2.0, 1.6, 1.0], dt=1)
    sim.seed(1)
    sim.simulate(20)


def main():
    temporal_test()


if __name__ == "__main__":
    import polars as pl
    import examine_transfers_for_sizes as esz

    snaps = load_snapshots("conc_tempo_14_detailed")
    sizes = esz.quick_read("concordant_networks/size_14.csv")
    pfind = pl.read_csv("probability_of_final_stay_by_shuffled_campus.csv")
    pfind = {
        h: p
        for h, p in zip(
            *(pfind.select(x).to_series().to_list() for x in ("loc", "final_stay"))
        )
    }
    parameters = (0.2,)

    sim = SnapshotNoveauSimulation(sizes, snaps, pfind, parameters)
    sim.seed(3, 5)
