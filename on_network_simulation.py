"""High dim metapopulation SIS model for AMR infection"""
"""Discrete time simulation"""
"""for the purpose of determining if temporality matters via simulaiton study"""
import numpy as np
import igraph as ig
import glob
import pathlib
from scipy import sparse 

_EPS = 1e-14

class Simulation():
    def __init__(self, hospital_sizes, transition_matrix, parameters, dt=1.0, remove_diagonal=True):
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
        if remove_diagonal: transition_matrix -= np.diag(np.diag(transition_matrix))
        self.PP = transition_matrix
        self.parameters = parameters
        self.dt = dt
        self.history = self.state
        self.ts = [0.0]

    def reset(self):
        """Reset the state of the simulation"""
        self.state = self.history[:,0:1]
        self.history = self.state
        self.ts = [0.0]

    def seed(self, n_seedings=1, seed_value=1, wipe=True, rng_seed=None):
        """Sets the initial condition, by seedig a number of hospitals with some number of infected each
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
            self.state[hospital,0] += seed_value

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
        n_rec = self.rng.poisson(gamma * I * self.dt)
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
    
    def simulate(self, until=100):
        """Performs simualtion by repeated stepping until the specified time
        Can terminate early if the system has no more infected individuals"""
        for ti in range(int(until / self.dt)):
            self.state = self.step()
            self.history = np.hstack([self.history, self.state])
            self.ts.append(self.ts[-1] + self.dt)
            if sum(self.state) < 1:
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
    def __init__(self, hospital_size_mapping, network, parameters, dt=1.0):
        # step size is determined by the temporal network
        transition_matrix, net_dt, dimensions = self.map_network_to_transition_matrix(hospital_size_mapping, network)
        self.DIMENSIONS = dimensions
        self.DT = net_dt
        hospital_sizes = [v for _,v in sorted(hospital_size_mapping.items())]
        hospital_sizes = np.concatenate([hospital_sizes, np.zeros(transition_matrix.shape[0] - len(hospital_sizes))])
        super().__init__(hospital_sizes, transition_matrix, parameters, dt=dt, remove_diagonal=False)
        self.current_stage = 0

    @staticmethod
    def map_network_to_transition_matrix(hospital_size_mapping: dict[int, int], network: ig.Graph):
        # setup an all leave situation
        times = {k: i for i,k in enumerate(sorted(set(network.vs['time'])))}
        NT = len(times)
        times_by_order = sorted(times.keys())
        DT = times_by_order[1] - times_by_order[0] # assumes regular time grid
        locs = {k: i for i,k in enumerate(hospital_size_mapping)}
        NLOC = len(locs)
        
        txn_matrix_data, txn_matrix_i, txn_matrix_j = [], [], []

        # structure is blocks where we have all locs at time 0, then all locs at time 1 etc
        for node in network.vs:
            nd_idx = NLOC*times[node['time']] + locs[node['loc']]
            # nd_idx = (locs[node['loc']], times[node['time']])

            for edge in node.out_edges():
                if edge.target == node.index:
                    continue
                else:
                    other = network.vs[edge.target]
                    # target_idx = (locs[other['loc']], times[other['time']])
                    target_idx = NLOC*times[other['time']] + locs[other['loc']]
                    out_num = edge['weight'] / hospital_size_mapping[node['loc']] / DT
                    txn_matrix_data.append(out_num)
                    txn_matrix_i.append(nd_idx)
                    txn_matrix_j.append(target_idx)

        transition_matrix = sparse.coo_array((txn_matrix_data, (txn_matrix_i, txn_matrix_j)), shape=(NLOC*NT, NLOC*NT))

        return transition_matrix.tocsr(), DT, {'NLOC': NLOC, 'NT': NT}
    
    def step(self):
        beta, gamma = self.parameters
        NLOC, NT = self.DIMENSIONS['NLOC'], self.DIMENSIONS['NT']
        N = self.N[:NLOC]

        # these are discrete steps that match the temporal network
        # jitter up by small number for computer epsilon
        tidx = int(self.ts[-1] / self.DT + 1e-8)
        # substep 1: check starting state
        XTHIS = tidx * NLOC
        XNEXT = (tidx + 1) * NLOC
        new_state = np.array(self.state)
        if self.current_stage < tidx:
            # move along prev state
            XPREV = self.current_stage * NLOC
            new_state[XTHIS:XNEXT, 0:1] += new_state[XPREV:XTHIS, 0:1]
            new_state[XTHIS:XNEXT, 0:1] = np.clip(new_state[XTHIS:XNEXT, 0:1], 0, N)
            new_state[XPREV:XTHIS, 0:1] = 0
            self.current_stage += 1
        current_state = new_state[XTHIS:XNEXT, 0:1] # by slicing, this creates a view, not a copy

        # substep 2: move people out that move into the future
        out_transitions = self.PP[XTHIS:XNEXT, XNEXT:] # these have units movers/hospital size per time unit
        # reshape here to get the axis alignment
        avg_freeze_rate = out_transitions * current_state.reshape((-1, 1))
        actual_freezers = sample_poisson_on_sparse(avg_freeze_rate * self.dt, self.rng)
        outgoing_time_travellers = actual_freezers.sum(axis=1).reshape((-1, 1))
        incoming_time_travellers = actual_freezers.sum(axis=0).reshape((-1, 1))
        # when adjusting states, we ensure that the num of ppl leaving is capped at the current remaining pop
        new_state[XTHIS:XNEXT,0:1] -= np.clip(outgoing_time_travellers, 0, new_state[XTHIS:XNEXT,0:1])
        new_state[XNEXT:,0:1] += incoming_time_travellers

        # substep 3: perform intra-timestep operations (movements/infection)
        I = current_state
        n_inf = self.rng.poisson(beta * (N-I) * I / N * self.dt)
        n_rec = self.rng.poisson(gamma * I * self.dt)
        mov_I = (self.PP[XTHIS:XNEXT, XTHIS:XNEXT] * I).todense()
        M_mov_I = self.rng.poisson(np.abs(mov_I) * self.dt) * np.sign(mov_I)
        n_mov_I = (M_mov_I.sum(axis=0) - M_mov_I.sum(axis=1)).reshape(I.shape)
        # clipping values at zero to prevent pathological behaviour
        # this might leak ppl out of/into the system if we are not careful, but it'll be quite small
        I_new = np.clip(I + n_inf - n_rec + n_mov_I, 0, N)
        current_new = I_new.reshape((-1, 1))
        new_state[XTHIS:XNEXT, 0:1] = current_new

        # substep 4: perform recovery step for time travellers
        I_future = new_state[XNEXT:,0:1]
        n_future_rec = self.rng.poisson(gamma * I_future * self.dt)
        I_future_new = np.clip(I_future - n_future_rec, 0, None)
        new_state[XNEXT:, 0:1] = I_future_new

        return new_state

    def seed(self, n_seedings=1, seed_value=1, wipe=True, rng_seed=None):
        # seed will wipe the memory by default
        self.rng = np.random.default_rng(rng_seed)

        if wipe:
            self.reset()

        for _ in range(n_seedings):
            hospital = int(self.rng.uniform(0, self.DIMENSIONS['NLOC']))
            self.state[hospital,0] += seed_value

    def extract_history(self):
        bins = np.arange(0, self.ts[-1]+1, self.DT)
        # map things so that we capture the correct boundaries, and also capture the initial condition
        tidxs = np.clip(np.digitize(self.ts, bins, right=True) - 1, 0, None)
        L = self.DIMENSIONS['NLOC']
        history = np.hstack([self.history[tidx*L:(tidx+1)*L, i:i+1] for i, tidx in enumerate(tidxs)])
        return history

class SnapshotNetworkSimulation(Simulation):
    """Extends Simulation in order to allow for swapping out of the transition matrix (PP)
    """

    def __init__(self, hospital_size_mapping, snapshots, parameters, dt=1.0):
        self.hospital_ordering = self.order_hospital_size_mapping(hospital_size_mapping)
        self.hospital_sizes = [size for _, size in sorted(hospital_size_mapping.items())]
        self.snapshot_times = sorted(snapshots.keys())
        self.snapshot_durations = np.diff(self.snapshot_times)
        self.transition_matrices = [
            self.make_transition_matrix_from_graph(snapshots[snapkey], duration) 
            for snapkey, duration in zip(self.snapshot_times, self.snapshot_durations)
        ]
        super().__init__(self.hospital_sizes, self.transition_matrices[0], parameters, dt=dt)
        self.current_index = 0

    @staticmethod
    def order_hospital_size_mapping(hospital_size_mapping):
        return {int(k):i for i,k in enumerate(sorted(hospital_size_mapping.keys()))}

    def step(self):
        step_res = super().step()
        if self.ts[-1] >= self.snapshot_times[self.current_index + 1]:
            # swap in new transition matrix
            self.current_index += 1
            self.PP = self.transition_matrices[self.current_index]
        return step_res

    def make_transition_matrix_from_graph(self, graph: ig.Graph, duration: float):

        graph_paste_order = [self.hospital_ordering[int(k)] for k in graph.vs['name']]
        A = graph.get_adjacency(attribute='weight')
        MAXN = max(self.hospital_ordering.values()) + 1
        ordered_adj = np.zeros((MAXN, MAXN))
        # fill the matrix with correclty pivoted A data
        # np.ix_ creates the mesh indices to do the correct index mapping
        ordered_adj[np.ix_(graph_paste_order, graph_paste_order)] = np.array(A.data)
        # no need to trim diag, that gets done by the super().__init__
        # here we massage the hospital sizes, and divide over to get probability of movement over the time step
        transition_matrix = ordered_adj / np.reshape(self.hospital_sizes, ((-1, 1))) / duration
        return transition_matrix

    @staticmethod
    def load_snapshots(rootpath):
        snapshots = dict()
        for graph in glob.glob(f"{rootpath}/*.graphml"):
            name = int(pathlib.Path(graph).stem)
            snapshots[name] = ig.Graph.Read_GraphML(graph)
        return snapshots

def stoch_sim_test():
    sim = Simulation(
            [100, 300, 200], 
            np.array([
                [0, 0.1, 0.05,], 
                [0.01, 0, 0.1], 
                [0.04, 0.1, 0]
            ]), 
            [2, 1.6, 0.2], 
            dt=0.1
        )
    sim.seed(1)
    sim.simulate(20)

def temporal_test():
    import graph_importer as gim
    sim = TemporalNetworkSimulation(
        [200, 300, 100],
        gim.make_graph('tiny_temporal_network.lgl'),
        [20., 16.]
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



def sample_poisson_on_sparse(sparse_rates, rng=None):
    """We want to sample poisson-distributed values, with mean values given by the values of a sparse array"""
    if rng is None:
        rng = np.random.default_rng()
    # convert to COO format
    sparse_rates = sparse_rates.tocoo()
    # pull sample based on the saprse amtrix non-zero values
    sample = rng.poisson(sparse_rates.data)
    # generate a new sparse matrix based on the sampled values
    sample_sparse = sparse.coo_array((sample, (sparse_rates.row, sparse_rates.col)), shape=sparse_rates.shape, dtype=sparse_rates.dtype)

    return sample_sparse


if __name__ == "__main__":
    main()