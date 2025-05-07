"""this is a base refactor of on_network_simulation"""

from typing import Iterable, Sequence, Mapping, SupportsFloat
import numpy as np
from .util import BlackHole
from .numba_sample import multinomial_sparse_full, truncated_poisson, multinomial_sample_sparse_collapsed
import h5py


class Simulation:
    def __init__(self, full_size: Iterable[SupportsFloat], parameters: Mapping, dt=1.0):
        self.N = np.asanyarray(full_size).reshape((-1, 1))
        self.NHOSP, _ = self.N.shape
        self.state = np.zeros_like(self.N)
        self.parameters = parameters
        self.dt = dt

        self.ts = [0.0]
        self._history = [
            self.state
        ]  # softref here should update as initial self.state updates in seed

        self.rng = np.random.default_rng(None)

    @property
    def history(self):
        return np.hstack(self._history)

    def export_history(self, to, identity, **kwargs):
        with h5py.File(to, mode="a") as h5f:
            g = h5f.create_group(str(identity))
            g.create_dataset("ts", data=self.ts, compression="gzip")
            g.create_dataset("history", data=self.history, compression="gzip")
            g.attrs.update(kwargs)

    def reset(self, soft=True):
        if soft:
            self.state = self._history[0]
        else:
            self.state = np.zeros_like(self.state)
        self._history = [self.state]
        self.ts = [0.0]

    def seed(self, n_seed_events=1, n_seed_number=1, rng_seed=None):
        """Sets the initial condition, by seeding a number of hospitals with some number of infected each
        Performs basic sanity checking post-seeding to prevent overflows.
        Does not reset the initial state.
        The rng seed can be specified.
        """
        if rng_seed is not None:
            self.rng = np.random.default_rng(rng_seed)

        for _ in range(n_seed_events):
            location = int(self.rng.uniform(0, len(self.state)))
            self.state[location, 0] += n_seed_number

        self.state = np.clip(self.state, 0, self.N)

    def fixedseed(self, location, number_to_seed=1, rng_seed=None):
        
        if rng_seed is not None:
            self.rng = np.random.default_rng(rng_seed)

        self.state[location, 0] += number_to_seed

        self.state = np.clip(self.state, 0, self.N)

    def step(self):
        """Performs a sinble time step of the simulation
        Returns the state at the end of the time step"""

        beta = self.parameters["beta"]
        gamma = self.parameters["gamma"]
        pstay = self.parameters["prob_final_stay"]
        WW = self.parameters["transition_matrix"]

        I = self.state

        # Model the events as Poisson, and presumed independent in a
        # time step of size dt
        n_inf = self.rng.poisson(beta * (self.N - I) * I / self.N * self.dt)
        n_rec = self.rng.poisson(gamma * I * self.dt)

        # recovery is bounded by the number of individuals in the state
        n_rec = np.clip(n_rec, 0, self.state)

        # update the state
        I_new = np.clip(I + n_inf - n_rec, 0, self.N)

        # compute movements
        # first, draw how many stay
        n_mov = self.rng.binomial(n_rec, 1 - pstay)
        # second, draw where they go
        # .flatten for broadcasting of the multinomial function numpy >1.22
        M_mov_I = self.rng.multinomial(n_mov.flatten(), WW)
        n_mov_I = M_mov_I.sum(axis=0)

        # reshape here since n_mov_I is 1D and will be implicitly broadcast
        I_new += n_mov_I.reshape(I_new.shape)

        return I_new

    def simulate(self, until=100, nostop=False):
        """Performs simualtion by repeated stepping until the specified time
        Can terminate early if the system has no more infected individuals"""
        for ti in range(int(until / self.dt)):
            self.state = self.step()
            self._history.append(self.state)
            self.ts.append(self.ts[-1] + self.dt)
            if not nostop and np.sum(self.state) < 1:
                print(f"Early termination: {self.ts[-1] = }")
                break


class SimulationWithMovers(Simulation):
    def __init__(self, *args, track_movement=False, **kwargs):
        super().__init__(*args, **kwargs)

        if track_movement:
            self.mover_in = []
            self.mover_out = []
        else:
            self.mover_in = BlackHole()
            self.mover_out = BlackHole()

    def export_history(self, to, identity, with_movers=False, **kwargs):
        if not with_movers:
            super().export_history(to=to, identity=identity, **kwargs)
        else:
            with h5py.File(to, mode="a") as h5f:
                g = h5f.create_group(str(identity))
                g.create_dataset("ts", data=self.ts, compression="gzip")
                g.create_dataset("history", data=self.history, compression="gzip")
                for movd in ("mover_out", "mover_in"):
                    data = np.squeeze(getattr(self, movd))
                    g.create_dataset(movd, data=data, compression="gzip")
                g.attrs.update(kwargs)

    def reset(self, soft=True):
        super().reset(soft=soft)

        self.mover_out = type(self.mover_out)()
        self.mover_in = type(self.mover_in)()


class SnapshotWithHome(SimulationWithMovers):
    def __init__(
        self,
        full_size: Iterable[SupportsFloat], 
        parameters: Mapping,
        timings: Sequence,
        track_movement=False,
        dt=1.0,
    ):
        super().__init__(
            full_size=full_size,
            parameters=parameters,
            track_movement=track_movement,
            dt=dt,
        )

        self.current_index = 0
        self.timings = timings

        self.shadow_state = np.zeros((self.NHOSP, self.NHOSP), dtype=np.int64)
        self.transient_shadow = np.zeros_like(self.shadow_state, dtype=np.int64)

    def reset(self, soft=True):
        super().reset(soft=soft)

        self.shadow_state = np.zeros((self.NHOSP, self.NHOSP), dtype=np.int64)
        self.transient_shadow = np.zeros_like(self.shadow_state, dtype=np.int64)
        self.current_index = 0

    def step(self):
        beta = self.parameters["beta"]
        gamma = self.parameters["gamma"]
        pstay = self.parameters["prob_final_stay"]
        TX_out = self.parameters["transition_matrix_out"]
        TX_in = self.parameters["transition_matrix_in"]

        I = self.state
        X = self.shadow_state
        Z = self.transient_shadow
        NARROW = I.shape
        WIDE = X.shape

        n_inf = self.rng.poisson(beta * (self.N - I) * I / self.N * self.dt)
        n_out = np.clip(
            self.rng.poisson(gamma[self.current_index] * I * self.dt), 0, I
        ).astype("int64")

        n_abandon = self.rng.binomial(n_out, pstay)
        n_remain = n_out - n_abandon
        self.mover_out.append(n_remain)

        move_to = multinomial_sparse_full(
            n_remain.flatten(), TX_out[self.current_index]
        )
        direct_move_to = move_to[:, : self.NHOSP].sum(axis=0).reshape(NARROW)
        indirect_move_to = move_to[:, self.NHOSP:].reshape(WIDE)

        # indirect_return_rate = TX_in[self.current_index] * X * self.dt
        # indirect_returns_raw = truncated_poisson(indirect_return_rate, X)
        indirect_return_prob = TX_in[self.current_index] * self.dt
        indirect_returns_raw = self.rng.binomial(n=X, p=indirect_return_prob)
        indirect_returns = indirect_returns_raw.sum(axis=0).reshape(NARROW)
        self.mover_in.append(indirect_returns)

        I_new = np.clip(
            I + n_inf - n_out + direct_move_to + indirect_returns, 0, self.N
        )
        self.shadow_state = X - indirect_returns_raw
        self.transient_shadow = Z + indirect_move_to

        if self.ts[-1] >= self.timings[self.current_index + 1]:
            # swap in new transition matrix
            self.current_index += 1
            self.shadow_state += self.transient_shadow
            self.transient_shadow = np.zeros_like(self.shadow_state)

        return I_new


class StaticWithHome(SimulationWithMovers):
    def __init__(
        self,
        full_size: Iterable[SupportsFloat], 
        parameters: Mapping,
        dt=1.0,
        track_movement=False,
    ):
        super().__init__(
            full_size=full_size,
            parameters=parameters,
            dt=dt,
            track_movement=track_movement,
        )

        self.shadow_state = np.zeros((self.NHOSP, self.NHOSP), dtype=np.int64)

    def reset(self, soft=True):
        super().reset(soft=soft)

        self.shadow_state = np.zeros((self.NHOSP, self.NHOSP), dtype=np.int64)

    def step(self):
        beta = self.parameters["beta"]
        gamma = self.parameters["gamma"]
        pstay = self.parameters["prob_final_stay"]
        TX_out = self.parameters["transition_matrix_out"]
        TX_in = self.parameters["transition_matrix_in"]

        I = self.state
        X = self.shadow_state
        NARROW = I.shape
        WIDE = X.shape

        n_inf = self.rng.poisson(beta * (self.N - I) * I / self.N * self.dt)
        n_out = np.clip(
            self.rng.poisson(gamma * I * self.dt), 0, I
        ).astype("int64")

        n_abandon = self.rng.binomial(n_out, pstay)
        n_remain = n_out - n_abandon
        self.mover_out.append(n_remain)

        move_to = multinomial_sparse_full(
            n_remain.flatten(), TX_out
        )
        direct_move_to = move_to[:, : self.NHOSP].sum(axis=0).reshape(NARROW)
        # this ascontiguousarray is to prevent copies in the np.add later
        indirect_move_to = np.ascontiguousarray(move_to[:, self.NHOSP:].reshape(WIDE))

        # indirect_return_rate = TX_in * X
        # indirect_returns_raw = truncated_poisson(indirect_return_rate * self.dt, X)
        indirect_return_prob = TX_in * self.dt
        indirect_returns_raw = self.rng.binomial(n=X, p=indirect_return_prob)
        indirect_returns = indirect_returns_raw.sum(axis=0).reshape(NARROW)
        self.mover_in.append(indirect_returns)

        I_new = np.clip(
            I + n_inf - n_out + direct_move_to + indirect_returns, 0, self.N
        )
        # self.shadow_state = self.shadow_state + indirect_move_to - indirect_returns_raw.T
        np.subtract(self.shadow_state, indirect_returns_raw, out=self.shadow_state)
        np.add(self.shadow_state, indirect_move_to, out=self.shadow_state)

        return I_new
    

class TemporalSim(SimulationWithMovers):
    
    def __init__(
            self, 
            full_size: Iterable[SupportsFloat],
            parameters: Mapping, 
            num_times: int,
            discretisation_size: float,
            track_movement=False,
            dt=1.0,
    ):
        super().__init__(
            full_size=full_size,
            parameters=parameters,
            track_movement=track_movement,
            dt=dt,
        )

        self.NT = num_times
        self.H = discretisation_size

        self.time_travellers = np.zeros(
            (self.NHOSP, self.NT),
            dtype=np.int64,
        )

    def step(self):

        beta = self.parameters['beta']
        gamma = self.parameters['gamma']
        prob_final = self.parameters['prob_final_stay']
        txn_mat = self.parameters['transition_matrix']

        # determine position in time
        t = self.ts[-1]
        # jitter here for machine eps
        tidx = int(t / self.H + 1e-8)
        XTHIS = tidx * self.NHOSP
        XNEXT = (tidx + 1) * self.NHOSP

        # re-introduce previously seen patients that were at home
        travellers = self.time_travellers[: ,tidx:tidx+1]
        next_time_boundary = (tidx + 1) * self.H
        remaining_time = next_time_boundary - t
        rel_time = np.clip(self.dt / remaining_time, 0, 1)
        movers = self.rng.binomial(travellers.astype(np.int64), rel_time)
        movers = np.clip(movers, 0, travellers)
        self.time_travellers[:, tidx : tidx + 1] = travellers - movers
        new_state = np.clip(self.state + movers, 0, self.N)
        self.mover_in.append(new_state - self.state)

        # perform mass-action infection dynamics
        I = self.state
        n_inf = self.rng.poisson(beta * (self.N - I) * I / self.N * self.dt).astype("int64")
        n_out = self.rng.poisson(gamma[:, tidx : tidx + 1] * I * self.dt)
        # truncated poisson: cannot have more people leave than are present
        n_out = np.clip(n_out, 0, I).astype("int64")
        new_state += n_inf - n_out

        # lose individuals that never return
        n_removed = self.rng.binomial(n_out, prob_final)
        n_retained = n_out - n_removed
        self.mover_out.append(n_retained)

        # determine when and where patients reappear
        M = txn_mat[XTHIS:XNEXT, XTHIS:]
        n_out_collapsed = multinomial_sample_sparse_collapsed(n_retained.flatten(), M)
        
        # move direct transfers
        direct_transfers = n_out_collapsed[:self.NHOSP]
        new_state += direct_transfers.reshape(*new_state.shape)

        # store indirect transfers
        indirect_transfers = n_out_collapsed[self.NHOSP:]
        indirect_movers_influx = indirect_transfers.reshape((self.NHOSP, -1), order="F")
        self.time_travellers[:, tidx + 1 :] += indirect_movers_influx

        # truncate state to location capacity
        new_state = np.clip(new_state, 0, self.N)

        return new_state

    def seed(self, n_seed_events=1, n_seed_number=1, unique_locations=False, strict_validity=True, rng_seed=None):
        self.rng = np.random.default_rng(rng_seed)

        if strict_validity:
            valid_hospitals = list(set(self.parameters['transition_matrix'][:self.NHOSP, :].nonzero()[0]))
        else:
            valid_hospitals = list(range(self.NHOSP))

        for hospital in self.rng.choice(valid_hospitals, n_seed_events, replace=not(unique_locations)):
            self.state[hospital, 0] += n_seed_number
        
        self.state = np.clip(self.state, 0, self.N)

    def reset(self, soft=True):
        super().reset(soft=soft)

        self.time_travellers = np.zeros_like(self.time_travellers, dtype=np.int64)

class SnapshotNaive(Simulation):
    def __init__(self, full_size: Iterable[SupportsFloat], parameters: Mapping, timings: Sequence, dt=1.0):
        super().__init__(
            full_size=full_size,
            parameters=parameters,
            dt=dt,
        )

        self.current_index = 0
        self.timings = np.array(timings).flatten()
        self.timings = np.concatenate([self.timings, [np.inf]])

    def swapover(self):
        # assume current index is correctly set
        self.parameters['transition_matrix'] = self.parameters['transition_matrices'][self.current_index]

        self.parameters['gamma'] = self.parameters['gammas'][self.current_index]

    def reset(self, soft=True):
        super().reset(soft=soft)
        self.current_index = 0

    def simulate(self, until=100, nostop=False):
        self.swapover()
        
        for ti in range(int(until / self.dt)):
            
            self.state = self.step()
            self._history.append(self.state)
            self.ts.append(self.ts[-1] + self.dt)

            if self.ts[-1] >= self.timings[self.current_index+1]:
                self.current_index += 1
                self.swapover()

            if not nostop and np.sum(self.state) < 1:
                print(f"Early termination: {self.ts[-1] =}")
                break
