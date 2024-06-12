"""this is a base refactor of on_network_simulation"""

from typing import Sequence, Mapping
import numpy as np
from util import BlackHole
from numba_sample import multinomial_sparse_full, truncated_poisson

class Simulation():
    def __init__(self, full_size: Sequence[int], parameters: Mapping, dt=1.0):
        self.N = np.asanyarray(full_size).reshape((-1, 1))
        self.NHOSP, _ = self.N.shape
        self.state = np.zeros_like(self.N)
        self.parameters = parameters
        self.dt = dt

        self.ts = [0.0]
        self._history = [self.state] # softref here should update as initial self.state updates in seed

    @property
    def history(self):
        return np.hstack(self._history)

    def export_history(self, to):
        np.savez_compressed(
            file = to,
            ts = self.ts,
            history = self.history,
        )

    def reset(self, soft=True):
        if soft:
            self.state = self._history[0]
        else:
            self.state = np.zeros_like(self.state)
        self._history = self.state
        self.ts = [0.0]

    def seed(self, n_seed_events=1, n_seed_number=1, rng_seed=None):
        """Sets the initial condition, by seeding a number of hospitals with some number of infected each
        Performs basic sanity checking post-seeding to prevent overflows.
        Does not reset the initial state.
        The rng seed can be specified.
        """
        self.rng = np.random.default_rng(rng_seed)

        for _ in range(n_seed_events):
            location = int(self.rng.uniform(0, len(self.state)))
            self.state[location, 0] += n_seed_number

        self.state = np.clip(self.state, 0, self.N)

    def step(self):
        """Performs a sinble time step of the simulation
        Returns the state at the end of the time step"""

        beta = self.parameters['beta']
        gamma = self.parameters['gamma']
        pstay = self.parameters['prob_final_stay']
        WW = self.parameters['weighting_matrix']

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
        n_mov = self.rng.binomial(n_rec, 1-pstay)
        # second, draw where they go
        M_mov_I = self.rng.multinomial(n_mov, WW)
        n_mov_I = M_mov_I.sum(axis=0)

        I_new += n_mov_I

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


class SnapshotWithHome(Simulation):
    def __init__(self, full_size: Sequence[int], parameters: Mapping, timings: Sequence, dt=1.0, track_movement=False):
        super().__init__(
            full_size=full_size,
            parameters=parameters,
            dt=dt,
        )

        self.current_index = 0
        self.timings = timings

        if track_movement:
            self.mover_in = []
            self.mover_out = []
        else:
            self.mover_in = BlackHole()
            self.mover_out = BlackHole()

        self.shadow_state = np.zeros((self.NHOSP, self.NHOSP), dtype=np.int64)
        self.transient_shadow = np.zeros_like(self.shadow_state, dtype=np.int64)

    def export_history(self, to, with_movers=False):

        move_info = dict()
        if with_movers:
            move_info = {
                'mover_in': self.mover_in,
                'mover_out': self.mover_out,
            }

        np.savez_compressed(
            file = to,
            ts = self.ts,
            history = self.history,
            **move_info
        )

    def reset(self, soft=True):
        super().reset(soft=soft)

        self.shadow_state = np.zeros((self.NHOSP, self.NHOSP), dtype=np.int64)
        self.transient_shadow = np.zeros_like(self.shadow_state, dtype=np.int64)
        self.current_index = 0
        self.mover_out = type(self.mover_out)()
        self.mover_in = type(self.mover_in)()

    def step(self):

        beta = self.parameters['beta']
        gamma = self.parameters['gamma']
        pstay = self.parameters['prob_final_stay']
        TX_out = self.parameters['transition_matrix_out']
        TX_in = self.parameters['transition_matrix_in']

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
        indirect_move_to = move_to[:, self.NHOSP :].reshape(WIDE)

        indirect_return_rate = TX_in[self.current_index] * X
        indirect_returns_raw = truncated_poisson(indirect_return_rate * self.dt, X)
        indirect_returns = indirect_returns_raw.sum(axis=1).reshape(NARROW)
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
