# simulate the prob dist of a (non-stationary) markov chain to determine hitting time

# class NonStationaryMarkovSimulation():
#     def __init__(self, *args):

#         self.transition_matrices = []
        


class StationaryMarkov():
    def __init__(self, generator):
        self.Q = generator

    def step(self, state, dt):
        dP = self.Q @ state
        return state + dt*dP
    
    def solve(self, initial_state, tspan, dt):
        t, t_end = tspan
        Pt = [initial_state]
        ts = [t]
        while t < t_end:
            Pt.append(self.step(Pt[-1], dt))
            t = t + dt
            ts.append(t)

        return ts, Pt