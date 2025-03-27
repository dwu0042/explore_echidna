import numpy as np

rng = np.random.default_rng()

N = 10_000
samples = rng.pareto(0.7, size=(N))
T = np.sum(samples)

Omega = 10  # threshold

small = samples[samples < Omega]
big = samples[samples >= Omega]

L = np.mean(big)
mu = T/N


def generate_until(generator, T):
    t = 0
    ts = []
    while t < T:
        dt = generator()
        t += dt
        ts.append(dt)
    return ts

naive_ts = generate_until(lambda : rng.exponential(mu), T)

S = (mu * N + L * len(big)) / len(small)
def improved_sample():
    base = rng.exponential(S)
    if rng.random() < (len(small) / N):
        return base
    else:
        return base + rng.exponential(L - S)

improved_ts = generate_until(improved_sample, T)

print(N, len(naive_ts), len(improved_ts))


