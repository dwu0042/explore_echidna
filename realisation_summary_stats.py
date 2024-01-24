import pickle
import numpy as np

def hitting_time(realisation, sizes, threshold=0.7):
    """Finds the approximate hitting time for each hospital reaching the given threshold of its capacity
    
    Inputs:
        realisation: SimulationResult-like object, with a .ts and .history field that contain the realisation state
        sizes: Ordered list of hospital capacities/sizes. Corresponds to the first axis of .history
        threshold: The hitting time threshold (realistion.history / sizes)

    Returns:
        hitting_time: the first time at which the threshold is exceeded
    """

    proportion_infected = realisation.history / np.reshape(sizes, (-1, 1))
    max_indices = np.argmax(proportion_infected > threshold, axis=1)
    return [(realisation.ts[maxi] if proportion_infected[i,maxi] > threshold else None) 
            for i,maxi in enumerate(max_indices) ]