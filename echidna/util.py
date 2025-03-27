from typing import Any, Protocol, TypeVar
import numpy as np
import numba as nb

class Iden():
    def __init__(self):
        pass

    def __getitem__(self, x):
        return x
    
    def __setitem__(self, x):
        raise TypeError("Identity objects do not support assignment")

class BlackHole():
    def __init__(self):
        pass

    def append(self, item):
        pass

    def __iter__(self):
        yield

@nb.njit()
def nparr_find(array: np.ndarray, item: Any):
    for idx, val in np.ndenumerate(array):
        if val == item:
            return idx
        
class NotFound(Exception):
    pass

T = TypeVar('T', covariant=True)
K = TypeVar('K', contravariant=True)
class SupportsGet(Protocol[K, T]):
    def __getitem__(self, key: K) -> T:
        ...