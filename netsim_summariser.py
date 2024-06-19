import h5py
import polars as pl
import numpy as np


class Summariser():

    def __init__(self, file):
        self.target_file = file
        self._open = False

    def __enter__(self):
        self._file = h5py.File(self.target_file)
        self._open = True
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self._open = False
        self._file.close()

    def ensure_open(self):
        if self._open:
            return
        else:
            self._file = h5py.File(self.target_file)
            self._open = True

    @staticmethod
    def hitting_time(target : h5py.Group, null=np.nan, **kwargs):
        mask = (target['history'][:] != 0)
        return np.where(mask.any(axis=1), mask.argmax(axis=1), null)

    @staticmethod
    def movement_out_total(target: h5py.Group, **kwargs):
        return target['mover_out'][:].sum(axis=1)
    
    @staticmethod
    def movement_in_total(target: h5py.Group, **kwargs):
        return target['mover_in'][:].sum(axis=1)

    @staticmethod
    def steady_state_infected(target: h5py.Group, **kwargs):
        init = target['history'][:,0].sum()
        return target['history'][:].sum(axis=0) / init
    
    @staticmethod
    def emptying_time(target: h5py.Group, null=np.nan, **kwargs):
        xs = target['history'][:].sum(axis=0)
        mask = (xs == 0)
        return mask.argmax() if mask.any() else null

    def collect(self, over=None, verbose=False):
        self.ensure_open()
        if over is None:
            over = list(self._file)

        results = []

        recall = lambda *x: None
        if verbose:
            recall = print

        for sim in over:
            recall(sim)
            target = self._file[sim]
            dx = dict(target.attrs)
            for metric in (
                self.hitting_time,
                self.movement_out_total,
                self.movement_in_total,
                self.steady_state_infected,
                self.emptying_time,
            ):
                dx[metric.__name__] = metric(target, null=np.nan)
            results.append(dx)

        return results