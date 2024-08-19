import h5py
import polars as pl
import numpy as np
import multiprocessing
from itertools import repeat

class Summariser():

    def __init__(self, file):
        self.target_file = file

    @staticmethod
    def hitting_time(target : h5py.Group, null=np.nan, **kwargs):
        mask = (target['history'][:] != 0)
        return list(np.where(mask.any(axis=1), mask.argmax(axis=1), null))

    @staticmethod
    def movement_out_total(target: h5py.Group, **kwargs):
        return list(target['mover_out'][:].sum(axis=1))
    
    @staticmethod
    def movement_in_total(target: h5py.Group, **kwargs):
        return list(target['mover_in'][:].sum(axis=1))

    @staticmethod
    def steady_state_infected(target: h5py.Group, **kwargs):
        init = target['history'][:,0].sum()
        normed = target['history'][:].sum(axis=0) / init
        return [np.mean(normed), np.std(normed)]
    
    @staticmethod
    def emptying_time(target: h5py.Group, null=np.nan, **kwargs):
        xs = target['history'][:].sum(axis=0)
        mask = (xs == 0)
        return mask.argmax() if mask.any() else null

    @property
    def groups(self):
        with h5py.File(self.target_file, 'r') as fp:
            grps = list(fp)
        return grps

    def compute_metrics(self, target_str, no_move=False, verbose=False):
        if verbose:
            print(target_str)

        with h5py.File(self.target_file, 'r') as fp:

            info = fp[target_str]

            dx = dict(info.attrs)

            metrics = [
                self.hitting_time,
                self.steady_state_infected,
                self.emptying_time,
            ]
            if not no_move:
                metrics += [
                    self.movement_out_total,
                    self.movement_in_total,
                ]

            for metric in metrics:
                    dx[metric.__name__] = metric(info, null=np.nan)

            return dx

    def collect(self, over=None, ncpus=4, no_move=False, verbose=False):
        if over is None:
            over = self.groups

        with multiprocessing.Pool(processes=ncpus) as pool:
            results = pool.starmap(self.compute_metrics, zip(over, repeat(no_move), repeat(verbose)))

        return results
    
    @staticmethod
    def results_to_polars(results, drop=('movement_out_total', 'movement_in_total')):

        df = pl.from_dicts(results).drop(*drop)
        return (
            df
            .with_columns(
                pl.col('hitting_time').list.to_struct(
                    fields=lambda idx: f"hitting_time_{idx}"
                ),
                pl.col('steady_state_infected').list.to_struct(
                    fields=['ssi_mean', 'ssi_std']
                )
            )
            .unnest(
                'hitting_time',
                'steady_state_infected',
            )
            .fill_nan(None)
        )
    