import h5py
import polars as pl
import numpy as np
import multiprocessing
from itertools import repeat, permutations
from functools import wraps
from scipy import stats

class Summariser():
    """Utility class to sumamrise metrics from an h5 database of simulation outputs"""

    def __init__(self, file):
        self.target_file = file

    @staticmethod
    def _sanitise(fn):
        @wraps(fn)
        def sanitised_fn(*args, **kwargs):
            return list(fn(*args, **kwargs))
        return sanitised_fn

    @staticmethod
    @_sanitise
    def hitting_time(target : h5py.Group, null=np.nan, **kwargs):
        """Yields list of hitting times to all locations"""
        mask = (target['history'][:] != 0)
        indices = np.where(mask.any(axis=1), mask.argmax(axis=1), null)
        times = np.nan * np.empty_like(indices)
        times[np.isfinite(indices)] = target['ts'][:][indices[np.isfinite(indices)].astype(int)]
        return times

    # @staticmethod
    # def extent(target: h5py.Group, time_limit=30, **kwargs):
    #     """Yields the number of locations hit by the time limit"""
    #     hitting_times = np.array(Summariser.hitting_time)
    #     return (
    #         sum(np.where(mask.any(axis=1), mask.argmax(axis=1), np.nan) < time_limit, -1),
    #         time_limit,
    #     )

    @staticmethod
    @_sanitise
    def movement_out_total(target: h5py.Group, **kwargs):
        """Yields the list of number of movements out of all locations"""
        return list(target['mover_out'][:].sum(axis=1))

    @staticmethod
    @_sanitise
    def movement_in_total(target: h5py.Group, **kwargs):
        """Yields the list of number of movements into all locations"""
        return list(target['mover_in'][:].sum(axis=1))

    @staticmethod
    def steady_state_infected(target: h5py.Group, **kwargs):
        """Yields the mean and standard deviation of the number of infected individuals (over locs)"""
        init = target['history'][:,0].sum()
        normed = target['history'][:].sum(axis=0) / init
        return [np.mean(normed), np.std(normed)]

    @staticmethod
    def emptying_time(target: h5py.Group, null=np.nan, **kwargs):
        """Yields the first time that no location has any infected individuals"""
        xs = target['history'][:].sum(axis=0)
        mask = (xs == 0)
        return mask.argmax() if mask.any() else null

    @property
    def groups(self):
        """The list of simulation keys (h5 groups)"""
        with h5py.File(self.target_file, 'r') as fp:
            grps = list(fp)
        return grps

    def compute_metrics(self, target_str, no_move=False, verbose=False, **kwargs):
        """Computes (all) metrics on a target simulation given by the key target_str
        
        Args:
            no_move: if True, does not return the movement metrics
            verbose: if True, also prints the target string to stdout
            **kwargs: keywords to pass into the metrics
                all metrics must have a **kwargs to gracefully ignore extra kwargs
        """
        if verbose:
            print(target_str)

        with h5py.File(self.target_file, 'r') as fp:

            info = fp[target_str]

            dx = dict(info.attrs)

            metrics = [
                self.hitting_time,
                self.steady_state_infected,
                self.emptying_time,
                # self.extent,
            ]
            if not no_move:
                metrics += [
                    self.movement_out_total,
                    self.movement_in_total,
                ]

            for metric in metrics:
                dx[metric.__name__] = metric(info, **kwargs)

            return dx
 
    def _packed_compute_metrics(self, target_str, no_move=False, verbose=False, kwargs=None):
        """helper function to pass kwargs to compute_metrics for paralllel starmapping"""
        if kwargs is None:
            kwargs = dict()

        return self.compute_metrics(target_str, no_move=no_move, verbose=verbose, **kwargs)

    def collect(self, over=None, ncpus=4, no_move=False, verbose=False, **kwargs):
        """Collects metrics over the given subset of simulations
        
        Args:
            over: groups/simulations to compute metrics over; if None, computes metrics over all groups
            ncpus: number of workers to use in parallel
            no_move: if True, does not compute movement metrics
            verbose: if True, prints the group being computed to stdout
            kwargs: arguments to pass to metrics
        """
        if over is None:
            over = self.groups

        with multiprocessing.Pool(processes=ncpus) as pool:
            results = pool.starmap(
                self._packed_compute_metrics,
                zip(over, repeat(no_move), repeat(verbose), repeat(kwargs)),
            )

        return results

    @staticmethod
    def results_to_polars(results, drop=('movement_out_total', 'movement_in_total'), strict_drop=True):
        """Forms the results from .collect into a polars dataframe.
        
        Args:
            drop: iterable of keys to drop
            strict_drop: if True, the keys to drop must exist, and an Exception is raised otherwise; 
                does not check if False 
        """
        metric_schema_overrides = {
            'hitting_time': pl.List(pl.Float64),
            'steady_state_infected': pl.List(pl.Float64),
            'emptying_time': pl.Float64,
            'movement_out_total': pl.List(pl.Float64),
            'movement_in_total': pl.List(pl.Float64),
        }

        df = pl.from_dicts(results, schema_overrides=metric_schema_overrides).drop(*drop, strict=strict_drop)
        return (
            df
            .with_columns(
                pl.col('hitting_time').list.to_struct(
                    fields=lambda idx: f"hitting_time_{idx}"
                ),
                pl.col('steady_state_infected').list.to_struct(
                    fields=['ssi_mean', 'ssi_std']
                ),
                # pl.col('extent').list.to_struct(
                #     fields=['extent', 'extent_time']
                # )
            )
            .unnest(
                'hitting_time',
                'steady_state_infected',
                # 'extent',
            )
            .fill_nan(None)
        )

    def metrics(self, over=None, ncpus=4, no_move=False, verbose=False, drop=(), strict_drop=True, **kwargs):

        results = self.collect(over=over, ncpus=ncpus, no_move=no_move, verbose=verbose, **kwargs)
        df = self.results_to_polars(results, drop=drop, strict_drop=strict_drop)

        return Metrics(df)


class Metrics(pl.DataFrame):

    def add_extent(self, extent_time=30):
        return self.with_columns(
            pl.concat_list(pl.selectors.starts_with('hitting_time'))
            .list.eval(pl.element().lt(extent_time))
            .list.sum().sub(1)
            .alias('extent'),
            pl.lit(extent_time).alias('extent_time')
        )


def hitting_time_comparison(df_left, df_right, maxv=8*365+1):

    test_results = []

    seeds = df_left.select(pl.col('seed').unique()).to_series().to_list()
    path_pairs = permutations(seeds, 2)
    for path_pair in path_pairs:
        seed_id, target_id = path_pair
        samples = [
            df.filter(pl.col('seed') == seed_id)
            .select(pl.col(f"hitting_time_{target_id}"))
            .to_series().to_numpy()

            for df in (df_left, df_right)
        ]

        censored_samples = [
            stats.CensoredData(
                uncensored=sample[np.isfinite(sample)],
                right=np.ones_like(sample[np.isnan(sample)])*maxv
            )
            for sample in samples
        ]

        raw_result = stats.logrank(*censored_samples, alternative='two-sided')
        test_results.append({
            'seed': seed_id,
            'target': target_id,
            'pvalue': raw_result.pvalue,
            'statistic': raw_result.statistic,
        })

    result_df = pl.from_dicts(test_results)

    return result_df
