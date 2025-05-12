# assumes that metrics are computed
from os import PathLike
import numpy as np
from pathlib import Path
import polars as pl
from matplotlib import pyplot as plt
import seaborn as sns

from echidna import hitting_time_multichain as mch

import hitting_time_analysis as hta

root = Path(__file__).resolve().parent.parent.resolve()

def compute_aggregated_metrics():
    metrics_files = {
        folder.stem: folder / "metrics_30s.parquet"
        for folder in (root / "simulations/zero_sims_resized").iterdir()
    }

    metrics = {
        label: pl.read_parquet(file)
        for label , file in metrics_files.items()
    }

    _hitting_time_columns = pl.selectors.starts_with("hitting_time_")
    agg_metrics = dict()
    hitting_time_dists = dict()
    extent_dists = dict()

    for model, df in metrics.items():
        agg_metric = (df
                    .unpivot(
                        on=_hitting_time_columns,
                        index=['seed', 'extent'],
                        variable_name='target',
                        value_name='hitting_time',
                    )
                    .with_columns(
                        target_seed = (
                            pl.col('target')
                            .str.strip_prefix('hitting_time_')
                            .str.to_integer()
                        )
                    )
                    .drop('target')
                    ) 
        agg_metrics[model] = agg_metric
        hitting_time_dists[model] = np.sort(agg_metric.select("hitting_time").to_series().to_numpy())
        extent_dists[model] = np.sort(agg_metric.select('extent').to_series().to_numpy())

    return {
        'agg': agg_metrics, 
        'hitting': hitting_time_dists, 
        'extent': extent_dists,
    }

def plot_hitting_time_dists(hitting_time_dists, ax=None):

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot()
    for model, dist in hitting_time_dists.items():
        ax.plot(dist, np.linspace(0, 1, len(dist), endpoint=False), label=model)

    ax.set_xlabel('hitting time')
    ax.set_ylabel('ECDF')

    ax.legend(loc='lower right')

    return ax

def compute_analytical_naive_static_hitting(outfile: str | PathLike | None = None, n_chains=30, scaling=1e3):

    Q = hta.get_naive_static_Q()
    hitting_times = mch.compute_all_multichain_hittings(
        Nchains=n_chains,
        Q=Q,
        scaling=scaling,
    )
    M, N = hitting_times.shape
    MM, NN = np.meshgrid(np.arange(M), np.arange(N), indexing='ij')
    analytical_hitting_df = pl.from_dict(
        {
            'source': MM.ravel(),
            'target': NN.ravel(),
            'hitting_time': hitting_times.ravel(),
        }
    )

    if outfile is not None:
        analytical_hitting_df.write_csv(outfile)

def plot_analytical_hitting(analytical_hitting_df, ax=None):
    
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot()
    
    analytical_hitting_arr = np.sort(
        analytical_hitting_df.select('hitting_time').to_series().to_numpy()
    )

    ax.plot(
        analytical_hitting_arr, 
        np.linspace(0, 1, len(analytical_hitting_arr)),
        linestyle='dashed',

    )

    return ax


def main_plot():

    agg_metrics = compute_aggregated_metrics()
    ax = plot_hitting_time_dists(agg_metrics['hitting'])

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    analytical_df = pl.read_csv(root / "outputs/hitting_time_analysis/numer_hitting_30s_nu_size.csv")
    ax = plot_analytical_hitting(analytical_df, ax=ax)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    ax.figure.savefig("outputs/zero_hitting_time_dur_upd.png")
    return ax

if __name__ == "__main__":

    analytical_file = root / "outputs/hitting_time_analysis/numer_hitting_30s_nu_size.csv"
    compute_analytical_naive_static_hitting(analytical_file)
    ax = main_plot()