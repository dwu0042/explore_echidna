from typing import Sequence
import netsim_summariser as nsu
import polars as pl
import numpy as np
import pickle
from matplotlib import pyplot as plt
from matplotlib import colors
import seaborn as sns
import hvplot.polars
import hvplot.networkx
import networkx as nx
import network_conversion as ntc
from scipy import stats
from collections import defaultdict

hvplot.extension('matplotlib')


def recompute_metrics(h5file):
    """Recompute a dataframe with the metrics from the raw simulation h5 file"""
    summariser = nsu.Summariser(h5file)
    raw_metrics = summariser.collect(no_move=False)
    metric_df = summariser.results_to_polars(raw_metrics)
    return metric_df

def unpickle(file):
    with open(file, 'rb') as fp:
        payload = pickle.load(fp)
    return payload

def finitize(arr, subs):
    """Replaces nan values in array with a substitute value"""
    return np.asanyarray([
        *arr[np.isfinite(arr)],
        *(np.ones_like(arr[np.isnan(arr)]) * subs),
    ])

def plot_ecdf(key, df, maxt, label, ax=None, color='C0'):
    """Plot the ECDF of a given hitting time"""
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot()
    
    kfrom, kto = key

    data = df.filter(pl.col('seed') == kfrom).select(pl.col(f"hitting_time_{kto}")).to_series().to_numpy()
    fdata = finitize(data, maxt)

    ecdf = ax.ecdf(fdata, color=color, label=label)

    return ax, ecdf

def plot_multiple_ecdf(key, dfs:Sequence[pl.DataFrame], maxt, labels: Sequence[str], ax=None):

    handles = []
    for i, (df, label) in enumerate(zip(dfs, labels)):
        ax, ecdf = plot_ecdf(key, df, maxt, label, ax=ax, color=f'C{i}')
        handles.append(ecdf)
    
    return ax, handles

def recompute_comparison_tests(df1, df2, nseeds=338, maxv=8*365+1, rescaling1=1, rescaling2=1):
    test_results = dict()

    for seed in range(nseeds):
        for cname in df1.select(pl.selectors.starts_with('hitting_time')).columns:
            cidx = int(cname.split('_')[-1])
            test_idx = (seed, cidx)
            if cidx == seed:
                continue
            
            samples = [
                df.filter(pl.col('seed') == seed).select(pl.col(cname) * rescaling).to_series().to_numpy()
                for df, rescaling in zip((df1, df2), (rescaling1, rescaling2))
            ]

            censored_samples = [
                stats.CensoredData(
                    uncensored=sample[np.isfinite(sample)],
                    right=np.ones_like(sample[np.isnan(sample)])*maxv
                )
                for sample in samples
            ]

            test_results[test_idx] = stats.logrank(*censored_samples, alternative='two-sided')
    
    return test_results
