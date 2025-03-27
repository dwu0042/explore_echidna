from typing import Iterable
import graph_importer as gim
import polars as pl
from scipy import stats

def extract_edge_df(G: gim.ig.Graph):
    """Extract a dataframe of edge weights w/ nodal attributes"""
    graph_data = [
        {
            'edge_weight': edge['weight'],
            'source_loc': edge.source_vertex['loc'],
            'sink_loc': edge.target_vertex['loc'],
            'source_time': edge.source_vertex['time'],
            'sink_time': edge.target_vertex['time'],
        }
        for edge in G.es
    ]

    df = (
        pl.from_records(graph_data)
        .with_columns(
            tau=pl.col('sink_time') - pl.col('source_time'),
        )
    )

    return df


def pull_series_tau_dist(df: pl.DataFrame):

    grouped_edges = df.group_by('source_time', 'tau').agg(pl.col('edge_weight').sum())
    return grouped_edges.partition_by('source_time')

def prep_taudist(df: pl.DataFrame):

    sorted_df = df.sort('tau').select(pl.col('tau').log(), pl.col('edge_weight').log())
    accept_df = sorted_df.filter(pl.col('tau').is_finite())

    tau = accept_df.select(pl.col('tau')).to_series().to_numpy()
    ews = accept_df.select(pl.col('edge_weight')).to_series().to_numpy()

    return tau, ews

def compute_tau_dists(df_series: Iterable[pl.DataFrame]):

    regression_results = []
    for df in df_series:
        key = df.select(pl.col('source_time').unique()).item()
        res = stats.linregress(*prep_taudist(df))
        regression_results.append(
            {
                'time': key,
                'slope': res.slope,
                'intercept': res.intercept,
                'rvalue': res.rvalue,
            }
        )

    return pl.from_dicts(regression_results)
