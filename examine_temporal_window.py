import polars as pl
import graph_importer as gim
from matplotlib import pyplot as plt

def load_graph(path):
    return gim.make_graph(path)

def hometime(graph):
    data = [{
                'fromloc': e.source_vertex['loc'], 
                'fromtime': e.source_vertex['time'], 
                'toloc': e.target_vertex['loc'], 
                'totime': e.target_vertex['time'], 
                'weight': e['weight']
            } for e in graph.es]

    df = pl.from_dicts(data)

    hometime = (df
                .filter(pl.col('fromloc').ne(pl.col('toloc')))
                .with_columns((pl.col('totime') - pl.col('fromtime')).alias('hometime'))
                .group_by('hometime').agg(pl.col('weight').sum().alias('count'))
                .sort('hometime')
    )

    return hometime

def plot_hometime(hometime, ax=None):
    if ax is None:
        fig, ax = plt.subplots()

    ax.loglog(
        hometime.select('hometime').to_series(),
        hometime.select('count').to_series(),
        marker='o',
        linestyle='none',
    )

    return ax


if __name__ == "__main__":
    import sys
    G = load_graph(sys.argv[1])
    ht = hometime(G)
    plot_hometime(ht)
    plt.show()