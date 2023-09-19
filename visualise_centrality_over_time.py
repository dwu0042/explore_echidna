import graph_importer as gim
import numpy as np
from matplotlib import pyplot as plt, colors
import polars as pl


def load_and_understand(graph_file):
    G = gim.make_graph(graph_file)
    stats = np.load(f"{graph_file}.stats.pkl", allow_pickle=True)

    raw_dict = {
            'loc': G.vs['loc'],
            'time': G.vs['time'],
    }

    raw_dict.update(stats)

    df = pl.from_dict(raw_dict).with_columns(pl.col('loc', 'time').cast(pl.Int32))

    return df

def temporal_plot(df, func='max'):
    tdf = getattr(df.groupby('time'), func)()
    xdf = tdf.drop('loc').with_columns(pl.exclude('time')/pl.exclude('time').max())

    plt.figure()
    timer = xdf.select('time').to_series().to_numpy()
    for col in ['harmonic', 'pagerank', 'betweenness', 'closeness']:
        plt.plot(timer, xdf.select(col).to_series().to_numpy(), 'o', label=col)
    plt.legend(loc='upper left', bbox_to_anchor=(1.01, 1))
    plt.tight_layout()

def squash_over_loc(df):
    xdf = df.groupby('loc').max().drop('time')
    xdf = xdf.with_columns(pl.exclude('loc')/pl.exclude('loc').max())

    xdf = xdf.sort('harmonic').with_row_count('rank')

    plt.figure()
    timer = xdf.select('rank').to_series().to_numpy()
    for col in ['harmonic', 'pagerank', 'betweenness', 'closeness']:
        plt.plot(timer, xdf.select(col).to_series().to_numpy(), 'o', label=col)
    plt.legend(loc='upper left', bbox_to_anchor=(1.01, 1))
    plt.tight_layout()


if __name__ == "__main__":
    from sys import argv

    if len(argv) > 1:
        df = load_and_understand(argv[1])
        temporal_plot(df, 'max')
        temporal_plot(df, 'mean')
        squash_over_loc(df)

        plt.show()
    else:
        print("enter a lgl file that you have parsed into stats")