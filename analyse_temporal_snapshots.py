# this will rely on clustering labelling

import polars as pl
import itertools
from sklearn import metrics as sklm
from matplotlib import pyplot as plt

def import_clusters(cluster_file = "conc_snapshots_deprc/conc_tempo_14_snapshots/results/infomap_clusters.csv"):
    return pl.read_csv(cluster_file)





def main():

    clusters = import_clusters()

    adjusted_clusters = clusters.fill_null(-1)

    persistence_functions = {
        'rand': sklm.adjusted_rand_score,
        'mutual_info': sklm.adjusted_mutual_info_score,
    }

    cl_persist_metric = {
        score_name: [
            score_function(
                adjusted_clusters.select(cl_left).to_series().to_numpy(),
                adjusted_clusters.select(cl_right).to_series().to_numpy(),
            )
            for cl_left, cl_right in itertools.pairwise(adjusted_clusters.columns[1:])
        ]
        for score_name, score_function in persistence_functions.items()
    }

    for score_name, score in cl_persist_metric.items():
        plt.plot(score, marker='.', label=score_name)
    plt.legend()





if __name__ == "__main__":
    main()
