import polars as pl
import igraph as ig
from typing import Mapping
import numpy as np
import sparse  # github.com/pydata/sparse
import glob
from collections import defaultdict


def extract_all_nodes(path: str):
    TN = ig.Graph.Read_Ncol(path)
    return {
        k: i
        for i, k in enumerate(
            sorted({int(x.split(",")[0].lstrip("(")) for x in TN.vs["name"]})
        )
    }


def export_all_nodes(all_nodes, outpath: str):
    pl.from_dict(
        {
            "key": all_nodes.keys(),
            "value": all_nodes.values(),
        }
    ).write_csv(outpath)


def import_all_nodes(path: str):
    basedict = pl.read_csv(path).to_dict(as_series=False)
    return dict(zip(basedict["key"], basedict["value"]))


def permutation_matrix(G: ig.Graph, all_nodes: Mapping):
    """Get a matrix that can map the reduced adjacency matrix of graph G to the full adjacency matrix represented by the order in all_nodes

    The idea is that we can left and right multiply the adjaceny matrix from G of size m x m
    with two matrices P and P.T of size n x m and m x n resp.
    so that we get a full n x n matrix"""

    available_nodes = [int(x) for x in G.vs["name"]]
    this_index = np.arange(len(available_nodes))
    that_index = [all_nodes[i] for i in available_nodes]
    P = sparse.COO(
        [this_index, that_index],
        np.ones_like(available_nodes),
        shape=(len(available_nodes), len(all_nodes)),
    )
    return P


def presence_matrix(G: ig.Graph, all_nodes: Mapping):
    P = permutation_matrix(G, all_nodes)
    # get out-degree adjacency matrix
    A = sparse.COO.from_scipy_sparse(G.get_adjacency_sparse())
    return P.T @ A @ P


def coherence_matrix(snapshot_folder, all_nodes):
    presences = []
    for snapshot in glob.glob(f"{snapshot_folder}/*.graphml"):
        G = ig.Graph.Read_GraphML(snapshot)
        A = presence_matrix(G, all_nodes)
        presences.append(A)

    return sparse.stack(presences, axis=2)


def temporal_coherence_distribution(C: sparse.COO):
    coherence = C.sum(axis=2).todense()  # collapses time axis
    valid_coherence = coherence[coherence > 0]  # this autocollapses the 2D matrix to 1D
    return valid_coherence / C.shape[-1]


def waiting_times(C: sparse.COO):
    # get the presence dict for piling
    presences = np.unravel_index(C.linear_loc(), C.shape)
    presence_df = defaultdict(list)
    for i, j, v in zip(*presences):
        presence_df[(i, j)].append(v)
    # convert to raw waiting times, filter singleton events
    waiting_df = {k: np.diff(sorted(v)) for k, v in presence_df.items() if len(v) > 1}
    return waiting_df


def burstiness(v):
    m = np.mean(v)
    s = np.std(v)
    return (s - m) / (s + m)


def burstiness_distribution(C: sparse.COO):
    W = waiting_times(C)
    distribution = [
        burstiness(v) for (k0, k1), v in W.items() if len(v) > 1 if k0 != k1
    ]
    return distribution
