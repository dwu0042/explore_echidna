import numpy as np
import igraph as ig
import polars as pl
from pathlib import Path
from collections import Counter
import argparse


def convert_distance_to_flow(G: ig.Graph):
    """invert weights. Needed to convert distance graph to weight graph for infomap"""
    G.es["weight"] = [1 / k for k in G.es["weight"]]


def infomap(G: ig.Graph, edge_weights="weight", assign=False, sort=True):
    """Performs infomap clustering on the graph, and then lables the clusters

    Returns
    -------
    LG: labelled graph
    """
    cluster_labels = -1 * np.ones((G.vcount(),), dtype=int)
    G_cc = G.connected_components()
    clus_idx_0 = 0
    for i, cc in enumerate(G_cc.subgraphs()):
        cz = cc.community_infomap(edge_weights=edge_weights)

        cluster_labels[G_cc[i]] = [m + clus_idx_0 for m in cz.membership]

        n_clus = len(cz.sizes())
        clus_idx_0 += n_clus

    if assign:
        G["cluster"] = list(cluster_labels.astype(int))

    # sort the clusters appropriately
    if sort:
        size_mapping = {
            c: (i + 1) for i, (c, _) in enumerate(Counter(cluster_labels).most_common())
        }
        cluster_labels = [size_mapping[c] for c in cluster_labels]

    return cluster_labels


def multi_infomap(G_base: str | Path, conv=True):
    """
    G_base: folder that contains mulitple graphml snapshot graphs
    conv: if True, takes the reciprocal of weights
    """
    df = pl.DataFrame(schema={"name": pl.Int64})

    G_folder = Path(G_base)
    for graph_file in G_folder.glob("*.graphml"):
        # load graph
        G = ig.Graph.Read_GraphML(str(graph_file))

        clus = f"cluster_{graph_file.stem}"

        # prep weights on edges correctly
        if conv:
            convert_distance_to_flow(G)

        # compute clustering
        df_i = pl.from_dict(
            {
                "name": G.vs["name"],
                clus: infomap(G, assign=False),
            }
        ).with_columns(pl.col("name").cast(pl.Int64))

        # append to end of dataframe
        df = df.join(df_i, on="name", how="outer")

    # sort column order
    df = df.select(["name", *sorted(df.columns[1:])])

    return df


def heat_wavelet(G: ig.Graph, tau=1.2, normalised=True):
    assert not G.is_directed(), "Laplacian not defined for directed networks"
    A = np.array(G.get_adjacency().data)
    D = np.diag(np.array(G.degree()))
    W = 1 / np.sqrt(D)
    if normalised:
        L = np.eye(A.shape[0]) - W @ A @ W
    else:
        L = D - A
    V, U = np.linalg.eig(L)
    S = np.exp(-tau * V)
    return U @ S @ U.T


def heat_wavelet_distance(G1: ig.Graph, G2: ig.Graph, tau=1.2, normalised=True):
    P1 = heat_wavelet(G1, tau=tau, normalised=normalised)
    P2 = heat_wavelet(G2, tau=tau, normalised=normalised)

    Del = P1 - P2
    *_, N = Del.shape
    DTD = Del.T @ Del

    return DTD.trace() / N


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_snapshots", metavar="FOLDER")
    parser.add_argument("method", choices=["infomap"])
    parser.add_argument(
        "-c",
        "--convert",
        action="store_true",
        help="if True, takes the reciprocal of the edge weights",
    )
    parser.add_argument("-o", "--output", default="cluster_by_name.csv")

    args = parser.parse_args()
    match args.method:
        case "infomap":
            clusters = multi_infomap(args.input_snapshots, conv=args.convert)
        case _:
            raise ValueError(f"Unknown method {args.method}")
    clusters.write_csv(args.output)


if __name__ == "__main__":
    main()
