import numpy as np
import igraph as ig
import polars as pl
from pathlib import Path
from itertools import pairwise, chain

def convert_distance_to_flow(G: ig.Graph):
    """invert weights. Needed to convert distance graph to weight graph for infomap"""
    G.es['weight'] = [1/k for k in G.es['weight']]

def infomap(G: ig.Graph, edge_weights='weight', assign=False):
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
        G['cluster'] = list(cluster_labels.astype(int))

    return cluster_labels

def multi_infomap(G_base: str|Path, conv=True):
    """
    G_base: folder that contains mulitple graphml snapshot graphs
    conv: if True, takes the reciprocal of weights
    """
    df = pl.DataFrame(schema={'name': pl.Int64})

    G_folder = Path(G_base)
    for graph_file in G_folder.glob("*.graphml"):
        # load graph
        G = ig.Graph.Read_GraphML(str(graph_file))

        clus = f'cluster_{graph_file.stem}'

        # prep weights on edges correctly
        if conv: convert_distance_to_flow(G)

        # compute clustering
        df_i = pl.from_dict({
            'name': G.vs['name'],
            clus: infomap(G, assign=False),
        }).with_columns(pl.col('name').cast(pl.Int64))

        # append to end of dataframe
        df = df.join(df_i, on='name', how='outer')

    # sort column order
    df = df.select(['name', *sorted(df.columns[1:])])

    return df

def map_multidf_to_sankey(df: pl.DataFrame):
    """ Brute-force map of a dataframe of clusterings of nodes to Sankey diagram
    """

    # compute n clusters, for sankey indexing
    n_clusters = df.max().to_numpy()[:,1:].flatten().cumsum()

    sankey_edges = pl.DataFrame(schema={
        'source': pl.Int32, 
        'target': pl.Int32, 
        'value': pl.Int32,
    })

    sankey_nodes = pl.DataFrame(schema={
        'node': pl.Int32,
        'x': pl.Float64,
        'y': pl.Float64,
    })

    # compute flows
    for i, ((n0, col0), (n1, col1)) in enumerate(pairwise(zip(chain([0], n_clusters), df.columns[1:]))):
        flows = (df.lazy().select(col0, col1)
                 .group_by('*')
                 .count()
                 .drop_nulls()
                 .select(
                     (pl.col(col0) + n0).alias('source').cast(pl.Int32),
                     (pl.col(col1) + n1).alias('target').cast(pl.Int32),
                     pl.col('count').alias('value').cast(pl.Int32),
                    )
        ).collect()
        sankey_edges = sankey_edges.vstack(flows)

    for i, (col, (n0, n1)) in enumerate(zip(df.columns[1:], pairwise(chain([0], n_clusters)))):
        nodes = (df.lazy()
                .select(col)
                .drop_nulls()
                .unique()
                .with_columns(
                    x=(i)/(len(df.columns)-1),
                    y=pl.col(col) / (n1 - n0),
                )
                .select(
                    (pl.col(col) + n0).alias('node').cast(pl.Int32),
                    pl.col('x').cast(pl.Float64),
                    pl.col('y').cast(pl.Float64),
                )).collect()
        sankey_nodes = sankey_nodes.vstack(nodes)

    return sankey_nodes, sankey_edges

def heat_wavelet(G: ig.Graph, tau=1.2, normalised=True):
    assert not G.is_directed(), "Laplacian not defined for directed networks"
    A = np.array(G.get_adjacency().data)
    D = np.diag(np.array(G.degree()))
    W = 1/np.sqrt(D)
    if normalised:
        L = np.eye(A.shape[0]) - W@A@W
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
