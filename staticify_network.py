import igraph as ig
from networkx import attracting_components
import polars as pl

import graph_importer as gim


def load_temporal_network(path):
    return gim.make_graph(path)


def convert_temporal_network_to_static_edge_list(
    temponet: ig.Graph, direct_threshold=4
) -> pl.DataFrame:
    """Convert a full temporal network into a dataframe of edges of the corresponding static network
    
    Differentiates between direct transfers and indirect transfers"""

    # extract network information
    edge_df = pl.from_dicts(
        [
            {
                "source_time": edge.source_vertex["time"],
                "source_loc": edge.source_vertex["loc"],
                "dest_time": edge.target_vertex["time"],
                "dest_loc": edge.target_vertex["loc"],
                "weight": edge["weight"],
            }
            for edge in temponet.es
        ]
    ).lazy()

    # add in column for direct transfer bool flag
    edge_df = edge_df.with_columns(
        direct=((pl.col("dest_time") - pl.col("source_time") < direct_threshold)),
        link_time=(pl.col('dest_time') - pl.col('source_time'))*pl.col('weight')
    )

    # combine edges across time
    edge_df = (
        edge_df.group_by("source_loc", "dest_loc", "direct")
        .sum()
        .select("source_loc", "dest_loc", "direct", "weight", "link_time")
        .group_by("source_loc", "dest_loc")
        .agg(
            pl.col("weight").filter(pl.col("direct")).alias("direct_weight"),
            pl.col("weight").filter(pl.col("direct").not_()).alias("indirect_weight"),
            pl.col("link_time").filter(pl.col("direct").not_()).alias("link_time"),
        )
        .with_columns(
            pl.col("direct_weight").list.sum(),
            pl.col("indirect_weight").list.sum(),
            pl.col("link_time").list.sum(),
        )
        .with_columns(
            (pl.col("link_time") / pl.col("indirect_weight")).alias('link_time')
        )
    )

    # collect and return
    return edge_df.collect()


def generate_graph_from_edge_list(edge_df: pl.DataFrame) -> ig.Graph:
    G = ig.Graph(directed=True)

    nodes = (
        edge_df.select(
            pl.col("source_loc")
            .implode()
            .list.set_union(pl.col("dest_loc").implode())
            .explode()
        )
        .to_series()
        .to_list()
    )
    node_index = {nd: i for i,nd in enumerate(nodes)}

    G.add_vertices(len(nodes), attributes={'node': nodes})

    mapped_edge_df = edge_df.with_columns(
        pl.col('source_loc').map_dict(node_index),
        pl.col('dest_loc').map_dict(node_index),
    )

    edge_list = mapped_edge_df.select('source_loc', 'dest_loc').to_dict()
    edges = zip(edge_list['source_loc'], edge_list['dest_loc'])

    edge_info = edge_df.to_dict()

    G.add_edges(
        es=list(edges),
        attributes=edge_info
    )

    return G
