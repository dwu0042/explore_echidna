import polars as pl
import graph_importer as gim # imports igraph

def construct_readmission_counts(file="./concordant_networks/temponet_1_365.lgl") -> pl.DataFrame:

    G = gim.make_graph(file)

    counts = [
        {
            'readmission_time': edge.target_vertex['time'] - edge.source_vertex['time'],
            'count': edge['weight'],
        }
        for edge in G.es
    ]

    return pl.from_dicts(counts).with_columns(pl.col('count').cast(pl.Int64))


if __name__ == "__main__":
    
    df = construct_readmission_counts(file="./concordant_networks/temponet_1_365.lgl")
    df.write_parquet("./concordant_networks/readmission_time.parquet")
