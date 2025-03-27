import numpy as np
import graph_importer as gim
import polars as pl
import igraph as ig
import glob
import pathlib
import warnings
from matplotlib import pyplot as plt

_metric_dict = {
    "pagerank": lambda G: G.pagerank(weights="weight"),
    "harmonic": lambda G: G.harmonic_centrality(weights="weight"),
    "betweenness": lambda G: G.betweenness(weights="weight"),
    "closeness": lambda G: G.closeness(weights="weight"),
    "eigenvector": lambda G: G.eigenvector_centrality(weights="weight"),
    "hubscore": lambda G: G.hub_score(weights="weight"),
    "authority": lambda G: G.authority_score(weights="weight"),
}


def outbound_snapshot(graph: ig.Graph, time, aggmode="sum"):
    included_vertices = graph.vs.select(time_eq=time)
    included_locations = sorted(set(included_vertices["loc"]))  # sorted returns list
    # loc_lookup = {loc: i for i,loc in enumerate(included_locations)}
    included_edges = graph.es.select(_source_in=included_vertices)

    return snapshot_from_locations_and_edges(
        time, included_locations, included_edges, aggmode
    )


def inbound_snapshot(graph: ig.Graph, time, aggmode="sum"):
    included_vertices = graph.vs.select(time_eq=time)
    included_locations = sorted(set(included_vertices["loc"]))  # sorted returns list
    # loc_lookup = {loc: i for i,loc in enumerate(included_locations)}
    included_edges = graph.es.select(_target_in=included_vertices)

    return snapshot_from_locations_and_edges(
        time, included_locations, included_edges, aggmode
    )


def bothbound_snapshot(graph: ig.Graph, time, aggmode="detailed"):
    included_vertices = graph.vs.select(time_eq=time)
    included_locations = sorted(set(included_vertices["loc"]))  # sorted returns list
    # loc_lookup = {loc: i for i,loc in enumerate(included_locations)}
    included_edges = graph.es.select(_incident_eq=included_vertices)

    return snapshot_from_locations_and_edges(
        time, included_locations, included_edges, aggmode
    )


def snapshot_from_locations_and_edges(
    time, included_locations, included_edges, aggmode="sum"
):
    edge_df = aggregate_edges(included_edges, time, aggmode)

    vertices = [{"name": location} for location in included_locations]

    snapshot = ig.Graph.DictList(vertices, edge_df, directed=True)

    return snapshot


def aggregate_edges(included_edges, time, method="sum"):
    if len(included_edges) < 1:
        warnings.warn(f"No valid edges at time {time}.")
        edge_info = []
    else:
        edge_info = pl.from_dicts(
            [
                {
                    "source": e.source_vertex["loc"],
                    "target": e.target_vertex["loc"],
                    "weight": e["weight"],
                    "diff_time": e.source_vertex["time"] != e.target_vertex["time"],
                    "source_now": e.source_vertex["time"] == time,
                    "target_now": e.target_vertex["time"] == time,
                }
                for e in included_edges
            ]
        ).group_by("source", "target")

        if method == "sum":
            edge_df = edge_info.sum().to_dicts()
        elif method == "invsum":
            edge_df = edge_info.agg(pl.col("weight").pow(-1).sum().pow(-1)).to_dicts()
        elif method == "detailed":
            # assume only with both-type projection
            edge_df = edge_info.agg(
                pl.col("weight")
                .filter(pl.col("diff_time").not_())
                .sum()
                .alias("weight"),
                pl.col("weight")
                .filter(pl.col("diff_time") & pl.col("source_now"))
                .sum()
                .alias("departures"),
                pl.col("weight")
                .filter(pl.col("diff_time") & pl.col("target_now"))
                .sum()
                .alias("arrivals"),
            ).to_dicts()
        else:
            raise NotImplemented(
                f"Unknown aggregation method for edge weights: {method}"
            )

    return edge_df


_snapshot_mode = {
    "out": outbound_snapshot,
    "in": inbound_snapshot,
    "both": bothbound_snapshot,
}


def make_snapshots(graph: ig.Graph, output_dir, mode="out", aggmode="sum"):
    times = sorted(set(graph.vs["time"]))
    durations = np.diff(times)
    if len(durations) < 1:
        raise ValueError("Not enough times to construct sane snapshots")

    snapper = _snapshot_mode[mode]

    for t, dt in zip(times, durations):
        snapshot = snapper(graph, t, aggmode=aggmode)
        snapshot['duration'] = dt
        snapshot.write_graphml(f"{output_dir}/{t:04}.graphml")


def read_snapshots(directory):
    snapshots = dict()
    for file in glob.glob((f"{directory}/*.graphml")):
        filepath = pathlib.Path(file)
        time = int(filepath.stem)
        snapshots[time] = ig.Graph.Read_GraphML(file)
    return snapshots


def metric_over_snapshots(snapshots: dict[int, ig.Graph], metric: str):
    metric_fn = _metric_dict[metric]
    dataset = [
        {
            str(name): metric_value
            for name, metric_value in zip(snap.vs["name"], metric_fn(snap))
        }
        for _time, snap in sorted(snapshots.items())
        if len(snap.es) > 0
    ]

    return pl.from_dicts(dataset)


def rank_metrics(metric_df: pl.DataFrame, ordered=True, descending=True, fill=False):
    dfx = metric_df.transpose().with_columns(
        pl.all().rank("min", descending=descending)
    )

    if fill:
        if fill is True:
            fill = dfx.shape[0] + 1
        dfx = dfx.fill_null(fill)

    if ordered:
        ranking_mean = dfx.mean(axis=1)

    dfx = dfx.with_columns(pl.Series(metric_df.columns).alias("loc"))

    if ordered:
        dfx = dfx.with_columns(meanrank=ranking_mean).sort("meanrank").drop("meanrank")

    return dfx


def snapshot_metric_ranks(
    snapshots: dict[int, ig.Graph],
    metric: str,
    ordered=True,
    descending=True,
    fill=False,
):
    metric_df = metric_over_snapshots(snapshots, metric)

    return rank_metrics(metric_df, ordered=ordered, descending=descending, fill=fill)


def plot_rankings(
    metric_ranks, ax=None, cmap="magma_r", fillcolor=None, cbar=True, labels=True
):
    metric_arr = metric_ranks.drop("loc").to_numpy()
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot()
    image = ax.imshow(metric_arr, cmap=cmap, interpolation="nearest")
    if labels:
        ax.set_xlabel("Time")
        ax.set_ylabel("Campus")
    if fillcolor is not None:
        ax.set_facecolor(fillcolor)
    if cbar:
        cb = ax.get_figure().colorbar(image, label="Ranking")
        cb.ax.invert_yaxis()
    return ax


def ranking_plot(
    snapshots: dict[int, ig.Graph],
    metric: str,
    ordered=True,
    descending=True,
    fill=False,
    ax=None,
    cmap="magma_r",
    fillcolor=None,
    cbar=True,
    labels=True,
):
    metric_ranks = snapshot_metric_ranks(
        snapshots, metric, ordered=ordered, descending=descending, fill=fill
    )
    plot_rankings(
        metric_ranks, ax=ax, cmap=cmap, fillcolor=fillcolor, cbar=cbar, labels=labels
    )


def multiranking_plot(
    snapshots, metrics, shape=(2, 2), timeunit="fortnights", **kwargs
):
    fig, azs = plt.subplots(*shape)
    axs = azs.flatten()

    for metric, ax in zip(metrics, axs):
        ranking_plot(snapshots, metric, ax=ax, cbar=False, labels=False, **kwargs)
        ax.set_title(metric.title())

    fig.supxlabel(f"Time [{timeunit}]")
    fig.supylabel("Campus")
    cb = fig.colorbar(ax.get_images()[0], ax=azs, label="Ranking")
    cb.ax.invert_yaxis()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("snapshot", "plot", "values"))
    parser.add_argument("input_path")
    parser.add_argument("-o", "--output", dest="output", default=None)
    parser.add_argument(
        "-s",
        "--snapshot_type",
        dest="snaptype",
        choices=_snapshot_mode.keys(),
        default="out",
    )
    parser.add_argument("-a", "--aggregation", dest="aggmode", default="sum")
    parser.add_argument(
        "-m", "--metric", dest="metric", choices=_metric_dict.keys(), default="harmonic"
    )
    parser.add_argument("-x", "--fillcolor", dest="fillcolor", default=None)
    parser.add_argument("--fill", action="store_true")
    parser.add_argument("-g", "--graphs", action="store_true")

    args = parser.parse_args()

    if args.mode == "snapshot":
        G = gim.make_graph(args.input_path)
        if args.output is None:
            base = args.input_path.replace(".", "_")
            output_dir = f"{base}_snapshots/"
        else:
            output_dir = args.output
        pathlib.Path(output_dir).mkdir(exist_ok=True, parents=True)

        mode = args.snaptype

        snapshots = make_snapshots(G, output_dir, mode=mode, aggmode=args.aggmode)
        print(f"{mode} snapshots ({args.aggmode}) made to {output_dir}, success")

    if args.mode == "plot":
        if args.graphs:
            input_path = args.input_path + "/graphs/"
        else:
            input_path = args.input_path
        snapshots = read_snapshots(input_path)
        ranking_plot(
            snapshots, metric=args.metric, fill=args.fill, fillcolor=args.fillcolor
        )
        if args.output is None:
            output_path = f"{args.input_path}/ranking_{args.metric}.png"
        else:
            output_path = args.output
        plt.savefig(output_path, dpi=600)

    if args.mode == "values":
        if args.graphs:
            input_path = args.input_path + "/graphs/"
        else:
            input_path = args.input_path
        snapshots = read_snapshots(input_path)
        values = metric_over_snapshots(snapshots, metric=args.metric)
        if args.output is None:
            outpath = f"{args.input_path}/values_{args.metric}.csv"
        else:
            outpath = args.output
        values.write_csv(outpath)
