# test the static probability theory

from typing import Hashable
import numpy as np
import polars as pl
import igraph as ig
import warnings

from collections import Counter

from pytz import BaseTzInfo

import staticify_network as stcfy
import projection_metrics as projm

import network_conversion as nconv
import network_simulation as nsiml

def make_graph_from_name_edges(name_edges, vertex_names=None):

    if vertex_names is None:
        vertex_names = sorted({name for vertices in name_edges.keys() for name in vertices})
    if isinstance(vertex_names, list | set):
        vertex_map = [{"name": name, "id": i} for i, name in enumerate(vertex_names)]
    elif isinstance(vertex_names, dict):
        vertex_map = [
            {"name": name, "id": i, **values}
            for i, (name, values) in enumerate(vertex_names.items())
        ]
    else:
        raise TypeError("vertex_names provided is not a list, set or dict")
    vertex_map_df = pl.from_dicts(vertex_map)

    edge_df = pl.from_dicts(
        [
            {"source_vertex": v0, "target_vertex": v1, **values}
            for (v0, v1), values in name_edges.items()
        ]
    )

    mapped_edge_df = (
        edge_df.join(
            vertex_map_df.select("name", "id"), left_on="source_vertex", right_on="name"
        )
        .rename({"id": "source_id"})
        .join(
            vertex_map_df.select("name", "id"), left_on="target_vertex", right_on="name"
        )
        .rename({"id": "target_id"})
        .with_columns(
            edge_name=pl.concat_list("source_vertex", "target_vertex"),
            edge_id=pl.concat_list("source_id", "target_id"),
        )
    )

    edge_list = mapped_edge_df.select("edge_id").to_series().to_list()

    vertex_attrs = vertex_map_df.sort("id").drop("id").to_dict(as_series=False)
    edge_attrs = mapped_edge_df.to_dict(as_series=False)

    return ig.Graph(
        n=len(vertex_map),
        edges=edge_list,
        directed=True,
        edge_attrs=edge_attrs,
        vertex_attrs=vertex_attrs,
    )


def basic_example():
    locs = ["A", "B", "C"]
    times = list(range(20))
    vertex_info = {
        f"{loc}{time}": {"loc": loc, "time": time} for time in times for loc in locs
    }

    edge_info = {
        ("A0", "C0"): {"weight": 1.0},
        **{(f"A{i}", f"B{i+1}"): {"weight": 1.0} for i in range(19)},
        ("A10", "C15"): {"weight": 1.0},
    }

    ordering = {loc: 10 for loc in locs}

    return {
        'graph': make_graph_from_name_edges(edge_info, vertex_names=vertex_info),
        'ordering': ordering
    }


def collapse_static(G: ig.Graph):
    # here we abuse the fact that we keep loc and time attrs on vertices
    return stcfy.temporal_to_static(
        G, direct_threshold=1
    )

def get_snapshots(G: ig.Graph):
    # use existing snapshot code
    times = sorted(set(G.vs["time"]))
    durations = np.diff(times)

    snapshots  = [projm.bothbound_snapshot(G, t, aggmode='detailed') for t in times[:-1]]

    for snapshot, t, dt in zip(snapshots, times, durations):
        snapshot['time'] = t
        snapshot['duration'] = dt

    return {snapshot['time']: snapshot for snapshot in snapshots}

def simulate_on(G: ig.Graph | dict[Hashable, ig.Graph], ordering, type='temporal', repeats=30):
    ordering = nconv.Ordering(ordering)
    prob_zero = np.array([0.0 for _ in ordering])
    base_params = {
        'beta': 0.0,
        'prob_final_stay': prob_zero
    }

    sim_hists = []
    move_in = []
    move_out = []
    match type:
        case "temporal":
            converter = nconv.TemporalNetworkConverter(
                G, ordering=ordering
            )
            parameters = converter.map_parameters(base_params)
            sim = nsiml.TemporalSim(
                full_size=converter.ordering.sizes,
                parameters=parameters,
                dt = 1.0,
                num_times=converter.NT,
                discretisation_size=converter.DT,
                track_movement=True,
            )
        case "snapshot":
            converter = nconv.SnapshotWithHomeConverter(G, ordering=ordering)
            parameters = converter.map_parameters(base_params)
            sim = nsiml.SnapshotWithHome(
                full_size=converter.ordering.sizes,
                parameters=parameters,
                timings=converter.snapshot_times,
                track_movement=True,
                dt=1.0
            )
        case "static":
            converter = nconv.StaticConverter(G, ordering=ordering, time_span=20)
            parameters = converter.map_parameters(base_params)
            sim = nsiml.StaticWithHome(
                full_size = converter.ordering.sizes,
                parameters=parameters,
                dt =1.0,
                track_movement=True
            )
        case _:
            raise ValueError(f"invalid type: {type}")
    sim.seed(n_seed_events=0, n_seed_number=0)
    for repeat in range(repeats):
        sim.reset(soft=False)
        sim.state[0] = 1
        sim.simulate(until=19, nostop=True)
        sim_hists.append(sim.history)
        move_in.append(sim.mover_in)
        move_out.append(sim.mover_out)
    return {
        'hists': sim_hists,
        'move_out': move_out,
        'move_in': move_in,
        'converter': converter,
        'sim': sim,
    }


def summarise_hist_outcome(hists):
    return Counter(tuple(np.where(s[:,-1])[0]) for s in hists)





if __name__ == "__main__":

    example = basic_example()
    G = example['graph']
    sizes = example['ordering']

    S = collapse_static(G)

    Ps = get_snapshots(G)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

        static_hists = simulate_on(S, ordering=sizes, type='static', repeats=1000)
        print('Static', summarise_hist_outcome(static_hists['hists']))

        snapshot_hists = simulate_on(Ps, ordering=sizes, type='snapshot', repeats=1000)
        print('Snapshot', summarise_hist_outcome(snapshot_hists['hists']))

        temporal_hists = simulate_on(G, ordering=sizes, type='temporal', repeats=1000)
        print('Temporal', summarise_hist_outcome(temporal_hists['hists']))