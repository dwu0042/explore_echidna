# script

import graph_importer as gim
import examine_transfers_for_sizes as esz
import on_network_simulation as ons
import igraph as ig
import numpy as np
import polars as pl
import multiprocessing
import pickle
from typing import Sequence

def load_temporal_graph(graph_file):
    return gim.make_graph(graph_file)

def find_sizes(size_file: None|str, temporal_graph, write=True):
    if size_file is None:
        szs = esz.rough_hospital_size(temporal_graph, 0, 5)
        if write:
            temp_file = "rough_size.temp.csv"
            esz.quick_write(szs, temp_file)
            print(f"Write rough hospital size file to {temp_file}")
    else:
        szs = esz.quick_read(size_file)

    return szs

def run_temporal_sim(G, sizes, prob_final_stay, parameters, num_seed_events=3, num_seed_indvs=5):

    temporal_sim = ons.TemporalNetworkSimulation(
        hospital_size_mapping=sizes,
        network=G,
        parameters=parameters,
        prob_final_stay=prob_final_stay,
        dt=1.0,
    )

    temporal_sim.seed(num_seed_events, num_seed_indvs)
    temporal_sim.simulate(5 * 365)
    temporal_sim = SimulationResult.minify(temporal_sim)
    return temporal_sim

def do_temporal_sim(graph_file, size_file=None, prob_final_file=None, parameters=(0.2, 1./28)):

    G = load_temporal_graph(graph_file)
    szs = find_sizes(size_file, G)
    prob_final_stay = load_prob_final_file(prob_final_file)
    return run_temporal_sim(G, szs, prob_final_stay, parameters)


def run_snapshot_sim(sizes, snapshots, prob_final, parameters, num_seed_events=3, num_seed_indvs=5):
    
    # snapshot_sim = ons.SnapshotNetworkSimulation(
    snapshot_sim = ons.SnapshotNoveauSimulation(
        hospital_size_mapping=sizes,
        snapshots=snapshots,
        prob_final=prob_final,
        parameters=(*parameters, 1.0),
        dt=1.0,
    )

    snapshot_sim.seed(num_seed_events, num_seed_indvs)
    snapshot_sim.simulate(5 * 365)

    # squeeze out excess memory
    snap_result = SimulationResult.minify(snapshot_sim)
    return snap_result

def load_prob_final_file(prob_final_file):
    prob_df = pl.read_csv(prob_final_file)
    return {hosp: prob for hosp, prob in zip(
        prob_df.select('loc').to_series().to_list(),
        prob_df.select('final_stay').to_series().to_list(),
    )}

def do_snapshot_sim(snapshot_file, size_file=None, graph_file=None, prob_final_file=None, parameters=(0.2, 1./28)):
    if size_file is None and graph_file is None:
        raise ValueError("Both size_file and graph_file cannnot be None")
    
    if size_file is None:
        G = load_temporal_graph(graph_file)
        sizes = find_sizes(size_file, G)
    else:
        sizes = find_sizes(size_file, None)

    snapshots = ons.load_snapshots(snapshot_file)
    prob_final = load_prob_final_file(prob_final_file)

    snapshot_sim = run_snapshot_sim(sizes, snapshots, prob_final, parameters)
    return snapshot_sim

def run_batched_temporal_simulations(graph_file, size_file=None, prob_final_file=None, parameters=(0.2, 1./365, 1./28), seed_options=(3, 5), n_runs=10):

    graph = gim.make_graph(graph_file)
    sizes = find_sizes(size_file, graph)
    prob_final = load_prob_final_file(prob_final_file)

    return [run_temporal_sim(graph, sizes, prob_final, parameters, *seed_options) for _ in range(n_runs)]

def run_batched_snapshot_simulations(snapshot_file, size_file=None, graph_file=None, prob_final_file=None, parameters=(0.2, 1./28), seed_options=(3, 5), n_runs=10):

    snapshots = ons.load_snapshots(snapshot_file)
    assert not (size_file is None and graph_file is None), "Need to provide either the hospital size file, or a temporal graph file."
    sizes = find_sizes(size_file, graph_file)
    prob_final = load_prob_final_file(prob_final_file)

    return [run_snapshot_sim(sizes, snapshots, prob_final, parameters, *seed_options) for _ in range(n_runs)]

def run_static_sim(graph, sizes, prob_final, parameters, n_seed_events, n_seeds_per_event):
    ordering = {hospital: i for i, (hospital, size) in enumerate(sorted(sizes.items()))}
    ordered_sizes = [size for hospital,size in sorted(sizes.items())]


    transition_matrix = ons.transition_matrix_from_graph(
        graph,
        ordering=ordering,
        scaling_per_node=ordered_sizes,
        global_scaling=(10*365),
        ordering_key='name',
        adjacency_attribute='weight',
        matrix_size=len(ordering)
    )

    prob_final_arr = np.array([prob_final[h] for h in sorted(sizes.keys())])

    sim = ons.Simulation(ordered_sizes, transition_matrix, prob_final_arr, parameters=parameters, dt=1)
    sim.seed(n_seed_events, n_seeds_per_event)
    sim.simulate(5 * 365)
    return sim

def run_static_simulations(static_graph_file, size_file=None, graph_file=None, prob_final_file=None, parameters=(0.2, 1./28), seed_options=(3, 5), n_runs=10):

    G = ig.Graph.Read_GraphML(static_graph_file)
    assert not (size_file is None and graph_file is None), "Need to provide either the hospital size file, or a temporal graph file."
    sizes = find_sizes(size_file, graph_file)
    prob_final = load_prob_final_file(prob_final_file)

    return [run_static_sim(G, sizes, prob_final, parameters, *seed_options) for _ in range(n_runs)]


class SimulationResult():
    def __init__(self, record):
        self.ts = record['ts']
        self.history = record['history']

    @classmethod
    def minify(cls, fullsim: ons.Simulation):
        return cls({
            'ts': fullsim.ts,
            'history': fullsim.history
        })

def export_batched_realisations(realisations: Sequence[ons.Simulation|SimulationResult], metadata, output_path):
    with open(output_path, 'wb') as fp:
        pickle.dump({
            'parameters': metadata['parameters'],
            'seeds': metadata['seeds'],
            'records': [{'ts': x.ts, 'history': x.history} for x in realisations]
        }, fp)

def import_batched_realisations(input_file):
    """Map the dictionary elements back to namespace-like objects"""
    with open(input_file, 'rb') as fp:
        results = pickle.load(fp)
    
    return {
        'parameters': results['parameters'],
        'seeds': results['seeds'],
        'records': [SimulationResult(record) for record in results['records']]
    }

def main(graph_file, snapshot_file, size_file=None, prob_file=None, parameters=(0.2, 1./365, 1./28)):
    import time
    # temporal_sim = do_temporal_sim(graph_file, size_file, parameters)
    
    # snapshot_sim = do_snapshot_sim(snapshot_file, size_file, graph_file, parameters)

    # return temporal_sim, snapshot_sim
    # return temporal_sim
    snapshot_sims = run_batched_snapshot_simulations(
        snapshot_file=snapshot_file,
        size_file=size_file,
        graph_file=graph_file,
        prob_final_file=prob_file,
        parameters=parameters,
        n_runs=20,
    )

    now = time.strftime("%Y_%m_%d_%H_%M", time.localtime())


    export_batched_realisations(snapshot_sims, 
                                {'parameters': parameters, 'seeds': (3, 5)},
                                f"conc_14_simulations/{now}.sim")


if __name__ == "__main__":
    main("./concordant_networks/temponet_14_365.lgl", "./conc_tempo_14_detailed", "./concordant_networks/size_14.csv", "probability_of_final_stay_by_shuffled_campus.csv", parameters=(0.2,))

    main("./concordant_networks/temponet_14_365.lgl", "./conc_tempo_14_detailed", "./concordant_networks/size_14.csv", "probability_of_final_stay_by_shuffled_campus.csv", parameters=(0.1,))

    main("./concordant_networks/temponet_14_365.lgl", "./conc_tempo_14_detailed", "./concordant_networks/size_14.csv", "probability_of_final_stay_by_shuffled_campus.csv", parameters=(0.05,))

