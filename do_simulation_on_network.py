# script

import graph_importer as gim
import examine_transfers_for_sizes as esz
import on_network_simulation as ons

import multiprocessing
import pickle

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

def run_temporal_sim(G, sizes, parameters, num_seed_events=3, num_seed_indvs=5):

    temporal_sim = ons.TemporalNetworkSimulation(
        hospital_size_mapping=sizes,
        network=G,
        parameters=parameters,
        dt=1.0,
    )

    temporal_sim.seed(num_seed_events, num_seed_indvs)
    temporal_sim.simulate(5 * 365)
    return temporal_sim

def do_temporal_sim(graph_file, size_file=None, parameters=(0.2, 1./28)):

    G = load_temporal_graph(graph_file)
    szs = find_sizes(size_file, G)
    return run_temporal_sim(G, szs, parameters)


def run_snapshot_sim(sizes, snapshots, parameters, num_seed_events=3, num_seed_indvs=5):
    
    snapshot_sim = ons.SnapshotNetworkSimulation(
        hospital_size_mapping=sizes,
        snapshots=snapshots,
        parameters=(*parameters, 1.0),
        dt=1.0,
    )

    snapshot_sim.seed(num_seed_events, num_seed_indvs)
    snapshot_sim.simulate(5 * 365)
    return snapshot_sim

def do_snapshot_sim(snapshot_file, size_file=None, graph_file=None, parameters=(0.2, 1./28)):
    if size_file is None and graph_file is None:
        raise ValueError("Both size_file and graph_file cannnot be None")
    
    if size_file is None:
        G = load_temporal_graph(graph_file)
        sizes = find_sizes(size_file, G)
    else:
        sizes = find_sizes(size_file, None)

    snapshots = ons.SnapshotNetworkSimulation.load_snapshots(snapshot_file)

    snapshot_sim = run_snapshot_sim(sizes, snapshots, parameters)
    return snapshot_sim


def main(graph_file, snapshot_file, size_file=None, parameters=(0.2, 1./365, 1./28)):
    
    temporal_sim = do_temporal_sim(graph_file, size_file, parameters)
    
    # snapshot_sim = do_snapshot_sim(snapshot_file, size_file, graph_file, parameters)

    # return temporal_sim, snapshot_sim
    return temporal_sim

def run_batched_temporal_simulations(graph_file, size_file=None, parameters=(0.2, 1./365, 1./28), seed_options=(3, 5), n_runs=10):

    graph = gim.make_graph(graph_file)
    sizes = find_sizes(size_file, graph)

    return [run_temporal_sim(graph, sizes, parameters, *seed_options) for _ in range(n_runs)]

def run_batched_snapshot_simulations(snapshot_file, size_file=None, graph_file=None, parameters=(0.2, 1./28), seed_options=(3, 5), n_runs=10):

    snapshots = ons.SnapshotNetworkSimulation.load_snapshots(snapshot_file)
    assert not (size_file is None and graph_file is None), "Need to provide either the hospital size file, or a temporal graph file."
    sizes = find_sizes(size_file, graph_file)

    return [run_snapshot_sim(sizes, snapshots, parameters, *seed_options) for _ in range(n_runs)]

class SimulationResult():
    def __init__(self, record):
        self.ts = record['ts']
        self.history = record['history']

def export_batched_realisations(realisations: list[ons.Simulation], metadata, output_path):
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

if __name__ == "__main__":
    main("./concordant_networks/temponet_14_365.lgl", "./conc_tempo_14_in", "./concordant_networks/size_14.csv")
