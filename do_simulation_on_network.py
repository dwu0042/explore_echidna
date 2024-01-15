# script

import graph_importer as gim
import examine_transfers_for_sizes as esz
import on_network_simulation as ons

import multiprocessing

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


def main(graph_file, snapshot_file, size_file=None, parameters=(0.2, 1./28)):
    
    temporal_sim = do_temporal_sim(graph_file, size_file, parameters)
    
    snapshot_sim = do_snapshot_sim(snapshot_file, size_file, graph_file, parameters)

    return temporal_sim, snapshot_sim

def multiproc_main(graph_file, snapshot_file, size_file=None, parameters=(0.2, 1./28)):

    base_graph = load_temporal_graph(graph_file)
    sizes = find_sizes(size_file, base_graph)
    snapshots = ons.SnapshotNetworkSimulation.load_snapshots(snapshot_file)

    ctx = multiprocessing.get_context('forkserver')




if __name__ == "__main__":
    main("./concordant_networks/temponet_14_365.lgl", "./conc_tempo_14_in", "./concordant_networks/size_14.csv")
