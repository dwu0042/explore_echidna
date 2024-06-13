"""new_newtork_sim"""
import numpy as np
import network_conversion as netcon
import network_simulation as netsim

def create_snaphome_parameter_converter(ordering_file, snapshots_directory, kerpow=-0.6, trunc=26):

    ordering = netcon.Ordering.from_file(ordering_file)

    converter = netcon.SnapshotWithHomeConverter.from_directory(snapshots_directory, ordering=ordering, kernel_expo=kerpow, trunc=trunc)

    return converter

def init_params(converter: netcon.SnapshotWithHomeConverter, prob_final_file, beta=1.0):

    prob_final = netcon.ColumnDict.from_prob_final_file(prob_final_file)
    pf_arr = prob_final.organise_by(converter.ordering)

    parameters = converter.map_parameters({
        'beta': beta,
        'prob_final_stay': pf_arr,
    })

    return parameters

def init_zero_params(converter, prob_final_file):

    prob_final = netcon.ColumnDict.from_prob_final_file(prob_final_file)
    pf_arr = prob_final.organise_by(converter.ordering)
    pf_zero = np.zeros_like(pf_arr)

    return converter.map_parameters({
        'beta': 0.0,
        'prob_final_stay': pf_zero,
    })

def create_sim(converter: netcon.SnapshotWithHomeConverter, parameters, seeds=1, numinf_perseed=1, dt=1.0, track_movement=False):

    sim = netsim.SnapshotWithHome(
        full_size=converter.ordering.sizes,
        parameters=parameters,
        timings=converter.snapshot_times,
        dt=dt,
        track_movement=track_movement,
    )

    sim.seed(
        n_seed_events=seeds,
        n_seed_number=numinf_perseed,
    )

    return sim


def simulate_sim_and_record(simulation: netsim.SnapshotWithHome, until=100, nostop=False, outfile=None, with_movers=False):

    simulation.simulate(until=until, nostop=nostop)

    simulation.export_history(outfile, with_movers=with_movers)





if __name__ == "__main__":
    import datetime

    # conv = create_snaphome_parameter_converter(
    #     ordering_file="./concordant_networks/size_14.csv",
    #     snapshots_directory="./conc_tempo_14_detailed/",
    # )

    import pickle

    with open("snapconverter.pkl", 'rb') as ifp:
        conv = pickle.load(ifp)

    params = init_zero_params(
        conv, "./probability_of_final_stay_by_shuffled_campus.csv"
    )

    sim = create_sim(
        conv, parameters=params, seeds=0, numinf_perseed=0, track_movement=True,
    )

    REPEATS = 20
    for _ in range(REPEATS):
        for i, v in enumerate(conv.ordering.sizes):
            print(i, v, sep=',', end=' ', flush=True)

            sim.reset(soft=False)

            sim.state[i] = v

            outname = f"zero_ss_new_sims/sim_{i}_{datetime.datetime.now().strftime('%y%m%d_%H%M%S')}.npz"
            simulate_sim_and_record(sim, until=8*365, nostop=True, outfile=outname, with_movers=True)