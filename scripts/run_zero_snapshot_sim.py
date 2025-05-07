"""new_newtork_sim"""
import numpy as np
from echidna import network_conversion as netcon
from echidna import network_simulation as netsim

def create_snaphome_parameter_converter(ordering_file, snapshots_directory):

    ordering = netcon.Ordering.from_file(ordering_file)

    converter = netcon.SnapshotWithHomeConverter.from_directory(snapshots_directory, ordering=ordering, infer_durations=True)

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

def create_sim(converter: netcon.SnapshotWithHomeConverter, parameters, seeds=1, numinf_perseed=1, dt=1.0, track_movement=False, pseudo_capacity=False):

    if pseudo_capacity:
        # fake the size of the hospitals
        full_sizes = 100_000 * np.ones_like(converter.ordering.sizes, dtype=int)
    else:
        full_sizes = converter.ordering.sizes

    sim = netsim.SnapshotWithHome(
        full_size=full_sizes,
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


def simulate_sim_and_record(simulation: netsim.SnapshotWithHome, simid, until=100, nostop=False, outfile=None, with_movers=False, **kwargs):

    simulation.simulate(until=until, nostop=nostop)

    simulation.export_history(outfile, identity=simid, with_movers=with_movers, **kwargs)





if __name__ == "__main__":
    import datetime
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent.resolve()

    conv = create_snaphome_parameter_converter(
        ordering_file= root / "data/concordant_networks/size_14_nu.csv",
        snapshots_directory= root / "data/conc_tempo_14_detailed/",
    )

    params = init_zero_params(
        conv, root / "data/concordant_networks/probability_of_final_stay_by_shuffled_campus.csv"
    )

    sim = create_sim(
        conv, parameters=params, seeds=0, numinf_perseed=0, track_movement=True, pseudo_capacity=True
    )

    REPEATS = 20
    for _ in range(REPEATS):
        print(end='\n', flush=True)
        for i, v in enumerate(conv.ordering.sizes):
            print(i, v, sep=',', end=' ', flush=True)

            sim.reset(soft=False) # non-soft reset sets all state to zero

            sim.state[i] = 30

            now = datetime.datetime.now()
            simdate = now.strftime('%y%m%d')
            simtime = now.strftime('%H%M%S')
            simid = int(simdate + simtime)
            outname = root / "simulations/zero_sims_resized/snapshot/sims_30s_pc_fa_r2.h5"
            simulate_sim_and_record(sim, simid=simid, until=8*365, nostop=True, outfile=outname, with_movers=True, simdate=simdate, simtime=simtime, seed=i)
