from pathlib import Path
import numpy as np
import datetime
from echidna import network_conversion as conv
from echidna import network_simulation as simu

root = Path(__file__).resolve().parent.parent.resolve()

def preload():

    ordering = conv.Ordering.from_file(root / "data/concordant_networks/size_14_nu.csv")
    converter = conv.TemporalNetworkConverter.from_file(
        str(root / "data/concordant_networks/temponet_14_365.lgl"), 
        ordering=ordering,
        weight='weight',
    )

    return ordering, converter

def gen_params(converter):

    prob_final = conv.ColumnDict.from_prob_final_file(root/ "data/concordant_networks/probability_of_final_stay_by_shuffled_campus.csv")
    prob_final_arr = prob_final.organise_by(converter.ordering)
    prob_final_zero =  np.zeros_like(prob_final_arr)

    return converter.map_parameters({
        'beta': 0.0,
        'prob_final_stay': prob_final_zero,
    })

def create_sim(
        converter: conv.TemporalNetworkConverter, 
        parameters, 
        pseudo_capacity=False,
        seeds=0, 
        numinfperseed=0, 
        dt=1.0, 
        track_movement=True,
):

    if pseudo_capacity:
        # fake the size of the hospitals
        full_sizes = 100_000 * np.ones_like(converter.ordering.sizes, dtype=int)
    else:
        full_sizes = converter.ordering.sizes


    sim = simu.TemporalSim(
        full_size=full_sizes,
        parameters=parameters,
        dt=dt,
        num_times=converter.NT,
        discretisation_size=converter.DT,
        track_movement=track_movement,
    )

    sim.seed(
        n_seed_events=seeds,
        n_seed_number=numinfperseed,
    )

    return sim

def simulate_sim_and_record(simulation: simu.TemporalSim, simid, until=100, nostop=False, outfile=None, with_movers=False, **kwargs):

    simulation.simulate(until=until, nostop=nostop)

    simulation.export_history(outfile, identity=simid, with_movers=with_movers, **kwargs)




def run_sims(n_reps=20):

    ordering, converter = preload()
    parameters = gen_params(converter)
    sim = create_sim(converter, parameters=parameters, pseudo_capacity=True, dt=0.5)


    for _n in range(n_reps):
        for i,v in enumerate(converter.ordering.sizes):
            print(i, v, sep=',', end=' ', flush=True)

            sim.reset(soft=False)
            sim.state[i] = 30

            now = datetime.datetime.now()
            simdate = now.strftime('%y%m%d')
            simtime = now.strftime('%H%M%S')
            simid = (i, _n, simdate, simtime)
            outname = root / "simulations/zero_sims_resized/temporal/sim_all_30s.h5"
            simulate_sim_and_record(sim, simid=simid, until=8*365, nostop=True, outfile=outname, with_movers=True, simdate=simdate, simtime=simtime, seed=i)

if __name__ == "__main__":

    run_sims()