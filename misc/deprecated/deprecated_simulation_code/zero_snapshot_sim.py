import numpy as np
import subprocess
import pathlib
import tarfile
import re
from functools import lru_cache

import graph_importer as gim
import on_network_simulation as ons
import do_simulation_on_network as dosim

def write_tracker_sim(sim:ons.SnapshotNoveauSimulation, out):
    with open(out, 'wb') as out_fp:
        np.savez_compressed(out_fp, 
                 t=sim.ts,
                 history=sim.history,
                 mover_out=np.asanyarray(sim.mover_out).squeeze(),
                 mover_in=np.asanyarray(sim.mover_in).squeeze(),
        )


if __name__ == "__main__":
    import datetime
    import cProfile

    # G = gim.make_graph("./concordant_networks/temponet_14_365.lgl")
    szs = dosim.find_sizes("./concordant_networks/size_14.csv", None)
    prob_final_skele = dosim.load_prob_final_file("./probability_of_final_stay_by_shuffled_campus.csv")
    # mess with prob_final
    prob_final = {k: 0.0 for k in prob_final_skele}
    sz_arr = np.array([v for k,v in sorted(szs.items())])

    snapshots = ons.load_snapshots("./conc_tempo_14_detailed/")

    tsim = ons.SnapshotNoveauSimulation(
        szs,
        snapshots, 
        prob_final,
        parameters=(0.0, 0.0),
        dt=0.2,
        track_movers=True,
    )

    # seed required to set the .rng attr
    tsim.seed(0, 0)
    


    REPEATS = 20
    for _ in range(REPEATS):
        for i, v in enumerate(sz_arr):
            print(i, v, sep=',', end=' ', flush=True)

            # clear state to zero
            tsim.reset()
            tsim.state[:] = 0
            tsim._history[0] = tsim.state

            # force seed at place i
            tsim.state[i] = v

            # simulate
            tsim.simulate(8*365, nostop=True)

            # record
            write_tracker_sim(tsim, f"zero_ss_sims/sim_{i}_{datetime.datetime.now().strftime('%y%m%d_%H%M%S')}.npz")

    # crunch("zero_sims", f"batch_{datetime.datetime.now().strftime('%y%m%d_%H%M%S')}", clear=True)
