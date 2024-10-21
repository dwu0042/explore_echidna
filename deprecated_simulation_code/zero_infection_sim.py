import numpy as np
import subprocess
import pathlib
import tarfile
import re
from functools import lru_cache

import graph_importer as gim
import on_network_simulation as ons
import do_simulation_on_network as dosim

def write_tracker_sim(sim:ons.TemporalNetworkSimulation, out):
    with open(out, 'wb') as out_fp:
        np.savez_compressed(out_fp, 
                 t=sim.ts,
                 history=sim.history,
                 mover_out=sim.mover_out,
                 mover_in=sim.mover_in)

def crunch(dir, output, prefix='sim_', clear=False):
    # assumes that files follow the pattern "dir/prefix*"
    # we use shell tar to avoid looping explicitly in tarfile library
    ret = subprocess.run(f"tar -czvf {dir}/{output}.tar.gz {dir}/{prefix}*", shell=True)
    ret.check_returncode()

    if clear:
        for path in pathlib.Path(dir).glob(f"{prefix}*"):
            path.unlink()
    return ret

class SimTarViewer():
    """Reads a .tar.gz file of multiple sims"""

    def __init__(self, path):
        self.tar = tarfile.open(path)
        names = self.tar.getnames()
        parts = [(int(re.match(r'.*sim_(\d+)_\d+_\d+.npz', name).group(1)), name) for name in names]
        self.names = [name for _,name in sorted(parts)]

    @lru_cache(20)
    def view(self, idx):
        return dict(np.load(self.tar.extractfile(self.names[idx])))

    def __getitem__(self, idx):
        return self.view(idx)
    
    def close(self):
        self.tar.close()

    def __enter__(self):
        return self
    
    def __exit__(self):
        self.close()

    def __iter__(self):
        for idx, _ in enumerate(self.names):
            yield self.view(idx)





if __name__ == "__main__":
    import datetime

    G = gim.make_graph("./concordant_networks/temponet_14_365.lgl")
    szs = dosim.find_sizes("./concordant_networks/size_14.csv", None)
    prob_final = dosim.load_prob_final_file("./probability_of_final_stay_by_shuffled_campus.csv")
    sz_arr = np.array([v for k,v in sorted(szs.items())])

    tsim = ons.TemporalNetworkSimulation(
        szs,
        G, 
        prob_final,
        parameters=(0.0, 0.0),
        dt=0.2,
        track_movers=True,
    )

    # seed required to set the .rng attr
    tsim.seed(0, 0)

    # need to adjust the probability of leakage via removal
    tsim.removal_rate = (1- tsim.prob_final_stay) * tsim.removal_rate
    tsim.prob_final_stay *= 0.0
    
    REPEATS = 5
    cur_delay = 0
    for delay in np.arange(18, 26*3, 4):
        tsim.delay(delay-cur_delay)
        cur_delay = delay
        for _ in range(REPEATS):
            for i, v in enumerate(sz_arr):
                print(i, v, sep=',', end=' ')

                # clear state to zero
                tsim.reset()
                tsim.state[:] = 0
                tsim._history[0] = tsim.state

                # force seed at place i
                tsim.state[i] = v

                # simulate
                tsim.simulate(5*365, nostop=True)

                # record
                write_tracker_sim(tsim, f"zero_temp_delay_sims/sim_{i}_{delay}_{datetime.datetime.now().strftime('%y%m%d_%H%M%S')}.npz")

        # crunch("zero_sims", f"batch_{datetime.datetime.now().strftime('%y%m%d_%H%M%S')}", clear=True)
