import datetime
import numpy as np
import igraph as ig
import network_conversion as ntc
import network_simulation as nts


def run_sims(
    simulation,
    n_size,
    n_reps = 30,
    outname = "temp_sims.h5"
):
    # assume simulation is already initialised via a .seed

    for nchains in [1, 5, 10, 30]:
        for rep in range(n_reps):
            for i in range(n_size):
                simulation.reset(soft=False)
                simulation.state[i] = nchains

                now = datetime.datetime.now()
                simdate = now.strftime('%y%m%d')
                simtime = now.strftime('%H%M%S')
                sim_index = i*n_reps + rep
                simid = f"{simdate}_{simtime}_{sim_index}"

                print(simid, end=' ', flush=True)

                simulation.simulate(until = 600, nostop=True)
                simulation.export_history(outname, identity=simid, seed=i, simdate=simdate, simtime=simtime, nchains=nchains)




def naive_case():
    graph = ig.Graph.Read_GraphML("small_case/graph.graphml")
    ordering = ntc.Ordering(
        {node['name']: node['capacity'] for node in graph.vs}
    )
    converter = ntc.NaiveStaticConverter(graph, ordering=ordering, time_span=28)

    prob_no_escape = np.zeros(len(ordering))
    parameters = converter.map_parameters(
        {
            'beta': 0.0,
            'prob_final_stay': prob_no_escape,
        }
    )

    full_sizes = 100_000 * np.ones_like(converter.ordering.sizes, dtype=int)

    simulation = nts.Simulation(full_size=full_sizes, parameters=parameters, dt=0.5)
    simulation.seed(n_seed_events=0, n_seed_number=0)

    run_sims(simulation, n_size=len(converter.ordering), n_reps=100, outname="small_case/simulations.h5") 

def improved_case():
    graph = ig.Graph.Read("small_case/bigraph.graphml") 
    ordering = ntc.Ordering(
        {node['name']: node['capacity'] for node in graph.vs}
    )
    converter = ntc.StaticConverter(graph, ordering=ordering, time_span=28, ordering_key='name')

    prob_no_escape = np.zeros(len(ordering))
    parameters = converter.map_parameters(
        {
            'beta': 0.0,
            'prob_final_stay': prob_no_escape
        }
    )

    full_sizes = 100_000 * np.ones_like(converter.ordering.sizes, dtype=int)

    simulation = nts.StaticWithHome(full_size=full_sizes, parameters=parameters, dt=0.5)
    simulation.seed(n_seed_events=0, n_seed_number=0)

    run_sims(simulation, n_size=len(converter.ordering), n_reps=100, outname="small_case/home_sims.h5")

func_maps = {
    'naive': naive_case,
    'improved': improved_case,
}
if __name__ == "__main__":
    from sys import argv
    
    if len(argv) > 1:
        _type = argv[1]
        if _type not in func_maps:
            raise ValueError(f"No simulation type [{_type}] defined. Valid types: {list(func_maps.keys())}")
    else:
        _type = 'naive'

    print("Running simulation type:", _type)

    func_maps[_type]()