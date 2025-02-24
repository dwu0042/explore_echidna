import datetime
import numpy as np
import igraph as ig
import network_conversion as ntc
import network_simulation as nts

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

for nchains in [1, 5, 10, 30]:
    for rep in range(N_REPS:= 100):
        for i,v in enumerate(converter.ordering.sizes):
            
            simulation.reset(soft=False)
            simulation.state[i] = nchains

            now = datetime.datetime.now()
            simdate = now.strftime('%y%m%d')
            simtime = now.strftime('%H%M%S')
            sim_index = i*N_REPS + rep
            simid = f"{simdate}_{simtime}_{sim_index}"
            outname = "small_case/simulations.h5"

            print(simid, end=' ', flush=True)

            simulation.simulate(until = 600, nostop=True)
            simulation.export_history(outname, identity=simid, seed=i, simdate=simdate, simtime=simtime, nchains=nchains)