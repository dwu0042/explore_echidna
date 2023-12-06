import igraph as ig
from collections import defaultdict

def determine_self_loops(G: ig.Graph):
    self_weights = defaultdict(int)
    for loc in set(G.vs['loc']):
        nds = G.vs.select(loc_eq=loc)
        self_weights[loc] = sum(G.es.select(_within=nds)['weight'])
    return self_weights

def determine_out_edges(G:ig.Graph):
    node_weights = {v.index: sum(e['weight'] for e in v.out_edges()) for v in G.vs}
    loc_weights = {loc: sum(node_weights[v.index] for v in G.vs.select(loc_eq=loc)) for loc in set(G.vs['loc'])}
    return loc_weights

def determine_self_prop(G: ig.Graph):
    self_loops = determine_self_loops(G)
    out_edges = determine_out_edges(G)
    return {loc: self_loops[loc]/out_edges[loc] for loc in self_loops}

def rough_hospital_size(G: ig.Graph):
    PROP_READMIT = 0.42 # persons/admission
    CHURN_RATE = 1 # hospitals/day
    TIME_SCALE = 3652 # days
    return {loc: (sz / PROP_READMIT / TIME_SCALE / CHURN_RATE) for loc,sz in determine_out_edges(G).items()}