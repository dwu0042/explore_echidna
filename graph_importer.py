import igraph as ig


def make_graph(filename: str) -> ig.Graph:
    G = ig.Graph.Read_Ncol(filename)
    G.vs["loc"] = [int(x.split(",")[0].strip("()")) for x in G.vs["name"]]
    G.vs["time"] = [int(x.split(",")[1].strip("()")) for x in G.vs["name"]]
    return G
