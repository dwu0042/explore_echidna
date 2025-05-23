from os import PathLike
from typing import Mapping
from pathlib import Path
import numpy as np
import igraph as ig
import polars as pl
from matplotlib import pyplot as plt, colors
from scipy import sparse

from echidna import network_conversion as ntc
from echidna import hitting_markov as hit
from echidna import examine_transfers_for_sizes as esz

root = Path(__file__).resolve().parent.parent.resolve()

def graph_read(filename: str):
    return ig.Graph.Read(filename)

## NAIVE STATIC MODEL
def get_naive_static_Q(
    graph: str | PathLike | ig.Graph = root / "data/concordant_networks/shuf_static_network.graphml",
    size_file: str | PathLike = root / "data/concordant_networks/size_14_nu.csv",
    duration_file: str | PathLike = root / "data/concordant_networks/durations_1.csv",
    default_time_span = (10*365),
    ordering_key = 'name',
    value_key = 'id',
):
    """input graph is either the path name or the grpah object"""

    if isinstance(graph, ig.GraphBase):
        G = graph
    else:
        G = ig.Graph.Read(graph)

    graph_map = {int(k):v for k,v in zip(G.vs[ordering_key], G.vs[value_key])}
    sorted_graph_map = {k: graph_map[k] for k in sorted(graph_map.keys())}
    graph_ordering = ntc.Ordering(sorted_graph_map)

    size_order = ntc.Ordering.from_file(size_file)
    sizes = np.array([size_order(i) for i in graph_ordering.order])

    durations = esz.quick_read(duration_file)
    duration_values = np.array([durations.get(i, default_time_span) 
                                for i in graph_ordering.order])

    denominator = sizes * duration_values

    # we use the naive construction, since the null set (never return) version creates 
    # results that misrepresent the amount of time it takes to move through the system
    # We also use the internal function in order to directly pass the transition matrix 
    # constructed, as opposed to the non-scaled weighted adjacency matrix
    T = ntc.transition_matrix_from_graph(G, 
                                        ordering=graph_ordering,
                                        scaling_per_node=denominator,
                                        ordering_key=ordering_key, 
                                        adjacency_attribute='weight',
                                        )
    Q = hit._Q_mat_naive(T)

    return Q


def compute_static_hitting_times(
    Q,
    plot = False,
):

    N, _ = Q.shape

    ## I wish I could speed this up, but I think we need the absorption
    hitting_times = [
        hit.solve_hitting_time(Q, [i]) for i in range(N)
    ]

    hitting_time_arr = np.hstack([h.reshape((-1, 1)) for h in hitting_times])

    ax = []
    if plot:
        axim = plt.matshow(hitting_time_arr, norm=colors.LogNorm())
        plt.colorbar(label='hitting time (days)')
        plt.xlabel('to')
        plt.ylabel('from')

        ax = [axim.axes]

    # plt.savefig("zstatic_no_home_time.png", dpi=360)
    return hitting_time_arr, ax


### STATIC MODEL WITH DELAYED RETURN


def extract_delayed_static_Qmat(
    graph: str|ig.Graph = "./concordant_networks/trimmed_static_base_1_threshold_4.graphml",
    size_mapping: str|Mapping = "./concordant_networks/size_14.csv",
    scaling = (10*365),
    eps = 1e-7,
    ordering_key = 'node', 
    value_key = 'id',
    cast_int = False, 
):
    if isinstance(graph, ig.GraphBase):
        Z = graph
    else:
        Z = ig.Graph.Read(graph)

    base_ordering_arr = Z.vs[ordering_key]
    if cast_int:
        ordering_arr = list(map(int, base_ordering_arr))
    else:
        ordering_arr = base_ordering_arr
    zgraph_map = dict(zip(ordering_arr, Z.vs[value_key]))
    zsorted_graph_map = {k: zgraph_map[k] for k in sorted(zgraph_map.keys())}
    zgraph_ordering = ntc.Ordering(zsorted_graph_map)
    zN = len(zgraph_ordering.sizes)

    if isinstance(size_mapping, Mapping):
        size_ordering = ntc.Ordering(size_mapping)
    else:
        size_ordering = ntc.Ordering.from_file(size_mapping)
    sizes = np.array([size_ordering(i) for i in zgraph_ordering.order])

    T_direct = ntc.transition_matrix_from_graph(Z, 
                                                ordering=zgraph_ordering,
                                                scaling_per_node=sizes,
                                                global_scaling=scaling, 
                                                ordering_key=ordering_key, 
                                                adjacency_attribute='direct_weight'
                                            )
    T_indirect = ntc.transition_matrix_from_graph(Z, 
                                                ordering=zgraph_ordering, 
                                                scaling_per_node=sizes,
                                                global_scaling=scaling, 
                                                ordering_key=ordering_key, 
                                                adjacency_attribute='indirect_weight'
                                            )
    T_return = ntc.transition_matrix_from_graph(Z, 
                                                ordering=zgraph_ordering, 
                                                ordering_key=ordering_key, 
                                                adjacency_attribute='link_time'
                                            )

    # build expanded state transition matrix
    # this matrix is usually too large to be an explicit array in memory e.g.
    # T = np.zeros((zN+zN**2, zN+zN**2))
    # so we will construct a sparse matrix
    # to prevent degeneracy (rank-deficiency), we will have one row per node and one row per edge

    # extract sparse matrix constructor data for direct (node-node) components
    T_direct_sparse = sparse.coo_array(T_direct)
    T_xs = list(T_direct_sparse.row)
    T_ys = list(T_direct_sparse.col)
    T_data = list(T_direct_sparse.data)

    MACHINE_EPS = eps

    # indirect transfers
    idx = zN - 1
    edge_table = dict()
    Tix, Tiy = [mat.flatten() for mat in np.indices(T_indirect.shape)]
    for (x,y,v) in zip(Tix, Tiy, T_indirect.flatten()):
        if np.abs(v) > MACHINE_EPS:
            idx += 1
            edge_table[(x, y)] = idx
            T_xs.append(x)
            T_ys.append(idx)
            T_data.append(v)


    # indirect return, map link times to number of observations
    Trx, Try = [mat.flatten() for mat in np.indices(T_indirect.shape)]
    for x,y,v in zip(Trx, Try, T_return.flatten()):
        if np.abs(v) > MACHINE_EPS:
            T_xs.append(edge_table[(x,y)])
            T_ys.append(y)
            T_data.append(1/v)

    T = sparse.coo_array((T_data, (T_xs, T_ys)))

    R = hit._Q_mat_sparse(T).tocsr()

    return R

def eval_delayed_static_hitting_times(R, plot=False):

    zN = R.shape[0]

    zhitting_times = [
        hit.solve_hitting_time(R, [i]) for i in range(zN)
    ]

    zhitting_time_arr = np.hstack([h.reshape((-1, 1))[:zN,:] for h in zhitting_times])

    axs = []
    if plot:
        axim1 = plt.matshow(zhitting_time_arr, norm=colors.LogNorm())
        plt.colorbar(label='hitting time (days)')
        plt.xlabel('to')
        plt.ylabel('from')
        # plt.savefig("z_z_s.png", dpi=360)
        ax1 = axim1.axes

        zresidence_times = -1/R.diagonal()
        plt.figure()
        ax2 = plt.hist(np.log10(zresidence_times), bins=31)
        plt.yscale('log')
        plt.xlabel('$log_{10}$ residence time (log-days)')
        # plt.savefig('zstatic_no_home_residence.png', dpi=360)

        axs = [ax1, ax2]

    return zhitting_time_arr, axs


def compute_delayed_static_hitting_times(
    graph_file = "./concordant_networks/trimmed_static_base_1_threshold_4.graphml",
    scaling = (10*365),
    eps = 1e-7,
    plot = False,
):
    R = extract_delayed_static_Qmat(
        graph_file=graph_file,
        scaling=scaling,
        eps=eps,
    )

    return eval_delayed_static_hitting_times(R, plot=plot)

def main():

    hitting_time_arr, _ = compute_static_hitting_times()
    zhitting_time_arr, _ = compute_delayed_static_hitting_times()

    # we know relative change is usually quite large
    relative_change = zhitting_time_arr / (hitting_time_arr+1) 
    plt.matshow(relative_change)
    plt.colorbar(label='relative change in hitting time')
    plt.xlabel('to')
    plt.ylabel('from')
    # plt.savefig("zrelative_change_hitting_time_home.png", dpi=360)

    plt.figure()
    plt.hist(relative_change.flatten(), bins=31)
    plt.xlabel("relative change")
    plt.ylabel("frequency")
    plt.yscale('log')



    # direct_movement_total_out = T_direct.sum(axis=1)
    # direct_movement_total_in = T_direct.sum(axis=0)

    # indirect_movement_total_out = T_indirect.sum(axis=1)
    # indirect_movement_total_in = T_indirect.sum(axis=0)


    # large_indices = np.unravel_index(np.argsort(relative_change, axis=None)[::-1][:5], relative_change.shape)
    # large_index_list = np.array(large_indices).T


    # plt.figure()
    # plt.scatter(direct_movement_total_out, direct_movement_total_in)
    # plt.xscale('log')
    # plt.yscale('log')


    # for indices, colour in zip(large_index_list, (f"C{i+1}" for i in range(5))):
    #     xs = direct_movement_total_out[list(indices)]
    #     ys = direct_movement_total_in[list(indices)]
    #     plt.annotate(
    #         "",
    #         xy = (xs[1], ys[1]),
    #         xytext = (xs[0], ys[0]),
    #         arrowprops={'headwidth': 6, 'facecolor': colour}
    #     )


if __name__ == "__main__":
    main()

def create_cached_hitting_times(
    fpath = "hitting_time_analysis/hitting_time_arrays.h5"
):
    naive_static_Q = get_naive_static_Q()
    hitting_time_arr, _ = compute_static_hitting_times(naive_static_Q)
    zhitting_time_arr, _ = compute_delayed_static_hitting_times()

    import h5py

    with h5py.File(fpath, 'w') as h5fp:
        h5fp.create_dataset('naive_static', data=hitting_time_arr)
        h5fp.create_dataset('delayed_static', data=zhitting_time_arr)