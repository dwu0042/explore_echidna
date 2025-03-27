import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colors
from matplotlib import animation
import seaborn as sns
import polars as pl
from polars import selectors as sel
from itertools import pairwise
import argparse


def bend_line(y_from: float, y_to: float, n_grid: int = 100):
    """Return an array that represents a bent line
    (from y_from to y_to) with lenght n_grid
    """

    N = int(n_grid // 3)
    M = n_grid + 2 * N - 2
    Z = int(M // 2)

    line_base = np.array(Z * [y_from] + Z * [y_to])
    window = 1 / N * np.ones(N)
    line = np.convolve(line_base, window, mode="valid")
    line = np.convolve(line, window, mode="valid")

    return line


def sankey_diagram(
    data: pl.DataFrame,
    save_to: str | None = None,
    block_scale=1.0,
    block_gap=0.1,
    level_separation=20.0,
    export="figure",
):
    """
    Creates a sankey diagram
    data should be structured as

    | name | cluster_1 | cluster_2 | cluster_3 | ...

    where named agents move from location_1 to location_2, for example.
    A location_* value can be Null (None), in which case they are not part of the system at that point in time

    We will return a matplotlib.pyplot Figure instance that contains a single Axes that has the Sankey diagram.
    """

    # Build block cache for each location
    # 1. Determine the size at each column
    cluster_size = (
        data.select(pl.exclude("name"))
        .melt(variable_name="snapshot")
        .group_by(pl.all())
        .count()
        .drop_nulls()
        .pivot(
            values="count", index="value", columns="snapshot", aggregate_function=None
        )
        .rename({"value": "cluster_id"})
        .sort("cluster_id")
        .with_columns(
            (pl.exclude("cluster_id") * block_scale * (1 + block_gap)).name.map(
                lambda x: f"{x}_size"
            ),
        )
    )

    # 2. For each column compute where the sankey blocks start and end
    cluster_blocks = cluster_size.with_columns(
        # cluster_block_end
        sel.ends_with("_size")
        .cumsum()
        .name.map(lambda x: f"{x.rstrip('_size')}_block_end"),
    ).with_columns(
        # cluster_block_start
        sel.ends_with("_block_end")
        .shift(1, fill_value=0)
        .name.map(lambda x: f"{x.strip('_end')}_start"),
    )
    # 2a. mapping a separate dataframe to subtract concordant columns (very non-scalable)
    cluster_blocks_prime = cluster_size.with_columns(
        # subtract the gap
        (pl.all() * 0).name.map(lambda x: x),
        (pl.exclude("cluster_id", "^.*_size$") * block_scale * block_gap).name.map(
            lambda x: f"{x}_block_end"
        ),
        (pl.exclude("cluster_id", "^.*_size$") * 0.0).name.map(
            lambda x: f"{x}_block_start"
        ),
    )

    cluster_blocks = (cluster_blocks - cluster_blocks_prime).select(
        "cluster_id", sel.ends_with("_block_start"), sel.ends_with("_block_end")
    )

    # 2c. resize the block gap so that we can use it later
    cluster_size = cluster_size.with_columns(pl.col("^.*_size$") / (1 + block_gap))

    if export == "cluster_size":
        return cluster_size
    print("blocks determined")

    # 3. generate the dataframe that has the amount of flow between clusters
    # also join so that we get the actual start and end of each flow to bend_line

    flows = pl.DataFrame(schema={"from": pl.Int64, "to": pl.Int64})

    for col0, col1 in pairwise(data.columns[1:]):
        base_name = col0.split("_")[1]
        this_flow = (
            data.select(col0, col1)
            .group_by(pl.all())
            .count()
            .rename({"count": base_name, col0: "from", col1: "to"})
            .drop_nulls()
            # attach size_left and size_right
            .join(
                cluster_size.select(
                    pl.col("cluster_id"),
                    pl.col(col0).alias("tot_left"),
                    pl.col(f"{col0}_size").alias("size_left"),
                ),
                left_on="from",
                right_on="cluster_id",
                how="left",
            )
            .join(
                cluster_size.select(
                    pl.col("cluster_id"),
                    pl.col(col1).alias("tot_right"),
                    pl.col(f"{col1}_size").alias("size_right"),
                ),
                left_on="to",
                right_on="cluster_id",
                how="left",
            )
            .join(
                cluster_blocks.select(
                    pl.col("cluster_id"),
                    pl.col(f"{col0}_block_start").alias("block_left"),
                ),
                left_on="from",
                right_on="cluster_id",
                how="left",
            )
            .join(
                cluster_blocks.select(
                    pl.col("cluster_id"),
                    pl.col(f"{col1}_block_start").alias("block_right"),
                ),
                left_on="to",
                right_on="cluster_id",
                how="left",
            )
        )

        # compute the relative block positions
        this_flow = (
            this_flow.sort("from", "to")
            .with_columns(
                (
                    (pl.col(base_name).cumsum().over("from") - pl.col(base_name))
                    / pl.col("tot_left")
                ).alias("left_cumul"),
                (
                    (pl.col(base_name).cumsum().over("to") - pl.col(base_name))
                    / pl.col("tot_right")
                ).alias("right_cumul"),
            )
            .with_columns(
                (
                    pl.col("left_cumul") * pl.col("size_left") + pl.col("block_left")
                ).alias("left_below"),
                (
                    (pl.col("left_cumul") + pl.col(base_name) / pl.col("tot_left"))
                    * pl.col("size_left")
                    + pl.col("block_left")
                ).alias("left_above"),
                (
                    pl.col("right_cumul") * pl.col("size_right") + pl.col("block_right")
                ).alias("right_below"),
                (
                    (pl.col("right_cumul") + pl.col(base_name) / pl.col("tot_right"))
                    * pl.col("size_right")
                    + pl.col("block_right")
                ).alias("right_above"),
            )
            .select(
                "from",
                "to",
                (
                    pl.col(
                        "left_below", "left_above", "right_below", "right_above"
                    ).name.map(lambda cat: f"{base_name}_{cat}")
                ),
            )
        )

        flows = flows.join(this_flow, on=("from", "to"), how="outer")

    flows = flows.sort("from", "to")

    if export == "flows":
        return flows
    print("flows computed")

    # 4. patch flows to figure

    fsize = np.array([cluster_size.shape[1] * 10, cluster_size.shape[0]])
    fsize = fsize / fsize[0] * 100
    fig = plt.figure(figsize=fsize, dpi=180, layout="compressed")
    ax = fig.add_subplot()
    ax.spines[["top", "bottom", "left", "right"]].set_visible(False)

    n_flow = 60

    for i, column_base in enumerate(data.columns[1:]):
        base_name = column_base.split("_")[1]
        level_x = i * level_separation
        x_arr = np.linspace(level_x, level_x + level_separation, n_flow)
        flow_partition = (
            flows.select(
                sel.starts_with(base_name).name.map(lambda x: x.lstrip(base_name + "_"))
            )
            .drop_nulls()
            .to_dicts()
        )
        clrs = sns.color_palette("hls", len(flow_partition))

        for flow_part, clr in zip(flow_partition, clrs):
            line_below = bend_line(
                flow_part["left_below"], flow_part["right_below"], n_flow
            )
            line_above = bend_line(
                flow_part["left_above"], flow_part["right_above"], n_flow
            )
            ax.fill_between(x_arr, line_below, line_above, color=clr, alpha=0.6)

    print("figure constructed")

    if save_to is not None:
        fig.savefig(save_to, dpi=360)
    else:
        return fig, ax


def flow_frame(data: pl.DataFrame):
    """Construct the dataframe of the number of hospitals that move from one cluster to another at each snapshot time"""
    flows = pl.DataFrame(schema={"from": pl.Int64, "to": pl.Int64})

    for col0, col1 in pairwise(data.columns[1:]):
        base_name = col0.split("_")[1]
        this_flow = (
            data.select(col0, col1)
            .group_by(pl.all())
            .count()
            .rename({"count": base_name, col0: "from", col1: "to"})
            .drop_nulls()
        )
        flows = flows.join(this_flow, on=("from", "to"), how="outer")

    return flows.sort("from", "to")


def remap_flow_fromto(flows: pl.DataFrame):
    """Remaps the identifiers (from/to) of the flows to indexable values"""

    index_map = (
        flows.select(pl.col("from", "to"))
        .melt()
        .select(pl.col("value").unique())
        .with_row_count()
    )
    return (
        flows.join(index_map, left_on="from", right_on="value", how="left")
        .drop("from")
        .rename({"row_nr": "from"})
        .join(index_map, left_on="to", right_on="value", how="left")
        .drop("to")
        .rename({"row_nr": "to"})
    )


def export_flow_matrix(data: pl.DataFrame):
    """Constructs the 3D matrix of flows of hospitals between clusters"""
    flows = flow_frame(data)
    flows = remap_flow_fromto(flows)
    num_clus = (
        flows.select(pl.col("from", "to"))
        .melt()
        .select(pl.col("value").unique())
        .shape[0]
    )
    flow_mat = np.zeros((num_clus, num_clus, len(flows.columns) - 2))
    flow_indices = flows.select("from", "to").to_dict(as_series=False)
    flow_mat[flow_indices["from"], flow_indices["to"]] = (
        flows.drop("from", "to").fill_null(0).to_numpy()
    )
    return flow_mat


def construct_flow_anim(data: pl.DataFrame, early_stop=0):
    """Constructs an animation of the flow matrices

    Parameters
    ----------
    early_stop: number of frames to drop off the end of the animation

    """
    flow_matrix = export_flow_matrix(data)
    sizer = (
        (data.drop("name").max() - 1)
        .select(pl.all().name.map(lambda x: x.split("_")[1]))
        .to_numpy()
        .flatten()
    )

    fig, ax = plt.subplots()
    ims = []
    for ii, sz in enumerate(sizer[: -(1 + early_stop)]):
        A = flow_matrix[:sz, :sz, ii]
        im = ax.imshow(A, norm=colors.PowerNorm(0.5), cmap="inferno", animated=True)
        if ii == 0:
            ax.imshow(
                A,
                norm=colors.PowerNorm(0.5),
                cmap="inferno",
            )
        ims.append([im])

    return animation.ArtistAnimation(
        fig, ims, interval=50, blit=True, repeat_delay=1000
    )


def num_clus(data: pl.DataFrame):
    """returns number of clusters at each snapshot
    clusters indexed from 1, so no need to +1 to get values"""
    return data.max().drop("name").to_numpy().flatten()


def pivot_matrix(A: np.ndarray, copy=True):
    """Pivot the columns of a matrix A so that the dominant elements of the submatrix are along the diagonal

    if copy is True, returns a matrix of the same size that is pivoted, along with the pivot vector
    else, only returns the pivot vector"""

    assert np.allclose(*A.shape), "Input matrix not square"

    if copy:
        B_ = np.array(A)
        A, B_ = B_, A

    p = np.zeros((A.shape[0]), dtype=int)
    for i in range(A.shape[0]):
        p[i] = int(np.argmax(A[i, i:]) + i)
        A[:, [i, p[i]]] = A[:, [p[i], i]]

    if copy:
        return A, p
    return p


def matrix_diagonalness_coefficient(A: np.ndarray):
    """Compute the "diagonalness" of a matrix, based on the sample correlation coefficient

    Based on a stackexchange post: math.stackexchange.com/questions/1392491
    """

    assert np.allclose(*A.shape), "Input matrix not square"

    NN = A.shape[0]
    J = np.ones((NN, 1))
    R = np.arange(NN).reshape((NN, 1))
    R2 = R**2

    n = J.T @ A @ J
    Sx = R.T @ A @ J
    Sy = J.T @ A @ R
    Sx2 = R2.T @ A @ J
    Sy2 = J.T @ A @ R2
    Sxy = R.T @ A @ R

    return ((n * Sxy - Sx * Sy) / np.sqrt((n * Sx2 - Sx**2) * (n * Sy2 - Sy**2))).item()


def matrix_offdiag_coefficient(A: np.ndarray):
    """Compute the "diagonalness" of a matrix, based on the norm of the non-diagonal part

    Based off the same post as above: math.stackexchange.com/questions/1392491
    """

    assert np.allclose(*A.shape), "Input matrix not square"

    Z = A - np.diag(np.diag(A))
    return 1 - (np.linalg.norm(Z) / np.linalg.norm(A))


def matrix_proportion_diagonal_coefficient(A: np.ndarray, axis=0):
    """Compute the diagonalness based on the size of the off-diagonal components cf full matrix"""

    assert np.allclose(*A.shape), "Input matrix not square"

    return np.nanmean(np.diagonal(A) / np.sum(A, axis=axis))


def extract_slice(A: np.ndarray, ii: int, pivot=False):
    slc = A[:, :, ii]
    if pivot:
        return pivot_matrix(slc, copy=True)[0]
    else:
        return slc


_methods = {
    "corr": matrix_diagonalness_coefficient,
    "norm": matrix_offdiag_coefficient,
    "diag": matrix_proportion_diagonal_coefficient,
}


def cluster_coherency(flow_mat: np.ndarray, pivot=False, method="corr"):
    return [
        _methods[method](extract_slice(flow_mat, ii, pivot=pivot))
        for ii in range(flow_mat.shape[-1])
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument()


if __name__ == "__main__":
    dfx = pl.read_csv("flow_reweighted_both_14_365/cluster_by_name.csv")
    sankey_diagram(
        dfx, save_to="flow_reweighted_both_14_365/alluvial_diagram_clusters_14.png"
    )
