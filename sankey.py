import numpy as np
from matplotlib import pyplot as plt
import polars as pl
from polars import selectors as sel
from itertools import pairwise

def bend_line(y_from: float, y_to: float, n_grid: int=100):
    """ Return an array that represents a bent line
    (from y_from to y_to) with lenght n_grid
    """

    N = int(n_grid // 3)
    M = n_grid + 2*N - 2
    Z = int(M // 2)

    line_base = np.array(Z*[y_from] + Z*[y_to])
    window = 1/N * np.ones(N)
    line = np.convolve(line_base, window, mode='valid')
    line = np.convolve(line, window, mode='valid')

    return line

def sank_blocks(cluster_value_counts: pl.DataFrame, scale=1.0, gap=0.02):
    """ Return an array representing the blocks of a level of a Sankey diagram

    The scale parameter is the size of a 1-hospital cluster. 
    The gap parameter defines the relative size of the gap on either side of a block, as compared to the block itself
    """

    cluster_freqs = (
        cluster_value_counts
        .rename({
            col: ('counts' if col != 'cluster_id' else 'cluster_id') for col in cluster_value_counts.columns
        })
        .drop_nulls()
        .sort('counts', descending=True)
    )

    cluster_blocks = (
        cluster_freqs.lazy()
        .with_columns(
            block_size=(1+gap) * scale * pl.col('counts'),
            block_gap=gap * scale * pl.col('counts'),
        )
        .with_columns(
            block_end=pl.col('block_size').cumsum(),
        )
        .with_columns(
            block_start=pl.col('block_end').shift_and_fill(0),
            block_end=(pl.col('block_end') - pl.col('block_gap')),
            block_size=(pl.col('block_size') - pl.col('block_gap')),
        )
        .select(
            cluster_col.name,
            'block_start',
            'block_end'
        )
        .collect()
    )

    return cluster_blocks



# def sankey_multiflow(data: pl.DataFrame):
#     pass

#     for in pairwise()
#     data.group_by()

def sankey_diagram(
        data: pl.DataFrame,
        block_scale = 1.0,
        block_gap = 0.02
    ):
    """
    Creates a sankey diagram
    data should be structured as 

    | name | location_1 | location_2 | location_3 | ...

    where named agents move from location_1 to location_2, for example.
    A location_* value can be Null (None), in which case they are not part of the system at that point in time

    We will return a matplotlib.pyplot Figure instance that contains a single Axes that has the Sankey diagram.
    """

    # Build block cache for each location
    # 1. Determine the size at each column
    cluster_size = (
        data
        .select(pl.exclude('name'))
        .melt(variable_name='snapshot')
        .group_by(pl.all()).count().drop_nulls()
        .pivot(values='count', index='value', columns='snapshot', aggregate_function=None)
        .rename({'value': 'cluster_id'}).sort('cluster_id')
    )
    # 2. For each column generate a DataFrame that will map to block sizes
    cluster_blocks = (
        cluster_size
        .with_columns(
            (pl.exclude('cluster_id') * block_scale).name.map(lambda x: f"{x}_size"),
            (pl.exclude('cluster_id') * block_scale * block_gap).name.map(lambda x: f"{x}_gap"),
        )
        .with_columns(
            sel.ends_with('_size').cumsum().name.map(lambda x: f"{x.rstrip('_size')}_block_end"),
        )
        .with_columns(
            sel.ends_with('_block_end').shift(1, fill_value=0).name.map(lambda x: f"{x.strip('_end')}_start"),
            sel.ends_with('_block_end') # need to subtract concordant cols, can't do that, so need diff method, prob using cluster_id as a constasnt index.
        )
    )
    pass


# plan of attack
"""
we have the dataframe df
we loop by column <- this should be parallelisable
for each column extract value counts
    these value counts are then passed into sank_blocks to determine their position and size
    place these into a blocks df
for each pair of columns
    compute the flows using a group_by .sum
    for each flow 
        compute the limits of the flow using the blocks df lookup
        place into new flows df
for each flow
    generate the flow line using bend_line
    plot using fill_between
for each item in blocks df
    fill in the area with a fill_between
return plot

data structures

IN: data ||> hosp_name | cluster_0 | cluster_1 | cluster_2 | ...

OUT: blocks ||> cluster_id | x_loc | y_min | y_max |
OUT: flows  ||> cluster_from | cluster_to | flow_value
OUT: plot ~> sankey diagram
"""