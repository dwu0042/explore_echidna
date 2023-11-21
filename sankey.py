import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
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

def sankey_diagram(
        data: pl.DataFrame,
        block_scale = 1.0,
        block_gap = 0.1,
        level_separation = 20.0,
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
        data
        .select(pl.exclude('name'))
        .melt(variable_name='snapshot')
        .group_by(pl.all()).count().drop_nulls()
        .pivot(values='count', index='value', columns='snapshot', aggregate_function=None)
        .rename({'value': 'cluster_id'}).sort('cluster_id')
        .with_columns(
            (pl.exclude('cluster_id') * block_scale * (1 + block_gap)).name.map(lambda x: f"{x}_size"),
        )
    )

    # 2. For each column compute where the sankey blocks start and end
    cluster_blocks = (
        cluster_size
        .with_columns(
            # cluster_block_end
            sel.ends_with('_size').cumsum().name.map(lambda x: f"{x.rstrip('_size')}_block_end"),
        )
        .with_columns(
            # cluster_block_start
            sel.ends_with('_block_end').shift(1, fill_value=0).name.map(lambda x: f"{x.strip('_end')}_start"),
        )
    )
    # 2a. mapping a separate dataframe to subtract concordant columns (very non-scalable)
    cluster_blocks_prime = cluster_size.with_columns(
        # subtract the gap
        (pl.all() * 0).name.map(lambda x: x),
        (pl.exclude('cluster_id', '^.*_size$') * block_scale * block_gap).name.map(lambda x: f"{x}_block_end"),
        (pl.exclude('cluster_id', '^.*_size$') * 0.0).name.map(lambda x: f"{x}_block_start")
    )

    cluster_blocks = (
        (cluster_blocks - cluster_blocks_prime)
        .select(
            'cluster_id',
            sel.ends_with('_block_start'),
            sel.ends_with('_block_end')
        )
    )

    # 2c. resize the block gap so that we can use it later
    cluster_size = (
        cluster_size
        .with_columns(pl.col('^.*_size$') / (1 + block_gap) )
    )

    print("blocks determined")

    # 3. generate the dataframe that has the amount of flow between clusters
    # also join so that we get the actual start and end of each flow to bend_line

    flows = pl.DataFrame(schema={'from': pl.Int64, 'to': pl.Int64})

    for col0, col1 in pairwise(data.columns[1:]):
        base_name = col0.split('_')[1]
        this_flow = (
            data.select(col0, col1)
            .group_by(pl.all()).count()
            .rename({
                'count': base_name,
                col0: 'from',
                col1: 'to'
            })
            .drop_nulls()
            # attach size_left and size_right
            .join(
                cluster_size.select(
                    pl.col('cluster_id'),
                    pl.col(col0).alias('tot_left'),
                    pl.col(f'{col0}_size').alias('size_left'),
                ),
                left_on='from',
                right_on='cluster_id',
                how='left',
            )
            .join(
                cluster_size.select(
                    pl.col('cluster_id'),
                    pl.col(col1).alias('tot_right'),
                    pl.col(f'{col1}_size').alias('size_right'),
                ),
                left_on='to',
                right_on='cluster_id',
                how='left',
            )
            .join(
                cluster_blocks.select(
                    pl.col('cluster_id'),
                    pl.col(f'{col0}_block_start').alias('block_left'),
                ),
                left_on='from',
                right_on='cluster_id',
                how='left',
            )
            .join(
                cluster_blocks.select(
                    pl.col('cluster_id'),
                    pl.col(f'{col1}_block_start').alias('block_right'),
                ),
                left_on='to',
                right_on='cluster_id',
                how='left',
            )
        )

        # compute the relative block positions
        this_flow = (
            this_flow
            .sort('from', 'to')
            .with_columns(
                (
                    (pl.col(base_name).cumsum().over('from') - pl.col(base_name)) 
                    / pl.col('tot_left')
                ).alias('left_cumul'),
                (
                    (pl.col(base_name).cumsum().over('to') - pl.col(base_name)) 
                    / pl.col('tot_right')
                ).alias('right_cumul'),
            )
            .with_columns(
                (pl.col('left_cumul') * pl.col('size_left') + pl.col('block_left')).alias('left_below'),
                ((pl.col('left_cumul') + pl.col(base_name)/pl.col('tot_left')) * pl.col('size_left') + pl.col('block_left')).alias('left_above'),
                (pl.col('right_cumul') * pl.col('size_right') + pl.col('block_right')).alias('right_below'),
                ((pl.col('right_cumul') + pl.col(base_name)/pl.col('tot_right')) * pl.col('size_right') + pl.col('block_right')).alias('right_above'),
            )
            .select(
                'from', 'to',
                (pl.col('left_below', 'left_above', 'right_below', 'right_above')
                 .name.map(lambda cat: f"{base_name}_{cat}"))
            )
        )

        flows = flows.join(this_flow, on=('from','to'), how='outer')

    flows = flows.sort('from', 'to')

    print("flows computed")

    # 4. patch flows to figure

    fsize = np.array([cluster_size.shape[1] * 10, cluster_size.shape[0]])
    fsize = fsize / fsize[0] * 100
    fig = plt.figure(figsize=fsize, dpi=180, layout='compressed')
    ax = fig.add_subplot()
    ax.spines[['top', 'bottom', 'left', 'right']].set_visible(False)

    n_flow = 60

    for i, column_base in enumerate(data.columns[1:]):
        base_name = column_base.split("_")[1]
        level_x = i * level_separation
        x_arr = np.linspace(level_x, level_x + level_separation, n_flow)
        flow_partition = flows.select(sel.starts_with(base_name).name.map(lambda x: x.lstrip(base_name+'_'))).drop_nulls().to_dicts()
        clrs = sns.color_palette('hls', len(flow_partition))

        for flow_part, clr in zip(flow_partition, clrs):
            line_below = bend_line(flow_part['left_below'], flow_part['right_below'], n_flow)
            line_above = bend_line(flow_part['left_above'], flow_part['right_above'], n_flow)
            ax.fill_between(x_arr, line_below, line_above, color=clr, alpha=0.6)

    print("figure constructed")

    fig.savefig('test.pdf')

    return fig, ax

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

if __name__ == "__main__":
    dfx = pl.read_csv("flow_reweighted_14/cluster_by_name.csv")
    sankey_diagram(dfx)