import numpy as np
from matplotlib import pyplot as plt
import polars as pl
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

def sank_blocks(block_sizes_map, relative_gap=1.0):
    """ Return an array representing the blocks of a level of a Sankey diagram
    """

    for name in block_sizes_map:


    return blocks



def sankey_multiflow(data: pl.DataFrame):
    pass

    for in pairwise()
    data.group_by()



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