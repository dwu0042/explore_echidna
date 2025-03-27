import numpy as np
from matplotlib import pyplot as plt, colors, cm
import seaborn as sns
import itertools

import do_simulation_on_network as dosim

def import_sim(sim_path, size_path):
    sim_data = dosim.import_batched_realisations(sim_path)
    sizes = dosim.find_sizes(size_path, None)
    size_arr = [sz for h,sz in sorted(sizes.items())]
    sim_data['sizes'] = sizes
    sim_data['size_arr'] = size_arr

    return sim_data

def plot_sim_timeseries(sim, ax=None, cap=365, **plot_args):
    
    if ax is None:
        _, ax = plt.subplots(layout='constrained')
    ax.plot(sim.ts[:cap], sim.history.T[:cap,:], **plot_args)
    return ax

def default_plot_batch(ssims, grid=None, cap=365, lw=0.7, alpha=0.5):

    with sns.plotting_context('talk'):
        if grid is not None:
            fig, axs = plt.subplots(
                *grid, 
                figsize=(14, 9), 
                sharex=True, sharey=True,
                squeeze=True, layout='constrained'
            )
            axs = axs.flatten()
            fig.supxlabel("Time (days)")
            fig.supylabel("Prevalence (per hospital)")
        else:
            axs = itertools.cycle([None])
        for sim, ax_ in zip(ssims['records'], axs):
            ax = plot_sim_timeseries(sim, cap=cap, ax=ax_, lw=lw, alpha=alpha)
            if ax_ is None:
                ax.set_xlabel('Time (days)')
                ax.set_ylabel("Prevalence (per hospital)") 
        


def done():
    plt.close('all')