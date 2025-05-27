from collections import defaultdict
import functools
from typing import Mapping
import numpy as np
from scipy import linalg, stats
from matplotlib import pyplot as plt
from matplotlib import dates as mpldates
import seaborn as sns
import polars as pl
import h5py
from pathlib import Path

# arrays are NxM, (observations are rows, columns are time)


root = Path(__file__).resolve().parent.parent.resolve()

def gather(write=True):
    histories = dict()
    for dir in (root / "simulations/zero_sims_resized/" ).iterdir():
        model = dir.stem
        fl = dir / "sim_all_30s.h5"
        with h5py.File(fl) as fp:
            grps = list(fp)
            presence = [fp[grp]['history'][:].sum(axis=0) for grp in grps]
        histories[model] = np.vstack(presence)
        if write:
            write_to_file = dir / "hosp_presence.csv"
            np.savetxt(write_to_file, histories[model], delimiter=',')
            print(f"written to {write_to_file}")
    return histories

def load():
    histories = {
        dir.stem: np.loadtxt(dir/"hosp_presence.csv", delimiter=',')
        for dir in (root / "simulations/zero_sims_resized/" ).iterdir()
    }

    return histories

_BASE_DATE = np.datetime64('2011-01', 'M')
_MAX_DATE = np.datetime64('2018-12', 'M')
def set_xdatelabels(hist, ax: plt.Axes, spacing=12, dt=1.0):
    # here, we want to get succinct date formats
    # we map the times onto the timedeltas of the simulations
    N = hist.shape[1]
    emp_max_date = (np.timedelta64(int(N*dt), 'D') + _BASE_DATE).astype('datetime64[M]')
    
    ax_max_date = np.min([emp_max_date, _MAX_DATE])

    dates = np.arange(_BASE_DATE, ax_max_date, np.timedelta64(spacing, 'M'))

    label_pos = (dates.astype('datetime64[D]').astype('int64') - _BASE_DATE.astype('datetime64[D]').astype('int64')) / dt

    ax.set_xticks(label_pos, dates.astype('datetime64[Y]'), rotation=45)

    return ax.get_xticks()

def heatmap(hist, nmax=30):
    # hist is 2D
    bins = np.arange(nmax+2)
    hist_heat = np.array([
        np.histogram(hist[:, i], bins=bins)[0]
        for i in range(hist.shape[1])
    ]).T

    return bins, hist_heat

@functools.wraps(heatmap)
def plot_heatmap(*args, **kwargs):
    bins, hist_heat = heatmap(*args, **kwargs)

    ax = sns.heatmap(hist_heat)

    return ax

def mega_heatmap(hists, vmax=1500):
    hms = {model: heatmap(hist_vals)[1] for model, hist_vals in hists.items()}
    fig, axs = plt.subplots(nrows=4, figsize=[16, 9], sharex=True)
    for ax, (k, hmv) in zip(axs.flatten(), hms.items()):
        if k == "temporal": hmv=hmv[:,::2]
        sns.heatmap(data=hmv[::-1,:], vmax=vmax, ax=ax, cmap='inferno', cbar=False)
        ax.set_title(k)
        ax.set_ylim([-0.5, 30.5])
    set_xdatelabels(hmv, ax)
    fig.subplots_adjust(hspace=0.4)
    fig.colorbar(ax.collections[0], ax=axs, label='Count', extend='max')

    return fig

if __name__ == "__main__":
    gather()