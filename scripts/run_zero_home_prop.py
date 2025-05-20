import functools
import numpy as np
from scipy import linalg, stats
from matplotlib import pyplot as plt
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
    fig, axs = plt.subplots(2, 2, figsize=[16, 9])
    for ax, (k, hmv) in zip(axs.flatten(), hms.items()):
        sns.heatmap(data=hmv[::-1,:], vmax=vmax, ax=ax, cbar=False)
        ax.set_title(k)
    fig.colorbar(ax.collections[0], ax=axs, label='Count', extend='max')
    fig.suptitle('Number at home')

    return fig



if __name__ == "__main__":
    gather()